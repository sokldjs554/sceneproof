package uplink

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"google.golang.org/grpc"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
	"github.com/sokldjs554/sceneproof/gateway/internal/status"
)

// DefaultChunkSize is the UploadEvidence chunk size (256 KiB): large enough
// to amortise per-message overhead, small enough to stay well under gRPC's
// 4 MiB default message limit and to keep memory flat on the device.
const DefaultChunkSize = 256 * 1024

// UploaderOptions configures the evidence uploader.
type UploaderOptions struct {
	Conn   *Conn
	Status *status.Registry
	Ledger *Ledger
	Logger *slog.Logger
	// Root is the evidence directory shared with the analyzer; paths in
	// events are relative to it.
	Root string
	// RetryWindow is how long a missing file is waited for (default 10m).
	RetryWindow time.Duration
	// SettleTime: a file whose mtime is younger than this is considered
	// still being written and is retried shortly (default 1s). The clip is
	// written after the CLOSED revision is acked; uploading it mid-write
	// would hand the server a truncated mp4.
	SettleTime time.Duration
	// ChunkSize for the stream (default DefaultChunkSize).
	ChunkSize int
	// RPCTimeout bounds one whole upload (default 2m).
	RPCTimeout time.Duration
	// Backoff between attempts on one file.
	Backoff BackoffConfig
}

type uploadJob struct {
	eventID, path string
	first, next   time.Time
	bo            *Backoff
}

// Uploader uploads evidence files referenced by acknowledged events.
//
// Jobs live in memory (the ledger makes them survive restarts). Each job is
// retried on its own backoff within RetryWindow because files appear at
// different times: key frames exist when the entry revision is acked, the
// clip only a few seconds after the close. After the window the file is
// given up on, logged, and marked in the ledger so it is not revived.
type Uploader struct {
	o   UploaderOptions
	log *slog.Logger

	mu   sync.Mutex
	jobs map[string]*uploadJob // ledgerKey → job
	wake chan struct{}
}

// NewUploader creates an uploader and re-schedules the ledger's outstanding
// files. Call Run to start it.
func NewUploader(o UploaderOptions) *Uploader {
	if o.RetryWindow <= 0 {
		o.RetryWindow = 10 * time.Minute
	}
	if o.SettleTime <= 0 {
		o.SettleTime = time.Second
	}
	if o.ChunkSize <= 0 {
		o.ChunkSize = DefaultChunkSize
	}
	if o.RPCTimeout <= 0 {
		o.RPCTimeout = 2 * time.Minute
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	u := &Uploader{o: o, log: o.Logger.With("component", "uploader"), jobs: map[string]*uploadJob{}, wake: make(chan struct{}, 1)}
	for _, w := range o.Ledger.Outstanding() {
		u.add(w[0], w[1])
	}
	return u
}

// Schedule queues every evidence file referenced by ev that the ledger does
// not already have.
func (u *Uploader) Schedule(ev *sceneproofv1.Event) {
	evd := ev.GetEvidence()
	if evd == nil {
		return
	}
	for _, kf := range evd.GetKeyFrames() {
		if p := kf.GetPath(); p != "" {
			u.add(ev.GetEventId(), p)
		}
	}
	if p := evd.GetClipPath(); p != "" {
		u.add(ev.GetEventId(), p)
	}
}

func (u *Uploader) add(eventID, path string) {
	if u.o.Ledger.Has(eventID, path) {
		return
	}
	if !filepath.IsLocal(path) {
		// A path that escapes the evidence root is never read: the analyzer
		// is trusted, but the shared directory is not a reason to turn the
		// gateway into a file exfiltration tool.
		u.log.Error("refusing evidence path outside the evidence root", "event_id", eventID, "path", path)
		return
	}
	k := ledgerKey(eventID, path)
	u.mu.Lock()
	if _, ok := u.jobs[k]; ok {
		u.mu.Unlock()
		return
	}
	now := time.Now()
	u.jobs[k] = &uploadJob{eventID: eventID, path: path, first: now, next: now, bo: NewBackoff(u.o.Backoff)}
	n := len(u.jobs)
	u.mu.Unlock()
	if err := u.o.Ledger.Want(eventID, path); err != nil {
		u.log.Warn("ledger write failed", "error", err.Error())
	}
	u.o.Status.SetUploadsPending(n)
	select {
	case u.wake <- struct{}{}:
	default:
	}
}

// Pending returns the number of files waiting.
func (u *Uploader) Pending() int {
	u.mu.Lock()
	defer u.mu.Unlock()
	return len(u.jobs)
}

func (u *Uploader) remove(j *uploadJob) {
	u.mu.Lock()
	delete(u.jobs, ledgerKey(j.eventID, j.path))
	n := len(u.jobs)
	u.mu.Unlock()
	u.o.Status.SetUploadsPending(n)
}

// nextDue returns the earliest due job (or nil) and how long until the next
// one is due.
func (u *Uploader) nextDue(now time.Time) (*uploadJob, time.Duration) {
	u.mu.Lock()
	defer u.mu.Unlock()
	var best *uploadJob
	wait := time.Minute
	for _, j := range u.jobs {
		if !j.next.After(now) {
			if best == nil || j.next.Before(best.next) {
				best = j
			}
			continue
		}
		if d := j.next.Sub(now); d < wait {
			wait = d
		}
	}
	return best, wait
}

// Run processes upload jobs until ctx ends.
func (u *Uploader) Run(ctx context.Context) {
	for ctx.Err() == nil {
		job, wait := u.nextDue(time.Now())
		if job == nil {
			t := time.NewTimer(wait)
			select {
			case <-ctx.Done():
				t.Stop()
				return
			case <-u.wake:
				t.Stop()
			case <-t.C:
			}
			continue
		}
		u.process(ctx, job)
	}
}

func (u *Uploader) process(ctx context.Context, j *uploadJob) {
	log := u.log.With("event_id", j.eventID, "path", j.path)
	full := filepath.Join(u.o.Root, j.path)
	st, err := os.Stat(full)
	now := time.Now()
	switch {
	case err != nil && errors.Is(err, os.ErrNotExist):
		if now.Sub(j.first) > u.o.RetryWindow {
			log.Error("evidence file never appeared; giving up", "waited", now.Sub(j.first).Round(time.Second).String())
			u.giveUp(j)
			return
		}
		j.next = now.Add(j.bo.Next())
		return
	case err != nil:
		log.Warn("stat evidence file", "error", err.Error())
		j.next = now.Add(j.bo.Next())
		return
	case st.IsDir():
		log.Error("evidence path is a directory; giving up")
		u.giveUp(j)
		return
	case now.Sub(st.ModTime()) < u.o.SettleTime:
		// Still being written; look again soon without burning backoff.
		j.next = now.Add(u.o.SettleTime)
		return
	}

	cc, gen, err := u.o.Conn.Acquire(ctx)
	if err != nil {
		return
	}
	res, sha, err := u.upload(ctx, cc, j, full, st.Size())
	if err != nil {
		if ctx.Err() != nil {
			return
		}
		if IsTransportError(err) {
			u.o.Conn.Fail(gen, err)
			j.next = time.Now().Add(500 * time.Millisecond) // Acquire does the real waiting
			return
		}
		j.next = time.Now().Add(j.bo.Next())
		log.Warn("upload failed; will retry", "error", err.Error())
		if time.Since(j.first) > u.o.RetryWindow {
			log.Error("upload kept failing past the retry window; giving up")
			u.giveUp(j)
		}
		return
	}
	if !res.GetOk() && !res.GetAlreadyPresent() {
		j.next = time.Now().Add(j.bo.Next())
		log.Warn("server rejected upload; will retry", "message", res.GetMessage())
		if time.Since(j.first) > u.o.RetryWindow {
			u.giveUp(j)
		}
		return
	}
	if err := u.o.Ledger.Done(j.eventID, j.path, sha); err != nil {
		log.Warn("ledger write failed", "error", err.Error())
	}
	u.remove(j)
	u.o.Status.IncUploadsDone()
	log.Info("evidence uploaded", "bytes", st.Size(), "sha256", sha, "already_present", res.GetAlreadyPresent())
}

func (u *Uploader) giveUp(j *uploadJob) {
	if err := u.o.Ledger.Forget(j.eventID, j.path); err != nil {
		u.log.Warn("ledger write failed", "error", err.Error())
	}
	u.remove(j)
	u.o.Status.IncUploadsFailed()
}

// upload streams exactly size bytes of the file in ChunkSize pieces. The
// digest is computed over the same bytes that are sent, and the byte count
// is fixed up front, so a file that grows during the upload (it should not;
// see SettleTime) still yields a self-consistent (sha256, total_size) pair.
// An empty file is sent as one empty chunk with last=true.
func (u *Uploader) upload(ctx context.Context, cc *grpc.ClientConn, j *uploadJob, full string, size int64) (*sceneproofv1.UploadResult, string, error) {
	f, err := os.Open(full)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()

	rctx, cancel := context.WithTimeout(ctx, u.o.RPCTimeout)
	defer cancel()
	stream, err := sceneproofv1.NewIngestClient(cc).UploadEvidence(rctx)
	if err != nil {
		return nil, "", err
	}
	h := sha256.New()
	buf := make([]byte, u.o.ChunkSize)
	var offset int64
	for {
		remaining := size - offset
		if remaining < 0 {
			remaining = 0
		}
		want := int64(len(buf))
		if remaining < want {
			want = remaining
		}
		n, rerr := io.ReadFull(f, buf[:want])
		if rerr != nil && !errors.Is(rerr, io.EOF) && !errors.Is(rerr, io.ErrUnexpectedEOF) {
			return nil, "", rerr
		}
		if int64(n) < want {
			// The file shrank under us; fail this attempt and let the retry
			// re-stat it.
			return nil, "", fmt.Errorf("evidence file shrank while uploading (%d < %d)", offset+int64(n), size)
		}
		h.Write(buf[:n])
		chunk := &sceneproofv1.EvidenceChunk{
			EventId: j.eventID,
			Path:    j.path,
			Data:    buf[:n],
			Offset:  offset,
		}
		offset += int64(n)
		last := offset >= size
		if last {
			chunk.Last = true
			chunk.Sha256 = hex.EncodeToString(h.Sum(nil))
			chunk.TotalSize = size
		}
		if err := stream.Send(chunk); err != nil {
			// Send returning an error means the server already ended the
			// stream; the real status comes from CloseAndRecv.
			break
		}
		if last {
			break
		}
	}
	res, err := stream.CloseAndRecv()
	if err != nil {
		return nil, "", err
	}
	return res, hex.EncodeToString(h.Sum(nil)), nil
}
