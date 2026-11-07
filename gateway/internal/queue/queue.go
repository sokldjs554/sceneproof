// Package queue implements the gateway's durable, bounded, coalescing event
// queue.
//
// Why a write-ahead log and not a database: the gateway runs on an edge
// device next to the analyzer. It needs exactly one property from storage:
// an Event that was answered with accepted=true must survive a crash or a
// power cut until the server has acknowledged it. An append-only log gives
// that with one write and one fsync per event, no schema, no background
// threads and a recovery procedure that is a single sequential read.
//
// Model:
//
//   - An Event is a complete snapshot of a zone episode; a higher revision
//     supersedes every lower one (see sceneproof.proto). The queue therefore
//     keeps at most one pending record per event_id (the newest revision) and
//     coalesces: a newer revision replaces the pending one in place, keeping
//     the FIFO position of the first enqueue so an episode cannot starve
//     behind events that arrived after it.
//   - Pending() yields events in first-enqueue order. Ack(event_id, revision)
//     removes the pending record only if revision >= the pending revision:
//     when a newer revision arrived while the older one was in flight, the
//     newer one must still be sent.
//   - The queue is bounded by live payload bytes. When the budget is
//     exceeded the oldest pending events are evicted (and counted) because on
//     a device that has been offline for a long time the most recent episodes
//     are the ones an operator still cares about; the oldest ones are the
//     most likely to be stale by the time the uplink returns.
//   - The on-disk file only grows while events are in flight. It is rewritten
//     with just the live records (compacted) when it is more than twice the
//     live payload size, and on startup when it contains dead records.
package queue

import (
	"container/list"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	"google.golang.org/protobuf/proto"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
)

const (
	// FileName is the WAL file inside Options.Dir.
	FileName = "events.wal"

	defaultMaxBytes        = 64 << 20
	defaultCompactMinBytes = 1 << 20
)

var (
	// ErrClosed is returned by every operation after Close.
	ErrClosed = errors.New("queue: closed")
	// ErrTooLarge is returned by Put when a single event exceeds the whole
	// budget. Accepting it would mean evicting everything else and still not
	// fitting, so the caller (the analyzer) is told instead of being lied to.
	ErrTooLarge = errors.New("queue: event larger than queue capacity")
	// ErrInvalid is returned for events without an event_id.
	ErrInvalid = errors.New("queue: event has no event_id")
)

// Options configures Open.
type Options struct {
	// Dir holds the WAL file; it is created if missing.
	Dir string
	// MaxBytes bounds the sum of pending payload bytes. 0 means 64 MiB.
	MaxBytes int64
	// DisableFsync skips fsync after PUT records. The zero value is the safe
	// choice: a PUT is what backs the accepted=true answer to the analyzer,
	// so by default it is forced to disk before that answer is sent. ACK and
	// DROP records are never fsynced: losing one only causes a re-send, and
	// the server deduplicates by (event_id, revision).
	DisableFsync bool
	// CompactMinBytes is the file size below which automatic compaction is
	// not attempted, so a tiny file is not rewritten after every ack.
	// 0 means 1 MiB. Compact() ignores it.
	CompactMinBytes int64
	// Logger receives recovery, eviction and compaction messages.
	Logger *slog.Logger
}

// PutResult describes what Put did with an event.
type PutResult struct {
	// Stored is true when the event is now pending (new or replacing an
	// older revision) and its PUT record has been written.
	Stored bool
	// Duplicate is true when a revision >= the submitted one was already
	// pending, so nothing was written.
	Duplicate bool
	// Evicted is the number of older events dropped to make room.
	Evicted int
}

// Stats is a point-in-time view used by /status, /metrics and UplinkStatus.
type Stats struct {
	Pending     int
	LiveBytes   int64 // payload bytes of pending events
	FileBytes   int64 // size of the WAL file on disk
	MaxBytes    int64
	Dropped     int64 // evicted since Open (not persisted across restarts)
	Compactions int64
	Truncated   bool // recovery found a corrupt tail and cut it off
}

type entry struct {
	ev   *sceneproofv1.Event
	size int64
	elem *list.Element
}

// Queue is safe for concurrent use.
type Queue struct {
	opts Options
	log  *slog.Logger
	path string

	mu          sync.Mutex
	f           *os.File // append handle; nil after Close
	fileBytes   int64
	liveBytes   int64
	index       map[string]*entry
	order       *list.List // *entry in first-enqueue order
	dropped     int64
	compactions int64
	truncated   bool
	closed      bool

	wake chan struct{}
}

// Open recovers the queue from Dir (creating it if needed), truncates a
// corrupt tail, applies the capacity bound and compacts if the file holds
// dead records.
func Open(opts Options) (*Queue, error) {
	if opts.MaxBytes <= 0 {
		opts.MaxBytes = defaultMaxBytes
	}
	if opts.CompactMinBytes <= 0 {
		opts.CompactMinBytes = defaultCompactMinBytes
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	if err := os.MkdirAll(opts.Dir, 0o755); err != nil {
		return nil, fmt.Errorf("queue: create dir: %w", err)
	}
	q := &Queue{
		opts:  opts,
		log:   opts.Logger.With("component", "queue"),
		path:  filepath.Join(opts.Dir, FileName),
		index: make(map[string]*entry),
		order: list.New(),
		wake:  make(chan struct{}, 1),
	}
	if err := q.recover(); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(q.path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("queue: open wal: %w", err)
	}
	q.f = f
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	q.fileBytes = st.Size()

	q.mu.Lock()
	defer q.mu.Unlock()
	// The budget may have been lowered since the last run.
	if n := q.evictLocked(); n > 0 {
		q.log.Warn("evicted events on startup to honour queue budget", "evicted", n)
	}
	if q.fileBytes > q.liveFrameBytesLocked() {
		if err := q.compactLocked(); err != nil {
			f.Close()
			return nil, err
		}
	}
	q.log.Info("queue opened", "path", q.path, "pending", q.order.Len(),
		"live_bytes", q.liveBytes, "file_bytes", q.fileBytes, "max_bytes", q.opts.MaxBytes)
	return q, nil
}

// recover replays the WAL into memory. It stops at the first bad record,
// logs it and truncates the file there: everything before it was fully
// written and checksummed, everything after it is a torn write or garbage,
// and there is no safe way to resynchronise inside a length-prefixed stream.
func (q *Queue) recover() error {
	if _, err := os.Stat(q.path); errors.Is(err, os.ErrNotExist) {
		return nil
	}
	bad := 0
	good, err := ScanFile(q.path, func(typ byte, payload []byte) error {
		switch typ {
		case recPut:
			ev := &sceneproofv1.Event{}
			if uerr := proto.Unmarshal(payload, ev); uerr != nil || ev.GetEventId() == "" {
				// A checksummed record with an unparsable body is not a torn
				// tail (the CRC matched); skip it rather than throw away the
				// valid records behind it.
				bad++
				return nil
			}
			q.applyPutLocked(ev, int64(len(payload)))
		case recAck, recDrop:
			id, rev, kerr := decodeKey(payload)
			if kerr != nil {
				bad++
				return nil
			}
			q.applyAckLocked(id, rev)
		default:
			bad++
		}
		return nil
	})
	if err != nil && !errors.Is(err, ErrCorrupt) {
		return fmt.Errorf("queue: recover: %w", err)
	}
	if err != nil {
		q.log.Warn("wal has a corrupt tail; truncating", "offset", good, "error", err.Error())
		if terr := os.Truncate(q.path, good); terr != nil {
			return fmt.Errorf("queue: truncate corrupt tail: %w", terr)
		}
		q.truncated = true
	}
	if bad > 0 {
		q.log.Warn("skipped unparsable wal records", "count", bad)
	}
	return nil
}

// Put stores ev (coalescing with a pending older revision) and returns once
// its PUT record is on disk. Only after Put returns may the caller tell the
// analyzer accepted=true.
func (q *Queue) Put(ev *sceneproofv1.Event) (PutResult, error) {
	if ev.GetEventId() == "" {
		return PutResult{}, ErrInvalid
	}
	payload, err := proto.Marshal(ev)
	if err != nil {
		return PutResult{}, fmt.Errorf("queue: marshal: %w", err)
	}
	size := int64(len(payload))

	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return PutResult{}, ErrClosed
	}
	if size > q.opts.MaxBytes {
		return PutResult{}, ErrTooLarge
	}
	if cur, ok := q.index[ev.EventId]; ok && cur.ev.GetRevision() >= ev.GetRevision() {
		return PutResult{Duplicate: true}, nil
	}
	if err := q.write(recPut, payload, !q.opts.DisableFsync); err != nil {
		return PutResult{}, err
	}
	q.applyPutLocked(ev, size)
	res := PutResult{Stored: true}
	res.Evicted = q.evictLocked()
	q.maybeCompactLocked()
	select {
	case q.wake <- struct{}{}:
	default:
	}
	return res, nil
}

// applyPutLocked updates the in-memory index; used by Put and by recovery.
func (q *Queue) applyPutLocked(ev *sceneproofv1.Event, size int64) {
	if cur, ok := q.index[ev.EventId]; ok {
		if cur.ev.GetRevision() >= ev.GetRevision() {
			return
		}
		// Coalesce: newer snapshot, same FIFO slot.
		q.liveBytes += size - cur.size
		cur.ev, cur.size = ev, size
		return
	}
	e := &entry{ev: ev, size: size}
	e.elem = q.order.PushBack(e)
	q.index[ev.EventId] = e
	q.liveBytes += size
}

// Ack records that the server holds revision (accepted or duplicate). The
// pending record is removed only when revision >= the pending one, because a
// newer snapshot that arrived during the round trip still has to be sent.
// It reports whether a pending record was removed.
func (q *Queue) Ack(eventID string, revision int32) (bool, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return false, ErrClosed
	}
	cur, ok := q.index[eventID]
	if !ok || cur.ev.GetRevision() > revision {
		return false, nil
	}
	if err := q.write(recAck, encodeKey(eventID, revision), false); err != nil {
		return false, err
	}
	q.removeLocked(cur)
	q.maybeCompactLocked()
	return true, nil
}

func (q *Queue) applyAckLocked(eventID string, revision int32) {
	if cur, ok := q.index[eventID]; ok && cur.ev.GetRevision() <= revision {
		q.removeLocked(cur)
	}
}

func (q *Queue) removeLocked(e *entry) {
	q.order.Remove(e.elem)
	delete(q.index, e.ev.EventId)
	q.liveBytes -= e.size
}

// evictLocked drops the oldest pending events until the live bytes fit the
// budget. Oldest-first is deliberate (see the package comment): after a long
// outage the newest episodes are the ones still worth reviewing.
func (q *Queue) evictLocked() int {
	n := 0
	for q.liveBytes > q.opts.MaxBytes && q.order.Len() > 0 {
		victim := q.order.Front().Value.(*entry)
		if q.f != nil {
			if err := q.write(recDrop, encodeKey(victim.ev.EventId, victim.ev.GetRevision()), false); err != nil {
				q.log.Error("write DROP record", "error", err)
			}
		}
		q.removeLocked(victim)
		q.dropped++
		n++
		q.log.Warn("queue over budget; evicted oldest event",
			"event_id", victim.ev.EventId, "revision", victim.ev.GetRevision(),
			"live_bytes", q.liveBytes, "max_bytes", q.opts.MaxBytes)
	}
	return n
}

// Pending returns the pending events in first-enqueue order (newest revision
// of each). The returned events are shared with the queue and must be
// treated as read-only.
func (q *Queue) Pending() []*sceneproofv1.Event {
	q.mu.Lock()
	defer q.mu.Unlock()
	out := make([]*sceneproofv1.Event, 0, q.order.Len())
	for el := q.order.Front(); el != nil; el = el.Next() {
		out = append(out, el.Value.(*entry).ev)
	}
	return out
}

// Len returns the number of pending events.
func (q *Queue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.order.Len()
}

// Wake returns a channel that receives a value after a Put stored an event.
// It is buffered with capacity one; consumers drain it and then call Pending.
func (q *Queue) Wake() <-chan struct{} { return q.wake }

// Stats returns a snapshot of the queue's counters.
func (q *Queue) Stats() Stats {
	q.mu.Lock()
	defer q.mu.Unlock()
	return Stats{
		Pending:     q.order.Len(),
		LiveBytes:   q.liveBytes,
		FileBytes:   q.fileBytes,
		MaxBytes:    q.opts.MaxBytes,
		Dropped:     q.dropped,
		Compactions: q.compactions,
		Truncated:   q.truncated,
	}
}

// write appends one framed record and optionally fsyncs.
func (q *Queue) write(typ byte, payload []byte, sync bool) error {
	if q.f == nil {
		return errors.New("queue: wal file is not open")
	}
	frame := encodeRecord(typ, payload)
	n, err := q.f.Write(frame)
	q.fileBytes += int64(n)
	if err != nil {
		return fmt.Errorf("queue: write wal: %w", err)
	}
	if sync {
		if err := q.f.Sync(); err != nil {
			return fmt.Errorf("queue: fsync wal: %w", err)
		}
	}
	return nil
}

// liveFrameBytesLocked is the file size a freshly compacted WAL would have.
func (q *Queue) liveFrameBytesLocked() int64 {
	return q.liveBytes + int64(q.order.Len())*(headerBytes+1)
}

// maybeCompactLocked rewrites the file when dead records dominate it. The 2x
// rule keeps amortised write cost at O(1) per event; the minimum size keeps
// an idle queue from rewriting a few-hundred-byte file on every ack.
func (q *Queue) maybeCompactLocked() {
	if q.fileBytes < q.opts.CompactMinBytes || q.fileBytes <= 2*q.liveFrameBytesLocked() {
		return
	}
	if err := q.compactLocked(); err != nil {
		q.log.Error("compaction failed; continuing with the current file", "error", err)
	}
}

// Compact rewrites the WAL with only the live PUT records, in FIFO order.
func (q *Queue) Compact() error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return ErrClosed
	}
	return q.compactLocked()
}

// compactLocked writes a new file next to the old one, fsyncs it, renames it
// over the WAL and fsyncs the directory, so at every instant exactly one
// complete file is on disk. The live revision is written at the position of
// its first enqueue, so replaying the compacted file yields the same order.
func (q *Queue) compactLocked() error {
	tmp := q.path + ".compact"
	w, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("queue: compact: %w", err)
	}
	var written, live int64
	for el := q.order.Front(); el != nil; el = el.Next() {
		e := el.Value.(*entry)
		payload, err := proto.Marshal(e.ev)
		if err != nil {
			w.Close()
			os.Remove(tmp)
			return fmt.Errorf("queue: compact marshal: %w", err)
		}
		frame := encodeRecord(recPut, payload)
		if _, err := w.Write(frame); err != nil {
			w.Close()
			os.Remove(tmp)
			return fmt.Errorf("queue: compact write: %w", err)
		}
		e.size = int64(len(payload))
		live += e.size
		written += int64(len(frame))
	}
	if err := w.Sync(); err != nil {
		w.Close()
		os.Remove(tmp)
		return fmt.Errorf("queue: compact fsync: %w", err)
	}
	if err := w.Close(); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("queue: compact close: %w", err)
	}
	if err := os.Rename(tmp, q.path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("queue: compact rename: %w", err)
	}
	if err := syncDir(q.opts.Dir); err != nil {
		q.log.Warn("fsync queue dir after compaction", "error", err)
	}
	if q.f != nil {
		q.f.Close()
	}
	f, err := os.OpenFile(q.path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		q.f = nil
		return fmt.Errorf("queue: reopen after compaction: %w", err)
	}
	before := q.fileBytes
	q.f = f
	q.fileBytes = written
	q.liveBytes = live
	q.compactions++
	q.log.Info("wal compacted", "before_bytes", before, "after_bytes", written, "pending", q.order.Len())
	return nil
}

// Close fsyncs and closes the WAL. Further operations return ErrClosed.
func (q *Queue) Close() error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return nil
	}
	q.closed = true
	if q.f == nil {
		return nil
	}
	serr := q.f.Sync()
	cerr := q.f.Close()
	q.f = nil
	if serr != nil {
		return serr
	}
	return cerr
}
