package uplink

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"log/slog"
	"net"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
	"github.com/sokldjs554/sceneproof/gateway/internal/queue"
	"github.com/sokldjs554/sceneproof/gateway/internal/status"
)

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
}

// fakeUpload is what the fake server stored for one evidence file.
type fakeUpload struct {
	data    []byte
	sha     string
	total   int64
	chunks  int
	already bool
}

// fakeIngest is an in-process review server: it deduplicates events by
// (event_id, max revision) exactly like the real one is specified to.
type fakeIngest struct {
	sceneproofv1.UnimplementedIngestServer
	mu         sync.Mutex
	maxRev     map[string]int32
	received   []*sceneproofv1.Event
	heartbeats []*sceneproofv1.DeviceStatus
	hbTimes    []time.Time
	uploads    map[string]*fakeUpload
	present    map[string]bool
	results    []*sceneproofv1.CommandResult
	watchers   int
	cmds       chan *sceneproofv1.Command
	publishErr error
}

func newFakeIngest() *fakeIngest {
	return &fakeIngest{
		maxRev:  map[string]int32{},
		uploads: map[string]*fakeUpload{},
		present: map[string]bool{},
		cmds:    make(chan *sceneproofv1.Command, 16),
	}
}

func uploadKey(eventID, path string) string { return eventID + "/" + path }

func (f *fakeIngest) PublishEvent(_ context.Context, ev *sceneproofv1.Event) (*sceneproofv1.Ack, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.publishErr != nil {
		return nil, f.publishErr
	}
	f.received = append(f.received, proto.Clone(ev).(*sceneproofv1.Event))
	ack := &sceneproofv1.Ack{EventId: ev.EventId, Revision: ev.Revision}
	if cur, ok := f.maxRev[ev.EventId]; ok && cur >= ev.Revision {
		ack.Duplicate = true
		return ack, nil
	}
	f.maxRev[ev.EventId] = ev.Revision
	ack.Accepted = true
	return ack, nil
}

func (f *fakeIngest) UploadEvidence(stream sceneproofv1.Ingest_UploadEvidenceServer) error {
	var (
		key, eventID, path string
		buf                []byte
		chunks             int
		offset             int64
		last               *sceneproofv1.EvidenceChunk
	)
	for {
		c, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		if key == "" {
			eventID, path = c.EventId, c.Path
			key = uploadKey(eventID, path)
		}
		if c.Offset != offset {
			return grpcstatus.Errorf(codes.InvalidArgument, "offset %d != %d", c.Offset, offset)
		}
		buf = append(buf, c.Data...)
		offset += int64(len(c.Data))
		chunks++
		if c.Last {
			last = c
		}
	}
	if last == nil {
		return grpcstatus.Error(codes.InvalidArgument, "no last chunk")
	}
	sum := sha256.Sum256(buf)
	sha := hex.EncodeToString(sum[:])
	if last.Sha256 != sha || last.TotalSize != int64(len(buf)) {
		return stream.SendAndClose(&sceneproofv1.UploadResult{EventId: eventID, Path: path, Ok: false, Message: "digest/size mismatch"})
	}
	f.mu.Lock()
	already := f.present[key]
	f.uploads[key] = &fakeUpload{data: buf, sha: sha, total: last.TotalSize, chunks: chunks, already: already}
	f.mu.Unlock()
	if already {
		return stream.SendAndClose(&sceneproofv1.UploadResult{EventId: eventID, Path: path, Ok: false, AlreadyPresent: true, Size: int64(len(buf))})
	}
	return stream.SendAndClose(&sceneproofv1.UploadResult{EventId: eventID, Path: path, Ok: true, Size: int64(len(buf))})
}

func (f *fakeIngest) Heartbeat(_ context.Context, ds *sceneproofv1.DeviceStatus) (*sceneproofv1.HeartbeatAck, error) {
	f.mu.Lock()
	f.heartbeats = append(f.heartbeats, proto.Clone(ds).(*sceneproofv1.DeviceStatus))
	f.hbTimes = append(f.hbTimes, time.Now())
	f.mu.Unlock()
	return &sceneproofv1.HeartbeatAck{ServerTs: timestamppb.Now()}, nil
}

func (f *fakeIngest) WatchCommands(_ *sceneproofv1.WatchRequest, stream sceneproofv1.Ingest_WatchCommandsServer) error {
	f.mu.Lock()
	f.watchers++
	f.mu.Unlock()
	defer func() {
		f.mu.Lock()
		f.watchers--
		f.mu.Unlock()
	}()
	for {
		select {
		case <-stream.Context().Done():
			return nil
		case cmd := <-f.cmds:
			if err := stream.Send(cmd); err != nil {
				return err
			}
		}
	}
}

func (f *fakeIngest) PublishCommandResult(_ context.Context, res *sceneproofv1.CommandResult) (*sceneproofv1.Empty, error) {
	f.mu.Lock()
	f.results = append(f.results, proto.Clone(res).(*sceneproofv1.CommandResult))
	f.mu.Unlock()
	return &sceneproofv1.Empty{}, nil
}

func (f *fakeIngest) receivedCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.received)
}

func (f *fakeIngest) receivedKeys() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]string, 0, len(f.received))
	for _, e := range f.received {
		out = append(out, e.EventId+"@"+itoa(int(e.Revision)))
	}
	return out
}

func (f *fakeIngest) upload(eventID, path string) *fakeUpload {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.uploads[uploadKey(eventID, path)]
}

func (f *fakeIngest) uploadCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.uploads)
}

func (f *fakeIngest) heartbeatCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.heartbeats)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b strings.Builder
	var digits []byte
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	if neg {
		b.WriteByte('-')
	}
	b.Write(digits)
	return b.String()
}

// fakeServer runs fakeIngest on a real TCP port and can be stopped and
// restarted on the same address.
type fakeServer struct {
	t    *testing.T
	addr string
	svc  *fakeIngest
	mu   sync.Mutex
	gs   *grpc.Server
}

func newFakeServer(t *testing.T) *fakeServer {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	s := &fakeServer{t: t, addr: lis.Addr().String(), svc: newFakeIngest()}
	s.serve(lis)
	t.Cleanup(s.stop)
	return s
}

func (s *fakeServer) serve(lis net.Listener) {
	gs := grpc.NewServer()
	sceneproofv1.RegisterIngestServer(gs, s.svc)
	s.mu.Lock()
	s.gs = gs
	s.mu.Unlock()
	go gs.Serve(lis)
}

func (s *fakeServer) stop() {
	s.mu.Lock()
	gs := s.gs
	s.gs = nil
	s.mu.Unlock()
	if gs != nil {
		gs.Stop()
	}
}

func (s *fakeServer) restart() {
	s.t.Helper()
	var lis net.Listener
	var err error
	for i := 0; i < 50; i++ {
		lis, err = net.Listen("tcp", s.addr)
		if err == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err != nil {
		s.t.Fatalf("relisten on %s: %v", s.addr, err)
	}
	s.serve(lis)
}

// harness wires a Conn (and optionally the loops) against a fake server.
type harness struct {
	t      *testing.T
	srv    *fakeServer
	q      *queue.Queue
	reg    *status.Registry
	conn   *Conn
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func newHarness(t *testing.T, srv *fakeServer) *harness {
	t.Helper()
	q, err := queue.Open(queue.Options{Dir: t.TempDir(), DisableFsync: true, Logger: testLogger()})
	if err != nil {
		t.Fatal(err)
	}
	reg := status.New("v-test", "edge-test")
	reg.SetQueueStatsFunc(q.Stats)
	ctx, cancel := context.WithCancel(context.Background())
	h := &harness{t: t, srv: srv, q: q, reg: reg, ctx: ctx, cancel: cancel}
	h.conn = NewConn(ConnOptions{
		Target:         srv.addr,
		Insecure:       true,
		Backoff:        BackoffConfig{Base: 20 * time.Millisecond, Max: 150 * time.Millisecond},
		ConnectTimeout: 500 * time.Millisecond,
		Logger:         testLogger(),
		Status:         reg,
	})
	h.go_(func() { h.conn.Run(ctx) })
	t.Cleanup(func() {
		cancel()
		h.wg.Wait()
		q.Close()
	})
	return h
}

func (h *harness) go_(f func()) {
	h.wg.Add(1)
	go func() {
		defer h.wg.Done()
		f()
	}()
}

func (h *harness) startSender(onAcked func(*sceneproofv1.Event)) {
	h.go_(func() {
		RunSender(h.ctx, SenderOptions{
			Queue: h.q, Conn: h.conn, Status: h.reg, Logger: testLogger(),
			OnAcked: onAcked, RPCTimeout: 2 * time.Second, IdlePoll: 50 * time.Millisecond,
			RejectRetry: 100 * time.Millisecond,
		})
	})
}

func waitFor(t *testing.T, timeout time.Duration, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timeout waiting for: %s", msg)
}

func mkEvent(id string, rev int32) *sceneproofv1.Event {
	return &sceneproofv1.Event{EventId: id, Revision: rev, DeviceId: "edge-test", CameraId: "cam-1", ZoneId: "z1"}
}
