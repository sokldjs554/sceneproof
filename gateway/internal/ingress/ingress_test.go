package ingress

import (
	"context"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	grpcstatus "google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
	"github.com/sokldjs554/sceneproof/gateway/internal/queue"
	"github.com/sokldjs554/sceneproof/gateway/internal/status"
)

type fakeResults struct {
	mu  sync.Mutex
	res []*sceneproofv1.CommandResult
}

func (f *fakeResults) Push(r *sceneproofv1.CommandResult) {
	f.mu.Lock()
	f.res = append(f.res, r)
	f.mu.Unlock()
}

type env struct {
	dir     string
	q       *queue.Queue
	reg     *status.Registry
	bus     *Bus
	results *fakeResults
	client  sceneproofv1.EdgeIngressClient
}

func setup(t *testing.T) *env {
	t.Helper()
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
	dir := t.TempDir()
	q, err := queue.Open(queue.Options{Dir: dir, DisableFsync: true, Logger: log})
	if err != nil {
		t.Fatal(err)
	}
	reg := status.New("v-test", "edge-test")
	reg.SetQueueStatsFunc(q.Stats)
	bus := NewBus(log)
	results := &fakeResults{}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := grpc.NewServer()
	New(Options{Queue: q, Status: reg, Bus: bus, Results: results, DeviceID: "edge-test", Logger: log}).Register(gs)
	go gs.Serve(lis)
	cc, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cc.Close()
		gs.Stop()
		q.Close()
	})
	return &env{dir: dir, q: q, reg: reg, bus: bus, results: results, client: sceneproofv1.NewEdgeIngressClient(cc)}
}

func walIDs(t *testing.T, dir string) []string {
	t.Helper()
	var ids []string
	_, err := queue.ScanFile(filepath.Join(dir, queue.FileName), func(typ byte, payload []byte) error {
		ev := &sceneproofv1.Event{}
		if typ == 1 && proto.Unmarshal(payload, ev) == nil {
			ids = append(ids, ev.EventId)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return ids
}

func TestSubmitEventAcceptedOnlyAfterWALWrite(t *testing.T) {
	e := setup(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ack, err := e.client.SubmitEvent(ctx, &sceneproofv1.Event{EventId: "ev-1", Revision: 1, CameraId: "cam"})
	if err != nil || !ack.Accepted || ack.Duplicate {
		t.Fatalf("ack=%v err=%v", ack, err)
	}
	// By the time the Ack is observed, the record is in the WAL on disk.
	if ids := walIDs(t, e.dir); len(ids) != 1 || ids[0] != "ev-1" {
		t.Fatalf("wal ids = %v", ids)
	}
	pend := e.q.Pending()
	if len(pend) != 1 || pend[0].DeviceId != "edge-test" {
		t.Fatalf("pending = %v (device_id should be filled in)", pend)
	}

	// Same revision again → duplicate, nothing written.
	ack, err = e.client.SubmitEvent(ctx, &sceneproofv1.Event{EventId: "ev-1", Revision: 1})
	if err != nil || ack.Accepted || !ack.Duplicate {
		t.Fatalf("dup ack=%v err=%v", ack, err)
	}
	if ids := walIDs(t, e.dir); len(ids) != 1 {
		t.Fatalf("duplicate wrote to wal: %v", ids)
	}
	// Newer revision → accepted and coalesced.
	ack, err = e.client.SubmitEvent(ctx, &sceneproofv1.Event{EventId: "ev-1", Revision: 2})
	if err != nil || !ack.Accepted {
		t.Fatalf("rev2 ack=%v err=%v", ack, err)
	}
	if e.q.Len() != 1 || e.q.Pending()[0].Revision != 2 {
		t.Fatalf("not coalesced: %v", e.q.Pending())
	}
	// Missing event_id → InvalidArgument.
	if _, err := e.client.SubmitEvent(ctx, &sceneproofv1.Event{}); grpcstatus.Code(err) != codes.InvalidArgument {
		t.Fatalf("err = %v", err)
	}
	// After the queue is closed the analyzer must get an error, not an Ack.
	e.q.Close()
	if _, err := e.client.SubmitEvent(ctx, &sceneproofv1.Event{EventId: "ev-2", Revision: 1}); grpcstatus.Code(err) != codes.Unavailable {
		t.Fatalf("after close err = %v", err)
	}
}

func TestReportStatusStored(t *testing.T) {
	e := setup(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ds := &sceneproofv1.DeviceStatus{AnalyzerRunId: "run-9", Cameras: []*sceneproofv1.CameraStatus{{CameraId: "cam-1"}}}
	if _, err := e.client.ReportStatus(ctx, ds); err != nil {
		t.Fatal(err)
	}
	got, at := e.reg.AnalyzerStatus()
	if got == nil || got.AnalyzerRunId != "run-9" || got.DeviceId != "edge-test" || at.IsZero() {
		t.Fatalf("stored = %v at %v", got, at)
	}
	// Only the latest is kept.
	e.client.ReportStatus(ctx, &sceneproofv1.DeviceStatus{AnalyzerRunId: "run-10"})
	got, _ = e.reg.AnalyzerStatus()
	if got.AnalyzerRunId != "run-10" {
		t.Fatalf("latest = %q", got.AnalyzerRunId)
	}
	hb := e.reg.DeviceStatus(time.Now())
	if hb.AnalyzerRunId != "run-10" || hb.Uplink == nil || hb.GatewayVersion != "v-test" {
		t.Fatalf("heartbeat merge = %v", hb)
	}
}

func TestSubscribeCommandsForwardsAndBuffersForLateSubscriber(t *testing.T) {
	e := setup(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Live subscriber receives forwarded commands.
	stream, err := e.client.SubscribeCommands(ctx, &sceneproofv1.WatchRequest{DeviceId: "edge-test"})
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, func() bool { return e.bus.Subscribers() == 1 }, "subscriber registered")
	e.bus.Publish(&sceneproofv1.Command{CommandId: "c1", Type: sceneproofv1.CommandType_COMMAND_TYPE_START_CAMERA})
	cmd, err := stream.Recv()
	if err != nil || cmd.CommandId != "c1" {
		t.Fatalf("recv = %v %v", cmd, err)
	}
	cancel()
	waitFor(t, func() bool { return e.bus.Subscribers() == 0 }, "subscriber gone")

	// No subscriber: commands are buffered (bounded), a late subscriber gets them in order.
	for i := 0; i < DefaultRecentLimit+3; i++ {
		e.bus.Publish(&sceneproofv1.Command{CommandId: "b" + string(rune('A'+i))})
	}
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	stream2, err := e.client.SubscribeCommands(ctx2, &sceneproofv1.WatchRequest{DeviceId: "edge-test"})
	if err != nil {
		t.Fatal(err)
	}
	for i := 3; i < DefaultRecentLimit+3; i++ {
		cmd, err := stream2.Recv()
		if err != nil {
			t.Fatal(err)
		}
		if want := "b" + string(rune('A'+i)); cmd.CommandId != want {
			t.Fatalf("buffered cmd %d = %s, want %s", i, cmd.CommandId, want)
		}
	}
	// A second late subscriber does not get them again (they were delivered).
	ctx3, cancel3 := context.WithCancel(context.Background())
	defer cancel3()
	stream3, err := e.client.SubscribeCommands(ctx3, &sceneproofv1.WatchRequest{DeviceId: "edge-test"})
	if err != nil {
		t.Fatal(err)
	}
	waitFor(t, func() bool { return e.bus.Subscribers() == 2 }, "two subscribers")
	e.bus.Publish(&sceneproofv1.Command{CommandId: "fanout"})
	for _, s := range []sceneproofv1.EdgeIngress_SubscribeCommandsClient{stream2, stream3} {
		cmd, err := s.Recv()
		if err != nil || cmd.CommandId != "fanout" {
			t.Fatalf("fanout recv = %v %v", cmd, err)
		}
	}
}

func TestBufferedCommandsExpire(t *testing.T) {
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
	b := NewBus(log)
	now := time.Now()
	b.now = func() time.Time { return now }
	b.Publish(&sceneproofv1.Command{CommandId: "old"})
	now = now.Add(DefaultRecentTTL + time.Second)
	b.Publish(&sceneproofv1.Command{CommandId: "fresh"})
	sub := b.Subscribe()
	select {
	case c := <-sub.C:
		if c.CommandId != "fresh" {
			t.Fatalf("got %s", c.CommandId)
		}
	default:
		t.Fatal("fresh command not delivered")
	}
	select {
	case c := <-sub.C:
		t.Fatalf("expired command delivered: %s", c.CommandId)
	default:
	}
}

func TestReportCommandResultForwarded(t *testing.T) {
	e := setup(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := e.client.ReportCommandResult(ctx, &sceneproofv1.CommandResult{CommandId: "c1", Ok: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := e.client.ReportCommandResult(ctx, &sceneproofv1.CommandResult{}); grpcstatus.Code(err) != codes.InvalidArgument {
		t.Fatalf("err = %v", err)
	}
	e.results.mu.Lock()
	defer e.results.mu.Unlock()
	if len(e.results.res) != 1 || e.results.res[0].CommandId != "c1" || e.results.res[0].DeviceId != "edge-test" {
		t.Fatalf("results = %v", e.results.res)
	}
}

func waitFor(t *testing.T, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timeout: %s", msg)
}
