package uplink

import (
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
	"github.com/sokldjs554/sceneproof/gateway/internal/status"
)

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestEventsFlowDisconnectBackoffAndResendOnce(t *testing.T) {
	srv := newFakeServer(t)
	h := newHarness(t, srv)
	var mu sync.Mutex
	var acked []string
	h.startSender(func(ev *sceneproofv1.Event) {
		mu.Lock()
		acked = append(acked, ev.EventId)
		mu.Unlock()
	})

	// Phase 1: events flow and are acked.
	for _, id := range []string{"e1", "e2", "e3"} {
		if _, err := h.q.Put(mkEvent(id, 1)); err != nil {
			t.Fatal(err)
		}
	}
	waitFor(t, 5*time.Second, func() bool { return h.q.Len() == 0 }, "queue drained")
	if got := srv.svc.receivedKeys(); !equal(got, []string{"e1@1", "e2@1", "e3@1"}) {
		t.Fatalf("server received %v", got)
	}
	if h.reg.UplinkState() != status.StateConnected {
		t.Fatalf("state = %s", h.reg.UplinkState())
	}
	mu.Lock()
	if !equal(acked, []string{"e1", "e2", "e3"}) {
		t.Fatalf("OnAcked = %v", acked)
	}
	mu.Unlock()

	// Phase 2: server goes away → DISCONNECTED, reconnects counted, backoff.
	srv.stop()
	waitFor(t, 5*time.Second, func() bool { return h.reg.UplinkState() != status.StateConnected }, "disconnect noticed")
	for _, id := range []string{"e4", "e5", "e6"} {
		h.q.Put(mkEvent(id, 1))
	}
	// Newer revisions of e5 arrive while offline: only rev 3 must be sent.
	h.q.Put(mkEvent("e5", 2))
	h.q.Put(mkEvent("e5", 3))
	h.q.Put(mkEvent("e7", 1))
	time.Sleep(400 * time.Millisecond)
	if n := srv.svc.receivedCount(); n != 3 {
		t.Fatalf("server received %d while down, want 3", n)
	}
	if h.q.Len() != 4 {
		t.Fatalf("pending = %d, want 4", h.q.Len())
	}
	us := h.reg.UplinkStatus()
	if us.Connected || us.ReconnectCount < 1 || us.LastError == "" {
		t.Fatalf("uplink status while down: %+v", us)
	}
	if us.ReconnectCount > 40 {
		t.Fatalf("reconnect attempts %d in 400ms: backoff is not working", us.ReconnectCount)
	}

	// Phase 3: server returns on the same port → pending resent exactly once.
	srv.restart()
	waitFor(t, 10*time.Second, func() bool { return h.q.Len() == 0 }, "queue drained after restart")
	want := []string{"e1@1", "e2@1", "e3@1", "e4@1", "e5@3", "e6@1", "e7@1"}
	waitFor(t, time.Second, func() bool { return srv.svc.receivedCount() >= len(want) }, "all resent")
	time.Sleep(100 * time.Millisecond) // give a wrongly duplicated send time to show up
	if got := srv.svc.receivedKeys(); !equal(got, want) {
		t.Fatalf("server received %v, want %v", got, want)
	}
	if h.reg.UplinkState() != status.StateConnected {
		t.Fatalf("state after restart = %s", h.reg.UplinkState())
	}
	us = h.reg.UplinkStatus()
	if us.AckedEvents != 7 || us.SentEvents != 7 || us.DuplicateAcks != 0 || us.PendingEvents != 0 || us.LastAckTs == nil {
		t.Fatalf("uplink status = %+v", us)
	}
}

func TestDuplicateAckRemovesFromQueue(t *testing.T) {
	srv := newFakeServer(t)
	// Server already has e1 rev 2 (e.g. from a previous gateway life).
	srv.svc.mu.Lock()
	srv.svc.maxRev["e1"] = 2
	srv.svc.mu.Unlock()
	h := newHarness(t, srv)
	h.startSender(nil)
	h.q.Put(mkEvent("e1", 1))
	waitFor(t, 5*time.Second, func() bool { return h.q.Len() == 0 }, "duplicate ack drains queue")
	if us := h.reg.UplinkStatus(); us.DuplicateAcks != 1 || us.AckedEvents != 1 {
		t.Fatalf("status = %+v", us)
	}
}

func TestRejectedEventDoesNotBlockOthers(t *testing.T) {
	srv := newFakeServer(t)
	h := newHarness(t, srv)
	h.startSender(nil)
	// Reject everything for a moment: events must stay queued, and the
	// connection must not be torn down for an application error.
	srv.svc.mu.Lock()
	srv.svc.publishErr = grpcstatus.Error(codes.InvalidArgument, "schema mismatch")
	srv.svc.mu.Unlock()
	h.q.Put(mkEvent("bad", 1))
	waitFor(t, 5*time.Second, func() bool { return h.reg.UplinkStatus().SentEvents >= 1 }, "attempt made")
	time.Sleep(50 * time.Millisecond)
	if h.q.Len() != 1 || h.reg.UplinkState() != status.StateConnected {
		t.Fatalf("len=%d state=%s", h.q.Len(), h.reg.UplinkState())
	}
	srv.svc.mu.Lock()
	srv.svc.publishErr = nil
	srv.svc.mu.Unlock()
	h.q.Put(mkEvent("good", 1))
	waitFor(t, 5*time.Second, func() bool { return h.q.Len() == 0 }, "both delivered after rejection cool-down")
}

func TestCommandWatcherAndResults(t *testing.T) {
	srv := newFakeServer(t)
	h := newHarness(t, srv)
	sink := &recordingSink{}
	h.go_(func() {
		RunCommandWatcher(h.ctx, WatcherOptions{Conn: h.conn, DeviceID: "edge-test", Sink: sink, Logger: testLogger(),
			Backoff: BackoffConfig{Base: 10 * time.Millisecond, Max: 50 * time.Millisecond}})
	})
	results := NewResults(ResultsOptions{Conn: h.conn, Status: h.reg, Logger: testLogger(), Capacity: 4})
	h.go_(func() { results.Run(h.ctx) })

	// Results pushed before the connection is up are forwarded once it is.
	results.Push(&sceneproofv1.CommandResult{CommandId: "c0", Ok: true})
	waitFor(t, 5*time.Second, func() bool {
		srv.svc.mu.Lock()
		defer srv.svc.mu.Unlock()
		return srv.svc.watchers == 1 && len(srv.svc.results) == 1
	}, "watcher connected and result forwarded")

	srv.svc.cmds <- &sceneproofv1.Command{CommandId: "c1", Type: sceneproofv1.CommandType_COMMAND_TYPE_START_CAMERA, CameraId: "cam-1", IssuedAt: timestamppb.Now()}
	waitFor(t, 5*time.Second, func() bool { return sink.count() == 1 }, "command forwarded")
	if sink.get(0).CommandId != "c1" {
		t.Fatalf("got %v", sink.get(0))
	}

	// Stream survives a server restart: reconnect, re-watch, keep receiving.
	srv.stop()
	waitFor(t, 5*time.Second, func() bool { return h.reg.UplinkState() != status.StateConnected }, "disconnected")
	results.Push(&sceneproofv1.CommandResult{CommandId: "c1", Ok: true})
	srv.restart()
	waitFor(t, 10*time.Second, func() bool {
		srv.svc.mu.Lock()
		defer srv.svc.mu.Unlock()
		return srv.svc.watchers == 1 && len(srv.svc.results) == 2
	}, "re-watched and queued result forwarded")
	srv.svc.cmds <- &sceneproofv1.Command{CommandId: "c2", Type: sceneproofv1.CommandType_COMMAND_TYPE_STOP_CAMERA, CameraId: "cam-1"}
	waitFor(t, 5*time.Second, func() bool { return sink.count() == 2 }, "command after restart forwarded")

	// Bounded result queue drops the oldest when full.
	srv.stop()
	waitFor(t, 5*time.Second, func() bool { return h.reg.UplinkState() != status.StateConnected }, "disconnected again")
	for i := 0; i < 6; i++ {
		results.Push(&sceneproofv1.CommandResult{CommandId: "x" + itoa(i)})
	}
	if results.Len() != 4 {
		t.Fatalf("results len = %d, want 4", results.Len())
	}
}

type recordingSink struct {
	mu   sync.Mutex
	cmds []*sceneproofv1.Command
}

func (r *recordingSink) Publish(c *sceneproofv1.Command) {
	r.mu.Lock()
	r.cmds = append(r.cmds, c)
	r.mu.Unlock()
}

func (r *recordingSink) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.cmds)
}

func (r *recordingSink) get(i int) *sceneproofv1.Command {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cmds[i]
}

func TestBackoffGrowsAndIsJittered(t *testing.T) {
	b := NewBackoff(BackoffConfig{Base: 100 * time.Millisecond, Max: 1 * time.Second})
	prevMax := time.Duration(0)
	for i := 0; i < 6; i++ {
		d := b.Next()
		nominal := 100 * time.Millisecond << i
		if nominal > time.Second {
			nominal = time.Second
		}
		if d < nominal/2 || d > nominal {
			t.Fatalf("attempt %d: %v outside [%v, %v]", i, d, nominal/2, nominal)
		}
		if nominal > prevMax {
			prevMax = nominal
		}
	}
	b.Reset()
	if d := b.Next(); d > 100*time.Millisecond {
		t.Fatalf("after reset %v", d)
	}
}
