package uplink

import (
	"testing"
	"time"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
)

func TestHeartbeatSentAtIntervalWithMergedStatus(t *testing.T) {
	srv := newFakeServer(t)
	h := newHarness(t, srv)
	h.reg.SetAnalyzerStatus(&sceneproofv1.DeviceStatus{
		AnalyzerRunId: "run-1",
		CpuPercent:    12.5,
		Cameras: []*sceneproofv1.CameraStatus{{
			CameraId: "cam-1", State: sceneproofv1.CameraState_CAMERA_STATE_STREAMING, ProcessedFps: 9.5,
		}},
	})
	h.q.Put(mkEvent("pending-1", 1)) // no sender running: stays pending, visible in UplinkStatus
	const interval = 50 * time.Millisecond
	h.go_(func() {
		RunHeartbeat(h.ctx, HeartbeatOptions{Conn: h.conn, Status: h.reg, Logger: testLogger(), Interval: interval, RPCTimeout: 2 * time.Second})
	})
	waitFor(t, 5*time.Second, func() bool { return srv.svc.heartbeatCount() >= 5 }, "5 heartbeats")

	srv.svc.mu.Lock()
	hbs := append([]*sceneproofv1.DeviceStatus(nil), srv.svc.heartbeats...)
	times := append([]time.Time(nil), srv.svc.hbTimes...)
	srv.svc.mu.Unlock()

	for i := 1; i < len(times); i++ {
		gap := times[i].Sub(times[i-1])
		if gap < interval/2 || gap > 10*interval {
			t.Fatalf("gap %d = %v, want about %v", i, gap, interval)
		}
	}
	for _, ds := range hbs {
		if ds.DeviceId != "edge-test" {
			t.Fatalf("device_id = %q", ds.DeviceId)
		}
		if ds.GatewayVersion != "v-test" {
			t.Fatalf("gateway_version = %q", ds.GatewayVersion)
		}
		if ds.Ts == nil || time.Since(ds.Ts.AsTime()) > 10*time.Second {
			t.Fatalf("ts = %v", ds.Ts)
		}
		if ds.AnalyzerRunId != "run-1" || ds.CpuPercent != 12.5 || len(ds.Cameras) != 1 || ds.Cameras[0].CameraId != "cam-1" {
			t.Fatalf("analyzer fields not merged: %v", ds)
		}
		if ds.Uplink == nil || !ds.Uplink.Connected || ds.Uplink.PendingEvents != 1 || ds.Uplink.QueueCapacityBytes == 0 {
			t.Fatalf("uplink not merged: %v", ds.Uplink)
		}
	}
	// The stored analyzer status must not have been mutated by merging.
	stored, _ := h.reg.AnalyzerStatus()
	if stored.Uplink != nil || stored.GatewayVersion != "" || stored.DeviceId != "" {
		t.Fatalf("stored analyzer status was mutated: %v", stored)
	}
}

func TestHeartbeatWithoutAnalyzerReport(t *testing.T) {
	srv := newFakeServer(t)
	h := newHarness(t, srv)
	h.go_(func() {
		RunHeartbeat(h.ctx, HeartbeatOptions{Conn: h.conn, Status: h.reg, Logger: testLogger(), Interval: 30 * time.Millisecond, RPCTimeout: 2 * time.Second})
	})
	waitFor(t, 5*time.Second, func() bool { return srv.svc.heartbeatCount() >= 2 }, "heartbeats without analyzer")
	srv.svc.mu.Lock()
	ds := srv.svc.heartbeats[0]
	srv.svc.mu.Unlock()
	if ds.DeviceId != "edge-test" || len(ds.Cameras) != 0 || ds.Uplink == nil || ds.GatewayVersion != "v-test" || ds.Ts == nil {
		t.Fatalf("empty heartbeat malformed: %v", ds)
	}

	// While disconnected no heartbeat is attempted; after reconnect they resume.
	srv.stop()
	waitFor(t, 5*time.Second, func() bool { return !h.reg.UplinkStatus().Connected }, "disconnected")
	n := srv.svc.heartbeatCount()
	time.Sleep(150 * time.Millisecond)
	if srv.svc.heartbeatCount() != n {
		t.Fatal("heartbeats counted while server down")
	}
	srv.restart()
	waitFor(t, 10*time.Second, func() bool { return srv.svc.heartbeatCount() > n }, "heartbeats resumed")
}
