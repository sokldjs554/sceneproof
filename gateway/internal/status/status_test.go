package status

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
	"github.com/sokldjs554/sceneproof/gateway/internal/queue"
)

func TestEndpoints(t *testing.T) {
	r := New("v1.2.3", "edge-x")
	r.SetQueueStatsFunc(func() queue.Stats { return queue.Stats{Pending: 3, LiveBytes: 1234, MaxBytes: 5000, Dropped: 2} })
	r.SetUplinkState(StateConnected)
	r.IncSent()
	r.IncSent()
	r.IncAcked(false)
	r.IncAcked(true)
	r.IncReconnects()
	r.IncUploadsDone()
	r.SetUploadsPending(4)
	r.SetLastError(errors.New("boom"))
	r.SetAnalyzerStatus(&sceneproofv1.DeviceStatus{AnalyzerRunId: "run-1", Cameras: []*sceneproofv1.CameraStatus{{CameraId: "cam-1"}}})
	srv := httptest.NewServer(r.Handler())
	defer srv.Close()

	get := func(path string) (int, string) {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, string(b)
	}

	if code, _ := get("/healthz"); code != 200 {
		t.Fatalf("healthz = %d", code)
	}
	if code, _ := get("/readyz"); code != 503 {
		t.Fatalf("readyz before ready = %d", code)
	}
	r.SetQueueOpen(true)
	r.SetGRPCUp(true)
	if code, _ := get("/readyz"); code != 200 {
		t.Fatalf("readyz = %d", code)
	}

	code, body := get("/status")
	if code != 200 {
		t.Fatalf("status = %d", code)
	}
	var snap Snapshot
	if err := json.Unmarshal([]byte(body), &snap); err != nil {
		t.Fatalf("status json: %v\n%s", err, body)
	}
	if snap.Version != "v1.2.3" || snap.DeviceID != "edge-x" || !snap.Ready || snap.Uplink.State != StateConnected ||
		snap.Uplink.SentEvents != 2 || snap.Uplink.AckedEvents != 2 || snap.Uplink.DuplicateAcks != 1 ||
		snap.Uplink.ReconnectCount != 1 || snap.Uplink.LastError != "boom" || snap.Queue.Pending != 3 ||
		snap.Uploads.Pending != 4 || snap.Uploads.Done != 1 || snap.AnalyzerReportedAt == nil {
		t.Fatalf("snapshot = %+v", snap)
	}
	if !strings.Contains(string(snap.Analyzer), `"analyzerRunId"`) || !strings.Contains(string(snap.Analyzer), `"run-1"`) {
		t.Fatalf("analyzer json = %s", snap.Analyzer)
	}

	code, body = get("/metrics")
	if code != 200 {
		t.Fatalf("metrics = %d", code)
	}
	for _, want := range []string{
		"sceneproof_gateway_queue_pending 3\n",
		"sceneproof_gateway_queue_bytes 1234\n",
		"sceneproof_gateway_events_sent_total 2\n",
		"sceneproof_gateway_events_acked_total 2\n",
		"sceneproof_gateway_events_dropped_total 2\n",
		"sceneproof_gateway_uplink_connected 1\n",
		"sceneproof_gateway_uplink_reconnects_total 1\n",
		"sceneproof_gateway_uploads_pending 4\n",
		"sceneproof_gateway_uploads_done_total 1\n",
		"# TYPE sceneproof_gateway_queue_pending gauge\n",
		"# TYPE sceneproof_gateway_events_sent_total counter\n",
		`sceneproof_gateway_info{version="v1.2.3",device_id="edge-x"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics missing %q\n%s", want, body)
		}
	}

	us := r.UplinkStatus()
	if !us.Connected || us.PendingEvents != 3 || us.DroppedEvents != 2 || us.QueueBytes != 1234 || us.QueueCapacityBytes != 5000 ||
		us.PendingUploads != 4 || us.LastAckTs == nil || us.ReconnectCount != 1 {
		t.Fatalf("uplink status = %+v", us)
	}
	ds := r.DeviceStatus(time.Now())
	if ds.DeviceId != "edge-x" || ds.GatewayVersion != "v1.2.3" || ds.Uplink == nil || len(ds.Cameras) != 1 || ds.Ts == nil {
		t.Fatalf("device status = %+v", ds)
	}
}
