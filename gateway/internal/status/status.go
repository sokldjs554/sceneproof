// Package status is the gateway's single source of truth for observability:
// uplink connection state and counters, the analyzer's latest DeviceStatus,
// readiness flags, and the HTTP endpoints that expose them.
//
// Every component writes its facts here instead of keeping private counters,
// so the heartbeat, /status and /metrics never disagree with each other.
package status

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	zonewatchv1 "github.com/sokldjs554/zonewatch/gateway/gen/zonewatchv1"
	"github.com/sokldjs554/zonewatch/gateway/internal/queue"
)

// Uplink connection states, as strings so they read well in JSON and logs.
const (
	StateDisconnected = "DISCONNECTED"
	StateConnecting   = "CONNECTING"
	StateConnected    = "CONNECTED"
)

// Registry holds the live state. The zero value is not usable; use New.
type Registry struct {
	version   string
	deviceID  string
	startedAt time.Time

	mu          sync.RWMutex
	state       string
	connected   bool
	lastError   string
	lastErrorAt time.Time
	lastAck     time.Time
	analyzer    *zonewatchv1.DeviceStatus
	analyzerAt  time.Time
	queueStats  func() queue.Stats

	sent           atomic.Int64
	acked          atomic.Int64
	duplicateAcks  atomic.Int64
	reconnects     atomic.Int64
	uploadsDone    atomic.Int64
	uploadsFailed  atomic.Int64
	uploadsPending atomic.Int64
	queueOpen      atomic.Bool
	grpcUp         atomic.Bool
}

// New creates a registry for the given build version and device id.
func New(version, deviceID string) *Registry {
	return &Registry{
		version:   version,
		deviceID:  deviceID,
		startedAt: time.Now(),
		state:     StateDisconnected,
	}
}

// SetQueueStatsFunc installs the live queue stats source; without it the
// queue section reports zeros.
func (r *Registry) SetQueueStatsFunc(f func() queue.Stats) {
	r.mu.Lock()
	r.queueStats = f
	r.mu.Unlock()
}

func (r *Registry) queue() queue.Stats {
	r.mu.RLock()
	f := r.queueStats
	r.mu.RUnlock()
	if f == nil {
		return queue.Stats{}
	}
	return f()
}

// SetUplinkState records the connection state machine's current state.
func (r *Registry) SetUplinkState(state string) {
	r.mu.Lock()
	r.state = state
	r.connected = state == StateConnected
	r.mu.Unlock()
}

// UplinkState returns the current state name.
func (r *Registry) UplinkState() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.state
}

// SetLastError records the most recent uplink error (nil clears nothing: the
// last error stays visible for diagnosis even after recovery, with its time).
func (r *Registry) SetLastError(err error) {
	if err == nil {
		return
	}
	r.mu.Lock()
	r.lastError = err.Error()
	r.lastErrorAt = time.Now()
	r.mu.Unlock()
}

// IncReconnects counts one reconnection attempt after a lost connection.
func (r *Registry) IncReconnects() { r.reconnects.Add(1) }

// IncSent counts one PublishEvent attempt (including ones that fail).
func (r *Registry) IncSent() { r.sent.Add(1) }

// IncAcked counts one server acknowledgement; duplicate marks "already had it".
func (r *Registry) IncAcked(duplicate bool) {
	r.acked.Add(1)
	if duplicate {
		r.duplicateAcks.Add(1)
	}
	r.mu.Lock()
	r.lastAck = time.Now()
	r.mu.Unlock()
}

// IncUploadsDone counts one evidence file the server now holds.
func (r *Registry) IncUploadsDone() { r.uploadsDone.Add(1) }

// IncUploadsFailed counts one evidence file given up on.
func (r *Registry) IncUploadsFailed() { r.uploadsFailed.Add(1) }

// SetUploadsPending sets the number of evidence files waiting to upload.
func (r *Registry) SetUploadsPending(n int) { r.uploadsPending.Store(int64(n)) }

// SetQueueOpen / SetGRPCUp drive /readyz.
func (r *Registry) SetQueueOpen(v bool) { r.queueOpen.Store(v) }
func (r *Registry) SetGRPCUp(v bool)    { r.grpcUp.Store(v) }

// SetAnalyzerStatus stores the latest DeviceStatus reported by the analyzer.
// It is not queued: only the newest report matters for a heartbeat.
func (r *Registry) SetAnalyzerStatus(ds *zonewatchv1.DeviceStatus) {
	r.mu.Lock()
	r.analyzer = ds
	r.analyzerAt = time.Now()
	r.mu.Unlock()
}

// AnalyzerStatus returns the latest analyzer report (nil if none) and when it
// was received.
func (r *Registry) AnalyzerStatus() (*zonewatchv1.DeviceStatus, time.Time) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.analyzer, r.analyzerAt
}

// UplinkStatus assembles the proto UplinkStatus from the counters and the
// live queue stats.
func (r *Registry) UplinkStatus() *zonewatchv1.UplinkStatus {
	qs := r.queue()
	r.mu.RLock()
	us := &zonewatchv1.UplinkStatus{
		Connected:          r.connected,
		PendingEvents:      int64(qs.Pending),
		SentEvents:         r.sent.Load(),
		AckedEvents:        r.acked.Load(),
		DroppedEvents:      qs.Dropped,
		DuplicateAcks:      r.duplicateAcks.Load(),
		ReconnectCount:     int32(r.reconnects.Load()),
		LastError:          r.lastError,
		QueueBytes:         qs.LiveBytes,
		QueueCapacityBytes: qs.MaxBytes,
		PendingUploads:     r.uploadsPending.Load(),
	}
	if !r.lastAck.IsZero() {
		us.LastAckTs = timestamppb.New(r.lastAck)
	}
	r.mu.RUnlock()
	return us
}

// DeviceStatus builds the heartbeat payload: the analyzer's latest report (a
// copy, so the stored one is never mutated) merged with the uplink status and
// the gateway version. When the analyzer has never reported, a status with no
// cameras is produced so the server still sees the device as online.
// ts is the gateway's send time; the freshness of analyzer data is visible
// through cameras[].last_frame_ts.
func (r *Registry) DeviceStatus(now time.Time) *zonewatchv1.DeviceStatus {
	r.mu.RLock()
	src := r.analyzer
	r.mu.RUnlock()
	var ds *zonewatchv1.DeviceStatus
	if src != nil {
		ds = proto.Clone(src).(*zonewatchv1.DeviceStatus)
	} else {
		ds = &zonewatchv1.DeviceStatus{}
	}
	if ds.DeviceId == "" {
		ds.DeviceId = r.deviceID
	}
	ds.Ts = timestamppb.New(now)
	ds.Uplink = r.UplinkStatus()
	ds.GatewayVersion = r.version
	return ds
}

// Snapshot is the JSON document served at /status.
type Snapshot struct {
	Version   string    `json:"version"`
	DeviceID  string    `json:"device_id"`
	StartedAt time.Time `json:"started_at"`
	UptimeS   float64   `json:"uptime_s"`
	Ready     bool      `json:"ready"`
	Uplink    struct {
		State          string     `json:"state"`
		Connected      bool       `json:"connected"`
		SentEvents     int64      `json:"sent_events"`
		AckedEvents    int64      `json:"acked_events"`
		DuplicateAcks  int64      `json:"duplicate_acks"`
		ReconnectCount int64      `json:"reconnect_count"`
		LastAckTs      *time.Time `json:"last_ack_ts,omitempty"`
		LastError      string     `json:"last_error,omitempty"`
		LastErrorAt    *time.Time `json:"last_error_at,omitempty"`
	} `json:"uplink"`
	Queue   queue.Stats `json:"queue"`
	Uploads struct {
		Pending int64 `json:"pending"`
		Done    int64 `json:"done"`
		Failed  int64 `json:"failed"`
	} `json:"uploads"`
	AnalyzerReportedAt *time.Time      `json:"analyzer_reported_at,omitempty"`
	Analyzer           json.RawMessage `json:"analyzer"`
}

// Snapshot returns the current state as a JSON-ready structure.
func (r *Registry) Snapshot() Snapshot {
	now := time.Now()
	qs := r.queue()
	r.mu.RLock()
	defer r.mu.RUnlock()
	s := Snapshot{
		Version:   r.version,
		DeviceID:  r.deviceID,
		StartedAt: r.startedAt,
		UptimeS:   now.Sub(r.startedAt).Seconds(),
		Ready:     r.queueOpen.Load() && r.grpcUp.Load(),
		Queue:     qs,
		Analyzer:  json.RawMessage("null"),
	}
	s.Uplink.State = r.state
	s.Uplink.Connected = r.connected
	s.Uplink.SentEvents = r.sent.Load()
	s.Uplink.AckedEvents = r.acked.Load()
	s.Uplink.DuplicateAcks = r.duplicateAcks.Load()
	s.Uplink.ReconnectCount = r.reconnects.Load()
	s.Uplink.LastError = r.lastError
	if !r.lastAck.IsZero() {
		t := r.lastAck
		s.Uplink.LastAckTs = &t
	}
	if !r.lastErrorAt.IsZero() {
		t := r.lastErrorAt
		s.Uplink.LastErrorAt = &t
	}
	s.Uploads.Pending = r.uploadsPending.Load()
	s.Uploads.Done = r.uploadsDone.Load()
	s.Uploads.Failed = r.uploadsFailed.Load()
	if r.analyzer != nil {
		t := r.analyzerAt
		s.AnalyzerReportedAt = &t
		if b, err := protojson.Marshal(r.analyzer); err == nil {
			s.Analyzer = b
		}
	}
	return s
}

// Handler returns the local HTTP API:
//
//	/healthz  200 while the process runs
//	/readyz   200 when the queue is open and the gRPC listener is up
//	/status   JSON Snapshot
//	/metrics  Prometheus text exposition
func (r *Registry) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		fmt.Fprintln(w, "ok")
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		if r.queueOpen.Load() && r.grpcUp.Load() {
			fmt.Fprintln(w, "ready")
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprintf(w, "not ready: queue_open=%v grpc_up=%v\n", r.queueOpen.Load(), r.grpcUp.Load())
	})
	mux.HandleFunc("/status", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		if err := enc.Encode(r.Snapshot()); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		r.writeMetrics(w)
	})
	return mux
}

// writeMetrics hand-writes the Prometheus exposition format: a handful of
// gauges and counters do not justify a client library dependency on a
// device binary.
func (r *Registry) writeMetrics(w http.ResponseWriter) {
	qs := r.queue()
	r.mu.RLock()
	connected := r.connected
	r.mu.RUnlock()
	b2i := func(b bool) int64 {
		if b {
			return 1
		}
		return 0
	}
	metric := func(name, typ, help string, v int64) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s %s\n%s %d\n", name, help, name, typ, name, v)
	}
	fmt.Fprintf(w, "# HELP zonewatch_gateway_info Build information.\n# TYPE zonewatch_gateway_info gauge\nzonewatch_gateway_info{version=%q,device_id=%q} 1\n", r.version, r.deviceID)
	metric("zonewatch_gateway_queue_pending", "gauge", "Events waiting in the durable queue.", int64(qs.Pending))
	metric("zonewatch_gateway_queue_bytes", "gauge", "Live payload bytes in the durable queue.", qs.LiveBytes)
	metric("zonewatch_gateway_queue_capacity_bytes", "gauge", "Configured queue budget in bytes.", qs.MaxBytes)
	metric("zonewatch_gateway_queue_file_bytes", "gauge", "Size of the WAL file on disk.", qs.FileBytes)
	metric("zonewatch_gateway_events_sent_total", "counter", "PublishEvent attempts.", r.sent.Load())
	metric("zonewatch_gateway_events_acked_total", "counter", "Events acknowledged by the server (accepted or duplicate).", r.acked.Load())
	metric("zonewatch_gateway_events_duplicate_acks_total", "counter", "Acks where the server already had the revision.", r.duplicateAcks.Load())
	metric("zonewatch_gateway_events_dropped_total", "counter", "Events evicted because the queue was full (since start).", qs.Dropped)
	metric("zonewatch_gateway_uplink_connected", "gauge", "1 when the uplink to the review server is connected.", b2i(connected))
	metric("zonewatch_gateway_uplink_reconnects_total", "counter", "Reconnection attempts after a lost uplink.", r.reconnects.Load())
	metric("zonewatch_gateway_uploads_pending", "gauge", "Evidence files waiting to be uploaded.", r.uploadsPending.Load())
	metric("zonewatch_gateway_uploads_done_total", "counter", "Evidence files the server holds.", r.uploadsDone.Load())
	metric("zonewatch_gateway_uploads_failed_total", "counter", "Evidence files given up on after the retry window.", r.uploadsFailed.Load())
	metric("zonewatch_gateway_ready", "gauge", "1 when /readyz would return 200.", b2i(r.queueOpen.Load() && r.grpcUp.Load()))
}
