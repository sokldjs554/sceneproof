package uplink

import (
	"context"
	"log/slog"
	"time"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
	"github.com/sokldjs554/sceneproof/gateway/internal/status"
)

// HeartbeatOptions configures RunHeartbeat.
type HeartbeatOptions struct {
	Conn     *Conn
	Status   *status.Registry
	Logger   *slog.Logger
	Interval time.Duration // default 2s
	// RPCTimeout bounds one Heartbeat RPC (default 10s, never less than
	// Interval). It is deliberately longer than the interval: a slow link
	// must not be mistaken for a dead one, and a lost TCP connection is
	// caught much sooner by the channel-state watcher in Conn. Ticks that
	// fire while a call is still in flight are simply skipped.
	RPCTimeout time.Duration
	// OnSent, if set, is called after every successful heartbeat (tests).
	OnSent func(ds *sceneproofv1.DeviceStatus, ack *sceneproofv1.HeartbeatAck)
}

// RunHeartbeat sends a DeviceStatus every Interval while connected.
//
// The payload is the analyzer's latest ReportStatus merged with UplinkStatus
// and the gateway version (status.Registry.DeviceStatus). It is sent even
// when the analyzer has never reported, because "the gateway is up and the
// queue is this deep" is useful on its own and lets the server mark the
// device online. Ticks while disconnected are skipped rather than queued: a
// heartbeat is only meaningful now, and the connection state machine already
// owns the reconnect backoff.
func RunHeartbeat(ctx context.Context, o HeartbeatOptions) {
	if o.Interval <= 0 {
		o.Interval = 2 * time.Second
	}
	if o.RPCTimeout <= 0 {
		o.RPCTimeout = 10 * time.Second
	}
	if o.RPCTimeout < o.Interval {
		o.RPCTimeout = o.Interval
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	log := o.Logger.With("component", "heartbeat")
	ticker := time.NewTicker(o.Interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		cc, gen, ok := o.Conn.Current()
		if !ok {
			continue
		}
		ds := o.Status.DeviceStatus(time.Now())
		rctx, cancel := context.WithTimeout(ctx, o.RPCTimeout)
		ack, err := sceneproofv1.NewIngestClient(cc).Heartbeat(rctx, ds)
		cancel()
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			if IsTransportError(err) {
				o.Conn.Fail(gen, err)
			} else {
				o.Status.SetLastError(err)
				log.Warn("heartbeat rejected", "error", err.Error())
			}
			continue
		}
		if o.OnSent != nil {
			o.OnSent(ds, ack)
		}
	}
}
