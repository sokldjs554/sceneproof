package uplink

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"
	"time"

	zonewatchv1 "github.com/sokldjs554/zonewatch/gateway/gen/zonewatchv1"
	"github.com/sokldjs554/zonewatch/gateway/internal/status"
)

// longLivedStream is how long a WatchCommands stream must have lived for
// its end to count as "normal" rather than as a failure.
const longLivedStream = 30 * time.Second

// CommandSink receives commands from the server (implemented by the ingress
// command bus, which fans them out to analyzer subscribers).
type CommandSink interface {
	Publish(cmd *zonewatchv1.Command)
}

// WatcherOptions configures RunCommandWatcher.
type WatcherOptions struct {
	Conn     *Conn
	DeviceID string
	Sink     CommandSink
	Logger   *slog.Logger
	// Backoff between stream re-establishments that are not caused by a
	// lost connection (server closed the stream, or refused it).
	Backoff BackoffConfig
}

// RunCommandWatcher keeps Ingest.WatchCommands open and forwards every
// Command to the sink until ctx ends.
//
// A transport error on the stream is reported to Conn (the whole connection
// is gone, every loop should reconnect together). Any other end of the
// stream — the server closed it, or rejected the request — is retried on
// this loop's own backoff, reset once a stream delivers a message, so a
// server that is up but refusing the watch is not polled in a tight loop.
// The gateway does not deduplicate commands by command_id: the proto makes
// commands idempotent on the analyzer side, and suppressing a re-sent command
// here could hide one the analyzer never saw.
func RunCommandWatcher(ctx context.Context, o WatcherOptions) {
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	log := o.Logger.With("component", "commands")
	bo := NewBackoff(o.Backoff)
	for ctx.Err() == nil {
		cc, gen, err := o.Conn.Acquire(ctx)
		if err != nil {
			return
		}
		sctx, cancel := context.WithCancel(ctx)
		stream, err := zonewatchv1.NewIngestClient(cc).WatchCommands(sctx, &zonewatchv1.WatchRequest{DeviceId: o.DeviceID})
		if err != nil {
			cancel()
			if ctx.Err() != nil {
				return
			}
			if IsTransportError(err) {
				o.Conn.Fail(gen, err)
				continue
			}
			d := bo.Next()
			log.Warn("WatchCommands refused; retrying", "error", err.Error(), "in", d.String())
			if !sleep(ctx, d) {
				return
			}
			continue
		}
		log.Info("watching commands", "device_id", o.DeviceID)
		received := 0
		opened := time.Now()
		for {
			cmd, err := stream.Recv()
			if err != nil {
				cancel()
				if ctx.Err() != nil {
					return
				}
				switch {
				case errors.Is(err, io.EOF):
					log.Info("command stream closed by server", "received", received)
				case IsTransportError(err):
					o.Conn.Fail(gen, err)
				default:
					log.Warn("command stream error", "error", err.Error())
				}
				break
			}
			received++
			bo.Reset()
			log.Info("command received", "command_id", cmd.GetCommandId(), "type", cmd.GetType().String(),
				"camera_id", cmd.GetCameraId())
			o.Sink.Publish(cmd)
		}
		if time.Since(opened) > longLivedStream {
			// A stream that lived this long was healthy; its end (a
			// server-side idle timeout, a rolling restart) is not a reason
			// to slow down the next watch.
			bo.Reset()
		}
		if _, _, ok := o.Conn.Current(); ok {
			// Connection still fine → the stream itself ended; back off a bit.
			d := bo.Next()
			if !sleep(ctx, d) {
				return
			}
		}
	}
}

// ResultsOptions configures the command-result forwarder.
type ResultsOptions struct {
	Conn   *Conn
	Status *status.Registry
	Logger *slog.Logger
	// Capacity bounds the in-memory queue (default 256). Results are not
	// durable: the server also learns the outcome from the next DeviceStatus
	// (camera state), so losing a result on restart is tolerable, while
	// losing an Event is not.
	Capacity int
	// RPCTimeout bounds one PublishCommandResult (default 10s).
	RPCTimeout time.Duration
	Backoff    BackoffConfig
}

// Results forwards CommandResults from the analyzer to the server.
type Results struct {
	o    ResultsOptions
	log  *slog.Logger
	mu   sync.Mutex
	q    []*zonewatchv1.CommandResult
	wake chan struct{}
}

// NewResults creates the forwarder; call Run to start it.
func NewResults(o ResultsOptions) *Results {
	if o.Capacity <= 0 {
		o.Capacity = 256
	}
	if o.RPCTimeout <= 0 {
		o.RPCTimeout = 10 * time.Second
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	return &Results{o: o, log: o.Logger.With("component", "results"), wake: make(chan struct{}, 1)}
}

// Push enqueues a result. When the queue is full the oldest result is dropped
// (and logged): newer outcomes are more relevant and the bound exists to
// protect memory while the server is unreachable.
func (r *Results) Push(res *zonewatchv1.CommandResult) {
	r.mu.Lock()
	if len(r.q) >= r.o.Capacity {
		dropped := r.q[0]
		r.q = r.q[1:]
		r.log.Warn("command result queue full; dropped oldest", "command_id", dropped.GetCommandId())
	}
	r.q = append(r.q, res)
	r.mu.Unlock()
	select {
	case r.wake <- struct{}{}:
	default:
	}
}

// Len returns the number of results waiting.
func (r *Results) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.q)
}

func (r *Results) peek() *zonewatchv1.CommandResult {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.q) == 0 {
		return nil
	}
	return r.q[0]
}

func (r *Results) pop(res *zonewatchv1.CommandResult) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.q) > 0 && r.q[0] == res {
		r.q = r.q[1:]
	}
}

// Run forwards results until ctx ends.
func (r *Results) Run(ctx context.Context) {
	bo := NewBackoff(r.o.Backoff)
	for ctx.Err() == nil {
		res := r.peek()
		if res == nil {
			select {
			case <-ctx.Done():
				return
			case <-r.wake:
			}
			continue
		}
		cc, gen, err := r.o.Conn.Acquire(ctx)
		if err != nil {
			return
		}
		rctx, cancel := context.WithTimeout(ctx, r.o.RPCTimeout)
		_, err = zonewatchv1.NewIngestClient(cc).PublishCommandResult(rctx, res)
		cancel()
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			if IsTransportError(err) {
				r.o.Conn.Fail(gen, err)
				continue // Acquire blocks until reconnected
			}
			// Rejected by the server: a result is advisory, so after a
			// short backoff give up on this one rather than block the rest.
			d := bo.Next()
			r.log.Warn("PublishCommandResult rejected", "command_id", res.GetCommandId(), "error", err.Error(), "retry_in", d.String())
			if !sleep(ctx, d) {
				return
			}
			if bo.Attempt() >= 5 {
				r.log.Error("dropping command result after repeated rejections", "command_id", res.GetCommandId())
				r.pop(res)
				bo.Reset()
			}
			continue
		}
		bo.Reset()
		r.pop(res)
		r.log.Info("command result forwarded", "command_id", res.GetCommandId(), "ok", res.GetOk())
	}
}
