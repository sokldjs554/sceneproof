package uplink

import (
	"context"
	"log/slog"
	"time"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
	"github.com/sokldjs554/sceneproof/gateway/internal/queue"
	"github.com/sokldjs554/sceneproof/gateway/internal/status"
)

// SenderOptions configures RunSender.
type SenderOptions struct {
	Queue  *queue.Queue
	Conn   *Conn
	Status *status.Registry
	Logger *slog.Logger
	// OnAcked is called after the queue has removed an acknowledged event
	// (used to schedule evidence uploads). May be nil.
	OnAcked func(ev *sceneproofv1.Event)
	// RPCTimeout bounds one PublishEvent (default 15s).
	RPCTimeout time.Duration
	// RejectRetry is how long an event the server *rejected* (non-transport
	// error) waits before being retried (default 30s).
	RejectRetry time.Duration
	// IdlePoll bounds the wait when nothing is pending (default 1s); the
	// queue's Wake channel normally ends the wait much sooner.
	IdlePoll time.Duration
}

// RunSender drains the durable queue to the server until ctx ends.
//
// Policy:
//   - Events go out in queue order, one at a time, on the shared connection.
//     Sequential sending keeps the server's view ordered per device and makes
//     "what is in flight" exactly one event, which keeps the duplicate story
//     simple (at most one re-send per reconnect).
//   - Ack.accepted and Ack.duplicate both mean the server holds this
//     revision, so both remove the event from the queue.
//   - A transport error reports the failure to Conn and waits for the next
//     connection; the event stays queued and is re-sent, and the server's
//     (event_id, revision) idempotency absorbs the duplicate if the first
//     attempt did reach it.
//   - A non-transport error (the server rejected the event) does not block
//     the queue: the event is skipped for RejectRetry and the next one is
//     tried. The gateway never drops an event for a rejection, because the
//     rejection may be transient (a rolling server upgrade); only the
//     capacity bound drops events.
func RunSender(ctx context.Context, o SenderOptions) {
	if o.RPCTimeout <= 0 {
		o.RPCTimeout = 15 * time.Second
	}
	if o.RejectRetry <= 0 {
		o.RejectRetry = 30 * time.Second
	}
	if o.IdlePoll <= 0 {
		o.IdlePoll = time.Second
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	log := o.Logger.With("component", "sender")
	notBefore := map[string]time.Time{} // event_id → earliest retry after a rejection

	for ctx.Err() == nil {
		cc, gen, err := o.Conn.Acquire(ctx)
		if err != nil {
			return
		}
		client := sceneproofv1.NewIngestClient(cc)

		pending := o.Queue.Pending()
		now := time.Now()
		if len(notBefore) > 0 {
			// Forget cool-downs for events that are no longer pending
			// (acked by a later pass, or evicted).
			live := make(map[string]struct{}, len(pending))
			for _, ev := range pending {
				live[ev.EventId] = struct{}{}
			}
			for id := range notBefore {
				if _, ok := live[id]; !ok {
					delete(notBefore, id)
				}
			}
		}
		var sentAny bool
		wait := o.IdlePoll
		for _, ev := range pending {
			if ctx.Err() != nil {
				return
			}
			if nb, ok := notBefore[ev.EventId]; ok {
				if now.Before(nb) {
					if d := nb.Sub(now); d < wait {
						wait = d
					}
					continue
				}
				delete(notBefore, ev.EventId)
			}
			// The connection may have changed while iterating; stop and
			// re-acquire so every attempt uses the generation it reports.
			if _, cur, ok := o.Conn.Current(); !ok || cur != gen {
				break
			}
			rctx, cancel := context.WithTimeout(ctx, o.RPCTimeout)
			o.Status.IncSent()
			ack, err := client.PublishEvent(rctx, ev)
			cancel()
			sentAny = true
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				if IsTransportError(err) {
					o.Conn.Fail(gen, err)
					break
				}
				o.Status.SetLastError(err)
				notBefore[ev.EventId] = time.Now().Add(o.RejectRetry)
				log.Error("server rejected event; will retry later",
					"event_id", ev.EventId, "revision", ev.Revision, "error", err.Error())
				continue
			}
			if !ack.GetAccepted() && !ack.GetDuplicate() {
				// Neither flag: the server did not store it and did not have
				// it. Treat like a rejection.
				notBefore[ev.EventId] = time.Now().Add(o.RejectRetry)
				log.Error("server neither accepted nor deduplicated event",
					"event_id", ev.EventId, "revision", ev.Revision, "message", ack.GetMessage())
				continue
			}
			removed, qerr := o.Queue.Ack(ev.EventId, ev.Revision)
			if qerr != nil {
				log.Error("queue ack failed", "event_id", ev.EventId, "error", qerr.Error())
				return
			}
			o.Status.IncAcked(ack.GetDuplicate())
			log.Debug("event acked", "event_id", ev.EventId, "revision", ev.Revision,
				"duplicate", ack.GetDuplicate(), "removed", removed)
			if o.OnAcked != nil {
				o.OnAcked(ev)
			}
		}
		if sentAny {
			// Something happened; look again right away (new revisions may
			// have arrived during the pass).
			continue
		}
		// Nothing was sent: either the queue is empty or everything is in
		// a rejection cool-down. Wait for a wake-up or the next due time.
		t := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			t.Stop()
			return
		case <-o.Queue.Wake():
			t.Stop()
		case <-t.C:
		}
	}
}
