package ingress

import (
	"log/slog"
	"sync"
	"time"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
)

const (
	// DefaultSubscriberBuffer is the per-subscriber channel depth.
	DefaultSubscriberBuffer = 64
	// DefaultRecentLimit / DefaultRecentTTL bound the commands kept while no
	// analyzer is subscribed.
	DefaultRecentLimit = 32
	DefaultRecentTTL   = 60 * time.Second
)

type recentCmd struct {
	cmd *sceneproofv1.Command
	at  time.Time
}

// Bus fans commands from the server out to analyzer subscribers.
//
// Each subscriber gets its own buffered channel so one slow stream cannot
// stall delivery to another (or block the uplink's watcher goroutine). When
// nobody is subscribed — the analyzer is restarting, which is exactly when
// the console is likely to send START_CAMERA — the most recent commands are
// held for a short time and handed to the first subscriber that connects.
// The buffer is deliberately small and short-lived: a command that is a
// minute old is more likely to be stale than wanted, and the server re-issues
// what still matters.
type Bus struct {
	log         *slog.Logger
	bufSize     int
	recentLimit int
	recentTTL   time.Duration
	now         func() time.Time

	mu     sync.Mutex
	subs   map[*Subscription]struct{}
	recent []recentCmd
}

// Subscription is one analyzer stream's view of the bus.
type Subscription struct {
	C   <-chan *sceneproofv1.Command
	ch  chan *sceneproofv1.Command
	bus *Bus
}

// NewBus creates a bus with the default limits.
func NewBus(log *slog.Logger) *Bus {
	if log == nil {
		log = slog.Default()
	}
	return &Bus{
		log:         log.With("component", "command-bus"),
		bufSize:     DefaultSubscriberBuffer,
		recentLimit: DefaultRecentLimit,
		recentTTL:   DefaultRecentTTL,
		now:         time.Now,
		subs:        map[*Subscription]struct{}{},
	}
}

// Publish delivers cmd to every subscriber, or buffers it when there is none.
func (b *Bus) Publish(cmd *sceneproofv1.Command) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.subs) == 0 {
		b.pruneLocked()
		b.recent = append(b.recent, recentCmd{cmd: cmd, at: b.now()})
		if len(b.recent) > b.recentLimit {
			b.recent = b.recent[len(b.recent)-b.recentLimit:]
		}
		b.log.Info("no analyzer subscribed; buffering command", "command_id", cmd.GetCommandId(), "buffered", len(b.recent))
		return
	}
	for s := range b.subs {
		s.deliver(cmd)
	}
}

// deliver pushes without blocking; when the subscriber's buffer is full the
// oldest queued command is dropped in favour of the newest.
func (s *Subscription) deliver(cmd *sceneproofv1.Command) {
	for {
		select {
		case s.ch <- cmd:
			return
		default:
		}
		select {
		case old := <-s.ch:
			s.bus.log.Warn("subscriber slow; dropped oldest buffered command", "command_id", old.GetCommandId())
		default:
		}
	}
}

func (b *Bus) pruneLocked() {
	cutoff := b.now().Add(-b.recentTTL)
	i := 0
	for i < len(b.recent) && b.recent[i].at.Before(cutoff) {
		i++
	}
	b.recent = b.recent[i:]
}

// Subscribe registers a new subscriber. Buffered recent commands (if any)
// are placed on its channel first and the buffer is cleared: they have been
// delivered.
func (b *Bus) Subscribe() *Subscription {
	s := &Subscription{ch: make(chan *sceneproofv1.Command, b.bufSize), bus: b}
	s.C = s.ch
	b.mu.Lock()
	defer b.mu.Unlock()
	b.pruneLocked()
	for _, r := range b.recent {
		s.deliver(r.cmd)
	}
	if n := len(b.recent); n > 0 {
		b.log.Info("delivered buffered commands to new subscriber", "count", n)
	}
	b.recent = nil
	b.subs[s] = struct{}{}
	return s
}

// Unsubscribe removes s; its channel is not closed (the owner stops reading).
func (b *Bus) Unsubscribe(s *Subscription) {
	b.mu.Lock()
	delete(b.subs, s)
	b.mu.Unlock()
}

// Subscribers returns the number of connected analyzer streams.
func (b *Bus) Subscribers() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.subs)
}
