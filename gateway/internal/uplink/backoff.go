package uplink

import (
	"math/rand/v2"
	"time"
)

// BackoffConfig describes an exponential backoff with jitter.
type BackoffConfig struct {
	Base   time.Duration // first delay (default 500ms)
	Max    time.Duration // cap (default 30s)
	Factor float64       // growth per attempt (default 2)
}

func (c BackoffConfig) withDefaults() BackoffConfig {
	if c.Base <= 0 {
		c.Base = 500 * time.Millisecond
	}
	if c.Max <= 0 {
		c.Max = 30 * time.Second
	}
	if c.Factor < 1 {
		c.Factor = 2
	}
	return c
}

// Backoff hands out delays: base * factor^n, capped at max, with "equal
// jitter" (uniform in [d/2, d]). Jitter matters because a site may run many
// devices behind one uplink: if the server comes back after an outage, the
// devices must not all reconnect in the same instant. The half-floor keeps
// the delay from collapsing to zero so a dead server is not hammered.
type Backoff struct {
	cfg     BackoffConfig
	attempt int
}

// NewBackoff creates a Backoff with defaults filled in.
func NewBackoff(cfg BackoffConfig) *Backoff {
	return &Backoff{cfg: cfg.withDefaults()}
}

// Next returns the delay to wait before the next attempt and advances.
func (b *Backoff) Next() time.Duration {
	d := float64(b.cfg.Base)
	for i := 0; i < b.attempt; i++ {
		d *= b.cfg.Factor
		if d >= float64(b.cfg.Max) {
			d = float64(b.cfg.Max)
			break
		}
	}
	if d > float64(b.cfg.Max) {
		d = float64(b.cfg.Max)
	}
	b.attempt++
	half := d / 2
	return time.Duration(half + rand.Float64()*half)
}

// Reset is called after a success so the next failure starts from Base.
func (b *Backoff) Reset() { b.attempt = 0 }

// Attempt returns how many delays have been handed out since the last Reset.
func (b *Backoff) Attempt() int { return b.attempt }

// sleep waits d or until ctx is done; it reports false when ctx ended.
func sleep(ctx interface{ Done() <-chan struct{} }, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}
