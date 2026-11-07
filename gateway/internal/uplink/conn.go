// Package uplink implements the gateway → review-server side: a connection
// state machine with backoff, the event sender, the heartbeat loop, the
// command watcher, the command-result forwarder and the evidence uploader.
// All loops share one Conn and report transport failures to it, so there is
// exactly one place that decides "we are disconnected" and one backoff.
package uplink

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	grpcstatus "google.golang.org/grpc/status"

	"github.com/sokldjs554/zonewatch/gateway/internal/status"
)

// ConnOptions configures the uplink connection.
type ConnOptions struct {
	// Target is the review server address (host:port).
	Target string
	// Insecure selects plaintext gRPC. Otherwise TLS with the system roots
	// is used (no client certificate / token: auth is out of scope for now).
	Insecure bool
	// TLS optionally overrides the TLS config (ignored when Insecure).
	TLS *tls.Config
	// Backoff between connection attempts.
	Backoff BackoffConfig
	// ConnectTimeout bounds one dial attempt (default 5s).
	ConnectTimeout time.Duration
	// DialOptions are appended to the gRPC dial options (tests use this).
	DialOptions []grpc.DialOption
	Logger      *slog.Logger
	Status      *status.Registry
}

// Conn owns the gRPC client connection and its state machine:
//
//	DISCONNECTED --(backoff elapsed)--> CONNECTING --(READY)--> CONNECTED
//	CONNECTED --(RPC transport error | channel leaves READY)--> DISCONNECTED
//
// Why not rely on grpc-go's built-in reconnection alone: it hides the state
// from us, uses its own backoff, and makes "reconnect_count" and "last_error"
// guesswork. Dialling a fresh ClientConn per attempt gives a crisp state the
// heartbeat can report and a single backoff that all loops respect. Each
// successful connection gets a generation number; a loop that observes a
// failure reports it with the generation it was using, so a stale report from
// a slow RPC cannot tear down a newer, healthy connection.
type Conn struct {
	opts ConnOptions
	log  *slog.Logger

	mu     sync.Mutex
	cc     *grpc.ClientConn
	gen    uint64
	ready  chan struct{} // closed when the current generation is CONNECTED
	failed chan struct{} // closed when the current generation has failed
	done   chan struct{} // closed when Run returns
}

// NewConn creates a Conn; call Run to start the state machine.
func NewConn(opts ConnOptions) *Conn {
	if opts.ConnectTimeout <= 0 {
		opts.ConnectTimeout = 5 * time.Second
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	if opts.Status == nil {
		opts.Status = status.New("", "")
	}
	return &Conn{
		opts:   opts,
		log:    opts.Logger.With("component", "uplink"),
		ready:  make(chan struct{}),
		failed: make(chan struct{}),
		done:   make(chan struct{}),
	}
}

// Run drives the state machine until ctx is cancelled.
func (c *Conn) Run(ctx context.Context) {
	defer close(c.done)
	bo := NewBackoff(c.opts.Backoff)
	first := true
	for {
		if !first {
			d := bo.Next()
			c.log.Info("uplink disconnected; retrying", "in", d.Round(time.Millisecond).String(), "attempt", bo.Attempt())
			if !sleep(ctx, d) {
				return
			}
			c.opts.Status.IncReconnects()
		}
		first = false
		c.opts.Status.SetUplinkState(status.StateConnecting)
		cc, err := c.dial(ctx)
		if err != nil {
			if ctx.Err() != nil {
				c.opts.Status.SetUplinkState(status.StateDisconnected)
				return
			}
			c.opts.Status.SetLastError(err)
			c.opts.Status.SetUplinkState(status.StateDisconnected)
			c.log.Warn("uplink connect failed", "target", c.opts.Target, "error", err.Error())
			continue
		}

		c.mu.Lock()
		c.cc = cc
		c.failed = make(chan struct{})
		failed := c.failed
		gen := c.gen
		close(c.ready)
		c.mu.Unlock()
		bo.Reset()
		c.opts.Status.SetUplinkState(status.StateConnected)
		c.log.Info("uplink connected", "target", c.opts.Target, "generation", gen)

		// Watch the channel itself: if the server goes away while no RPC is
		// in flight, the state leaves READY and we notice without waiting
		// for the next heartbeat to fail.
		watchCtx, cancelWatch := context.WithCancel(ctx)
		go c.watchState(watchCtx, cc, gen)

		select {
		case <-ctx.Done():
			cancelWatch()
			c.teardown(cc)
			c.opts.Status.SetUplinkState(status.StateDisconnected)
			return
		case <-failed:
			cancelWatch()
			c.teardown(cc)
			c.opts.Status.SetUplinkState(status.StateDisconnected)
		}
	}
}

func (c *Conn) teardown(cc *grpc.ClientConn) {
	c.mu.Lock()
	c.cc = nil
	c.gen++
	c.ready = make(chan struct{})
	c.mu.Unlock()
	cc.Close()
}

func (c *Conn) watchState(ctx context.Context, cc *grpc.ClientConn, gen uint64) {
	for {
		s := cc.GetState()
		if s != connectivity.Ready {
			c.Fail(gen, fmt.Errorf("channel state %s", s))
			return
		}
		if !cc.WaitForStateChange(ctx, s) {
			return
		}
	}
}

// dial creates a ClientConn and waits until it is READY or the connect
// timeout expires. grpc-go's own reconnect backoff is set short because our
// loop owns the long backoff: the connection is thrown away on failure.
func (c *Conn) dial(ctx context.Context) (*grpc.ClientConn, error) {
	var creds credentials.TransportCredentials
	if c.opts.Insecure {
		creds = insecure.NewCredentials()
	} else {
		creds = credentials.NewTLS(c.opts.TLS)
	}
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(creds),
		grpc.WithConnectParams(grpc.ConnectParams{
			Backoff: backoff.Config{
				BaseDelay:  200 * time.Millisecond,
				Multiplier: 1.6,
				Jitter:     0.2,
				MaxDelay:   c.opts.ConnectTimeout,
			},
			MinConnectTimeout: c.opts.ConnectTimeout,
		}),
	}
	opts = append(opts, c.opts.DialOptions...)
	cc, err := grpc.NewClient(c.opts.Target, opts...)
	if err != nil {
		return nil, fmt.Errorf("new client: %w", err)
	}
	cc.Connect()
	dctx, cancel := context.WithTimeout(ctx, c.opts.ConnectTimeout)
	defer cancel()
	for {
		s := cc.GetState()
		switch s {
		case connectivity.Ready:
			return cc, nil
		case connectivity.Shutdown:
			cc.Close()
			return nil, errors.New("channel shut down during dial")
		case connectivity.TransientFailure:
			// Fail fast: grpc-go would retry on its own schedule, but the
			// Conn loop owns the backoff, so hand the attempt back now.
			cc.Close()
			return nil, fmt.Errorf("connect to %s failed (%s)", c.opts.Target, s)
		case connectivity.Idle:
			cc.Connect()
		}
		if !cc.WaitForStateChange(dctx, s) {
			cc.Close()
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, fmt.Errorf("connect timeout after %s (state %s)", c.opts.ConnectTimeout, s)
		}
	}
}

// Acquire blocks until the uplink is CONNECTED (or ctx ends) and returns the
// connection with its generation.
func (c *Conn) Acquire(ctx context.Context) (*grpc.ClientConn, uint64, error) {
	for {
		c.mu.Lock()
		cc, gen, ready := c.cc, c.gen, c.ready
		c.mu.Unlock()
		if cc != nil {
			return cc, gen, nil
		}
		select {
		case <-ctx.Done():
			return nil, 0, ctx.Err()
		case <-ready:
		}
	}
}

// Current returns the connection without blocking; ok is false while
// disconnected.
func (c *Conn) Current() (cc *grpc.ClientConn, gen uint64, ok bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cc, c.gen, c.cc != nil
}

// Fail reports a transport failure observed on generation gen. Reports for
// an older generation are ignored.
func (c *Conn) Fail(gen uint64, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cc == nil || gen != c.gen {
		return
	}
	select {
	case <-c.failed:
		// already failed
	default:
		c.opts.Status.SetLastError(err)
		c.log.Warn("uplink failure", "generation", gen, "error", err.Error())
		close(c.failed)
	}
}

// Done is closed when Run has returned and the connection is closed.
func (c *Conn) Done() <-chan struct{} { return c.done }

// IsTransportError tells whether an RPC error means "the server is not
// reachable" (→ reconnect) as opposed to "the server rejected this request"
// (→ do not tear the connection down; retry the request later).
// DeadlineExceeded is treated as transport: our RPC deadlines are generous,
// so hitting one means the server or network is stalled.
func IsTransportError(err error) bool {
	if err == nil {
		return false
	}
	switch grpcstatus.Code(err) {
	case codes.Unavailable, codes.DeadlineExceeded:
		return true
	}
	return false
}
