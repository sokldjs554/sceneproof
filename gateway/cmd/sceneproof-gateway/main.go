// Command sceneproof-gateway is the on-device edge gateway: it accepts events
// and status from the analyzer over EdgeIngress, stores events durably, and
// forwards everything to the review server over Ingest with reconnection,
// evidence upload, heartbeat and command relay.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"google.golang.org/grpc"

	"github.com/sokldjs554/sceneproof/gateway/internal/ingress"
	"github.com/sokldjs554/sceneproof/gateway/internal/queue"
	"github.com/sokldjs554/sceneproof/gateway/internal/status"
	"github.com/sokldjs554/sceneproof/gateway/internal/uplink"
	verpkg "github.com/sokldjs554/sceneproof/gateway/internal/version"
)

// version is set at build time: -ldflags "-X main.version=<v>".
var version = ""

// envPrefix is the prefix of environment variables that back every flag:
// --queue-dir ↔ SP_GW_QUEUE_DIR. Flags win over environment.
const envPrefix = "SP_GW_"

func envName(flagName string) string {
	return envPrefix + strings.ToUpper(strings.ReplaceAll(flagName, "-", "_"))
}

func envString(name, def string) string {
	if v, ok := os.LookupEnv(envName(name)); ok {
		return v
	}
	return def
}

func envBool(name string, def bool) bool {
	if v, ok := os.LookupEnv(envName(name)); ok {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}

func envDuration(name string, def time.Duration) time.Duration {
	if v, ok := os.LookupEnv(envName(name)); ok {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

func envInt64(name string, def int64) int64 {
	if v, ok := os.LookupEnv(envName(name)); ok {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			return n
		}
	}
	return def
}

type config struct {
	listen            string
	server            string
	deviceID          string
	queueDir          string
	queueMaxBytes     int64
	queueFsync        bool
	evidenceRoot      string
	httpAddr          string
	heartbeatInterval time.Duration
	uploadRetryWindow time.Duration
	insecure          bool
	logLevel          string
	showVersion       bool
}

func parseFlags(args []string) (config, error) {
	var c config
	fs := flag.NewFlagSet("sceneproof-gateway", flag.ContinueOnError)
	fs.StringVar(&c.listen, "listen", envString("listen", ":50051"), "EdgeIngress gRPC listen address (analyzer side)")
	fs.StringVar(&c.server, "server", envString("server", "127.0.0.1:50052"), "review server Ingest address host:port")
	fs.StringVar(&c.deviceID, "device-id", envString("device-id", "edge-01"), "device id reported to the server")
	fs.StringVar(&c.queueDir, "queue-dir", envString("queue-dir", "./data/gateway"), "directory for the durable event queue and upload ledger")
	fs.Int64Var(&c.queueMaxBytes, "queue-max-bytes", envInt64("queue-max-bytes", 64<<20), "queue budget in live payload bytes; oldest events are evicted beyond it")
	fs.BoolVar(&c.queueFsync, "queue-fsync", envBool("queue-fsync", true), "fsync the WAL after every stored event")
	fs.StringVar(&c.evidenceRoot, "evidence-root", envString("evidence-root", "./data/evidence"), "evidence directory shared with the analyzer")
	fs.StringVar(&c.httpAddr, "http", envString("http", ":9100"), "local HTTP address for /healthz /readyz /status /metrics (empty disables)")
	fs.DurationVar(&c.heartbeatInterval, "heartbeat-interval", envDuration("heartbeat-interval", 2*time.Second), "heartbeat period")
	fs.DurationVar(&c.uploadRetryWindow, "upload-retry-window", envDuration("upload-retry-window", 10*time.Minute), "how long to wait for a missing evidence file")
	fs.BoolVar(&c.insecure, "insecure", envBool("insecure", false), "plaintext gRPC to the server (otherwise TLS with system roots)")
	fs.StringVar(&c.logLevel, "log-level", envString("log-level", "info"), "debug|info|warn|error")
	fs.BoolVar(&c.showVersion, "version", false, "print version and exit")
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(args); err != nil {
		return c, err
	}
	return c, nil
}

func main() {
	verpkg.Set(version)
	cfg, err := parseFlags(os.Args[1:])
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			os.Exit(0)
		}
		os.Exit(2)
	}
	if cfg.showVersion {
		fmt.Println("sceneproof-gateway " + verpkg.Full())
		return
	}
	os.Exit(run(cfg))
}

func run(cfg config) int {
	var level slog.Level
	if err := level.UnmarshalText([]byte(cfg.logLevel)); err != nil {
		level = slog.LevelInfo
	}
	log := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(log)
	log = log.With("device_id", cfg.deviceID)
	log.Info("sceneproof-gateway starting", "version", verpkg.Get(), "listen", cfg.listen, "server", cfg.server,
		"queue_dir", cfg.queueDir, "evidence_root", cfg.evidenceRoot, "insecure", cfg.insecure)

	reg := status.New(verpkg.Get(), cfg.deviceID)

	q, err := queue.Open(queue.Options{
		Dir:          cfg.queueDir,
		MaxBytes:     cfg.queueMaxBytes,
		DisableFsync: !cfg.queueFsync,
		Logger:       log,
	})
	if err != nil {
		log.Error("open queue", "error", err.Error())
		return 1
	}
	defer q.Close()
	reg.SetQueueStatsFunc(q.Stats)
	reg.SetQueueOpen(true)

	ledger, err := uplink.OpenLedger(cfg.queueDir)
	if err != nil {
		log.Error("open upload ledger", "error", err.Error())
		return 1
	}
	defer ledger.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	conn := uplink.NewConn(uplink.ConnOptions{
		Target:   cfg.server,
		Insecure: cfg.insecure,
		Logger:   log,
		Status:   reg,
	})
	uploader := uplink.NewUploader(uplink.UploaderOptions{
		Conn: conn, Status: reg, Ledger: ledger, Logger: log,
		Root: cfg.evidenceRoot, RetryWindow: cfg.uploadRetryWindow,
	})
	results := uplink.NewResults(uplink.ResultsOptions{Conn: conn, Status: reg, Logger: log})
	bus := ingress.NewBus(log)

	// EdgeIngress listener (analyzer side).
	lis, err := net.Listen("tcp", cfg.listen)
	if err != nil {
		log.Error("listen", "addr", cfg.listen, "error", err.Error())
		return 1
	}
	gs := grpc.NewServer()
	ingress.New(ingress.Options{
		Queue: q, Status: reg, Bus: bus, Results: results, DeviceID: cfg.deviceID, Logger: log,
	}).Register(gs)

	var wg sync.WaitGroup
	start := func(name string, f func(context.Context)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f(ctx)
			log.Debug("loop stopped", "loop", name)
		}()
	}
	start("conn", conn.Run)
	start("sender", func(ctx context.Context) {
		uplink.RunSender(ctx, uplink.SenderOptions{Queue: q, Conn: conn, Status: reg, Logger: log, OnAcked: uploader.Schedule})
	})
	start("heartbeat", func(ctx context.Context) {
		uplink.RunHeartbeat(ctx, uplink.HeartbeatOptions{Conn: conn, Status: reg, Logger: log, Interval: cfg.heartbeatInterval})
	})
	start("commands", func(ctx context.Context) {
		uplink.RunCommandWatcher(ctx, uplink.WatcherOptions{Conn: conn, DeviceID: cfg.deviceID, Sink: bus, Logger: log})
	})
	start("results", results.Run)
	start("uploader", uploader.Run)

	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Info("EdgeIngress listening", "addr", lis.Addr().String())
		reg.SetGRPCUp(true)
		if err := gs.Serve(lis); err != nil {
			log.Error("grpc serve", "error", err.Error())
		}
		reg.SetGRPCUp(false)
	}()

	var httpSrv *http.Server
	if cfg.httpAddr != "" {
		httpSrv = &http.Server{Addr: cfg.httpAddr, Handler: reg.Handler(), ReadHeaderTimeout: 5 * time.Second}
		wg.Add(1)
		go func() {
			defer wg.Done()
			log.Info("http listening", "addr", cfg.httpAddr)
			if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Error("http serve", "error", err.Error())
			}
		}()
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigCh
	log.Info("shutting down", "signal", sig.String())

	// Order matters: stop admitting events first (so the analyzer sees
	// errors and keeps its snapshots), then stop the loops, then close the
	// WAL and ledger so every accepted event and finished upload is on disk.
	reg.SetGRPCUp(false)
	stopped := make(chan struct{})
	go func() { gs.GracefulStop(); close(stopped) }()
	select {
	case <-stopped:
	case <-time.After(5 * time.Second):
		log.Warn("grpc graceful stop timed out; forcing")
		gs.Stop()
	}
	cancel()
	if httpSrv != nil {
		sctx, scancel := context.WithTimeout(context.Background(), 3*time.Second)
		httpSrv.Shutdown(sctx)
		scancel()
	}
	wg.Wait()
	pending := q.Stats().Pending
	if err := ledger.Close(); err != nil {
		log.Warn("close ledger", "error", err.Error())
	}
	if err := q.Close(); err != nil {
		log.Error("close queue", "error", err.Error())
		return 1
	}
	log.Info("sceneproof-gateway stopped", "pending_events", pending, "pending_uploads", uploader.Pending())
	return 0
}
