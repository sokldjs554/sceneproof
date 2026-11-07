package main

import (
	"testing"
	"time"
)

func TestFlagsWithEnvFallback(t *testing.T) {
	t.Setenv("ZW_GW_SERVER", "10.0.0.5:50052")
	t.Setenv("ZW_GW_QUEUE_MAX_BYTES", "1024")
	t.Setenv("ZW_GW_HEARTBEAT_INTERVAL", "7s")
	t.Setenv("ZW_GW_INSECURE", "true")
	t.Setenv("ZW_GW_QUEUE_FSYNC", "false")
	cfg, err := parseFlags([]string{"--device-id", "edge-42", "--server", "flag-wins:1"})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.server != "flag-wins:1" || cfg.deviceID != "edge-42" || cfg.queueMaxBytes != 1024 ||
		cfg.heartbeatInterval != 7*time.Second || !cfg.insecure || cfg.queueFsync {
		t.Fatalf("cfg = %+v", cfg)
	}
	if cfg.listen != ":50051" || cfg.httpAddr != ":9100" || cfg.uploadRetryWindow != 10*time.Minute || cfg.queueDir != "./data/gateway" {
		t.Fatalf("defaults = %+v", cfg)
	}
}
