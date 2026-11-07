// Package version exposes the gateway build version to every component
// (heartbeat, /status, logs) from one place.
package version

import (
	"runtime"
	"sync/atomic"
)

// Version is the build version. It defaults to "dev" and is overridden at
// startup from main's -ldflags "-X main.version=..." value via Set, so the
// Makefile only has to know one symbol.
var current atomic.Pointer[string]

func init() {
	v := "dev"
	current.Store(&v)
}

// Set overrides the version; empty strings are ignored.
func Set(v string) {
	if v == "" {
		return
	}
	current.Store(&v)
}

// Get returns the current version string.
func Get() string { return *current.Load() }

// Full returns the version together with the Go runtime and platform, for
// --version output and the /status endpoint.
func Full() string {
	return Get() + " (" + runtime.Version() + " " + runtime.GOOS + "/" + runtime.GOARCH + ")"
}
