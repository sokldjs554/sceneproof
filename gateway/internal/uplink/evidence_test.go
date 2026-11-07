package uplink

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
	"time"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
)

func writeFile(t *testing.T, root, rel string, n int) []byte {
	t.Helper()
	full := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	data := make([]byte, n)
	rand.Read(data)
	if err := os.WriteFile(full, data, 0o644); err != nil {
		t.Fatal(err)
	}
	// Make the file look settled (older than SettleTime).
	old := time.Now().Add(-time.Minute)
	os.Chtimes(full, old, old)
	return data
}

func shaHex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

func startUploader(h *harness, root, ledgerDir string, window time.Duration) (*Uploader, *Ledger) {
	h.t.Helper()
	ledger, err := OpenLedger(ledgerDir)
	if err != nil {
		h.t.Fatal(err)
	}
	up := NewUploader(UploaderOptions{
		Conn: h.conn, Status: h.reg, Ledger: ledger, Logger: testLogger(), Root: root,
		RetryWindow: window, SettleTime: 20 * time.Millisecond, RPCTimeout: 5 * time.Second,
		Backoff: BackoffConfig{Base: 10 * time.Millisecond, Max: 40 * time.Millisecond},
	})
	h.go_(func() { up.Run(h.ctx) })
	h.t.Cleanup(func() { ledger.Close() })
	return up, ledger
}

func evidenceEvent(id string, frames []string, clip string) *sceneproofv1.Event {
	ev := mkEvent(id, 2)
	ev.Evidence = &sceneproofv1.Evidence{ClipPath: clip}
	for _, p := range frames {
		ev.Evidence.KeyFrames = append(ev.Evidence.KeyFrames, &sceneproofv1.KeyFrame{Role: "entry", Path: p})
	}
	return ev
}

func TestUploadChunkingShaAndAlreadyPresent(t *testing.T) {
	srv := newFakeServer(t)
	h := newHarness(t, srv)
	root := t.TempDir()
	up, ledger := startUploader(h, root, t.TempDir(), time.Minute)

	frame := writeFile(t, root, "ev1/entry.jpg", 300*1024) // 2 chunks
	clip := writeFile(t, root, "ev1/clip.mp4", 4*DefaultChunkSize+1)
	empty := writeFile(t, root, "ev1/empty.bin", 0)
	present := writeFile(t, root, "ev1/present.jpg", 1000)
	srv.svc.mu.Lock()
	srv.svc.present[uploadKey("ev1", "ev1/present.jpg")] = true
	srv.svc.mu.Unlock()

	up.Schedule(evidenceEvent("ev1", []string{"ev1/entry.jpg", "ev1/empty.bin", "ev1/present.jpg"}, "ev1/clip.mp4"))
	waitFor(t, 10*time.Second, func() bool { return up.Pending() == 0 && srv.svc.uploadCount() == 4 }, "all uploaded")

	cases := []struct {
		path   string
		data   []byte
		chunks int
	}{
		{"ev1/entry.jpg", frame, 2},
		{"ev1/clip.mp4", clip, 5},
		{"ev1/empty.bin", empty, 1},
		{"ev1/present.jpg", present, 1},
	}
	for _, c := range cases {
		got := srv.svc.upload("ev1", c.path)
		if got == nil {
			t.Fatalf("%s not uploaded", c.path)
		}
		if got.chunks != c.chunks {
			t.Errorf("%s chunks = %d, want %d", c.path, got.chunks, c.chunks)
		}
		if got.sha != shaHex(c.data) || string(got.data) != string(c.data) || got.total != int64(len(c.data)) {
			t.Errorf("%s content/digest mismatch", c.path)
		}
		if !ledger.Has("ev1", c.path) {
			t.Errorf("%s missing from ledger", c.path)
		}
	}
	if !srv.svc.upload("ev1", "ev1/present.jpg").already {
		t.Fatal("present file should have been answered already_present")
	}
	snap := h.reg.Snapshot()
	if snap.Uploads.Done != 4 || snap.Uploads.Pending != 0 || snap.Uploads.Failed != 0 {
		t.Fatalf("uploads = %+v", snap.Uploads)
	}
	if h.reg.UplinkStatus().PendingUploads != 0 {
		t.Fatal("pending_uploads should be 0")
	}
	// Unsafe paths are refused without touching the server.
	up.Schedule(evidenceEvent("evil", []string{"../../etc/passwd"}, "/etc/hostname"))
	if up.Pending() != 0 {
		t.Fatal("unsafe paths were scheduled")
	}
}

func TestMissingFileRetriedThenUploadedAndGivenUp(t *testing.T) {
	srv := newFakeServer(t)
	h := newHarness(t, srv)
	root := t.TempDir()
	up, ledger := startUploader(h, root, t.TempDir(), 400*time.Millisecond)

	up.Schedule(evidenceEvent("ev2", []string{"ev2/late.jpg"}, "ev2/never.mp4"))
	time.Sleep(120 * time.Millisecond)
	if up.Pending() != 2 || srv.svc.uploadCount() != 0 {
		t.Fatalf("pending=%d uploads=%d before file exists", up.Pending(), srv.svc.uploadCount())
	}
	if h.reg.UplinkStatus().PendingUploads != 2 {
		t.Fatal("pending_uploads not reported")
	}
	data := writeFile(t, root, "ev2/late.jpg", 5000)
	waitFor(t, 5*time.Second, func() bool { return srv.svc.upload("ev2", "ev2/late.jpg") != nil }, "late file uploaded")
	if got := srv.svc.upload("ev2", "ev2/late.jpg"); got.sha != shaHex(data) {
		t.Fatal("digest mismatch")
	}
	// The clip never appears: after the retry window it is given up on.
	waitFor(t, 5*time.Second, func() bool { return up.Pending() == 0 }, "never.mp4 given up")
	if snap := h.reg.Snapshot(); snap.Uploads.Failed != 1 || snap.Uploads.Done != 1 {
		t.Fatalf("uploads = %+v", snap.Uploads)
	}
	if srv.svc.uploadCount() != 1 {
		t.Fatalf("uploads = %d", srv.svc.uploadCount())
	}
	if !ledger.Has("ev2", "ev2/never.mp4") {
		t.Fatal("given-up file should be recorded so it is not revived")
	}
}

func TestNoReuploadAfterRestartAndOutstandingResumed(t *testing.T) {
	srv := newFakeServer(t)
	h := newHarness(t, srv)
	root := t.TempDir()
	ledgerDir := t.TempDir()
	up, ledger := startUploader(h, root, ledgerDir, time.Minute)

	writeFile(t, root, "ev3/entry.jpg", 2048)
	ev := evidenceEvent("ev3", []string{"ev3/entry.jpg"}, "ev3/clip.mp4") // clip not written yet
	up.Schedule(ev)
	waitFor(t, 5*time.Second, func() bool { return srv.svc.upload("ev3", "ev3/entry.jpg") != nil }, "entry uploaded")
	waitFor(t, time.Second, func() bool { return up.Pending() == 1 }, "clip pending")

	// "Restart": stop the first uploader's ledger, open a fresh one on the
	// same directory and build a new uploader from it.
	h.cancel()
	h.wg.Wait()
	ledger.Close()
	h2 := newHarness(t, srv)
	up2, ledger2 := startUploader(h2, root, ledgerDir, time.Minute)
	if !ledger2.Has("ev3", "ev3/entry.jpg") {
		t.Fatal("ledger lost the completed upload")
	}
	if up2.Pending() != 1 {
		t.Fatalf("outstanding clip not resumed: pending=%d", up2.Pending())
	}
	// Re-scheduling the same event (a re-sent revision) adds nothing.
	up2.Schedule(ev)
	if up2.Pending() != 1 {
		t.Fatalf("re-schedule changed pending to %d", up2.Pending())
	}
	time.Sleep(100 * time.Millisecond)
	if n := srv.svc.uploadCount(); n != 1 {
		t.Fatalf("entry.jpg re-uploaded: uploads=%d", n)
	}
	// The clip appears after the restart and is uploaded by the new process.
	clip := writeFile(t, root, "ev3/clip.mp4", 70000)
	waitFor(t, 5*time.Second, func() bool { return srv.svc.upload("ev3", "ev3/clip.mp4") != nil }, "clip uploaded after restart")
	if srv.svc.upload("ev3", "ev3/clip.mp4").sha != shaHex(clip) {
		t.Fatal("clip digest mismatch")
	}
	if srv.svc.upload("ev3", "ev3/entry.jpg").chunks != 1 || srv.svc.uploadCount() != 2 {
		t.Fatalf("unexpected uploads: %d", srv.svc.uploadCount())
	}
}
