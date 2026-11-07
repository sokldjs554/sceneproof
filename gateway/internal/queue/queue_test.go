package queue

import (
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
)

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
}

func openTest(t *testing.T, dir string, maxBytes int64) *Queue {
	t.Helper()
	q, err := Open(Options{Dir: dir, MaxBytes: maxBytes, DisableFsync: true, Logger: testLogger()})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	return q
}

func mkEvent(id string, rev int32, pad int) *sceneproofv1.Event {
	return &sceneproofv1.Event{
		EventId:  id,
		Revision: rev,
		DeviceId: "edge-01",
		CameraId: "cam-1",
		ZoneId:   "zone-a",
		Source:   strings.Repeat("x", pad),
	}
}

func ids(evs []*sceneproofv1.Event) []string {
	out := make([]string, 0, len(evs))
	for _, e := range evs {
		out = append(out, e.EventId)
	}
	return out
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestPutCoalesceAckOrder(t *testing.T) {
	q := openTest(t, t.TempDir(), 0)
	defer q.Close()

	for _, id := range []string{"a", "b", "c"} {
		res, err := q.Put(mkEvent(id, 1, 10))
		if err != nil || !res.Stored {
			t.Fatalf("put %s: %+v %v", id, res, err)
		}
	}
	// Newer revision of "a" coalesces in place; "a" stays first.
	res, err := q.Put(mkEvent("a", 3, 10))
	if err != nil || !res.Stored {
		t.Fatalf("put a rev3: %+v %v", res, err)
	}
	// Older/equal revision is a duplicate and writes nothing.
	res, err = q.Put(mkEvent("a", 2, 10))
	if err != nil || !res.Duplicate || res.Stored {
		t.Fatalf("put a rev2 should be duplicate: %+v %v", res, err)
	}
	pend := q.Pending()
	if !equalStrings(ids(pend), []string{"a", "b", "c"}) {
		t.Fatalf("order = %v", ids(pend))
	}
	if pend[0].Revision != 3 {
		t.Fatalf("a revision = %d, want 3", pend[0].Revision)
	}
	// Ack of an older revision must not remove the newer pending one.
	if removed, _ := q.Ack("a", 2); removed {
		t.Fatal("ack rev2 removed pending rev3")
	}
	if removed, _ := q.Ack("a", 3); !removed {
		t.Fatal("ack rev3 did not remove")
	}
	if removed, _ := q.Ack("b", 7); !removed {
		t.Fatal("ack with higher revision should remove")
	}
	if removed, _ := q.Ack("zzz", 1); removed {
		t.Fatal("ack unknown removed something")
	}
	if got := ids(q.Pending()); !equalStrings(got, []string{"c"}) {
		t.Fatalf("pending = %v", got)
	}
	if _, err := q.Put(&sceneproofv1.Event{}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("empty id err = %v", err)
	}
	select {
	case <-q.Wake():
	default:
		t.Fatal("expected wake signal after puts")
	}
}

func TestEvictionWhenOverBudget(t *testing.T) {
	dir := t.TempDir()
	one, _ := proto.Marshal(mkEvent("probe", 1, 100))
	max := int64(len(one))*4 + 2 // room for ~4 events
	q := openTest(t, dir, max)

	for i := 0; i < 10; i++ {
		res, err := q.Put(mkEvent(string(rune('a'+i)), 1, 100))
		if err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
		if i >= 4 && res.Evicted != 1 {
			t.Fatalf("put %d evicted %d, want 1", i, res.Evicted)
		}
	}
	st := q.Stats()
	if st.LiveBytes > max {
		t.Fatalf("live %d > max %d", st.LiveBytes, max)
	}
	if st.Dropped != 6 {
		t.Fatalf("dropped = %d, want 6", st.Dropped)
	}
	want := []string{"g", "h", "i", "j"}
	if got := ids(q.Pending()); !equalStrings(got, want) {
		t.Fatalf("pending = %v, want %v", got, want)
	}
	if _, err := q.Put(mkEvent("huge", 1, int(max)+10)); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("oversize err = %v", err)
	}
	q.Close()

	// Eviction is durable: the dropped events do not come back.
	q2 := openTest(t, dir, max)
	defer q2.Close()
	if got := ids(q2.Pending()); !equalStrings(got, want) {
		t.Fatalf("after reopen pending = %v, want %v", got, want)
	}
}

func TestRecoveryAfterReopen(t *testing.T) {
	dir := t.TempDir()
	q := openTest(t, dir, 0)
	q.Put(mkEvent("e1", 1, 20))
	q.Put(mkEvent("e2", 1, 20))
	q.Put(mkEvent("e3", 1, 20))
	q.Put(mkEvent("e2", 5, 30)) // coalesce
	q.Ack("e1", 1)
	q.Put(mkEvent("e4", 2, 20))
	q.Put(mkEvent("e3", 0, 20)) // duplicate (older)
	before := q.Pending()
	if err := q.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := q.Put(mkEvent("late", 1, 1)); !errors.Is(err, ErrClosed) {
		t.Fatalf("put after close err = %v", err)
	}

	q2 := openTest(t, dir, 0)
	defer q2.Close()
	after := q2.Pending()
	if !equalStrings(ids(before), ids(after)) {
		t.Fatalf("order before %v after %v", ids(before), ids(after))
	}
	if !equalStrings(ids(after), []string{"e2", "e3", "e4"}) {
		t.Fatalf("pending = %v", ids(after))
	}
	for i := range before {
		if !proto.Equal(before[i], after[i]) {
			t.Fatalf("event %d differs after reopen", i)
		}
	}
	if after[0].Revision != 5 {
		t.Fatalf("e2 revision = %d", after[0].Revision)
	}
	st := q2.Stats()
	if st.Pending != 3 || st.LiveBytes <= 0 {
		t.Fatalf("stats = %+v", st)
	}
}

func TestCorruptTailTolerated(t *testing.T) {
	dir := t.TempDir()
	q := openTest(t, dir, 0)
	q.Put(mkEvent("k1", 1, 40))
	q.Put(mkEvent("k2", 1, 40))
	q.Close()
	path := filepath.Join(dir, FileName)
	goodSize, err := ScanFile(path, func(byte, []byte) error { return nil })
	if err != nil {
		t.Fatalf("clean file reported corrupt: %v", err)
	}

	// Case 1: a torn record (header + half a body) at the tail.
	f, _ := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
	full := encodeRecord(recPut, mustMarshal(t, mkEvent("k3", 1, 40)))
	f.Write(full[:len(full)/2])
	f.Close()
	if _, err := ScanFile(path, func(byte, []byte) error { return nil }); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("expected ErrCorrupt, got %v", err)
	}
	q2 := openTest(t, dir, 0)
	if got := ids(q2.Pending()); !equalStrings(got, []string{"k1", "k2"}) {
		t.Fatalf("pending = %v", got)
	}
	if !q2.Stats().Truncated {
		t.Fatal("Truncated flag not set")
	}
	// Writes after recovery land on a clean file.
	q2.Put(mkEvent("k4", 1, 40))
	q2.Close()
	if _, err := ScanFile(path, func(byte, []byte) error { return nil }); err != nil {
		t.Fatalf("file not clean after recovery+put: %v", err)
	}

	// Case 2: a complete-looking record with a bad CRC, followed by garbage.
	f, _ = os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
	bad := encodeRecord(recPut, mustMarshal(t, mkEvent("k5", 1, 40)))
	bad[4] ^= 0xff
	f.Write(bad)
	f.Write([]byte("garbage garbage garbage"))
	f.Close()
	q3 := openTest(t, dir, 0)
	defer q3.Close()
	if got := ids(q3.Pending()); !equalStrings(got, []string{"k1", "k2", "k4"}) {
		t.Fatalf("pending = %v", got)
	}
	st, _ := os.Stat(path)
	if st.Size() < goodSize {
		t.Fatalf("file shrank below the good prefix: %d < %d", st.Size(), goodSize)
	}
}

func TestCompactionKeepsOnlyLiveRecords(t *testing.T) {
	dir := t.TempDir()
	q := openTest(t, dir, 0)
	for i := 0; i < 50; i++ {
		q.Put(mkEvent("c"+string(rune('A'+i%26))+string(rune('a'+i/26)), int32(i%3+1), 200))
	}
	// Supersede and ack most of them so the file is mostly dead records.
	pend := q.Pending()
	for i, ev := range pend {
		if i%5 == 0 {
			q.Put(mkEvent(ev.EventId, ev.Revision+10, 200))
		} else {
			q.Ack(ev.EventId, ev.Revision)
		}
	}
	livePending := q.Pending()
	stBefore := q.Stats()
	if err := q.Compact(); err != nil {
		t.Fatalf("compact: %v", err)
	}
	stAfter := q.Stats()
	if stAfter.Compactions != stBefore.Compactions+1 {
		t.Fatalf("compactions %d -> %d", stBefore.Compactions, stAfter.Compactions)
	}
	if stAfter.FileBytes >= stBefore.FileBytes {
		t.Fatalf("file did not shrink: %d -> %d", stBefore.FileBytes, stAfter.FileBytes)
	}
	// Only PUT records of live events remain, in FIFO order.
	var kinds []byte
	var seen []string
	_, err := ScanFile(filepath.Join(dir, FileName), func(typ byte, payload []byte) error {
		kinds = append(kinds, typ)
		ev := &sceneproofv1.Event{}
		if err := proto.Unmarshal(payload, ev); err != nil {
			return err
		}
		seen = append(seen, ev.EventId)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range kinds {
		if k != recPut {
			t.Fatalf("non-PUT record %d after compaction", k)
		}
	}
	if !equalStrings(seen, ids(livePending)) {
		t.Fatalf("records %v != live %v", seen, ids(livePending))
	}
	if len(seen) != 10 {
		t.Fatalf("live records = %d, want 10", len(seen))
	}
	// Queue keeps working and recovering after compaction.
	q.Put(mkEvent("post", 1, 10))
	q.Close()
	q2 := openTest(t, dir, 0)
	defer q2.Close()
	want := append(ids(livePending), "post")
	if got := ids(q2.Pending()); !equalStrings(got, want) {
		t.Fatalf("after reopen %v want %v", got, want)
	}
}

func TestAutomaticCompactionAndStartupCompaction(t *testing.T) {
	dir := t.TempDir()
	q, err := Open(Options{Dir: dir, DisableFsync: true, CompactMinBytes: 1, Logger: testLogger()})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		q.Put(mkEvent("x", int32(i+1), 100)) // each PUT supersedes the previous one
	}
	st := q.Stats()
	if st.Compactions == 0 {
		t.Fatalf("expected automatic compaction, stats %+v", st)
	}
	if st.FileBytes > 2*(st.LiveBytes+headerBytes+1) {
		t.Fatalf("file %d not compacted relative to live %d", st.FileBytes, st.LiveBytes)
	}
	q.Close()

	// Append dead records with a queue whose threshold is high, then reopen:
	// startup compaction should leave only live records.
	q2 := openTest(t, dir, 0)
	q2.Put(mkEvent("y", 1, 100))
	q2.Ack("y", 1)
	q2.Close()
	q3 := openTest(t, dir, 0)
	defer q3.Close()
	st3 := q3.Stats()
	if st3.FileBytes != st3.LiveBytes+int64(st3.Pending)*(headerBytes+1) {
		t.Fatalf("startup compaction left dead bytes: %+v", st3)
	}
	if got := ids(q3.Pending()); !equalStrings(got, []string{"x"}) || q3.Pending()[0].Revision != 20 {
		t.Fatalf("pending = %v", got)
	}
}

func mustMarshal(t *testing.T, m proto.Message) []byte {
	t.Helper()
	b, err := proto.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
