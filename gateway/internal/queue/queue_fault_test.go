package queue

// What the queue promises when the disk misbehaves.
//
// The gateway answers accepted=true to the analyzer only after Put returned
// without an error. From then on the event must come back after a restart
// until the server has acknowledged it. These tests make writes fail the way
// a full disk does (a write that stops part-way, an fsync that fails, a file
// that cannot be cut back) and then reopen the queue from the same folder
// with a healthy disk.
//
// The faults are injected through Options.OpenFile: the queue opens every
// file it writes through that function, and the tests hand it a file whose
// Write, Sync and Truncate can be told to fail.

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
)

var errDiskFull = errors.New("injected: no space left on device")

// disk decides which operations fail. The zero value never fails.
type disk struct {
	mu sync.Mutex
	// failWrites is the number of coming Write calls that fail. A failing
	// Write still puts the first `partial` bytes into the file (all but the
	// last byte when partial is negative), the way a write stops when the
	// disk fills up in the middle of it.
	failWrites int
	partial    int
	failSyncs  int // coming Sync calls that fail
	// failTruncates is the number of coming Truncate calls that fail.
	failTruncates int
	failOpens     int    // coming OpenFile calls that fail ...
	onlySuffix    string // ... when the file name ends with this (empty: every file)
}

func (d *disk) open(name string, flag int, perm os.FileMode) (File, error) {
	d.mu.Lock()
	fail := d.failOpens > 0 && strings.HasSuffix(name, d.onlySuffix)
	if fail {
		d.failOpens--
	}
	d.mu.Unlock()
	if fail {
		return nil, errDiskFull
	}
	f, err := os.OpenFile(name, flag, perm)
	if err != nil {
		return nil, err
	}
	return &faultyFile{File: f, d: d, name: name}, nil
}

type faultyFile struct {
	*os.File
	d    *disk
	name string
}

func (f *faultyFile) Write(p []byte) (int, error) {
	f.d.mu.Lock()
	fail := f.d.failWrites > 0 && strings.HasSuffix(f.name, f.d.onlySuffix)
	keep := f.d.partial
	if fail {
		f.d.failWrites--
	}
	f.d.mu.Unlock()
	if !fail {
		return f.File.Write(p)
	}
	if keep < 0 || keep >= len(p) {
		keep = len(p) - 1
	}
	n, err := f.File.Write(p[:keep])
	if err != nil {
		return n, err
	}
	return n, errDiskFull
}

func (f *faultyFile) Sync() error {
	f.d.mu.Lock()
	fail := f.d.failSyncs > 0 && strings.HasSuffix(f.name, f.d.onlySuffix)
	if fail {
		f.d.failSyncs--
	}
	f.d.mu.Unlock()
	if fail {
		return errDiskFull
	}
	return f.File.Sync()
}

func (f *faultyFile) Truncate(size int64) error {
	f.d.mu.Lock()
	fail := f.d.failTruncates > 0 && strings.HasSuffix(f.name, f.d.onlySuffix)
	if fail {
		f.d.failTruncates--
	}
	f.d.mu.Unlock()
	if fail {
		return errDiskFull
	}
	return f.File.Truncate(size)
}

func openOn(t *testing.T, dir string, d *disk, fsync bool) *Queue {
	t.Helper()
	q, err := Open(Options{Dir: dir, MaxBytes: 1 << 20, DisableFsync: !fsync, CompactMinBytes: 1 << 30,
		Logger: testLogger(), OpenFile: d.open})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	return q
}

func idRevs(evs []*sceneproofv1.Event) []string {
	out := make([]string, 0, len(evs))
	for _, e := range evs {
		out = append(out, fmt.Sprintf("%s@%d", e.EventId, e.Revision))
	}
	return out
}

// The defect this file was written for. A Put fails with part of its record
// in the file; the events accepted after it must still be there after a
// restart, and recovery must not have had to cut anything off.
func TestEventsAcceptedAfterAFailedWriteSurviveRestart(t *testing.T) {
	record := len(encodeRecord(recPut, mustMarshal(t, mkEvent("b", 1, 40))))
	for _, partial := range []int{0, 1, 3, 4, 7, headerBytes, headerBytes + 1, record / 2, record - 1} {
		t.Run(fmt.Sprintf("%d_of_%d_bytes_written", partial, record), func(t *testing.T) {
			dir := t.TempDir()
			d := &disk{}
			q := openOn(t, dir, d, false)
			if _, err := q.Put(mkEvent("a", 1, 40)); err != nil {
				t.Fatalf("put a: %v", err)
			}
			d.failWrites, d.partial = 1, partial
			if res, err := q.Put(mkEvent("b", 1, 40)); err == nil {
				t.Fatalf("put b: the write failed, so Put must return an error (got %+v)", res)
			}
			if got := idRevs(q.Pending()); !equalStrings(got, []string{"a@1"}) {
				t.Fatalf("after the failed put, pending = %v, want [a@1]", got)
			}
			// the disk has room again
			if res, err := q.Put(mkEvent("c", 1, 40)); err != nil || !res.Stored {
				t.Fatalf("put c: %+v, %v", res, err)
			}
			if res, err := q.Put(mkEvent("b", 1, 40)); err != nil || !res.Stored || res.Duplicate {
				t.Fatalf("put b again (the analyzer retries): %+v, %v", res, err)
			}
			st, err := os.Stat(filepath.Join(dir, FileName))
			if err != nil {
				t.Fatal(err)
			}
			if got := q.Stats().FileBytes; got != st.Size() {
				t.Fatalf("Stats.FileBytes = %d, the file is %d bytes", got, st.Size())
			}
			if err := q.Close(); err != nil {
				t.Fatalf("close: %v", err)
			}

			q2 := openTest(t, dir, 1<<20)
			defer q2.Close()
			if got, want := idRevs(q2.Pending()), []string{"a@1", "c@1", "b@1"}; !equalStrings(got, want) {
				t.Fatalf("after restart, pending = %v, want %v: an accepted event was lost", got, want)
			}
			if q2.Stats().Truncated {
				t.Fatalf("recovery had to cut off a corrupt tail; a failed write must not leave one behind accepted records")
			}
		})
	}
}

// The same with a crash instead of a clean shutdown right after the failed
// write: nothing accepted is lost. (Whether recovery reports a cut-off tail
// here depends on when the queue repairs the file; both are fine.)
func TestCrashRightAfterAFailedWrite(t *testing.T) {
	dir := t.TempDir()
	d := &disk{}
	q := openOn(t, dir, d, false)
	for _, id := range []string{"a", "b"} {
		if _, err := q.Put(mkEvent(id, 1, 40)); err != nil {
			t.Fatalf("put %s: %v", id, err)
		}
	}
	d.failWrites, d.partial = 1, 11
	if _, err := q.Put(mkEvent("c", 1, 40)); err == nil {
		t.Fatal("put c: want an error")
	}
	// no Close: the process dies here

	q2 := openTest(t, dir, 1<<20)
	defer q2.Close()
	if got, want := idRevs(q2.Pending()), []string{"a@1", "b@1"}; !equalStrings(got, want) {
		t.Fatalf("after the crash, pending = %v, want %v", got, want)
	}
}

// An fsync that fails means the record may not be on the disk: Put must not
// claim it is, and the event must not be pending.
func TestFailedFsyncIsNotAccepted(t *testing.T) {
	dir := t.TempDir()
	d := &disk{}
	q := openOn(t, dir, d, true)
	defer q.Close()
	if _, err := q.Put(mkEvent("a", 1, 40)); err != nil {
		t.Fatalf("put a: %v", err)
	}
	d.failSyncs = 1
	if res, err := q.Put(mkEvent("b", 1, 40)); err == nil {
		t.Fatalf("put b: fsync failed, so Put must return an error (got %+v)", res)
	}
	if got := idRevs(q.Pending()); !equalStrings(got, []string{"a@1"}) {
		t.Fatalf("pending = %v, want [a@1]", got)
	}
	if res, err := q.Put(mkEvent("b", 1, 40)); err != nil || !res.Stored {
		t.Fatalf("put b again: %+v, %v", res, err)
	}
}

// A failed Ack leaves the event pending (it is sent again; the server
// deduplicates), and does not damage what is accepted afterwards.
func TestFailedAckKeepsTheEventAndTheFile(t *testing.T) {
	dir := t.TempDir()
	d := &disk{}
	q := openOn(t, dir, d, false)
	for _, id := range []string{"a", "b"} {
		if _, err := q.Put(mkEvent(id, 1, 40)); err != nil {
			t.Fatalf("put %s: %v", id, err)
		}
	}
	d.failWrites, d.partial = 1, 6
	if removed, err := q.Ack("a", 1); err == nil || removed {
		t.Fatalf("ack a: removed=%v err=%v, want an error and removed=false", removed, err)
	}
	if got := idRevs(q.Pending()); !equalStrings(got, []string{"a@1", "b@1"}) {
		t.Fatalf("pending = %v, want [a@1 b@1]", got)
	}
	if _, err := q.Put(mkEvent("c", 1, 40)); err != nil {
		t.Fatalf("put c: %v", err)
	}
	if removed, err := q.Ack("b", 1); err != nil || !removed {
		t.Fatalf("ack b: removed=%v err=%v", removed, err)
	}
	if err := q.Close(); err != nil {
		t.Fatal(err)
	}
	q2 := openTest(t, dir, 1<<20)
	defer q2.Close()
	if got, want := idRevs(q2.Pending()), []string{"a@1", "c@1"}; !equalStrings(got, want) {
		t.Fatalf("after restart, pending = %v, want %v", got, want)
	}
}

// A compaction that cannot write its new file changes nothing: the queue
// keeps working on the old file and leaves no half-written file behind.
func TestFailedCompactionChangesNothing(t *testing.T) {
	for name, breakIt := range map[string]func(d *disk){
		"cannot_create_the_new_file": func(d *disk) { d.failOpens = 1 },
		"write_fails":                func(d *disk) { d.failWrites, d.partial = 1, 5 },
		"fsync_fails":                func(d *disk) { d.failSyncs = 1 },
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			d := &disk{}
			q := openOn(t, dir, d, false)
			for _, id := range []string{"a", "b", "c"} {
				if _, err := q.Put(mkEvent(id, 1, 40)); err != nil {
					t.Fatalf("put %s: %v", id, err)
				}
			}
			if _, err := q.Ack("b", 1); err != nil {
				t.Fatal(err)
			}
			breakIt(d)
			if err := q.Compact(); err == nil {
				t.Fatal("Compact: want an error")
			}
			d.mu.Lock()
			d.failWrites, d.failSyncs, d.failOpens = 0, 0, 0
			d.mu.Unlock()
			if got := idRevs(q.Pending()); !equalStrings(got, []string{"a@1", "c@1"}) {
				t.Fatalf("pending = %v, want [a@1 c@1]", got)
			}
			if res, err := q.Put(mkEvent("e", 1, 40)); err != nil || !res.Stored {
				t.Fatalf("put e after the failed compaction: %+v, %v", res, err)
			}
			if err := q.Close(); err != nil {
				t.Fatal(err)
			}
			files, err := os.ReadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			if len(files) != 1 || files[0].Name() != FileName {
				names := []string{}
				for _, f := range files {
					names = append(names, f.Name())
				}
				t.Fatalf("the queue folder holds %v, want only %s", names, FileName)
			}
			q2 := openTest(t, dir, 1<<20)
			defer q2.Close()
			if got, want := idRevs(q2.Pending()), []string{"a@1", "c@1", "e@1"}; !equalStrings(got, want) {
				t.Fatalf("after restart, pending = %v, want %v", got, want)
			}
		})
	}
}

// Many random operations on a disk that fails at random. Whatever happened,
// two things hold after a clean shutdown and a restart on a healthy disk:
//
//   - every event whose Put returned Stored and that was not acknowledged
//     since is pending, at that revision or a newer one;
//   - an event whose Ack succeeded, and that was not offered again since, is
//     not pending.
func TestRandomFaultsNeverLoseAnAcceptedEvent(t *testing.T) {
	for seed := int64(1); seed <= 40; seed++ {
		seed := seed
		t.Run(fmt.Sprintf("seed_%d", seed), func(t *testing.T) {
			rng := rand.New(rand.NewSource(seed))
			dir := t.TempDir()
			d := &disk{}
			q, err := Open(Options{Dir: dir, MaxBytes: 1 << 20, DisableFsync: seed%2 == 0, CompactMinBytes: 2048,
				Logger: testLogger(), OpenFile: d.open})
			if err != nil {
				t.Fatalf("open: %v", err)
			}
			promised := map[string]int32{} // accepted and not acknowledged since
			acked := map[string]bool{}     // acknowledged and not offered again since
			revision := map[string]int32{} // newest revision ever offered
			for step := 0; step < 300; step++ {
				d.mu.Lock()
				d.failWrites, d.failSyncs, d.failTruncates, d.failOpens = 0, 0, 0, 0
				switch rng.Intn(6) {
				case 0:
					d.failWrites, d.partial = 1+rng.Intn(2), rng.Intn(120)-20
				case 1:
					d.failSyncs = 1
				case 2:
					d.failWrites, d.partial, d.failTruncates = 1, rng.Intn(60), 1+rng.Intn(2)
				case 3:
					d.failOpens = 1
				}
				d.mu.Unlock()

				id := fmt.Sprintf("ev-%d", rng.Intn(12))
				switch rng.Intn(10) {
				case 0, 1, 2:
					if revision[id] == 0 {
						continue
					}
					removed, err := q.Ack(id, revision[id])
					if err == nil && removed {
						delete(promised, id)
						acked[id] = true
					}
				case 3:
					_ = q.Compact()
				default:
					revision[id]++
					delete(acked, id) // even a failed Put may leave its record in the file
					res, err := q.Put(mkEvent(id, revision[id], 10+rng.Intn(200)))
					if err == nil && res.Stored {
						promised[id] = revision[id]
					}
				}
			}
			d.mu.Lock()
			d.failWrites, d.failSyncs, d.failTruncates, d.failOpens = 0, 0, 0, 0
			d.mu.Unlock()
			if err := q.Close(); err != nil {
				t.Fatalf("close: %v", err)
			}

			q2 := openTest(t, dir, 1<<20)
			defer q2.Close()
			pending := map[string]int32{}
			for _, ev := range q2.Pending() {
				pending[ev.EventId] = ev.Revision
			}
			for id, rev := range promised {
				if got, ok := pending[id]; !ok || got < rev {
					t.Errorf("%s was accepted at revision %d and not acknowledged; after restart: pending=%v revision=%d", id, rev, ok, got)
				}
			}
			for id := range acked {
				if rev, ok := pending[id]; ok {
					t.Errorf("%s was acknowledged and not offered again, but revision %d is pending after restart", id, rev)
				}
			}
		})
	}
}
