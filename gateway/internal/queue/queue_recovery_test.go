package queue

// The file format, and what recovery makes of a file that was cut off or
// damaged. The format is fixed: gateways in the field have files written by
// the version before this one, and those must still open.

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// frame builds a record by hand, straight from the format in the README:
// uint32 length (type byte + payload), uint32 CRC-32 (IEEE) of type+payload,
// the type byte, the payload; integers little-endian.
func frame(typ byte, payload []byte) []byte {
	body := append([]byte{typ}, payload...)
	out := make([]byte, 8, 8+len(body))
	binary.LittleEndian.PutUint32(out[0:4], uint32(len(body)))
	binary.LittleEndian.PutUint32(out[4:8], crc32.ChecksumIEEE(body))
	return append(out, body...)
}

// key is the payload of an ACK or DROP record: uint32 revision, then the id.
func key(id string, revision int32) []byte {
	out := make([]byte, 4, 4+len(id))
	binary.LittleEndian.PutUint32(out, uint32(revision))
	return append(out, id...)
}

func readWAL(t *testing.T, dir string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, FileName))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestRecordsAreWrittenInTheDocumentedFormat(t *testing.T) {
	dir := t.TempDir()
	q, err := Open(Options{Dir: dir, MaxBytes: 300, DisableFsync: true, CompactMinBytes: 1 << 30, Logger: testLogger()})
	if err != nil {
		t.Fatal(err)
	}
	a, b, c := mkEvent("alpha", 3, 100), mkEvent("beta", 1, 100), mkEvent("gamma", 7, 100)
	want := []byte{}
	if _, err := q.Put(a); err != nil {
		t.Fatal(err)
	}
	want = append(want, frame(1, mustMarshal(t, a))...)
	if _, err := q.Put(b); err != nil {
		t.Fatal(err)
	}
	want = append(want, frame(1, mustMarshal(t, b))...)
	if removed, err := q.Ack("beta", 1); err != nil || !removed {
		t.Fatalf("ack: %v %v", removed, err)
	}
	want = append(want, frame(2, key("beta", 1))...)
	if _, err := q.Put(b); err != nil {
		t.Fatal(err)
	}
	want = append(want, frame(1, mustMarshal(t, b))...)
	// the third event does not fit the 300-byte budget: the oldest one is dropped
	res, err := q.Put(c)
	if err != nil || res.Evicted != 1 {
		t.Fatalf("put gamma: %+v, %v (want one eviction)", res, err)
	}
	want = append(want, frame(1, mustMarshal(t, c))...)
	want = append(want, frame(3, key("alpha", 3))...)
	if err := q.Close(); err != nil {
		t.Fatal(err)
	}
	if got := readWAL(t, dir); !bytes.Equal(got, want) {
		t.Fatalf("the file differs from the documented format\n got %d bytes: %x\nwant %d bytes: %x", len(got), got, len(want), want)
	}
}

func TestEncodeRecordMatchesTheFormat(t *testing.T) {
	for _, payload := range [][]byte{{}, {0}, []byte("hello"), bytes.Repeat([]byte{0xAB}, 70000)} {
		if got, want := encodeRecord(recAck, payload), frame(2, payload); !bytes.Equal(got, want) {
			t.Fatalf("encodeRecord(recAck, %d bytes) = %x..., want %x...", len(payload), got[:9], want[:9])
		}
	}
}

// A file written by the version before this one (testdata/v1.wal, kept as
// bytes) opens to the same pending events. It holds every kind of record,
// one record of a type this version does not know and one whose checksum is
// right but whose event cannot be parsed: both are skipped, not treated as
// the end of the file.
func TestOpensAFileFromThePreviousVersion(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "v1.wal"))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, FileName), data, 0o644); err != nil {
		t.Fatal(err)
	}
	q := openTest(t, dir, 1<<20)
	defer q.Close()
	if got, want := idRevs(q.Pending()), []string{"door-1@4", "dock-7@1", "yard-2@2", "gate-9@1"}; !equalStrings(got, want) {
		t.Fatalf("pending = %v, want %v", got, want)
	}
	if q.Stats().Truncated {
		t.Fatal("the file has no corrupt tail, but recovery cut something off")
	}
}

// Power can go at any byte. Whatever prefix of the file is left, reopening
// gives exactly the state after the last record that is completely in it.
func TestRecoveryFromEveryCutPoint(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	dir := t.TempDir()
	q, err := Open(Options{Dir: dir, MaxBytes: 1 << 20, DisableFsync: true, CompactMinBytes: 1 << 30, Logger: testLogger()})
	if err != nil {
		t.Fatal(err)
	}
	type state struct {
		end     int64
		pending []string
	}
	states := []state{{0, []string{}}}
	revision := map[string]int32{}
	for step := 0; step < 60; step++ {
		id := fmt.Sprintf("ev-%d", rng.Intn(8))
		switch rng.Intn(4) {
		case 0:
			// also acknowledges older revisions and unknown events: those write nothing
			if _, err := q.Ack(id, revision[id]-int32(rng.Intn(2))); err != nil {
				t.Fatal(err)
			}
		default:
			if rng.Intn(5) > 0 {
				revision[id]++ // otherwise the same revision again: a duplicate, nothing is written
			}
			if _, err := q.Put(mkEvent(id, revision[id], rng.Intn(60))); err != nil && !errors.Is(err, ErrInvalid) {
				t.Fatal(err)
			}
		}
		states = append(states, state{q.Stats().FileBytes, idRevs(q.Pending())})
	}
	if err := q.Close(); err != nil {
		t.Fatal(err)
	}
	data := readWAL(t, dir)
	if int64(len(data)) != states[len(states)-1].end {
		t.Fatalf("Stats.FileBytes said %d, the file is %d bytes", states[len(states)-1].end, len(data))
	}

	cuts := map[int64]bool{}
	for _, s := range states {
		for _, delta := range []int64{-1, 0, 1, 3, 4, 7, 8, 9, 12} {
			if c := s.end + delta; c >= 0 && c <= int64(len(data)) {
				cuts[c] = true
			}
		}
	}
	for cut := range cuts {
		want := states[0]
		for _, s := range states {
			if s.end <= cut {
				want = s
			}
		}
		sub := t.TempDir()
		if err := os.WriteFile(filepath.Join(sub, FileName), data[:cut], 0o644); err != nil {
			t.Fatal(err)
		}
		q2 := openTest(t, sub, 1<<20)
		got, torn := idRevs(q2.Pending()), q2.Stats().Truncated
		q2.Close()
		if !equalStrings(got, want.pending) {
			t.Fatalf("file cut at byte %d of %d: pending = %v, want %v (the state at byte %d)", cut, len(data), got, want.pending, want.end)
		}
		if torn != (cut != want.end) {
			t.Fatalf("file cut at byte %d (last whole record ends at %d): Stats.Truncated = %v", cut, want.end, torn)
		}
		// what is left on disk is a clean file again
		if _, err := ScanFile(filepath.Join(sub, FileName), func(byte, []byte) error { return nil }); err != nil {
			t.Fatalf("file cut at byte %d: after Open the file still does not scan cleanly: %v", cut, err)
		}
	}
}

// Put, Ack, Pending and Stats from many goroutines at once (run with -race).
func TestConcurrentUse(t *testing.T) {
	dir := t.TempDir()
	q := openTest(t, dir, 1<<20)
	const writers, perWriter = 8, 50
	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < perWriter; i++ {
				id := fmt.Sprintf("w%d-%d", w, i)
				if _, err := q.Put(mkEvent(id, 1, 20)); err != nil {
					t.Errorf("put %s: %v", id, err)
				}
				if i%2 == 0 {
					if _, err := q.Ack(id, 1); err != nil {
						t.Errorf("ack %s: %v", id, err)
					}
				}
				_ = q.Pending()
				_ = q.Stats()
			}
		}(w)
	}
	wg.Wait()
	select {
	case <-q.Wake():
	default:
		t.Error("Wake: no signal although events were stored")
	}
	if got, want := q.Len(), writers*perWriter/2; got != want {
		t.Fatalf("Len = %d, want %d", got, want)
	}
	if err := q.Close(); err != nil {
		t.Fatal(err)
	}
	q2 := openTest(t, dir, 1<<20)
	defer q2.Close()
	if got, want := q2.Len(), writers*perWriter/2; got != want {
		t.Fatalf("after restart, Len = %d, want %d", got, want)
	}
}

// Any bytes at all: reading them never panics, never trusts more than is
// there, and the part that was reported good reads back the same.
//
//	go test ./internal/queue -run '^$' -fuzz FuzzScan -fuzztime 30s
func FuzzScan(f *testing.F) {
	good := append(frame(1, []byte("payload")), frame(2, key("ev", 3))...)
	f.Add(good)
	f.Add(good[:len(good)-3])
	f.Add(append(append([]byte{}, good...), 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0))
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0})
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, data []byte) {
		type rec struct {
			typ     byte
			payload string
		}
		var first []rec
		n, err := scan(bytes.NewReader(data), func(typ byte, payload []byte) error {
			first = append(first, rec{typ, string(payload)})
			return nil
		})
		if n < 0 || n > int64(len(data)) {
			t.Fatalf("good = %d for %d bytes of input", n, len(data))
		}
		if err == nil && n != int64(len(data)) {
			t.Fatalf("no error, but only %d of %d bytes were read", n, len(data))
		}
		if err != nil && !errors.Is(err, ErrCorrupt) {
			t.Fatalf("error %v does not wrap ErrCorrupt", err)
		}
		var again []rec
		m, err := scan(bytes.NewReader(data[:n]), func(typ byte, payload []byte) error {
			again = append(again, rec{typ, string(payload)})
			return nil
		})
		if err != nil || m != n {
			t.Fatalf("the good part (%d bytes) does not read back cleanly: good=%d err=%v", n, m, err)
		}
		if len(first) != len(again) {
			t.Fatalf("first read gave %d records, the good part gives %d", len(first), len(again))
		}
		for i := range first {
			if first[i] != again[i] {
				t.Fatalf("record %d differs between the two reads", i)
			}
		}
	})
}

// A queue folder with any bytes in the file opens, and leaves a clean file.
//
//	go test ./internal/queue -run '^$' -fuzz FuzzOpen -fuzztime 30s
func FuzzOpen(f *testing.F) {
	if data, err := os.ReadFile(filepath.Join("testdata", "v1.wal")); err == nil {
		f.Add(data)
		f.Add(data[:len(data)/2])
	}
	f.Add(frame(1, []byte("not an event")))
	f.Add([]byte{9, 0, 0, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		dir := t.TempDir()
		path := filepath.Join(dir, FileName)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		q, err := Open(Options{Dir: dir, MaxBytes: 1 << 20, DisableFsync: true, Logger: testLogger()})
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		before := idRevs(q.Pending())
		if err := q.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := ScanFile(path, func(byte, []byte) error { return nil }); err != nil {
			t.Fatalf("after Open and Close the file does not scan cleanly: %v", err)
		}
		q2, err := Open(Options{Dir: dir, MaxBytes: 1 << 20, DisableFsync: true, Logger: testLogger()})
		if err != nil {
			t.Fatalf("second Open: %v", err)
		}
		defer q2.Close()
		if after := idRevs(q2.Pending()); !equalStrings(before, after) {
			t.Fatalf("pending changed between two opens: %v then %v", before, after)
		}
		if q2.Stats().Truncated {
			t.Fatal("the second Open found a corrupt tail again")
		}
	})
}
