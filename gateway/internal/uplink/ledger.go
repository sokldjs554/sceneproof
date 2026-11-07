package uplink

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// LedgerFileName is the upload ledger inside the queue directory.
const LedgerFileName = "uploads.ledger"

// Ledger is the small append-only file that remembers evidence uploads
// across restarts. Two line kinds, tab-separated:
//
//	W\t<event_id>\t<path>              wanted: an acked event referenced this file
//	D\t<event_id>\t<path>\t<sha256>    done: the server holds it (uploaded or already_present)
//
// Why both: "done" alone answers "don't re-upload after restart". "wanted"
// lets a restart re-schedule files whose event was acked but whose upload had
// not finished when the process died (typically the clip, which is written
// seconds after the CLOSED revision is acked); without it those files would be
// lost silently. Entries are keyed by (event_id, path): evidence files are
// written once by the analyzer and never rewritten in place, so a present key
// means the content the server has is the content on disk.
type Ledger struct {
	path string
	mu   sync.Mutex
	f    *os.File
	done map[string]string // key → sha256
	want map[string]struct{}
}

func ledgerKey(eventID, path string) string { return eventID + "\x00" + path }

// OpenLedger loads (or creates) the ledger file at dir/uploads.ledger.
func OpenLedger(dir string) (*Ledger, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	l := &Ledger{
		path: filepath.Join(dir, LedgerFileName),
		done: map[string]string{},
		want: map[string]struct{}{},
	}
	if f, err := os.Open(l.path); err == nil {
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 64*1024), 1<<20)
		for sc.Scan() {
			parts := strings.Split(sc.Text(), "\t")
			switch {
			case len(parts) == 3 && parts[0] == "W":
				l.want[ledgerKey(parts[1], parts[2])] = struct{}{}
			case len(parts) == 4 && parts[0] == "D":
				l.done[ledgerKey(parts[1], parts[2])] = parts[3]
			}
			// Anything else (a torn last line) is ignored: worst case is one
			// extra upload, which the server answers with already_present.
		}
		f.Close()
		if err := sc.Err(); err != nil {
			return nil, fmt.Errorf("ledger: read: %w", err)
		}
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	f, err := os.OpenFile(l.path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	l.f = f
	return l, nil
}

// Has reports whether (eventID, path) was uploaded already.
func (l *Ledger) Has(eventID, path string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	_, ok := l.done[ledgerKey(eventID, path)]
	return ok
}

// Want records that the file should be uploaded; it is a no-op if the file is
// done or already wanted.
func (l *Ledger) Want(eventID, path string) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	k := ledgerKey(eventID, path)
	if _, ok := l.done[k]; ok {
		return nil
	}
	if _, ok := l.want[k]; ok {
		return nil
	}
	l.want[k] = struct{}{}
	return l.appendLine("W\t" + eventID + "\t" + path + "\n")
}

// Done records a completed upload.
func (l *Ledger) Done(eventID, path, sha256 string) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	k := ledgerKey(eventID, path)
	l.done[k] = sha256
	delete(l.want, k)
	return l.appendLine("D\t" + eventID + "\t" + path + "\t" + sha256 + "\n")
}

// Forget drops a wanted entry without marking it done (gave up). It is
// recorded as done with an empty digest so a restart does not revive it.
func (l *Ledger) Forget(eventID, path string) error {
	return l.Done(eventID, path, "")
}

// Outstanding returns wanted-but-not-done (event_id, path) pairs.
func (l *Ledger) Outstanding() [][2]string {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([][2]string, 0, len(l.want))
	for k := range l.want {
		if _, ok := l.done[k]; ok {
			continue
		}
		id, p, _ := strings.Cut(k, "\x00")
		out = append(out, [2]string{id, p})
	}
	return out
}

func (l *Ledger) appendLine(s string) error {
	if l.f == nil {
		return fmt.Errorf("ledger: closed")
	}
	_, err := l.f.WriteString(s)
	return err
}

// Close flushes and closes the ledger file.
func (l *Ledger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.f == nil {
		return nil
	}
	err := l.f.Sync()
	if cerr := l.f.Close(); err == nil {
		err = cerr
	}
	l.f = nil
	return err
}
