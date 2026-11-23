// Package queue implements the gateway's durable, bounded, coalescing event
// queue.
//
// Why a write-ahead log and not a database: the gateway runs on an edge
// device next to the analyzer. It needs exactly one property from storage:
// an Event that was answered with accepted=true must survive a crash or a
// power cut until the server has acknowledged it. An append-only log gives
// that with one write and one fsync per event, no schema, no background
// threads and a recovery procedure that is a single sequential read.
//
// Model:
//
//   - An Event is a complete snapshot of a zone episode; a higher revision
//     supersedes every lower one (see sceneproof.proto). The queue therefore
//     keeps at most one pending record per event_id (the newest revision) and
//     coalesces: a newer revision replaces the pending one in place, keeping
//     the FIFO position of the first enqueue so an episode cannot starve
//     behind events that arrived after it.
//   - Pending() yields events in first-enqueue order. Ack(event_id, revision)
//     removes the pending record only if revision >= the pending revision:
//     when a newer revision arrived while the older one was in flight, the
//     newer one must still be sent.
//   - The queue is bounded by live payload bytes. When the budget is
//     exceeded the oldest pending events are evicted (and counted) because on
//     a device that has been offline for a long time the most recent episodes
//     are the ones an operator still cares about; the oldest ones are the
//     most likely to be stale by the time the uplink returns.
//   - The on-disk file only grows while events are in flight. It is rewritten
//     with just the live records (compacted) when it is more than twice the
//     live payload size, and on startup when it contains dead records.
//   - A write can fail (a full disk stops it part-way; an fsync can fail).
//     The operation then returns an error and changes nothing in memory, and
//     whatever it left in the file must never stand in front of a record
//     that is accepted later: recovery stops at the first record it cannot
//     trust, so everything behind such a record would be lost.
//
// The behaviour is fixed by the tests in this folder and by gateway/README.md
// (sections 3.1, 3.2 and 4). This file holds the public surface; the bodies
// are to be written.
package queue

import (
	"errors"
	"io"
	"log/slog"
	"os"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
)

const (
	// FileName is the WAL file inside Options.Dir.
	FileName = "events.wal"

	defaultMaxBytes        = 64 << 20
	defaultCompactMinBytes = 1 << 20
)

var (
	// ErrClosed is returned by every operation after Close.
	ErrClosed = errors.New("queue: closed")
	// ErrTooLarge is returned by Put when a single event exceeds the whole
	// budget. Accepting it would mean evicting everything else and still not
	// fitting, so the caller (the analyzer) is told instead of being lied to.
	ErrTooLarge = errors.New("queue: event larger than queue capacity")
	// ErrInvalid is returned for events without an event_id.
	ErrInvalid = errors.New("queue: event has no event_id")
)

// File is the part of *os.File the queue uses for a file it writes.
type File interface {
	io.Writer
	Sync() error
	Truncate(size int64) error
	Close() error
}

// Options configures Open.
type Options struct {
	// Dir holds the WAL file; it is created if missing.
	Dir string
	// MaxBytes bounds the sum of pending payload bytes. 0 means 64 MiB.
	MaxBytes int64
	// DisableFsync skips fsync after PUT records. The zero value is the safe
	// choice: a PUT is what backs the accepted=true answer to the analyzer,
	// so by default it is forced to disk before that answer is sent. ACK and
	// DROP records are never fsynced: losing one only causes a re-send, and
	// the server deduplicates by (event_id, revision).
	DisableFsync bool
	// CompactMinBytes is the file size below which automatic compaction is
	// not attempted, so a tiny file is not rewritten after every ack.
	// 0 means 1 MiB. Compact() ignores it.
	CompactMinBytes int64
	// Logger receives recovery, eviction and compaction messages.
	// nil means slog.Default().
	Logger *slog.Logger
	// OpenFile opens every file the queue writes: the WAL for appending and
	// the new file of a compaction. nil means os.OpenFile. Tests replace it
	// to make Write, Sync and Truncate fail the way a full disk does.
	OpenFile func(name string, flag int, perm os.FileMode) (File, error)
}

// PutResult describes what Put did with an event.
type PutResult struct {
	// Stored is true when the event is now pending (new or replacing an
	// older revision) and its PUT record has been written.
	Stored bool
	// Duplicate is true when a revision >= the submitted one was already
	// pending, so nothing was written.
	Duplicate bool
	// Evicted is the number of older events dropped to make room.
	Evicted int
}

// Stats is a point-in-time view used by /status, /metrics and UplinkStatus.
type Stats struct {
	Pending     int
	LiveBytes   int64 // payload bytes of pending events
	FileBytes   int64 // size of the WAL file on disk
	MaxBytes    int64
	Dropped     int64 // evicted since Open (not persisted across restarts)
	Compactions int64
	Truncated   bool // recovery found a corrupt tail and cut it off
}

// Queue is safe for concurrent use.
type Queue struct {
	// to be written
}

// Open recovers the queue from Dir (creating it if needed), truncates a
// corrupt tail, applies the capacity bound and compacts if the file holds
// dead records.
//
// Recovery reads the file once from the start and applies each record the
// way Put and Ack would. It stops at the first record it cannot trust and
// cuts the file there (Stats.Truncated). A record whose checksum is right
// but whose content cannot be used (an event that does not parse, a key that
// is too short, an unknown type) is skipped: it is not a torn write, and the
// records behind it are good.
func Open(opts Options) (*Queue, error) {
	panic("queue: Open is not implemented")
}

// Put stores ev (coalescing with a pending older revision) and returns once
// its PUT record is on disk. Only after Put returns without an error may the
// caller tell the analyzer accepted=true.
//
// Evicting for capacity writes a DROP record for each victim, oldest first.
func (q *Queue) Put(ev *sceneproofv1.Event) (PutResult, error) {
	panic("queue: Put is not implemented")
}

// Ack records that the server holds revision (accepted or duplicate). The
// pending record is removed only when revision >= the pending one, because a
// newer snapshot that arrived during the round trip still has to be sent.
// It reports whether a pending record was removed; when nothing is removed,
// nothing is written.
func (q *Queue) Ack(eventID string, revision int32) (bool, error) {
	panic("queue: Ack is not implemented")
}

// Pending returns the pending events in first-enqueue order (newest revision
// of each). The returned events are shared with the queue and must be
// treated as read-only.
func (q *Queue) Pending() []*sceneproofv1.Event {
	panic("queue: Pending is not implemented")
}

// Len returns the number of pending events.
func (q *Queue) Len() int {
	panic("queue: Len is not implemented")
}

// Wake returns a channel that receives a value after a Put stored an event.
// It is buffered with capacity one; consumers drain it and then call Pending.
func (q *Queue) Wake() <-chan struct{} {
	panic("queue: Wake is not implemented")
}

// Stats returns a snapshot of the queue's counters.
func (q *Queue) Stats() Stats {
	panic("queue: Stats is not implemented")
}

// Compact rewrites the WAL with only the live PUT records, in FIFO order, so
// that replaying the new file yields the same pending list. At every instant
// one complete file is on disk; a compaction that fails changes nothing and
// leaves no other file in Dir.
//
// It also runs by itself: after a Put or an Ack when the file is at least
// CompactMinBytes and more than twice the size a compacted file would have,
// and in Open when the file holds dead records.
func (q *Queue) Compact() error {
	panic("queue: Compact is not implemented")
}

// Close fsyncs and closes the WAL. Further operations return ErrClosed;
// a second Close returns nil.
func (q *Queue) Close() error {
	panic("queue: Close is not implemented")
}
