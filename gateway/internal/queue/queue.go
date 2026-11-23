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
// (sections 3.1, 3.2 and 4). This file holds the queue and its recovery logic.
package queue

import (
	"container/list"
	"encoding/binary"

	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	"google.golang.org/protobuf/proto"

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
	mu             sync.Mutex
	opts           Options
	path           string
	wal            File
	dir            *os.File
	needDirSync    bool
	committedBytes int64 // last good offset; FileBytes may include a partial write
	stats          Stats
	fifo           list.List
	byID           map[string]*entry
	wake           chan struct{}
}

// Each pending event has one entry. Updating it keeps its position in the FIFO.
type entry struct {
	event    *sceneproofv1.Event
	payload  []byte
	position *list.Element
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
	if opts.MaxBytes < 0 || opts.CompactMinBytes < 0 {
		return nil, errors.New("queue: negative capacity or compaction threshold")
	}
	if opts.MaxBytes == 0 {
		opts.MaxBytes = defaultMaxBytes
	}
	if opts.CompactMinBytes == 0 {
		opts.CompactMinBytes = defaultCompactMinBytes
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	if opts.OpenFile == nil {
		opts.OpenFile = func(name string, flags int, mode os.FileMode) (File, error) {
			return os.OpenFile(name, flags, mode)
		}
	}
	if err := os.MkdirAll(opts.Dir, 0755); err != nil {
		return nil, err
	}
	dir, err := os.Open(opts.Dir)
	if err != nil {
		return nil, err
	}
	q := &Queue{
		opts:  opts,
		dir:   dir,
		path:  filepath.Join(opts.Dir, FileName),
		byID:  make(map[string]*entry),
		wake:  make(chan struct{}, 1),
		stats: Stats{MaxBytes: opts.MaxBytes},
	}
	q.wal, err = opts.OpenFile(q.path, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0600)
	if err != nil {
		dir.Close()
		return nil, err
	}
	if err := q.recover(); err != nil {
		q.wal.Close()
		dir.Close()
		return nil, err
	}
	return q, nil
}

// Put stores ev (coalescing with a pending older revision) and returns once
// its PUT record is on disk. Only after Put returns without an error may the
// caller tell the analyzer accepted=true.
//
// Evicting for capacity writes a DROP record for each victim, oldest first.
func (q *Queue) Put(ev *sceneproofv1.Event) (PutResult, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if err := q.prepareWAL(); err != nil {
		return PutResult{}, err
	}
	id := ev.GetEventId()
	if id == "" {
		return PutResult{}, ErrInvalid
	}
	old := q.byID[id]
	if old != nil && old.event.Revision >= ev.Revision {
		return PutResult{Duplicate: true}, nil
	}
	size := int64(proto.Size(ev))
	if size > q.opts.MaxBytes || size >= maxRecordBytes {
		return PutResult{}, ErrTooLarge
	}
	payload, err := proto.Marshal(ev)
	if err != nil {
		return PutResult{}, err
	}
	// Log the whole operation before changing the FIFO.
	if err := q.writePutAndDrops(ev, payload); err != nil {
		return PutResult{}, err
	}
	q.store(proto.CloneOf(ev), payload)
	evicted := q.evictOverBudget()
	q.maybeCompact()
	_, stored := q.byID[id]
	if stored {
		select {
		case q.wake <- struct{}{}:
		default:
		}
	}
	return PutResult{Stored: stored, Evicted: evicted}, nil
}

// An update still occupies its original FIFO slot. Use its new size and
// revision when deciding which DROP records follow this PUT.
func (q *Queue) writePutAndDrops(ev *sceneproofv1.Event, payload []byte) error {
	if err := q.appendRecord(recPut, payload); err != nil {
		return err
	}
	bytesAfterPut := q.stats.LiveBytes + int64(len(payload))
	replaced := q.byID[ev.EventId]
	if replaced != nil {
		bytesAfterPut -= int64(len(replaced.payload))
	}
	for pos := q.fifo.Front(); bytesAfterPut > q.opts.MaxBytes; pos = pos.Next() {
		item := pos.Value.(*entry)
		victim, size := item.event, len(item.payload)
		if item == replaced {
			victim, size = ev, len(payload)
		}
		if err := q.appendRecord(recDrop, keyPayload(victim.EventId, victim.Revision)); err != nil {
			return err
		}
		bytesAfterPut -= int64(size)
	}
	q.committedBytes = q.stats.FileBytes
	return nil
}

// Ack records that the server holds revision (accepted or duplicate). The
// pending record is removed only when revision >= the pending one, because a
// newer snapshot that arrived during the round trip still has to be sent.
// It reports whether a pending record was removed; when nothing is removed,
// nothing is written.
func (q *Queue) Ack(eventID string, revision int32) (bool, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if err := q.prepareWAL(); err != nil {
		return false, err
	}
	e := q.byID[eventID]
	if e == nil || revision < e.event.Revision {
		return false, nil
	}
	if err := q.appendRecord(recAck, keyPayload(eventID, revision)); err != nil {
		return false, err
	}
	q.committedBytes = q.stats.FileBytes
	q.remove(e)
	q.maybeCompact()
	return true, nil
}

// Pending returns the pending events in first-enqueue order (newest revision
// of each). The returned events are shared with the queue and must be
// treated as read-only.
func (q *Queue) Pending() []*sceneproofv1.Event {
	q.mu.Lock()
	defer q.mu.Unlock()
	events := make([]*sceneproofv1.Event, 0, q.fifo.Len())
	for e := q.fifo.Front(); e != nil; e = e.Next() {
		events = append(events, e.Value.(*entry).event)
	}
	return events
}

// Len returns the number of pending events.
func (q *Queue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.fifo.Len()
}

// Wake returns a channel that receives a value after a Put stored an event.
// It is buffered with capacity one; consumers drain it and then call Pending.
func (q *Queue) Wake() <-chan struct{} {
	return q.wake
}

// Stats returns a snapshot of the queue's counters.
func (q *Queue) Stats() Stats {
	q.mu.Lock()
	defer q.mu.Unlock()
	st := q.stats
	st.Pending = q.fifo.Len()
	return st
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
	q.mu.Lock()
	defer q.mu.Unlock()
	if err := q.prepareWAL(); err != nil {
		return err
	}
	return q.compactLocked()
}

// Close fsyncs and closes the WAL. Further operations return ErrClosed;
// a second Close returns nil.
func (q *Queue) Close() error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.wal == nil {
		return nil
	}
	repairErr := q.prepareWAL()
	syncErr := q.wal.Sync()
	closeErr := q.wal.Close()
	dirErr := q.dir.Close()
	q.wal = nil
	return errors.Join(repairErr, syncErr, closeErr, dirErr)
}

func (q *Queue) store(ev *sceneproofv1.Event, payload []byte) {
	item := q.byID[ev.EventId]
	if item == nil {
		item = &entry{}
		item.position = q.fifo.PushBack(item)
		q.byID[ev.EventId] = item
	}
	q.stats.LiveBytes -= int64(len(item.payload))
	item.event = ev
	item.payload = payload
	q.stats.LiveBytes += int64(len(payload))
}

func (q *Queue) remove(item *entry) {
	delete(q.byID, item.event.EventId)
	q.fifo.Remove(item.position)
	q.stats.LiveBytes -= int64(len(item.payload))
}

func (q *Queue) evictOverBudget() int {
	count := 0
	for q.stats.LiveBytes > q.opts.MaxBytes {
		oldest := q.fifo.Front().Value.(*entry)
		q.opts.Logger.Warn("evicting pending event", "event_id", oldest.event.EventId)
		q.remove(oldest)
		q.stats.Dropped++
		count++
	}
	return count
}

func (q *Queue) replay(typ byte, payload []byte) error {
	switch typ {
	case recPut:
		ev := new(sceneproofv1.Event)
		if err := proto.Unmarshal(payload, ev); err != nil || ev.EventId == "" {
			return nil
		}
		old := q.byID[ev.EventId]
		if old == nil || ev.Revision > old.event.Revision {
			q.store(ev, payload)
		}
	case recAck, recDrop:
		if len(payload) <= 4 {
			return nil
		}
		revision := int32(binary.LittleEndian.Uint32(payload[:4]))
		id := string(payload[4:])
		item := q.byID[id]
		if item != nil && revision >= item.event.Revision {
			q.remove(item)
		}
	}
	return nil
}

func (q *Queue) compactBytes() int64 {
	return q.stats.LiveBytes + int64(q.fifo.Len())*(headerBytes+1)
}

func (q *Queue) recover() error {
	// A crash before rename can leave a staging file. Only events.wal is authoritative.
	if err := os.Remove(q.path + ".tmp"); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	good, err := ScanFile(q.path, q.replay)
	if err != nil {
		if !errors.Is(err, ErrCorrupt) {
			return err
		}
		if err := q.wal.Truncate(good); err != nil {
			return err
		}
		if err := q.wal.Sync(); err != nil {
			return err
		}
		q.stats.Truncated = true
		q.opts.Logger.Warn("truncated corrupt WAL tail", "good_bytes", good)
	}
	q.committedBytes = good
	q.stats.FileBytes = good
	q.evictOverBudget()
	if good > q.compactBytes() {
		return q.compactLocked()
	}
	return q.dir.Sync()
}

// Repair before the next operation. A failed truncate leaves the trusted
// boundary intact; appending behind that boundary would hide later records on replay.
func (q *Queue) prepareWAL() error {
	if q.wal == nil {
		return ErrClosed
	}
	if q.needDirSync {
		if err := q.dir.Sync(); err != nil {
			return err
		}
		q.needDirSync = false
	}
	if q.stats.FileBytes != q.committedBytes {
		if err := q.wal.Truncate(q.committedBytes); err != nil {
			return err
		}
		q.stats.FileBytes = q.committedBytes
	}
	return nil
}

func (q *Queue) appendRecord(typ byte, payload []byte) error {
	n, err := writeRecord(q.wal, typ, payload)
	q.stats.FileBytes += int64(n)
	if err == nil && typ == recPut && !q.opts.DisableFsync {
		err = q.wal.Sync()
	}
	if err != nil {
		// Roll back to before the PUT (or ACK), including any completed DROP records.
		repairErr := q.prepareWAL()
		return errors.Join(err, repairErr)
	}
	return nil
}

func (q *Queue) maybeCompact() {
	if q.stats.FileBytes >= q.opts.CompactMinBytes && q.stats.FileBytes > 2*q.compactBytes() {
		if err := q.compactLocked(); err != nil {
			q.opts.Logger.Warn("WAL compaction failed", "error", err)
		}
	}
}

func (q *Queue) compactLocked() error {
	path := q.path + ".tmp"
	next, err := q.opts.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	err = q.writeLiveRecords(next)
	if err == nil {
		err = os.Rename(path, q.path)
	}
	if err != nil {
		next.Close()
		os.Remove(path)
		return err
	}
	// After rename the new file is the WAL, even if syncing the directory fails.
	old := q.wal
	q.wal = next
	q.committedBytes = q.compactBytes()
	q.stats.FileBytes = q.committedBytes
	q.stats.Compactions++
	q.needDirSync = true
	syncErr := q.prepareWAL()
	closeErr := old.Close()
	if err := errors.Join(syncErr, closeErr); err != nil {
		return err
	}
	q.opts.Logger.Info("compacted WAL", "bytes", q.committedBytes)
	return nil
}

func (q *Queue) writeLiveRecords(file File) error {
	for pos := q.fifo.Front(); pos != nil; pos = pos.Next() {
		item := pos.Value.(*entry)
		if _, err := writeRecord(file, recPut, item.payload); err != nil {
			return err
		}
	}
	return file.Sync()
}
