package queue

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
)

// WAL record layout (all integers little-endian):
//
//	uint32 length   number of bytes that follow the crc (1 type byte + payload)
//	uint32 crc32    IEEE CRC over type||payload
//	byte   type     recPut | recAck | recDrop
//	bytes  payload  PUT: proto-marshalled Event; ACK/DROP: uint32 revision + event_id
//
// The length comes first so a reader can skip to the next record; the CRC
// protects against a torn write (power loss mid-record) and bit rot. A record
// is only trusted when its full length is present AND the CRC matches, which
// is what lets recovery stop cleanly at the first bad record instead of
// guessing where the next one begins.
const (
	recPut  byte = 1 // payload = serialized sceneproofv1.Event
	recAck  byte = 2 // payload = uint32 revision | event_id (server has this revision)
	recDrop byte = 3 // payload = uint32 revision | event_id (evicted for capacity)

	headerBytes = 8 // length + crc
	// maxRecordBytes caps a single record so a corrupt length field cannot
	// make recovery allocate gigabytes. gRPC rejects messages > 4 MiB by
	// default anyway, so a legitimate Event is far below this.
	maxRecordBytes = 64 << 20
)

// ErrCorrupt marks a WAL record that could not be trusted (short read, bad
// length or CRC mismatch). Recovery treats everything from that offset on as
// garbage and truncates the file there.
var ErrCorrupt = errors.New("queue: corrupt wal record")

// encodeRecord frames one record; the result is written with a single
// Write so that a crash can only produce a torn tail, never a torn middle.
func encodeRecord(typ byte, payload []byte) []byte {
	n := 1 + len(payload)
	buf := make([]byte, headerBytes+n)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(n))
	buf[8] = typ
	copy(buf[9:], payload)
	binary.LittleEndian.PutUint32(buf[4:8], crc32.ChecksumIEEE(buf[8:]))
	return buf
}

// encodeKey encodes the ACK/DROP payload: revision followed by the event id.
func encodeKey(eventID string, revision int32) []byte {
	buf := make([]byte, 4+len(eventID))
	binary.LittleEndian.PutUint32(buf[0:4], uint32(revision))
	copy(buf[4:], eventID)
	return buf
}

func decodeKey(payload []byte) (eventID string, revision int32, err error) {
	if len(payload) < 4 {
		return "", 0, fmt.Errorf("%w: key payload too short (%d bytes)", ErrCorrupt, len(payload))
	}
	return string(payload[4:]), int32(binary.LittleEndian.Uint32(payload[0:4])), nil
}

// ScanFile reads WAL records from path in order and calls fn for each one.
// It returns the number of bytes that were verified good. If the file ends
// with a truncated or corrupt record, the returned error wraps ErrCorrupt and
// good is the offset at which the file should be truncated. Any error from fn
// aborts the scan and is returned as-is.
//
// It is exported so tests (and operators' tooling) can inspect a live WAL
// without going through Open.
func ScanFile(path string, fn func(typ byte, payload []byte) error) (good int64, err error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return scan(f, fn)
}

func scan(rd io.Reader, fn func(typ byte, payload []byte) error) (good int64, err error) {
	r := bufio.NewReaderSize(rd, 1<<20)
	hdr := make([]byte, headerBytes)
	for {
		if _, err := io.ReadFull(r, hdr); err != nil {
			if errors.Is(err, io.EOF) {
				return good, nil // clean end of file
			}
			if errors.Is(err, io.ErrUnexpectedEOF) {
				return good, fmt.Errorf("%w: truncated header at offset %d", ErrCorrupt, good)
			}
			return good, err
		}
		n := binary.LittleEndian.Uint32(hdr[0:4])
		want := binary.LittleEndian.Uint32(hdr[4:8])
		if n == 0 || n > maxRecordBytes {
			return good, fmt.Errorf("%w: bad length %d at offset %d", ErrCorrupt, n, good)
		}
		body := make([]byte, n)
		if _, err := io.ReadFull(r, body); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return good, fmt.Errorf("%w: truncated body at offset %d", ErrCorrupt, good)
			}
			return good, err
		}
		if got := crc32.ChecksumIEEE(body); got != want {
			return good, fmt.Errorf("%w: crc mismatch at offset %d (want %08x got %08x)", ErrCorrupt, good, want, got)
		}
		if err := fn(body[0], body[1:]); err != nil {
			return good, err
		}
		good += int64(headerBytes) + int64(n)
	}
}

// syncDir fsyncs a directory so that a rename inside it is durable.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
