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
//
// The format is fixed: files written by earlier versions must still open
// (internal/queue/testdata/v1.wal).
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

// encodeRecord frames one record in the layout above.
func encodeRecord(typ byte, payload []byte) []byte {
	record := make([]byte, headerBytes+1+len(payload))
	body := record[headerBytes:]
	body[0] = typ
	copy(body[1:], payload)
	binary.LittleEndian.PutUint32(record, uint32(len(body)))
	binary.LittleEndian.PutUint32(record[4:], crc32.ChecksumIEEE(body))
	return record
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
	return scan(bufio.NewReader(f), fn)
}

// scan is ScanFile for any reader. A length of 0 or above maxRecordBytes is
// corrupt. It must not trust the length field before the bytes are there.
func scan(rd io.Reader, fn func(typ byte, payload []byte) error) (good int64, err error) {
	var header [headerBytes]byte
	for {
		_, err := io.ReadFull(rd, header[:])
		switch err {
		case io.EOF:
			return good, nil
		case io.ErrUnexpectedEOF:
			return good, fmt.Errorf("%w: partial header at %d", ErrCorrupt, good)
		case nil:
		default:
			return good, err
		}
		length := int64(binary.LittleEndian.Uint32(header[:]))
		if length < 1 || length > maxRecordBytes {
			return good, fmt.Errorf("%w: length %d at %d", ErrCorrupt, length, good)
		}
		// Grow with bytes actually read, even if the header claims 64 MiB.
		body, err := io.ReadAll(io.LimitReader(rd, length))
		if err != nil {
			return good, err
		}
		if int64(len(body)) != length || crc32.ChecksumIEEE(body) != binary.LittleEndian.Uint32(header[4:]) {
			return good, fmt.Errorf("%w: incomplete or damaged body at %d", ErrCorrupt, good)
		}
		good += headerBytes + length
		if err := fn(body[0], body[1:]); err != nil {
			return good, err
		}
	}
}
