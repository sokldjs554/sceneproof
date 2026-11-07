import numpy as np

from zonewatch.edge.frames import Frame, FrameQueue, FrameRing


def mk(i, media_s=None):
    return Frame(i, media_s if media_s is not None else i / 10, 1000 + i / 10, np.zeros((48, 64, 3), np.uint8))


def test_queue_drops_oldest_when_full_and_counts():
    q = FrameQueue(capacity=2)
    assert q.put(mk(0)) and q.put(mk(1))
    assert not q.put(mk(2))  # evicts frame 0
    s = q.stats()
    assert s.depth == 2 and s.dropped == 1 and s.received == 3
    assert q.get(timeout=0).index == 1
    assert q.get(timeout=0).index == 2
    assert q.get(timeout=0) is None


def test_ring_bounded_by_seconds_and_bytes():
    r = FrameRing(max_seconds=1.0, max_bytes=10 << 20, max_side=64, jpeg_quality=50)
    for i in range(30):  # 3 s at 10 fps
        r.append(mk(i))
    s = r.stats()
    assert s.seconds <= 1.0 + 1e-6 and s.entries <= 11 and s.evicted_by_time > 0
    tiny = FrameRing(max_seconds=100, max_bytes=3000, max_side=64)
    for i in range(30):
        tiny.append(mk(i))
    assert tiny.stats().bytes <= 3000 and tiny.stats().evicted_by_bytes > 0


def test_ring_pin_keeps_history_until_byte_cap():
    r = FrameRing(max_seconds=0.5, max_bytes=10 << 20, max_side=64)
    r.pinned_from = 0.0
    for i in range(30):
        r.append(mk(i))
    assert r.stats().entries == 30  # pin protects everything
    r.pinned_from = None
    r.append(mk(30))
    assert r.stats().seconds <= 0.5 + 1e-6


def test_ring_slice_and_nearest():
    r = FrameRing(max_seconds=10, max_bytes=10 << 20, max_side=64)
    for i in range(20):
        r.append(mk(i))
    assert [e.index for e in r.slice(0.5, 0.8)] == [5, 6, 7, 8]
    assert r.nearest(0.74).index == 7
