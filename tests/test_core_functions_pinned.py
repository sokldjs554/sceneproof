"""Behaviour pinned for eight core functions of the analyser: is a point in the zone, where the feet are, overlapping
boxes (NMS), the frame queue, and the evidence ring.

The expected values were recorded from the implementation the product was measured with (commit 1733a4c), before
those functions were written again by hand (PR #3). An implementation that passes analyses video the same way.
"""

import threading
import time

import numpy as np

from sceneproof.edge.detector import nms_xyxy
from sceneproof.edge.frames import Frame, FrameQueue, FrameRing
from sceneproof.edge.zones import anchor_point, point_in_polygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
L_SHAPE = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]


def mk(i, media_s=None):
    return Frame(i, media_s if media_s is not None else i / 10, 1000 + i / 10, np.zeros((48, 64, 3), np.uint8))


# ----------------------------------------------------------------------------- task A: is the point in the zone
def test_a_point_inside_and_outside_in_either_drawing_direction():
    for poly in (SQUARE, SQUARE[::-1]):
        assert point_in_polygon(5, 5, poly)
        assert not point_in_polygon(15, 5, poly) and not point_in_polygon(-1, 5, poly)
        assert not point_in_polygon(5, 15, poly) and not point_in_polygon(5, -1, poly)


def test_a_point_in_the_notch_of_a_concave_zone_is_outside():
    assert point_in_polygon(2, 8, L_SHAPE) and point_in_polygon(8, 2, L_SHAPE)
    assert not point_in_polygon(8, 8, L_SHAPE)


def test_a_points_exactly_on_the_border_follow_one_fixed_rule():
    # smaller-x and smaller-y borders count as inside, larger-x and larger-y borders do not
    assert point_in_polygon(0, 5, SQUARE) and point_in_polygon(5, 0, SQUARE) and point_in_polygon(0, 0, SQUARE)
    assert not point_in_polygon(10, 5, SQUARE) and not point_in_polygon(5, 10, SQUARE)
    assert not point_in_polygon(10, 0, SQUARE) and not point_in_polygon(10, 10, SQUARE) and not point_in_polygon(0, 10, SQUARE)


def test_a_recorded_answers_for_a_camera_zone():
    zone = [(0.2 * 640, 0.5 * 360), (0.8 * 640, 0.45 * 360), (0.95 * 640, 0.95 * 360), (0.1 * 640, 0.9 * 360)]
    pts = np.round(np.random.default_rng(11).uniform([0, 0], [640, 360], size=(64, 2)), 1)
    got = "".join("1" if point_in_polygon(float(x), float(y), zone) else "0" for x, y in pts)
    assert got == "0000010111011100000000000100000000000110000001111001000000000110"


def test_a_anchor_is_where_the_feet_are_as_plain_floats():
    x, y = anchor_point(np.array([10, 20, 31, 80], np.float32))
    assert (x, y) == (20.5, 80.0) and type(x) is float and type(y) is float


# ----------------------------------------------------------------------------- task B: overlapping boxes
def test_b_no_boxes_gives_an_empty_integer_array():
    keep = nms_xyxy(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.5)
    assert keep.dtype == np.int64 and keep.shape == (0,)


def test_b_kept_boxes_come_back_best_first():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 0, 110, 10]], np.float32)
    keep = nms_xyxy(boxes, np.array([0.2, 0.9, 0.5], np.float32), 0.5)
    assert keep.dtype == np.int64 and keep.tolist() == [1, 2, 0]


def test_b_a_box_is_removed_only_when_overlap_is_above_the_threshold():
    pair = np.array([[0, 0, 10, 10], [0, 0, 10, 5]], np.float32)  # overlap (IoU) is one half
    scores = np.array([0.9, 0.8], np.float32)
    assert nms_xyxy(pair, scores, 0.5).tolist() == [0, 1]   # not above 0.5: both stay
    assert nms_xyxy(pair, scores, 0.49).tolist() == [0]


def test_b_only_a_kept_box_removes_others():
    chain = np.array([[0, 0, 10, 10], [6, 0, 16, 10], [12, 0, 22, 10]], np.float32)  # left-middle and middle-right overlap
    assert nms_xyxy(chain, np.array([0.9, 0.8, 0.7], np.float32), 0.2).tolist() == [0, 2]  # middle is gone, so right stays
    assert nms_xyxy(chain, np.array([0.7, 0.9, 0.8], np.float32), 0.2).tolist() == [1]     # middle removes both


def test_b_recorded_answers_for_forty_boxes():
    rng = np.random.default_rng(7)
    xy = rng.uniform(0, 100, size=(40, 2))
    wh = rng.uniform(10, 40, size=(40, 2))
    boxes = np.concatenate([xy, xy + wh], 1).astype(np.float32)
    scores = rng.uniform(0.05, 1.0, 40).astype(np.float32)
    assert nms_xyxy(boxes, scores, 0.3).tolist() == [
        9, 38, 24, 1, 20, 22, 2, 17, 16, 18, 7, 12, 11, 4, 6, 33, 39, 5, 10, 15, 37, 8, 29, 0, 34, 3, 14, 26]
    assert nms_xyxy(boxes, scores, 0.5).tolist() == [
        9, 38, 24, 1, 20, 22, 2, 17, 21, 16, 18, 35, 7, 12, 11, 4, 6, 33, 31, 39, 23, 5, 10, 15, 37, 8, 28, 29, 0, 13,
        34, 19, 3, 14, 25, 32, 26, 36, 27]


# ----------------------------------------------------------------------------- task C: the frame queue
def test_c_live_queue_never_blocks_and_reports_what_it_dropped():
    q = FrameQueue(capacity=2)
    assert q.put(mk(0)) is True and q.put(mk(1)) is True
    assert q.put(mk(2)) is False and q.put(mk(3)) is False   # each put pushed the oldest waiting frame out
    s = q.stats()
    assert (s.received, s.dropped, s.taken, s.depth) == (4, 2, 0, 2)
    assert [q.get(timeout=0).index, q.get(timeout=0).index] == [2, 3]
    assert q.stats().taken == 2 and len(q) == 0


def test_c_get_gives_up_after_the_timeout_and_wakes_up_for_a_new_frame():
    q = FrameQueue(capacity=2)
    t0 = time.perf_counter()
    assert q.get(timeout=0.05) is None
    assert time.perf_counter() - t0 < 1.0
    threading.Timer(0.05, lambda: q.put(mk(7))).start()
    t0 = time.perf_counter()
    f = q.get(timeout=5)
    assert f is not None and f.index == 7 and time.perf_counter() - t0 < 2.0   # did not sleep the whole timeout


def test_c_lossless_queue_makes_the_producer_wait_instead_of_dropping():
    q = FrameQueue(capacity=1, lossless=True)
    results: list[bool] = []

    def producer():
        for i in range(5):
            results.append(q.put(mk(i)))

    t = threading.Thread(target=producer)
    t.start()
    time.sleep(0.1)
    assert len(q) == 1 and len(results) == 1   # the second put is waiting for room
    got = []
    while len(got) < 5:
        f = q.get(timeout=2)
        assert f is not None
        got.append(f.index)
    t.join(2)
    assert not t.is_alive() and got == [0, 1, 2, 3, 4] and results == [True] * 5
    s = q.stats()
    assert (s.received, s.dropped, s.taken) == (5, 0, 5)


def test_c_closing_releases_a_waiting_producer_and_the_remaining_frames_can_still_be_read():
    q = FrameQueue(capacity=1, lossless=True)
    assert q.put(mk(0)) is True
    results: list[bool] = []
    t = threading.Thread(target=lambda: results.append(q.put(mk(1))))
    t.start()
    time.sleep(0.1)
    assert t.is_alive()          # waiting: the queue is full
    q.close()
    t.join(2)
    assert not t.is_alive() and results == [False]   # the frame was not queued
    assert q.stats().received == 2 and len(q) == 1
    assert q.put(mk(2)) is False                     # a closed lossless queue takes nothing more
    assert q.get(timeout=5).index == 0
    t0 = time.perf_counter()
    assert q.get(timeout=5) is None                  # closed and empty: answer at once
    assert time.perf_counter() - t0 < 1.0


# ----------------------------------------------------------------------------- task D: the evidence ring
def indices(ring: FrameRing) -> list[int]:
    return [e.index for e in ring.slice(-1e9, 1e9)]


def test_d_ring_keeps_the_last_max_seconds_of_video():
    r = FrameRing(max_seconds=1.0, max_bytes=10 << 20, max_side=64)
    for i in range(21):   # 0.0 .. 2.0 s
        r.append(mk(i))
    assert indices(r) == list(range(10, 21))   # the frame exactly max_seconds old is still kept
    s = r.stats()
    assert (s.evicted_by_time, s.evicted_by_bytes) == (10, 0)
    assert s.bytes == sum(len(e.jpeg) for e in r.slice(-1e9, 1e9))


def test_d_a_pin_keeps_older_frames_but_never_shortens_the_ring():
    r = FrameRing(max_seconds=1.0, max_bytes=10 << 20, max_side=64)
    r.pinned_from = 0.5        # an open event still needs frames from 0.5 s on
    for i in range(21):
        r.append(mk(i))
    assert indices(r) == list(range(5, 21)) and r.stats().evicted_by_time == 5
    r = FrameRing(max_seconds=1.0, max_bytes=10 << 20, max_side=64)
    r.pinned_from = 1.8        # a pin newer than the time limit changes nothing
    for i in range(21):
        r.append(mk(i))
    assert indices(r) == list(range(10, 21))


def test_d_the_byte_limit_wins_over_the_pin_and_drops_oldest_first():
    one = len(FrameRing(max_side=64).encode(mk(0).image)[0])
    r = FrameRing(max_seconds=100, max_bytes=one * 5, max_side=64)
    r.pinned_from = 0.0
    for i in range(12):
        r.append(mk(i))
    s = r.stats()
    assert indices(r) == [7, 8, 9, 10, 11]
    assert (s.bytes, s.evicted_by_bytes, s.evicted_by_time) == (one * 5, 7, 0)


def test_d_slice_includes_both_ends_and_nearest_prefers_the_earlier_frame_on_a_tie():
    r = FrameRing(max_seconds=100, max_bytes=10 << 20, max_side=64)
    assert r.nearest(1.0) is None and r.slice(0, 10) == []
    for i, t in enumerate([0.0, 0.5, 1.0, 2.0]):
        r.append(mk(i, t))
    assert [e.index for e in r.slice(0.5, 1.0)] == [1, 2] and r.slice(1.1, 1.9) == []
    assert r.nearest(0.5).index == 1 and r.nearest(0.6).index == 1
    assert [r.nearest(0.25).index, r.nearest(0.75).index, r.nearest(1.5).index] == [0, 1, 2]   # exactly in between
    assert r.nearest(-5).index == 0 and r.nearest(99).index == 3
