import numpy as np

from sceneproof.edge.tracker import ByteTracker, STrack, TrackerParams, iou_matrix, linear_assignment


def box(cx, cy, w=40, h=100):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def setup_function():
    STrack.reset_ids()


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], np.float32)
    iou = iou_matrix(a, b)
    assert np.isclose(iou[0, 0], 1.0)
    assert np.isclose(iou[0, 1], 25 / 175)
    assert iou[0, 2] == 0


def test_linear_assignment_gate():
    cost = np.array([[0.1, 0.9], [0.95, 0.2]], np.float32)
    m, ur, uc = linear_assignment(cost, 0.5)
    assert m == [(0, 0), (1, 1)] and ur == [] and uc == []
    m, ur, uc = linear_assignment(np.array([[0.9]], np.float32), 0.5)
    assert m == [] and ur == [0] and uc == [0]


def test_single_object_keeps_id_and_is_tentative_first():
    t = ByteTracker(TrackerParams(track_buffer=5))
    # frame 1: activated immediately (frame_id == 1 rule)
    out = t.update(np.stack([box(100, 100)]), np.array([0.9]))
    assert len(out) == 1 and out[0].track_id == 1
    ids = set()
    for i in range(2, 30):
        out = t.update(np.stack([box(100 + 3 * i, 100)]), np.array([0.9]))
        ids.update(o.track_id for o in out)
    assert ids == {1}


def test_new_object_after_first_frame_needs_second_match():
    t = ByteTracker()
    t.update(np.zeros((0, 4), np.float32), np.zeros(0, np.float32))  # frame 1 empty
    out = t.update(np.stack([box(50, 50)]), np.array([0.9]))  # frame 2: tentative, not reported
    assert out == []
    out = t.update(np.stack([box(52, 50)]), np.array([0.9]))  # frame 3: confirmed
    assert len(out) == 1


def test_low_score_detection_bridges_occlusion():
    """High-score for a while, then only low-score boxes (partial occlusion): id must survive."""
    t = ByteTracker(TrackerParams(track_thresh=0.5, low_thresh=0.1, track_buffer=10))
    for i in range(10):
        out = t.update(np.stack([box(100 + 2 * i, 200)]), np.array([0.9]))
    tid = out[0].track_id
    for i in range(10, 20):  # occluded: detector only gives 0.2 confidence
        out = t.update(np.stack([box(100 + 2 * i, 200)]), np.array([0.2]))
        assert len(out) == 1 and out[0].track_id == tid
    out = t.update(np.stack([box(140, 200)]), np.array([0.9]))
    assert out[0].track_id == tid


def test_lost_track_recovers_id_within_buffer_and_not_after():
    t = ByteTracker(TrackerParams(track_buffer=5))
    for i in range(5):
        out = t.update(np.stack([box(100, 100 + i)]), np.array([0.9]))
    tid = out[0].track_id
    for _ in range(3):  # gone for 3 frames (< buffer)
        assert t.update(np.zeros((0, 4), np.float32), np.zeros(0, np.float32)) == []
    out = t.update(np.stack([box(100, 108)]), np.array([0.9]))
    assert len(out) == 1 and out[0].track_id == tid
    for _ in range(8):  # gone beyond buffer
        t.update(np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
    t.update(np.stack([box(100, 120)]), np.array([0.9]))
    out = t.update(np.stack([box(100, 121)]), np.array([0.9]))
    assert len(out) == 1 and out[0].track_id != tid


def test_two_crossing_objects_keep_distinct_ids():
    t = ByteTracker()
    ids_a, ids_b = set(), set()
    for i in range(40):
        a = box(50 + 5 * i, 100)   # moves right
        b = box(250 - 5 * i, 110)  # moves left, crosses a around i=20
        out = t.update(np.stack([a, b]), np.array([0.9, 0.9]))
        for o in out:
            cy = (o.xyxy[1] + o.xyxy[3]) / 2  # a stays at y=100, b at y=110: label by height, not x
            (ids_a if abs(cy - 100) < abs(cy - 110) else ids_b).add(o.track_id)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b
