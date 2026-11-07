import numpy as np

from zonewatch.edge.zones import CloseReason, FrameMeta, Transition, ZoneMonitor, ZoneSpec, anchor_point, point_in_polygon

SQUARE = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]


class T:
    def __init__(self, tid, cx, cy, w=20, h=60, score=0.9):
        self.track_id = tid
        self.xyxy = np.array([cx - w / 2, cy - h, cx + w / 2, cy], np.float32)  # anchor (feet) = (cx, cy)
        self.score = score


def meta(i, fps=10.0):
    return FrameMeta(i, i / fps, 1000.0 + i / fps, 200, 200)


def test_point_in_polygon():
    assert point_in_polygon(0.5, 0.5, SQUARE)
    assert not point_in_polygon(0.1, 0.5, SQUARE)
    concave = [(0, 0), (1, 0), (1, 1), (0.5, 0.5), (0, 1)]  # notch at the top centre
    assert point_in_polygon(0.2, 0.7, concave)
    assert not point_in_polygon(0.2, 0.9, concave)
    assert not point_in_polygon(0.5, 0.9, concave)


def test_anchor_is_bottom_centre():
    assert anchor_point(np.array([0, 0, 10, 20])) == (5.0, 20.0)


def zone(**kw):
    base = dict(zone_id="z1", name="z", polygon=SQUARE, entry_confirm_frames=3, exit_confirm_frames=2,
                dwell_seconds=1.0, lost_grace_seconds=0.5)
    base.update(kw)
    return ZoneSpec(**base)


def test_entry_is_debounced_then_opens_dwells_and_exits():
    m = ZoneMonitor([zone()], "cam")
    ev = []
    # frames 0-1 inside: not enough
    for i in range(2):
        ev += m.update([T(1, 100, 100)], meta(i))
    assert ev == []
    ev += m.update([T(1, 100, 100)], meta(2))  # 3rd consecutive -> OPENED
    assert [e.transition for e in ev] == [Transition.OPENED]
    ep = ev[0].episode
    assert ep.entry_frame == 2 and ep.revision == 1
    ev = []
    for i in range(3, 14):  # stays inside 1.1 s -> DWELL at >= 1.0 s after entry
        ev += m.update([T(1, 100, 100)], meta(i))
    assert [e.transition for e in ev] == [Transition.DWELL]
    assert ep.dwell_escalated and ep.revision == 2
    ev = []
    ev += m.update([T(1, 10, 100)], meta(14))  # outside once: not yet
    assert ev == []
    ev += m.update([T(1, 10, 100)], meta(15))  # outside twice -> CLOSED
    assert [e.transition for e in ev] == [Transition.CLOSED]
    assert ep.closed and ep.close_reason == CloseReason.EXITED and ep.exit_frame == 15
    assert ep.revision == 3
    assert abs(ep.dwell_seconds - (15 - 2) / 10) < 1e-6
    assert m.open_episodes == {}


def test_flicker_at_edge_does_not_open():
    m = ZoneMonitor([zone()], "cam")
    ev = []
    for i in range(20):  # alternate inside/outside
        ev += m.update([T(1, 100 if i % 2 == 0 else 10, 100)], meta(i))
    assert ev == []


def test_lost_track_closes_after_grace_with_last_seen_time():
    m = ZoneMonitor([zone()], "cam")
    ev = []
    for i in range(5):
        ev += m.update([T(1, 100, 100)], meta(i))
    ep = ev[0].episode
    ev = []
    for i in range(5, 12):  # track vanished; grace = 0.5s = 5 frames
        ev += m.update([], meta(i))
    assert [e.transition for e in ev] == [Transition.CLOSED]
    assert ep.close_reason == CloseReason.TRACK_LOST
    assert ep.exit_frame == 4  # last frame actually seen inside


def test_episode_id_is_deterministic():
    a = ZoneMonitor([zone()], "cam")
    b = ZoneMonitor([zone()], "cam")
    ea = eb = None
    for i in range(3):
        ea = a.update([T(7, 100, 100)], meta(i)) or ea
        eb = b.update([T(7, 100, 100)], meta(i)) or eb
    assert ea[0].episode.episode_id == eb[0].episode.episode_id


def test_two_tracks_two_episodes_and_close_all():
    m = ZoneMonitor([zone()], "cam")
    ev = []
    for i in range(3):
        ev += m.update([T(1, 100, 100), T(2, 120, 110)], meta(i))
    assert len(ev) == 2 and len(m.open_episodes) == 2
    ev = m.close_all(CloseReason.STREAM_ENDED, meta(3))
    assert len(ev) == 2 and all(e.episode.close_reason == CloseReason.STREAM_ENDED for e in ev)


def test_time_based_debounce_is_fps_independent():
    z = zone(entry_confirm_seconds=0.3, exit_confirm_seconds=0.5, entry_confirm_frames=1, exit_confirm_frames=1)
    for fps in (10.0, 30.0):
        m = ZoneMonitor([z], "cam")
        ev = []
        i = 0
        while not ev:
            ev += m.update([T(1, 100, 100)], meta(i, fps))
            i += 1
        assert abs(ev[0].episode.entry_media_s - 0.3) < 1.0 / fps + 1e-9, fps
        ev = []
        j = i
        while not ev:
            ev += m.update([T(1, 10, 100)], meta(j, fps))
            j += 1
        assert abs((ev[0].episode.exit_media_s or 0) - (i / fps + 0.5)) < 1.0 / fps + 1e-9, fps


def test_box_clipped_at_frame_bottom_holds_state():
    """A person walking at the bottom edge (feet out of view) must not end the episode."""
    m = ZoneMonitor([zone()], "cam")
    ev = []
    for i in range(3):
        ev += m.update([T(1, 100, 100)], meta(i))
    assert [e.transition for e in ev] == [Transition.OPENED]
    ev = []
    for i in range(3, 40):  # box bottom at the frame edge (y=200): anchor would be on the boundary
        ev += m.update([T(1, 100, 200)], meta(i))
    assert ev == [] and len(m.open_episodes) == 1
    ev = []
    for i in range(40, 43):  # clearly outside now
        ev += m.update([T(1, 10, 100)], meta(i))
    assert [e.transition for e in ev] == [Transition.CLOSED]
