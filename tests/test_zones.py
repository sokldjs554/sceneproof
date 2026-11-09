import numpy as np

from sceneproof.edge.zones import (
    CloseReason,
    FrameMeta,
    Transition,
    ZoneMonitor,
    ZoneSpec,
    anchor_point,
    point_in_polygon,
)

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


def _run(m, frames):
    ev = []
    for i, tracks in frames:
        ev += m.update(tracks, meta(i))
    return ev


def test_rejoin_continues_episode_when_tracker_starts_a_new_id():
    """Crowd case: the tracker loses person 1 and a moment later starts track 7 for the same person nearby."""
    frames = [(i, [T(1, 100, 100)]) for i in range(10)] + [(i, []) for i in range(10, 13)] \
        + [(i, [T(7, 104, 102)]) for i in range(13, 25)] + [(i, [T(7, 10, 100)]) for i in range(25, 30)]
    m = ZoneMonitor([zone(rejoin=True)], "cam")
    ev = _run(m, frames)
    assert [e.transition for e in ev] == [Transition.OPENED, Transition.DWELL, Transition.CLOSED]
    ep = ev[0].episode
    assert ep.track_id == 1 and ep.rejoined_track_ids == [7] and ep.close_reason == CloseReason.EXITED
    # without rejoin the same input is two episodes (the split the benchmark counted as a duplicate)
    ev2 = _run(ZoneMonitor([zone()], "cam"), frames)
    assert [e.transition for e in ev2].count(Transition.OPENED) == 2


def test_rejoin_ignores_tracks_that_existed_before_the_loss_or_are_far():
    # track 2 was already visible next to person 1 -> a different person, must not inherit the episode
    frames = [(i, [T(1, 100, 100), T(2, 110, 100)]) for i in range(10)] + [(i, [T(2, 110, 100)]) for i in range(10, 20)]
    m = ZoneMonitor([zone(rejoin=True)], "cam")
    ev = _run(m, frames)
    ep1 = next(e.episode for e in ev if e.transition == Transition.OPENED and e.episode.track_id == 1)
    assert ep1.closed and ep1.close_reason == CloseReason.TRACK_LOST and ep1.rejoined_track_ids == []
    # a new track far away (> 1 box height from where person 1 vanished) opens its own episode
    frames = [(i, [T(1, 60, 100)]) for i in range(10)] + [(i, [T(9, 140, 140)]) for i in range(10, 20)]
    ev = _run(ZoneMonitor([zone(rejoin=True)], "cam"), frames)
    assert [e.episode.track_id for e in ev if e.transition == Transition.OPENED] == [1, 9]


def test_exclusion_area_blocks_opening_but_holds_an_open_episode():
    exclude = [[(0.55, 0.40), (0.70, 0.40), (0.70, 0.60), (0.55, 0.60)]]  # e.g. a mannequin inside the zone
    m = ZoneMonitor([zone(exclude=exclude)], "cam")
    ev = _run(m, [(i, [T(5, 125, 100)]) for i in range(30)])  # standing still on the mannequin spot
    assert ev == []
    # a real person who entered elsewhere and walks across the spot keeps one open episode
    m = ZoneMonitor([zone(exclude=exclude)], "cam")
    frames = [(i, [T(1, 80, 100)]) for i in range(5)] + [(i, [T(1, 125, 100)]) for i in range(5, 15)] \
        + [(i, [T(1, 80, 100)]) for i in range(15, 18)]
    ev = _run(m, frames)
    assert [e.transition for e in ev] == [Transition.OPENED, Transition.DWELL]
    assert not ev[0].episode.closed


def test_zone_settings_survive_the_command_round_trip():
    """Zones sent from the server (START_CAMERA) keep time-based debounce, exclusion areas and rejoin settings."""
    from sceneproof.edge.zone_proto import fill_pb, from_pb
    from sceneproof.proto.sceneproof.v1 import sceneproof_pb2 as pb

    z = zone(entry_confirm_seconds=0.3, exit_confirm_seconds=0.8, rejoin=True, rejoin_dist_rel=0.8,
             exclude=[[(0.5, 0.5), (0.6, 0.5), (0.6, 0.6)]])
    back = from_pb(pb.ZoneSpec.FromString(fill_pb(z, pb.ZoneSpec()).SerializeToString()))
    assert back.entry_confirm_seconds == 0.3 and back.exit_confirm_seconds == 0.8
    assert back.rejoin and abs(back.rejoin_dist_rel - 0.8) < 1e-6 and len(back.exclude) == 1
    assert [tuple(round(v, 6) for v in p) for p in back.exclude[0]] == [(0.5, 0.5), (0.6, 0.5), (0.6, 0.6)]


def test_rejoin_expects_a_walking_person_further_along():
    """A person walking right disappears behind something and comes out 30 px further: rejoined because the expected
    position moves on at the walking speed; a standing-still expectation (the last position) would reject it."""
    frames = [(i, [T(1, 55 + 6 * i, 100, h=40)]) for i in range(10)] + [(i, []) for i in range(10, 14)] \
        + [(i, [T(4, 55 + 6 * i, 100, h=40)]) for i in range(14, 16)]
    ev = _run(ZoneMonitor([zone(rejoin=True, rejoin_dist_rel=0.5)], "cam"), frames)
    opened = [e for e in ev if e.transition == Transition.OPENED]
    assert len(opened) == 1 and opened[0].episode.rejoined_track_ids == [4] and not opened[0].episode.closed
