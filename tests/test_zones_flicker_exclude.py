"""구역 판정: 한 프레임 이탈로는 사건이 닫히지 않고, 제외 영역(마네킹 자리)에서는 사건이 열리지 않는다.  도우미(T, meta, zone)는 tests/test_zones.py와 같다. """

import numpy as np

from sceneproof.edge.zones import FrameMeta, Transition, ZoneMonitor, ZoneSpec

SQUARE = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]   # 200x200 화면에서 x,y 50~150 px


class T:
    def __init__(self, tid, cx, cy, w=20, h=60, score=0.9):
        self.track_id = tid
        self.xyxy = np.array([cx - w / 2, cy - h, cx + w / 2, cy], np.float32)  # 발 위치 = (cx, cy)
        self.score = score


def meta(i, fps=10.0):
    return FrameMeta(i, i / fps, 1000.0 + i / fps, 200, 200)


def zone(**kw):
    base = dict(zone_id="z1", name="z", polygon=SQUARE, entry_confirm_frames=3, exit_confirm_frames=2,
                dwell_seconds=1.0, lost_grace_seconds=0.5)
    base.update(kw)
    return ZoneSpec(**base)


def test_one_frame_outside_does_not_close_the_episode():
    """구역 안에서 사건이 열린 뒤, 1프레임만 밖에 나갔다가(exit_confirm_frames=2보다 짧음) 다시 들어오면
    사건이 닫히지 않고 같은 사건이 이어져야 한다."""
    m = ZoneMonitor([zone()], "cam")
    events = []
    for i in range(3):
        events += m.update([T(1, 100, 100)], meta(i))
    assert [e.transition for e in events] == [Transition.OPENED]
    episode = events[0].episode
    events += m.update([T(1, 10, 100)], meta(3))
    for i in range(4, 8):
        events += m.update([T(1, 100, 100)], meta(i))
    assert [e.transition for e in events].count(Transition.OPENED) == 1
    assert Transition.CLOSED not in [e.transition for e in events]
    assert len(m.open_episodes) == 1
    assert next(iter(m.open_episodes.values())) is episode


def test_excluded_area_never_opens_but_the_rest_of_the_zone_does():
    """제외 영역(마네킹 자리) 안에 발이 계속 있으면 사건이 열리지 않아야 하고,
    같은 사람이 제외 영역 밖이지만 구역 안으로 옮기면 사건이 열려야 한다."""
    exclude = [[(0.45, 0.45), (0.55, 0.45), (0.55, 0.55), (0.45, 0.55)]]  # 화면 가운데 작은 네모(90~110 px)
    m = ZoneMonitor([zone(exclude=exclude)], "cam")
    events = []
    for i in range(20):
        events += m.update([T(1, 100, 100)], meta(i))
    assert Transition.OPENED not in [e.transition for e in events]
    assert len(m.open_episodes) == 0
    for i in range(20, 23):
        events += m.update([T(1, 130, 130)], meta(i))
    assert [e.transition for e in events].count(Transition.OPENED) == 1
    assert len(m.open_episodes) == 1
