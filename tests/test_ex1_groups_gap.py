"""과제 1 — 거의 이어진 사건도 묶기 (src/sceneproof/server/groups.py)

concurrent_groups(events, now=None, gap_s=0.0)에 gap_s를 추가합니다.
gap_s 초 이하로 떨어진 사건(앞 사건이 끝나고 gap_s 초 안에 다음 사건이 시작)도 같은 묶음으로 봅니다.
영상 시각과 실제 시각 두 쪽 모두에 같은 여유를 적용합니다. gap_s=0.0이면 지금과 똑같이 동작해야 합니다.
"""

from sceneproof.server.groups import concurrent_groups
import pytest


def ev(eid, m0, m1, w0=None, w1=None, cam="c", zone="z"):
    w0 = 1000.0 + m0 if w0 is None else w0
    w1 = (None if m1 is None else 1000.0 + m1) if w1 is None else w1
    return {"event_id": eid, "camera_id": cam, "zone_id": zone, "media_entry_s": m0, "media_exit_s": m1,
            "entry_ts": w0, "exit_ts": w1}


def test_default_is_unchanged():
    # b starts 0.5 s after a ends: not concurrent without a gap
    assert concurrent_groups([ev("a", 0, 5), ev("b", 5.5, 9)], now=2000) == {}
    assert concurrent_groups([ev("a", 0, 5), ev("b", 5.5, 9)], now=2000, gap_s=0.0) == {}


def test_small_gap_joins_and_large_gap_does_not():
    events = [ev("a", 0, 5), ev("b", 5.8, 9), ev("c", 10.5, 12)]
    g = concurrent_groups(events, now=2000, gap_s=1.0)
    assert g["a"]["event_ids"] == ["a", "b"]          # 0.8 s gap -> same group
    assert "c" not in g                                 # 1.5 s after b ends -> separate


def test_gap_applies_to_wall_clock_too():
    # same media gap (0.5 s) but the second run happened an hour later on the wall clock
    events = [ev("run1", 0, 5, 1000, 1005), ev("run2", 5.5, 9, 4600, 4603)]
    assert concurrent_groups(events, now=9000, gap_s=1.0) == {}


def test_overlapping_events_still_group_with_a_gap():
    g = concurrent_groups([ev("a", 0, 5), ev("b", 4, 8)], now=2000, gap_s=2.0)
    assert g["a"]["size"] == 2


def test_negative_gap_is_rejected_even_without_events():
    with pytest.raises(ValueError, match="gap_s"):
        concurrent_groups([], gap_s=-1)


def test_touching_events_respect_default_and_positive_gap():
    events = [ev("a", 0, 5), ev("b", 5, 9)]
    assert concurrent_groups(events, now=2000, gap_s=0) == {}
    assert concurrent_groups(events, now=2000, gap_s=1e-9)["a"]["size"] == 2
