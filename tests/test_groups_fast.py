"""Task F: concurrent_groups gives the same groups as before, without comparing every pair of events in a zone.

`reference_groups` below states the rule pair by pair. It is the specification of the result, not the way to
compute it: comparing every pair is exactly what makes the listing slow when a zone has thousands of events.
"""

import random
import time

from sceneproof.server.groups import concurrent_groups

NOW = 2_000_000.0


def ev(eid, m0, m1, w0=None, w1=None, cam="c", zone="z"):
    """m0/m1: seconds in the video; w0/w1: wall clock (None for an event that is still open)."""
    if w0 is None:
        w0 = 1_000_000.0 + m0
        w1 = None if m1 is None else 1_000_000.0 + m1
    return {"event_id": eid, "camera_id": cam, "zone_id": zone, "media_entry_s": m0, "media_exit_s": m1,
            "entry_ts": w0, "exit_ts": w1}


def _span(e):
    media_start = float(e.get("media_entry_s") or 0.0)
    media_end = float("inf") if e.get("media_exit_s") is None else float(e["media_exit_s"])
    wall_start = float(e.get("entry_ts") or 0.0)
    wall_end = max(NOW, wall_start) if e.get("exit_ts") is None else float(e["exit_ts"])
    return media_start, media_end, wall_start, wall_end


def _linked(a, b, gap):
    if (a.get("camera_id"), a.get("zone_id")) != (b.get("camera_id"), b.get("zone_id")):
        return False
    a_ms, a_me, a_ws, a_we = _span(a)
    b_ms, b_me, b_ws, b_we = _span(b)
    media = (a_ms < b_me and b_ms < a_me) if gap == 0 else (a_ms - b_me <= gap and b_ms - a_me <= gap)
    return media and a_ws - b_we <= gap and b_ws - a_we <= gap


def reference_groups(events, gap=0.0):
    parent = list(range(len(events)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(len(events)):
        for j in range(i):
            if _linked(events[i], events[j], gap):
                parent[find(i)] = find(j)
    members = {}
    for i, e in enumerate(events):
        members.setdefault(find(i), []).append(e)
    out = {}
    for group in members.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda e: (float(e.get("media_entry_s") or 0.0), e["event_id"]))
        ids = [e["event_id"] for e in group]
        for eid in ids:
            out[eid] = {"group_id": ids[0], "size": len(ids), "event_ids": ids}
    return out


def random_events(seed, n=400):
    """Three cameras, two zones, two separate analysis runs of the same videos, and a few events still open."""
    rng = random.Random(seed)
    events = []
    for i in range(n):
        m0 = round(rng.uniform(0, 900), 2)
        m1 = round(m0 + rng.uniform(0.3, 25), 2)
        run_start = 1_000_000.0 + 5000.0 * rng.randrange(2)
        still_open = rng.random() < 0.02
        events.append(ev(f"e{i:04d}", m0, None if still_open else m1, run_start + m0, None if still_open else run_start + m1,
                         cam=f"cam{rng.randrange(3)}", zone=f"z{rng.randrange(2)}"))
    return events


def test_f_same_groups_as_the_pairwise_rule_on_random_events():
    for seed in range(5):
        events = random_events(seed)
        for gap in (0.0, 1.5):
            want = reference_groups(events, gap)
            assert want, "the random data must contain groups"
            assert concurrent_groups(events, now=NOW, gap_s=gap) == want, (seed, gap)


def test_f_the_order_of_the_input_does_not_matter():
    events = random_events(11)
    want = reference_groups(events, 1.0)
    shuffled = list(events)
    random.Random(3).shuffle(shuffled)
    assert concurrent_groups(shuffled, now=NOW, gap_s=1.0) == want
    assert concurrent_groups(list(reversed(events)), now=NOW, gap_s=1.0) == want


def test_f_an_open_event_joins_everything_that_comes_later_in_its_zone():
    events = [ev("open", 10, None), ev("b", 300, 305), ev("c", 800, 801), ev("other-zone", 300, 305, zone="z2")]
    got = concurrent_groups(events, now=NOW)
    assert got == reference_groups(events)
    assert got["open"]["event_ids"] == ["open", "b", "c"] and "other-zone" not in got


def test_f_events_that_start_at_the_same_moment():
    events = [ev(f"s{i}", 50.0, 50.0 + i) for i in range(1, 6)] + [ev("after", 60.0, 61.0)]
    assert concurrent_groups(events, now=NOW) == reference_groups(events)
    assert concurrent_groups(events, now=NOW, gap_s=5.0) == reference_groups(events, 5.0)


def test_f_an_event_recorded_with_its_end_before_its_start_is_judged_by_the_same_rule():
    # bad data must not be grouped differently than before: both wall-clock conditions are still checked
    events = [ev("a", 10, 20, 1000.0, 1010.0), ev("bad", 12, 18, 1012.0, 900.0), ev("c", 15, 25, 1005.0, 1015.0)]
    for gap in (0.0, 2.0):
        assert concurrent_groups(events, now=NOW, gap_s=gap) == reference_groups(events, gap)


def _sequential(n):
    """n events one after another in one zone: each lasts 1 s and the next starts 1 s later. Nothing overlaps."""
    return [ev(f"q{i:05d}", 2.0 * i, 2.0 * i + 1.0) for i in range(n)]


def _best_time(events):
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        assert concurrent_groups(events, now=NOW) == {}
        best = min(best, time.perf_counter() - t0)
    return best


def test_f_time_grows_about_in_step_with_the_number_of_events_when_nothing_overlaps():
    small, large = _best_time(_sequential(2000)), _best_time(_sequential(8000))
    # four times the events: comparing every pair takes about sixteen times as long, a sweep about four times
    assert large < 8 * small + 0.02, f"2000 events {small * 1e3:.0f} ms, 8000 events {large * 1e3:.0f} ms"
