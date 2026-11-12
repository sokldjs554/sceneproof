"""Show related events together without merging or hiding their individual records.

Media time locates events in a video; wall-clock time keeps separate analyses of
the same video apart. Events in the same camera and zone form a group when they
overlap on both clocks. Links are transitive: A-B and B-C connect A, B and C,
even when A and C do not directly overlap.
"""

import time
from collections import defaultdict


def concurrent_groups(events: list[dict], now: float | None = None, gap_s: float = 0.0) -> dict[str, dict]:
    """Return group metadata for events connected by concurrent intervals.

    Events must share a camera and zone, and overlap in both media and wall
    time. A positive ``gap_s`` also joins intervals separated by at most that
    many seconds on both clocks. At zero, touching media intervals do not
    overlap, while touching wall-clock intervals do. Open events extend to
    the present. Negative ``gap_s`` raises ValueError.
    """
    if gap_s < 0:
        raise ValueError("gap_s must be non-negative")
    if now is None:
        now = time.time()
    events = list(events)
    parents = list(range(len(events)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def media_overlaps(a, b):
        a_start = float(a.get("media_entry_s") or 0.0)
        b_start = float(b.get("media_entry_s") or 0.0)
        a_end = float("inf") if a.get("media_exit_s") is None else float(a["media_exit_s"])
        b_end = float("inf") if b.get("media_exit_s") is None else float(b["media_exit_s"])
        if gap_s == 0:
            return a_start < b_end and b_start < a_end
        return a_start - b_end <= gap_s and b_start - a_end <= gap_s

    def wall_overlaps(a, b):
        a_start = float(a.get("entry_ts") or 0.0)
        b_start = float(b.get("entry_ts") or 0.0)
        a_end = max(now, a_start) if a.get("exit_ts") is None else float(a["exit_ts"])
        b_end = max(now, b_start) if b.get("exit_ts") is None else float(b["exit_ts"])
        return a_start - b_end <= gap_s and b_start - a_end <= gap_s

    by_zone: dict[tuple, list[int]] = defaultdict(list)
    for i, a in enumerate(events):
        zone_key = (a.get("camera_id"), a.get("zone_id"))
        for j in by_zone[zone_key]:
            b = events[j]
            if media_overlaps(a, b) and wall_overlaps(a, b):
                parents[find(i)] = find(j)
        by_zone[zone_key].append(i)

    components = {}
    for i, event in enumerate(events):
        components.setdefault(find(i), []).append(event)

    result = {}
    for component in components.values():
        if len(component) < 2:
            continue
        component.sort(key=lambda event: (float(event.get("media_entry_s") or 0.0), event["event_id"]))
        ids = [event["event_id"] for event in component]
        group = {"group_id": ids[0], "size": len(ids), "event_ids": ids}
        for event_id in ids:
            result[event_id] = group
    return result
