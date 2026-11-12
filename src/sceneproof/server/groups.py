"""Group events that overlap in the same camera and zone."""

import time


def concurrent_groups(events, now=None, gap_s: float = 0.0):
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
        a_end = float("inf") if a["media_exit_s"] is None else a["media_exit_s"]
        b_end = float("inf") if b["media_exit_s"] is None else b["media_exit_s"]
        if gap_s == 0:
            return a["media_entry_s"] < b_end and b["media_entry_s"] < a_end
        return a["media_entry_s"] - b_end <= gap_s and b["media_entry_s"] - a_end <= gap_s

    def wall_overlaps(a, b):
        a_end = now if a["exit_ts"] is None else a["exit_ts"]
        b_end = now if b["exit_ts"] is None else b["exit_ts"]
        return a["entry_ts"] - b_end <= gap_s and b["entry_ts"] - a_end <= gap_s

    for i, a in enumerate(events):
        for j in range(i):
            b = events[j]
            if (a["camera_id"] == b["camera_id"] and a["zone_id"] == b["zone_id"]
                    and media_overlaps(a, b) and wall_overlaps(a, b)):
                parents[find(i)] = find(j)

    components = {}
    for i, event in enumerate(events):
        components.setdefault(find(i), []).append(event)

    result = {}
    for component in components.values():
        if len(component) < 2:
            continue
        component.sort(key=lambda event: (event["media_entry_s"], event["event_id"]))
        ids = [event["event_id"] for event in component]
        group = {"group_id": ids[0], "size": len(ids), "event_ids": ids}
        for event_id in ids:
            result[event_id] = group
    return result
