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

    by_zone: dict[tuple, list[int]] = defaultdict(list)
    spans = []
    for i, event in enumerate(events):
        media_start = float(event.get("media_entry_s") or 0.0)
        media_end_value = event.get("media_exit_s")
        media_end = float("inf") if media_end_value is None else float(media_end_value)
        wall_start = float(event.get("entry_ts") or 0.0)
        wall_end_value = event.get("exit_ts")
        wall_end = max(now, wall_start) if wall_end_value is None else float(wall_end_value)
        spans.append((media_start, media_end, wall_start, wall_end))
        zone_key = (event.get("camera_id"), event.get("zone_id"))
        for j in by_zone[zone_key]:
            other_media_start, other_media_end, other_wall_start, other_wall_end = spans[j]
            if gap_s == 0:
                media_match = media_start < other_media_end and other_media_start < media_end
            else:
                media_match = media_start - other_media_end <= gap_s and other_media_start - media_end <= gap_s
            if (media_match and wall_start - other_wall_end <= gap_s
                    and other_wall_start - wall_end <= gap_s):
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
