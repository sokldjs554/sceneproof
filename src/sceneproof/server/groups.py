"""Group events that share a camera, zone, and sufficiently close time intervals."""

from __future__ import annotations

import time
from collections import deque


def concurrent_groups(events: list[dict], now: float | None = None, gap_s: float = 0.0) -> dict[str, dict]:
    """Return group details for linked events; gap_s is the allowed gap in seconds on both clocks."""
    if gap_s < 0:
        raise ValueError("gap_s must be non-negative")

    current_time = time.time() if now is None else now
    neighbors = [[] for _ in events]

    def connected(first: dict, second: dict) -> bool:
        if (first.get("camera_id"), first.get("zone_id")) != (second.get("camera_id"), second.get("zone_id")):
            return False

        first_media_start = float(first.get("media_entry_s") or 0.0)
        second_media_start = float(second.get("media_entry_s") or 0.0)
        first_media_end = first.get("media_exit_s")
        second_media_end = second.get("media_exit_s")
        first_media_end = float("inf") if first_media_end is None else float(first_media_end)
        second_media_end = float("inf") if second_media_end is None else float(second_media_end)
        latest_start = max(first_media_start, second_media_start)
        earliest_end = min(first_media_end, second_media_end)
        # Zero preserves the existing strict media-time boundary; a positive gap is inclusive.
        media_match = latest_start < earliest_end if gap_s == 0 else latest_start <= earliest_end + gap_s
        if not media_match:
            return False

        first_wall_start = float(first.get("entry_ts") or 0.0)
        second_wall_start = float(second.get("entry_ts") or 0.0)
        first_wall_end = first.get("exit_ts")
        second_wall_end = second.get("exit_ts")
        first_wall_end = max(current_time, first_wall_start) if first_wall_end is None else float(first_wall_end)
        second_wall_end = max(current_time, second_wall_start) if second_wall_end is None else float(second_wall_end)
        return max(first_wall_start, second_wall_start) <= min(first_wall_end, second_wall_end) + gap_s

    for left in range(len(events)):
        for right in range(left + 1, len(events)):
            if connected(events[left], events[right]):
                neighbors[left].append(right)
                neighbors[right].append(left)

    result: dict[str, dict] = {}
    visited: set[int] = set()
    for start in range(len(events)):
        if start in visited:
            continue
        pending = deque([start])
        visited.add(start)
        component = []
        while pending:
            index = pending.popleft()
            component.append(events[index])
            for neighbor in neighbors[index]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        if len(component) < 2:
            continue
        component.sort(key=lambda event: (float(event.get("media_entry_s") or 0.0), event["event_id"]))
        ids = [event["event_id"] for event in component]
        details = {"group_id": ids[0], "size": len(ids), "event_ids": ids}
        result.update({event_id: details for event_id in ids})
    return result
