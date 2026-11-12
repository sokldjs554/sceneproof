"""Events that were inside the same zone at the same time.

The zone monitor makes one event per tracked person, which is also how the benchmark ground truth counts. People
who mark the same video count a group that is inside together as ONE event (docs/eval/bench/findings.md, agreement
round 2), and one person whose track id changed can also appear as two overlapping events. The review console
shows such events together so a reviewer can look at them, and if they decide so handle them, at once. Nothing is
merged or hidden here: every event keeps its own id and review state.

Two events are concurrent when they share camera and zone and their intervals overlap both in media time (the
video's own clock) and in wall-clock time. The second condition keeps two analyses of the same file, run at
different times, apart. An open event (no exit yet) counts as still inside. Groups are the connected components of
that relation, so A-B and B-C put A, B and C in one group even if A and C never overlapped.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict


def _overlap(a0: float, a1: float, b0: float, b1: float, gap_s: float = 0.0) -> bool:
    # Preserve strict overlap at zero; any positive tolerance includes touching intervals.
    if gap_s > 0:
        return a0 <= b1 + gap_s and b0 <= a1 + gap_s
    return a0 < b1 and b0 < a1


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def concurrent_groups(events: list[dict], now: float | None = None, gap_s: float = 0.0) -> dict[str, dict]:
    """event_id -> {"group_id", "size", "event_ids"} for events in a group of two or more.

    ``events`` need event_id, camera_id, zone_id, entry_ts / exit_ts (epoch seconds, exit may be None) and
    media_entry_s / media_exit_s (exit may be None). ``event_ids`` is ordered by media entry time and the group id
    is the first of them, so it does not change when a later member closes.
    ``gap_s`` is the allowed gap in seconds on both clocks, to join short tracking interruptions.
    """
    if gap_s < 0:
        raise ValueError("gap_s must be non-negative")
    now = time.time() if now is None else now
    by_zone: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        by_zone[(e.get("camera_id"), e.get("zone_id"))].append(e)
    out: dict[str, dict] = {}
    for evs in by_zone.values():
        if len(evs) < 2:
            continue
        parent = list(range(len(evs)))
        spans = []
        for e in evs:
            m0 = float(e.get("media_entry_s") or 0.0)
            m1 = math.inf if e.get("media_exit_s") is None else float(e["media_exit_s"])
            w0 = float(e.get("entry_ts") or 0.0)
            w1 = max(now, w0) if e.get("exit_ts") is None else float(e["exit_ts"])
            spans.append((m0, m1, w0, w1))
        for i in range(len(evs)):
            for j in range(i + 1, len(evs)):
                a, b = spans[i], spans[j]
                if _overlap(a[0], a[1], b[0], b[1], gap_s) and (a[2] <= b[3] + gap_s and b[2] <= a[3] + gap_s):
                    parent[_find(parent, i)] = _find(parent, j)
        comps: dict[int, list[int]] = defaultdict(list)
        for i in range(len(evs)):
            comps[_find(parent, i)].append(i)
        for members in comps.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda k: (spans[k][0], evs[k]["event_id"]))
            ids = [evs[k]["event_id"] for k in members]
            info = {"group_id": ids[0], "size": len(ids), "event_ids": ids}
            for eid in ids:
                out[eid] = info
    return out
