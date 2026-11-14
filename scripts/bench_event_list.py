"""What the `concurrent_only` filter of GET /api/events costs, and how often the two listing modes disagree.

The default listing groups concurrent events inside the returned page. `concurrent_only=true` computes the groups
over ALL stored events first, so that review-state filters and page boundaries cannot break a group — and therefore
reads every event on every request. This script seeds N synthetic closed events into a temporary review server,
times both requests (in-process, no network), and counts the events whose `concurrent` info differs between the modes —
without a filter, and when listing only the events still waiting for review while part of each group is in review.

    python scripts/bench_event_list.py --events 5000 --out docs/eval/raw/api_event_list_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from sceneproof.server.app import create_app  # noqa: E402
from sceneproof.server.config import ServerSettings  # noqa: E402


def synthetic_event(i: int, cameras: int, rng: random.Random, span_s: float) -> dict:
    m0 = rng.uniform(0, span_s)
    m1 = m0 + rng.uniform(1, 20)
    return {"event_id": f"e{i}", "revision": 3, "device_id": "d", "camera_id": f"c{i % cameras}", "zone_id": "z", "zone_name": "Z",
            "kind": "EVENT_KIND_ZONE_ENTRY", "status": "EVENT_STATUS_CLOSED", "track_id": i, "entry_ts": 1e6 + m0, "exit_ts": 1e6 + m1,
            "detected_at": 1e6 + m0, "media_entry_s": m0, "media_exit_s": m1, "dwell_seconds": m1 - m0, "source": "synthetic",
            "run": {}, "evidence": {"key_frames": []}}


def timed(client: TestClient, params: dict, repeats: int) -> tuple[dict, dict]:
    times, body = [], {}
    for _ in range(repeats):
        t0 = time.perf_counter()
        body = client.get("/api/events", params=params).json()
        times.append((time.perf_counter() - t0) * 1e3)
    return body, {"median_ms": round(statistics.median(times), 1), "min_ms": round(min(times), 1), "max_ms": round(max(times), 1)}


def compare_modes(client: TestClient, params: dict) -> dict:
    """The same events seen through both modes (largest page the API allows): how many carry different group info."""
    a = {x["event_id"]: x["concurrent"] for x in client.get("/api/events", params={**params, "limit": 2000}).json()["items"]}
    b = {x["event_id"]: x["concurrent"] for x in
         client.get("/api/events", params={**params, "limit": 2000, "concurrent_only": "true"}).json()["items"]}
    common = [k for k in b if k in a]
    return {"compared_events": len(common), "different": sum(1 for k in common if a[k] != b[k]),
            "no_group_info_in_default": sum(1 for k in common if a[k] is None)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=5000)
    ap.add_argument("--cameras", type=int, default=5)
    ap.add_argument("--span-s", type=float, default=36000.0, help="media time span the events are spread over")
    ap.add_argument("--limit", type=int, default=500, help="page size of the timed requests")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--in-review-share", type=float, default=0.3, help="share of events moved to in_review for the filter comparison")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tmp = tempfile.mkdtemp(prefix="sp-event-list-")
    app = create_app(ServerSettings(data_dir=f"{tmp}/srv", mode="replay", vlm_provider="none", console_dir=f"{tmp}/no-console"))
    with TestClient(app) as client:
        store = app.state.sp.store
        for i in range(args.events):
            store.upsert_event(synthetic_event(i, args.cameras, rng, args.span_s))
        page, t_default = timed(client, {"limit": args.limit}, args.repeats)
        only, t_only = timed(client, {"limit": args.limit, "concurrent_only": "true"}, args.repeats)
        unfiltered = compare_modes(client, {})
        # move a share of the events to "in_review", then list only the ones still waiting: the default mode groups
        # among the listed rows, so a group whose other members are in review loses its group info there
        moved = [f"e{i}" for i in range(args.events) if rng.random() < args.in_review_share]
        for eid in moved:
            client.post(f"/api/events/{eid}/review", json={"state": "in_review", "actor": "bench", "note": "", "resolution": None})
        filtered = compare_modes(client, {"state": "needs_review"})
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = {
        "what": "GET /api/events timed in-process (TestClient) on synthetic closed events; median of repeats",
        "git": git, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "events": args.events, "cameras": args.cameras, "media_span_s": args.span_s, "page_limit": args.limit,
        "repeats": args.repeats, "seed": args.seed,
        "default": {**t_default, "items": page["count"], "items_with_group_info": sum(1 for x in page["items"] if x["concurrent"])},
        "concurrent_only": {**t_only, "items": only["count"]},
        "slowdown_x": round(t_only["median_ms"] / t_default["median_ms"], 1) if t_default["median_ms"] else None,
        "group_info_default_vs_concurrent_only": {
            "why": "default groups among the rows it returns; concurrent_only groups over all events before filtering and paging",
            "no_filter": unfiltered,
            "state_needs_review_filter": {**filtered, "events_moved_to_in_review": len(moved), "in_review_share": args.in_review_share},
        },
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
