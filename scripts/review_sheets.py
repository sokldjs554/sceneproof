#!/usr/bin/env python
"""Builds visual review sheets for ground-truth annotation.

For a run (events.jsonl + evidence dir) it writes one JPEG per event with the key frames tiled
(context_before / entry / dwell / exit / context_after, annotated) and a header, so a human (or a
reviewer agent) can confirm/reject each episode quickly. Also writes an in-zone occupancy strip
per video (how many tracked people were inside the zone per second) to spot missed episodes.

Usage: python scripts/review_sheets.py --events data/runs/eval-draft/eval-campus.jsonl \
          --evidence data/evidence-eval --out data/runs/eval-draft/sheets/eval-campus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-w", type=int, default=384)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latest: dict[str, dict] = {}
    with open(args.events, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ev = json.loads(line)
                if ev["event_id"] not in latest or ev.get("revision", 0) > latest[ev["event_id"]].get("revision", 0):
                    latest[ev["event_id"]] = ev
    rows = []
    for i, ev in enumerate(sorted(latest.values(), key=lambda e: e.get("media_entry_s", 0)), 1):
        kfs = {k["role"]: k for k in ev.get("evidence", {}).get("key_frames", [])}
        tiles = []
        for role in ("context_before", "entry", "dwell", "exit", "context_after"):
            k = kfs.get(f"{role}_annot") or kfs.get(role)
            if k and (Path(args.evidence) / k["path"]).exists():
                img = cv2.imread(str(Path(args.evidence) / k["path"]))
            else:
                img = np.full((216, args.tile_w, 3), 40, np.uint8)
            h, w = img.shape[:2]
            img = cv2.resize(img, (args.tile_w, int(h * args.tile_w / w)))
            label = f"{role} {k.get('media_time_s', 0):.1f}s" if k else f"{role} (none)"
            cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        hh = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
        strip = np.concatenate(tiles, axis=1)
        header = np.full((34, strip.shape[1], 3), 255, np.uint8)
        txt = (f"#{i} {ev['event_id'][:8]} track={ev.get('track_id')} {ev.get('kind', '')[11:]} "
               f"entry={ev.get('media_entry_s', 0):.1f}s exit={ev.get('media_exit_s', 0):.1f}s "
               f"dwell={ev.get('dwell_seconds', 0):.1f}s close={ev.get('close_reason', '')[13:]}")
        cv2.putText(header, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        sheet = np.concatenate([header, strip], axis=0)
        cv2.imwrite(str(out / f"{i:03d}_{ev['event_id'][:8]}.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        rows.append({"n": i, "event_id": ev["event_id"], "track_id": ev.get("track_id"), "kind": ev.get("kind"),
                     "entry_s": ev.get("media_entry_s"), "exit_s": ev.get("media_exit_s"),
                     "dwell_s": ev.get("dwell_seconds"), "close_reason": ev.get("close_reason")})
    (out / "index.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} sheets -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
