#!/usr/bin/env python
"""Independent recall check for ground-truth drafting: run the detector on EVERY frame (no tracker,
no debounce, low threshold) and record how many persons have their anchor point inside the zone.
Writes <out>.json (per-frame counts, contiguous occupied segments) and <out>.png (timeline strip).
Segments not covered by a drafted event are candidate misses to inspect by hand.

    python scripts/zone_occupancy.py --video data/videos/campus-walkway.mp4 --config configs/eval.yaml \
        --camera eval-campus --variant yolox_s --conf 0.2 --out data/runs/eval-draft/occupancy/eval-campus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from sceneproof.edge.detector import build_detector
from sceneproof.edge.zones import ZoneSpec, anchor_point


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--variant", default="yolox_s")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--min-segment-s", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cam = next(c for c in cfg["cameras"] if c["camera_id"] == args.camera)
    zones = [ZoneSpec(z["zone_id"], z["name"], [tuple(p) for p in z["polygon"]]) for z in cam["zones"]]
    det = build_detector({"variant": args.variant, "model": f"models/{args.variant}.onnx", "conf_threshold": args.conf, "intra_op_threads": 4})
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    counts, anywhere = [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        d = det.detect(f)
        h, w = f.shape[:2]
        n = 0
        for b in d.boxes:
            ax, ay = anchor_point(b)
            if any(z.contains_px(ax, ay, w, h) for z in zones):
                n += 1
        counts.append(n)
        anywhere.append(len(d.boxes))
        i += 1
    segs = []
    start = None
    for k, c in enumerate([*counts, 0]):
        if c > 0 and start is None:
            start = k
        elif c == 0 and start is not None:
            if (k - start) / fps >= args.min_segment_s:
                segs.append({"start_s": round(start / fps, 2), "end_s": round((k - 1) / fps, 2), "max_count": int(max(counts[start:k]))})
            start = None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"video": args.video, "fps": fps, "frames": i, "variant": args.variant, "conf": args.conf,
                                                    "segments": segs, "in_zone_per_frame": counts, "persons_per_frame": anywhere}))
    # strip: 10 px per second, height 60
    secs = int(np.ceil(i / fps))
    strip = np.full((80, max(10, secs * 10), 3), 255, np.uint8)
    for k, c in enumerate(counts):
        x = int(k / fps * 10)
        if c > 0:
            cv2.line(strip, (x, 60), (x, 60 - min(c, 5) * 10), (0, 0, 255), 1)
        elif anywhere[k] > 0:
            cv2.line(strip, (x, 78), (x, 72), (180, 180, 180), 1)
    for s in range(0, secs, 10):
        cv2.putText(strip, str(s), (s * 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.imwrite(str(out.with_suffix(".png")), strip)
    print(json.dumps({"video": args.video, "frames": i, "segments": segs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
