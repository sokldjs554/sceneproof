"""Event-level evaluation on the held-out videos.

For each camera in configs/eval.yaml: run the analyser offline (every frame, no drops), load the
ground truth YAML, match predicted episodes to GT, and write per-video + aggregate metrics with the
exact run conditions (model, backend, thresholds, config hash, git sha).

    python -m sceneproof.eval.run_eval --config configs/eval.yaml --gt data/ground_truth \
        --out docs/eval/raw/event_eval_tiny.json [--variant yolox_nano]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from sceneproof.edge.agent import run_offline
from sceneproof.edge.config import EdgeConfig

from .gt import GroundTruth
from .metrics import match_events, preds_from_jsonl, summarize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--gt", default="data/ground_truth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-dir", default="data/runs/eval")
    ap.add_argument("--variant", default=None, help="override detector variant (model path follows)")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--cameras", default="")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--tolerance", type=float, default=1.0)
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.config).read_text())
    if args.variant:
        data["detector"]["variant"] = args.variant
        data["detector"]["model"] = f"models/{args.variant}.onnx"
    if args.conf is not None:
        data["detector"]["conf_threshold"] = args.conf
    cfg = EdgeConfig.model_validate(data)
    tag = f"{cfg.detector.variant}-c{cfg.detector.conf_threshold}"
    runs_dir = Path(args.runs_dir) / tag
    runs_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cameras = [c for c in cfg.cameras if not args.cameras or c.camera_id in args.cameras.split(",")]
    per_video = {}
    agg = {"tp": 0, "fp": 0, "missed": 0, "duplicates": 0, "boundary_hits": 0, "duration_s": 0.0, "gt_positives": 0}
    for cam in cameras:
        gt_path = Path(args.gt) / f"{cam.camera_id}.yaml"
        if not gt_path.exists():
            print(f"[eval] no ground truth for {cam.camera_id} ({gt_path}); skipping", flush=True)
            continue
        gt = GroundTruth.load(gt_path)
        jsonl = runs_dir / f"{cam.camera_id}.jsonl"
        if jsonl.exists():
            jsonl.unlink()
        print(f"[eval] {cam.camera_id} ({cam.source}) with {tag} ...", flush=True)
        t0 = time.time()
        run = run_offline(cfg, cam.camera_id, str(jsonl), run_id=f"eval-{tag}-{cam.camera_id}")
        preds = preds_from_jsonl(str(jsonl))
        duration = gt.duration_s or (run["frames"] / cam.fps_hint)
        m = match_events(preds, gt, args.min_overlap, args.tolerance)
        s = summarize(m, gt, duration)
        s["run"] = {"frames": run["frames"], "wall_s": run["wall_s"], "fps": run["fps"], "timings": run["timings"],
                    "detector": run["detector"]}
        s["gt_annotation"] = gt.annotation
        s["predictions"] = [{"event_id": p.event_id, "start_s": p.start_s, "end_s": p.end_s, "dwell": p.dwell,
                             "track_id": p.track_id, **p.extra} for p in preds]
        per_video[cam.camera_id] = s
        for k in ("tp", "fp", "missed", "duplicates", "boundary_hits", "gt_positives"):
            agg[k] += s[k]
        agg["duration_s"] += duration
        print(f"[eval]   P={s['precision']} R={s['recall']} F1={s['f1']} FP/h={s['false_alarms_per_hour']} "
              f"tp={s['tp']} fp={s['fp']} miss={s['missed']} dup={s['duplicates']} boundary={s['boundary_hits']} ({time.time() - t0:.0f}s)", flush=True)
    tp, fp, miss = agg["tp"], agg["fp"], agg["missed"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + miss) if tp + miss else 0.0
    agg.update({"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
                "false_alarms_per_hour": round(fp / (agg["duration_s"] / 3600), 2) if agg["duration_s"] else None})
    out = {"tag": tag, "git": git, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": data,
           "matching": {"min_overlap_s": args.min_overlap, "start_tolerance_s": args.tolerance},
           "aggregate": agg, "per_video": per_video}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[eval] aggregate P={agg['precision']} R={agg['recall']} F1={agg['f1']} FP/h={agg['false_alarms_per_hour']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
