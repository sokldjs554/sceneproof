"""Re-match saved predictions (from a run_eval JSON) against the ground truth without re-running the analyser.
Used when the matching rule or a ground-truth file changes.

    python -m zonewatch.eval.rescore docs/eval/raw/event_eval_yolox_tiny.json --gt data/ground_truth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .gt import GroundTruth
from .metrics import PredEvent, match_events, summarize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--gt", default="data/ground_truth")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--tolerance", type=float, default=1.0)
    args = ap.parse_args()
    d = json.loads(Path(args.path).read_text())
    agg = {"tp": 0, "fp": 0, "missed": 0, "duplicates": 0, "boundary_hits": 0, "duration_s": 0.0, "gt_positives": 0}
    for cam, s in d["per_video"].items():
        gt = GroundTruth.load(Path(args.gt) / f"{cam}.yaml")
        preds = [PredEvent(p["event_id"], p["start_s"], p["end_s"], p["dwell"], p.get("track_id", 0), "",
                           {k: p[k] for k in ("close_reason", "revision") if k in p}) for p in s["predictions"]]
        m = match_events(preds, gt, args.min_overlap, args.tolerance)
        new = summarize(m, gt, s["duration_s"])
        for k in ("run", "gt_annotation", "predictions"):
            new[k] = s[k]
        d["per_video"][cam] = new
        for k in ("tp", "fp", "missed", "duplicates", "boundary_hits", "gt_positives"):
            agg[k] += new[k]
        agg["duration_s"] += s["duration_s"]
    tp, fp, miss = agg["tp"], agg["fp"], agg["missed"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + miss) if tp + miss else 0.0
    agg.update({"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
                "false_alarms_per_hour": round(fp / (agg["duration_s"] / 3600), 2) if agg["duration_s"] else None})
    d["aggregate"] = agg
    d["matching"] = {"min_overlap_s": args.min_overlap, "start_tolerance_s": args.tolerance, "rescored": True}
    Path(args.path).write_text(json.dumps(d, indent=1, ensure_ascii=False))
    print(json.dumps({"tag": d["tag"], **{k: agg[k] for k in ("precision", "recall", "f1", "false_alarms_per_hour", "tp", "fp", "missed", "duplicates", "boundary_hits")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
