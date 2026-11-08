"""CV-only vs CV+VLM comparison on the same events, plus the human-grading rubric.

Inputs: a server data dir (SQLite) that holds the evaluated events (replayed from an eval run) and
their explanations, and the ground-truth YAMLs with a label per GT event
(intrusion | boundary_pass | occluded | unsure) and the prediction→GT matching from run_eval.

What is measured automatically
  * coverage: explanations done / invalid / failed / timeout / skipped, latency p50/p95, cost
  * "needs_attention" vs matched GT label: CV-only treats every event as needs_attention (that is
    what the console shows without a VLM); CV+VLM uses the model's categorical assessment. We
    report, per assessment value, how many matched intrusions / false positives / boundary passes
    fall into it. This tells whether the VLM separates true intrusions from CV false alarms —
    the only quality claim the VLM can make on top of CV.
  * zone_relation vs GT: for TP events the model should say clearly_inside/near_boundary; for FP
    events (no person in zone per GT) it should not say clearly_inside.
What needs a human (rubric written to <out>-rubric.csv, one row per explanation)
  * observations match the frames (0/1), hallucinated facts (0/1 + note), uncertainties sensible (0/1),
    Korean readability (1-3). Evaluator name/date columns are left empty on purpose.

    python -m sceneproof.eval.vlm_eval --data-dir data/server-eval --events-jsonl data/runs/eval-bundle-yolox_tiny/events.jsonl \
        --out docs/eval/raw/vlm_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from sceneproof.server.db import Store


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--events-jsonl", default=None,
                    help="events.jsonl of the SAME offline run that was replayed into the server; matched against --gt here. "
                         "Event ids are derived from (camera, zone, track, entry frame), so ids from another run/machine do not line up.")
    ap.add_argument("--gt", default="data/ground_truth")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--event-eval", default=None, help="fallback: JSON written by run_eval (prediction→GT matching) for the same run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.events_jsonl and not args.event_eval:
        ap.error("one of --events-jsonl (preferred) or --event-eval is required")
    store = Store(Path(args.data_dir) / "sceneproof.db")
    rows = store.list_events(limit=100000)
    # prediction outcome per event_id: tp | fp | duplicate | boundary
    outcome: dict[str, str] = {}
    missed_total = 0
    if args.events_jsonl:
        from sceneproof.eval.gt import GroundTruth
        from sceneproof.eval.metrics import match_events, preds_from_jsonl

        for cam in sorted({r["camera_id"] for r in rows}):
            gt_path = Path(args.gt) / f"{cam}.yaml"
            if not gt_path.exists():
                print(f"warning: no ground truth for camera {cam} ({gt_path}); its events stay unmatched", file=sys.stderr)
                continue
            m = match_events(preds_from_jsonl(args.events_jsonl, cam), GroundTruth.load(gt_path), args.min_overlap, args.tolerance)
            outcome.update({pid: "tp" for pid, _ in m.tp})
            outcome.update({pid: "fp" for pid in m.fp})
            outcome.update({pid: "duplicate" for pid, _ in m.duplicates})
            outcome.update({pid: "boundary" for pid, _ in m.boundary_hits})
            missed_total += len(m.missed)
        matching_source = f"{args.events_jsonl} vs {args.gt} (min_overlap={args.min_overlap}s, tolerance={args.tolerance}s)"
    else:
        ev_eval = json.loads(Path(args.event_eval).read_text())
        for s in ev_eval["per_video"].values():
            d = s["details"]
            outcome.update({pid: "tp" for pid, _ in d["tp"]})
            outcome.update({pid: "fp" for pid in d["fp"]})
            outcome.update({pid: "duplicate" for pid, _ in d["duplicates"]})
            outcome.update({pid: "boundary" for pid, _ in d["boundary_hits"]})
            missed_total += len(d["missed"])
        matching_source = args.event_eval
    n_unmatched = sum(1 for r in rows if r["event_id"] not in outcome)
    if rows and n_unmatched == len(rows):
        print("warning: none of the server events matched the prediction ids — the matching input is from a different run", file=sys.stderr)
    expl = store.latest_explanation_map([r["event_id"] for r in rows])
    status_counts: dict[str, int] = {}
    lat = []
    cost = 0.0
    by_outcome_assess: dict[str, dict[str, int]] = {}
    by_outcome_zone: dict[str, dict[str, int]] = {}
    rubric = []
    provider = model = prompt = ""
    for r in rows:
        x = expl.get(r["event_id"])
        if not x:
            status_counts["none"] = status_counts.get("none", 0) + 1
            continue
        provider, model, prompt = x["provider"], x["model"], x["prompt_version"]
        status_counts[x["status"]] = status_counts.get(x["status"], 0) + 1
        if x.get("latency_ms"):
            lat.append(x["latency_ms"])
        inp = json.loads(x["input_json"]) if x.get("input_json") else {}
        cost += float(inp.get("cost_usd", 0.0))
        oc = outcome.get(r["event_id"], "unmatched")
        res = json.loads(x["result_json"]) if x.get("result_json") else None
        if res:
            by_outcome_assess.setdefault(oc, {})[res["assessment"]] = by_outcome_assess.setdefault(oc, {}).get(res["assessment"], 0) + 1
            by_outcome_zone.setdefault(oc, {})[res["zone_relation"]] = by_outcome_zone.setdefault(oc, {}).get(res["zone_relation"], 0) + 1
        rubric.append({"event_id": r["event_id"], "camera_id": r["camera_id"], "cv_outcome_vs_gt": oc, "vlm_status": x["status"],
                       "assessment": res["assessment"] if res else "", "zone_relation": res["zone_relation"] if res else "",
                       "scene_summary": res["scene_summary"] if res else "", "observations": " | ".join(o["text"] for o in res["observations"]) if res else "",
                       "uncertainties": " | ".join(res["uncertainties"]) if res else "",
                       "grader_observations_match_frames_0_1": "", "grader_hallucination_0_1": "", "grader_hallucination_note": "",
                       "grader_uncertainties_sensible_0_1": "", "grader_readability_1_3": "", "grader_name": "", "grader_date": ""})
    n_assessed = sum(sum(v.values()) for v in by_outcome_assess.values())
    tp_flagged = by_outcome_assess.get("tp", {}).get("needs_attention", 0)
    tp_total = sum(by_outcome_assess.get("tp", {}).values())
    fp_cleared = by_outcome_assess.get("fp", {}).get("likely_benign", 0)
    fp_total = sum(by_outcome_assess.get("fp", {}).values())
    out = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "provider": provider, "model": model, "prompt_version": prompt,
        "matching": {"source": matching_source, "events": len(rows), "matched": len(rows) - n_unmatched, "unmatched": n_unmatched,
                     "by_cv_outcome": {k: sum(1 for r in rows if outcome.get(r["event_id"], "unmatched") == k)
                                       for k in ("tp", "fp", "duplicate", "boundary", "unmatched")},
                     "gt_intrusions_missed_by_cv": missed_total},
        "events": len(rows), "explanations_by_status": status_counts,
        "latency_ms": {"p50": float(np.percentile(lat, 50)) if lat else None, "p95": float(np.percentile(lat, 95)) if lat else None, "n": len(lat)},
        "estimated_cost_usd": round(cost, 4),
        "cv_only": {"policy": "every event shown as needs_attention (확인 필요)", "events_needing_review": len(rows),
                    "true_intrusions_flagged": sum(1 for r in rows if outcome.get(r["event_id"]) == "tp"),
                    "cv_false_alarms_flagged": sum(1 for r in rows if outcome.get(r["event_id"]) == "fp")},
        "cv_plus_vlm": {"assessment_by_cv_outcome": by_outcome_assess, "zone_relation_by_cv_outcome": by_outcome_zone,
                        "true_intrusions_kept_as_needs_attention": f"{tp_flagged}/{tp_total}",
                        "cv_false_alarms_marked_likely_benign": f"{fp_cleared}/{fp_total}", "assessed_events": n_assessed},
        "human_rubric_csv": str(Path(args.out).with_suffix("")) + "-rubric.csv",
        "notes": ["assessment is categorical; no confidence numbers are produced or evaluated",
                  "a VLM 'likely_benign' on a GT intrusion is a miss that CV-only would not have made: report it, do not hide it"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    with open(out["human_rubric_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rubric[0].keys()) if rubric else ["event_id"])
        w.writeheader()
        w.writerows(rubric)
    print(json.dumps({k: out[k] for k in ("matching", "explanations_by_status", "latency_ms", "cv_plus_vlm")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
