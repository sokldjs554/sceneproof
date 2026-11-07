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

    python -m zonewatch.eval.vlm_eval --data-dir data/server-eval --event-eval docs/eval/raw/event_eval_tiny.json \
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

from zonewatch.server.db import Store


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--event-eval", required=True, help="JSON written by run_eval (prediction→GT matching)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    store = Store(Path(args.data_dir) / "zonewatch.db")
    ev_eval = json.loads(Path(args.event_eval).read_text())
    # prediction outcome per event_id
    outcome: dict[str, str] = {}
    for cam, s in ev_eval["per_video"].items():
        d = s["details"]
        for pid, _ in d["tp"]:
            outcome[pid] = "tp"
        for pid in d["fp"]:
            outcome[pid] = "fp"
        for pid, _ in d["duplicates"]:
            outcome[pid] = "duplicate"
        for pid, _ in d["boundary_hits"]:
            outcome[pid] = "boundary"
    rows = store.list_events(limit=100000)
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
        "event_eval": args.event_eval, "events": len(rows), "explanations_by_status": status_counts,
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
    print(json.dumps({k: out[k] for k in ("explanations_by_status", "latency_ms", "cv_plus_vlm")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
