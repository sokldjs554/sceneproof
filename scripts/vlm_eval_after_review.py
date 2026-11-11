"""Re-matches the 19 events of the CV vs CV+VLM comparison against the REVIEWED ground truth.

The CI run's events.jsonl was not kept, but every event's (track id, entry frame) is recovered exactly from its
deterministic id (scripts/build_review_pages.py), and its exit frame from the re-run of the same pipeline (17 of 19;
for the 2 events the re-run does not reproduce, the event is taken as 1 s long — enough for the start-time rule).
The VLM's assessments are the ones recorded in docs/eval/raw/vlm_eval-rubric.csv (v1.3) and
vlm_eval_v1.2-ko-rubric.csv (v1.2, same 19 event ids); nothing is re-generated.

    python scripts/vlm_eval_after_review.py   # -> docs/eval/raw/vlm_eval_reviewed_gt.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_review_pages import _recover_ids, _rerun_events  # noqa: E402
from sceneproof.eval.gt import GroundTruth  # noqa: E402
from sceneproof.eval.metrics import PredEvent, match_events  # noqa: E402


def main() -> int:
    rubric = list(csv.DictReader((ROOT / "docs" / "eval" / "raw" / "vlm_eval-rubric.csv").open(encoding="utf-8")))
    cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text())
    ids = _recover_ids(rubric, cfg)
    local = _rerun_events(ROOT / "data" / "runs" / "review-rerun")
    by_cam: dict[str, list] = defaultdict(list)
    approx = []
    for r in rubric:
        cam, eid = r["camera_id"], r["event_id"]
        fps = yaml.safe_load((ROOT / "data" / "ground_truth" / f"{cam}.yaml").read_text())["fps"]
        _tid, f0 = ids[eid]
        twin = min((e for e in local if e["camera_id"] == cam and abs(int(e["entry_frame_index"]) - f0) <= 6),
                   key=lambda e: abs(int(e["entry_frame_index"]) - f0), default=None)
        if twin is not None and twin.get("exit_frame_index") is not None:
            f1 = int(twin["exit_frame_index"]) + (f0 - int(twin["entry_frame_index"]))
        else:
            f1 = f0 + round(fps)
            approx.append(eid)
        by_cam[cam].append(PredEvent(eid, f0 / fps, f1 / fps, (f1 - f0) / fps >= 5, 0, "", {}))
    outcome = {}
    for cam, preds in by_cam.items():
        m = match_events(preds, GroundTruth.load(ROOT / "data" / "ground_truth" / f"{cam}.yaml"), 0.5, 1.0)
        for pid, _g in m.tp:
            outcome[pid] = "tp"
        for pid, _g in m.duplicates:
            outcome[pid] = "duplicate"
        for pid in m.fp:
            outcome[pid] = "fp"
        for pid, _g in m.boundary_hits:
            outcome[pid] = "boundary"
    v12_path = ROOT / "docs" / "eval" / "raw" / "vlm_eval_v1.2-ko-rubric.csv"
    v12 = {r["event_id"]: r["assessment"] for r in csv.DictReader(v12_path.open(encoding="utf-8"))} if v12_path.exists() else {}
    rows = []
    for r in rubric:
        rows.append({"event_id": r["event_id"], "camera_id": r["camera_id"], "outcome_draft_gt": r["cv_outcome_vs_gt"],
                     "outcome_reviewed_gt": outcome.get(r["event_id"], "boundary"), "assessment": r["assessment"],
                     "assessment_v1_2": v12.get(r["event_id"]), "exit_time_approximate": r["event_id"] in approx})

    def tables(key: str) -> dict:
        table = defaultdict(Counter)
        for x in rows:
            if x[key] is not None:
                table[x["outcome_reviewed_gt"]][x[key]] += 1
        tp, fp = table["tp"], table["fp"]
        return {"assessment_by_cv_outcome_reviewed_gt": {k: dict(v) for k, v in table.items()},
                "true_intrusions_kept_as_needs_attention": f"{tp['needs_attention']}/{sum(tp.values())}",
                "true_intrusions_marked_likely_benign": f"{tp['likely_benign']}/{sum(tp.values())}",
                "cv_false_alarms_marked_likely_benign": f"{fp['likely_benign']}/{sum(fp.values())}"}

    out = {"note": "same 19 explanations (prompt v1.3, and v1.2 for comparison); only the ground truth changed (human review 2026-11-11)",
           **tables("assessment"),
           "prompt_v1_2": tables("assessment_v1_2") if v12 else None,
           "changed_outcomes": [x for x in rows if x["outcome_draft_gt"] != x["outcome_reviewed_gt"]],
           "events": rows}
    path = ROOT / "docs" / "eval" / "raw" / "vlm_eval_reviewed_gt.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    keys = ("assessment_by_cv_outcome_reviewed_gt", "true_intrusions_kept_as_needs_attention", "true_intrusions_marked_likely_benign",
            "cv_false_alarms_marked_likely_benign")
    print("v1.3", json.dumps({k: out[k] for k in keys}, ensure_ascii=False))
    if out["prompt_v1_2"]:
        print("v1.2", json.dumps(out["prompt_v1_2"], ensure_ascii=False))
    for x in out["changed_outcomes"]:
        print("changed:", x["camera_id"], x["event_id"][:8], x["outcome_draft_gt"], "->", x["outcome_reviewed_gt"], "| VLM:", x["assessment"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
