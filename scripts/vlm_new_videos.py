"""CV+VLM on new videos (configs/bench/vlm_new_videos.yaml): build the inputs, select the events.

    python scripts/vlm_new_videos.py prepare --work data/runs/vlm-new   # edge config + ground truth YAMLs
    # run the product analyser on every camera of <work>/edge.yaml  -> <work>/events_all.jsonl
    python scripts/vlm_new_videos.py select --work data/runs/vlm-new    # -> <work>/bundle/events.jsonl
The paid step and the report are in scripts/run_vlm_new_videos.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sceneproof.eval.bench.data import load_boxes, load_sequences  # noqa: E402
from sceneproof.eval.bench.episodes import gt_episodes, load_zones, zone_specs  # noqa: E402
from sceneproof.eval.gt import GroundTruth  # noqa: E402
from sceneproof.eval.metrics import match_events, preds_from_jsonl  # noqa: E402

PLAN = ROOT / "configs" / "bench" / "vlm_new_videos.yaml"


def prepare(work: Path) -> None:
    plan = yaml.safe_load(PLAN.read_text())
    zc = load_zones()
    seqs = load_sequences()
    base = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text())
    cams, gt_dir = [], work / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for sid in plan["videos"]:
        s = seqs.get(sid)
        if s is None:
            raise SystemExit(f"{sid}: bench data missing (python scripts/fetch_bench.py --ids {sid})")
        site = zc["sites"][s.site]
        zones = [{"zone_id": z["zone_id"], "name": z["name"], "polygon": z["polygon"],
                  "entry_confirm_seconds": zc["rules"]["entry_confirm_seconds"], "exit_confirm_seconds": zc["rules"]["exit_confirm_seconds"],
                  "dwell_seconds": zc["rules"]["dwell_seconds"], "lost_grace_seconds": zc["rules"]["lost_grace_seconds"]}
                 for z in site["zones"]]  # no `exclude`: see the plan
        cams.append({"camera_id": sid, "name": f"{s.site} · {sid}", "source": f"file://{s.video}", "split": "eval",
                     "realtime": False, "fps_hint": s.fps, "zones": zones})
        eps = gt_episodes(s, load_boxes(s), zone_specs(s.site, zc))
        label = {"intrusion": "intrusion", "boundary_pass": "boundary_pass", "ignore": "unsure"}
        events = [{"id": e.id, "start_s": round(e.start_s, 3), "end_s": round(e.end_s, 3), "label": label.get(e.label, "unsure"),
                   "dwell": bool(e.dwell), "zone_id": e.zone_id} for e in sorted(eps, key=lambda e: e.start_s)]
        gt = {"video": s.video.name, "fps": s.fps, "duration_s": s.duration_s, "zones": site["zones"], "rules": zc["rules"],
              "events": events, "annotation": {"method": "dataset hand-drawn boxes -> product zone rule (eval/bench/episodes.py)",
                                               "status": "derived", "source": s.dataset}}
        (gt_dir / f"{sid}.yaml").write_text(yaml.safe_dump(gt, allow_unicode=True, sort_keys=False))
        print(f"[prepare] {sid}: {sum(e['label'] == 'intrusion' for e in events)} intrusions, {len(events)} GT episodes")
    cfg = {**base, "evidence_root": str(work / "evidence"), "cameras": cams}
    (work / "edge.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    print(f"[prepare] {work / 'edge.yaml'}: {len(cams)} cameras")


def select(work: Path) -> None:
    plan = yaml.safe_load(PLAN.read_text())
    last: dict[str, dict] = {}
    for line in (work / "events_all.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["event_id"] not in last or int(e.get("revision", 0)) >= int(last[e["event_id"]].get("revision", 0)):
            last[e["event_id"]] = e
    closed = [e for e in last.values() if e.get("status") == "EVENT_STATUS_CLOSED"]
    # CV false alarms first (events matching no ground-truth episode, same matching as the report)
    fp_ids: list[str] = []
    for sid in plan["videos"]:
        m = match_events(preds_from_jsonl(str(work / "events_all.jsonl"), sid), GroundTruth.load(work / "gt" / f"{sid}.yaml"), 0.5, 1.0)
        fp_ids += m.fp
    by_id = {e["event_id"]: e for e in closed}
    picked: list[dict] = sorted((by_id[i] for i in fp_ids if i in by_id),
                                key=lambda e: (plan["videos"].index(e["camera_id"]), float(e.get("media_entry_s", 0))))
    picked = picked[: plan["max_false_alarms"]]
    taken = {e["event_id"] for e in picked}
    per_video = {sid: sorted((e for e in closed if e["camera_id"] == sid and e["event_id"] not in taken),
                             key=lambda e: float(e.get("media_entry_s", 0))) for sid in plan["videos"]}
    rank = 0
    while len(picked) < plan["max_events"] and any(len(v) > rank for v in per_video.values()):
        for sid in plan["videos"]:
            if len(per_video[sid]) > rank and len(picked) < plan["max_events"]:
                picked.append(per_video[sid][rank])
        rank += 1
    out = work / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in picked))
    summary = {sid: f"{sum(e['camera_id'] == sid for e in picked)}/{sum(e['camera_id'] == sid for e in closed)}" for sid in plan["videos"]}
    (work / "selection.json").write_text(json.dumps({"closed_events": len(closed), "picked": len(picked),
                                                     "false_alarms_in_run": len(fp_ids), "false_alarms_picked": len(taken),
                                                     "picked_per_video": summary, "event_ids": [e["event_id"] for e in picked]}, indent=1))
    print(f"[select] {len(picked)} of {len(closed)} closed events ({len(taken)} of {len(fp_ids)} CV false alarms first): {summary}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "select"])
    ap.add_argument("--work", default=str(ROOT / "data" / "runs" / "vlm-new"))
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    (prepare if a.step == "prepare" else select)(work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
