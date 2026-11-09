"""Benchmark runner: event accuracy (per site, seen vs unseen sites) and person-detection AP for one or more
detector candidates on one split.

    # development (any number of times)
    python -m sceneproof.eval.bench.run --split dev --candidate tiny-coco:models/yolox_tiny.onnx
    # the final measurement: every candidate in configs/bench/final.yaml, ONCE (writes docs/eval/bench/test_lock.json)
    python -m sceneproof.eval.bench.run --split test --final

A second --final is refused unless --rerun-reason "..." is given; the reason is appended to the lock file, so
the history of the held-out set is visible in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

from sceneproof.edge.detector import build_detector

from .data import CONFIG_DIR, ROOT, load_boxes, load_sequences
from .detection import APAccumulator
from .episodes import gt_episodes, load_zones, run_analyzer, zone_specs
from .match import Match, match, summarize
from .split import load_split

OUT_DIR = ROOT / "docs" / "eval" / "bench"
LOCK = OUT_DIR / "test_lock.json"


def _sha(path: Path, n: int = 12) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n] if path.exists() else ""


def _git() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _merge(ms: list[Match]) -> Match:
    out = Match()
    for m in ms:
        for k in ("tp", "duplicates", "neutral", "fp", "missed", "start_err_s", "end_err_s"):
            getattr(out, k).extend(getattr(m, k))
        out.dwell_agree += m.dwell_agree
        out.dwell_total += m.dwell_total
    return out


def evaluate(candidate: dict, split_name: str, seq_ids: list[str], det_stride: int, edge_cfg: dict, log=print) -> dict:
    seqs = load_sequences()
    zc = load_zones()
    split = load_split()
    det_cfg = {**edge_cfg["detector"], "variant": candidate["variant"], "model": candidate["model"],
               "backend": candidate.get("backend", "onnxruntime"), "conf_threshold": 0.05}
    if candidate.get("input_size"):
        det_cfg["input_size"] = candidate["input_size"]
    op = float(candidate.get("conf_threshold", edge_cfg["detector"]["conf_threshold"]))
    detector = build_detector(det_cfg)
    per_seq, matches_by, ap_by = {}, defaultdict(list), defaultdict(APAccumulator)
    dur_by, gt_by = defaultdict(float), defaultdict(int)
    t_all = time.time()
    for sid in seq_ids:
        s = seqs.get(sid)
        if s is None:
            log(f"[bench] {sid}: data missing (run scripts/fetch_bench.py)")
            continue
        has_zone = s.site in zc["sites"]
        specs = zone_specs(s.site, zc) if has_zone else []
        boxes = load_boxes(s)
        r = run_analyzer(s, specs, detector, edge_cfg["tracker"], op, det_stride)
        for f, (db, ds) in r.det_samples.items():
            gtb = boxes.get(f, [])
            ap_by[s.site].add(db, ds, gtb, op)
            ap_by["all"].add(db, ds, gtb, op)
        entry = {"site": s.site, "frames": r.frames, "duration_s": round(s.duration_s, 1), "wall_s": round(r.wall_s, 1)}
        if has_zone:
            gts = gt_episodes(s, boxes, specs)
            m = match(r.episodes, gts)
            n_gt = sum(1 for g in gts if g.label == "intrusion")
            entry.update(summarize(m, s.duration_s, n_gt))
            entry["gt_events"] = [{"id": g.id, "label": g.label, "track": g.track_id, "start_s": round(g.start_s, 2),
                                   "end_s": round(g.end_s, 2), "dwell": g.dwell} for g in gts]
            entry["predictions"] = [{"id": p.id, "track": p.track_id, "start_s": round(p.start_s, 2), "end_s": round(p.end_s, 2),
                                     "dwell": p.dwell, "close": p.close_reason} for p in r.episodes]
            entry["outcome"] = {"tp": m.tp, "duplicates": m.duplicates, "neutral": m.neutral, "fp": m.fp, "missed": m.missed}
            group = "unseen_sites" if s.site in split["unseen_sites"] else "seen_sites"
            for key in (s.site, group, "all"):
                matches_by[key].append(m)
                dur_by[key] += s.duration_s
                gt_by[key] += n_gt
            log(f"[bench] {candidate['name']} {sid}: gt={n_gt} tp={len(m.tp)} fp={len(m.fp)} miss={len(m.missed)} "
                f"dup={len(m.duplicates)} ({r.frames} frames, {r.wall_s:.0f}s)")
        per_seq[sid] = entry
    events = {k: summarize(_merge(v), dur_by[k], gt_by[k]) for k, v in matches_by.items()}
    return {"candidate": {**candidate, "model_sha256_12": _sha(ROOT / candidate["model"]), "op_conf": op,
                          "detector_info": detector.info.__dict__},
            "events": dict(sorted(events.items())), "detection": {k: v.result() for k, v in sorted(ap_by.items())},
            "per_sequence": per_seq, "wall_s": round(time.time() - t_all, 1)}


def parse_candidate(spec: str) -> dict:
    """name:model_path[:variant[:input_w]] e.g. tiny-coco:models/yolox_tiny.onnx:yolox_tiny"""
    parts = spec.split(":")
    name, model = parts[0], parts[1]
    variant = parts[2] if len(parts) > 2 else ("yolox_s" if "yolox_s" in model else "yolox_nano" if "nano" in model else "yolox_tiny")
    c = {"name": name, "model": model, "variant": variant}
    if len(parts) > 3:
        c["input_size"] = [int(parts[3]), int(parts[3])]
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev", "test"], required=True)
    ap.add_argument("--candidate", action="append", default=[], help="name:model_path[:variant[:input]] (dev/train only)")
    ap.add_argument("--final", action="store_true", help="required for --split test; uses configs/bench/final.yaml")
    ap.add_argument("--rerun-reason", default="")
    ap.add_argument("--ids", default="", help="restrict to these sequence ids (dev/train only)")
    ap.add_argument("--det-stride", type=int, default=5, help="score detection AP on every Nth frame (0 = off)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    edge_cfg = yaml.safe_load((ROOT / "configs" / "edge.yaml").read_text())
    split = load_split()
    ids = split["sequences"][args.split]
    if args.split == "test":
        if not args.final or args.candidate or args.ids:
            print("the test split is measured only with --final (all candidates from configs/bench/final.yaml, all sequences)",
                  file=sys.stderr)
            return 2
        final = yaml.safe_load((CONFIG_DIR / "final.yaml").read_text())
        candidates = final["candidates"]
        if LOCK.exists():
            lock = json.loads(LOCK.read_text())
            if not args.rerun_reason:
                print(f"test split already measured at {lock['runs'][0]['opened']} — refusing. Pass --rerun-reason to "
                      f"measure again (it will be recorded in {LOCK.relative_to(ROOT)}).", file=sys.stderr)
                return 3
        else:
            lock = {"policy": "the test split is measured once; every further run must give a reason", "runs": []}
        lock["runs"].append({"opened": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": _git(), "split_sha256_16": split["sha256_16"],
                             "zones_sha12": _sha(CONFIG_DIR / "zones.yaml"), "final_sha12": _sha(CONFIG_DIR / "final.yaml"),
                             "candidates": [c["name"] for c in candidates], "reason": args.rerun_reason or "first measurement"})
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    else:
        candidates = [parse_candidate(c) for c in args.candidate] or [parse_candidate("tiny-coco:models/yolox_tiny.onnx")]
        if args.ids:
            ids = [i for i in ids if i in args.ids.split(",")]
    results = {}
    for c in candidates:
        results[c["name"]] = evaluate(c, args.split, ids, args.det_stride, edge_cfg)
    out = {"split": args.split, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": _git(),
           "split_sha256_16": split["sha256_16"], "zones_sha12": _sha(CONFIG_DIR / "zones.yaml"),
           "zone_rules": load_zones()["rules"], "tracker": edge_cfg["tracker"],
           "matching": {"min_overlap_s": 0.5, "start_tolerance_s": 1.0, "max_foot_dist_rel": 0.5},
           "sequences": ids, "results": results}
    path = Path(args.out) if args.out else OUT_DIR / "raw" / f"bench_{args.split}{'_final' if args.final else ''}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    if args.split == "test":
        lock = json.loads(LOCK.read_text())
        lock["runs"][-1]["completed"] = out["time"]
        lock["runs"][-1]["result_file"] = str(path.relative_to(ROOT))
        LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    for name, r in results.items():
        a = r["events"].get("all", {})
        print(f"[bench] {name} {args.split}: P={a.get('precision')} R={a.get('recall')} F1={a.get('f1')} "
              f"FA/h={a.get('false_alarms_per_hour')} AP50={r['detection'].get('all', {}).get('ap50')} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
