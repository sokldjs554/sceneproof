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


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _git() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _dirty() -> list[str]:
    """Tracked files with uncommitted changes (the code that runs is then not exactly the recorded commit)."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def preflight(ids: list[str]) -> list[str]:
    """Problems that would make a measurement silently incomplete: missing data, or a video whose frame count
    differs from its ground truth according to the fetch manifest."""
    seqs = load_sequences(require_data=False)
    problems = []
    for sid in ids:
        s = seqs.get(sid)
        if s is None or not s.video.exists() or not s.gt_path.exists():
            problems.append(f"{sid}: video or ground truth missing (run scripts/fetch_bench.py)")
            continue
        m = s.extra.get("manifest", {})
        if not m:
            problems.append(f"{sid}: no manifest entry (re-run scripts/fetch_bench.py --ids {sid})")
        elif m.get("error") or not m.get("frames_match_gt", False):
            problems.append(f"{sid}: manifest says {m.get('error') or 'frame count differs from ground truth'}")
    return problems


def _merge(ms: list[Match]) -> Match:
    out = Match()
    for m in ms:
        for k in ("tp", "duplicates", "neutral", "fp", "missed", "start_err_s", "end_err_s"):
            getattr(out, k).extend(getattr(m, k))
        out.dwell_agree += m.dwell_agree
        out.dwell_total += m.dwell_total
    return out


def evaluate(candidate: dict, split_name: str, seq_ids: list[str], det_stride: int, edge_cfg: dict, log=print,
             strict: bool = False) -> dict:
    """strict (the final measurement): a missing sequence or a decoded frame count that differs from the expected
    one is an error, not a log line — an incomplete measurement must not look complete."""
    model_sha = _sha(ROOT / candidate["model"])  # hash the file that is about to be measured, not the one present later
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
            if strict:
                raise RuntimeError(f"{sid}: data missing")
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
        if r.frames != s.frames:
            msg = f"{sid}: decoded {r.frames} frames, expected {s.frames}"
            if strict:
                raise RuntimeError(msg)
            log(f"[bench] WARNING {msg}")
        entry = {"site": s.site, "frames": r.frames, "frames_expected": s.frames, "duration_s": round(s.duration_s, 1),
                 "wall_s": round(r.wall_s, 1)}
        if has_zone:
            gts = gt_episodes(s, boxes, specs)
            m = match(r.episodes, gts)
            n_gt = sum(1 for g in gts if g.label == "intrusion")
            entry.update(summarize(m, s.duration_s, n_gt))
            entry["gt_events"] = [{"id": g.id, "label": g.label, "track": g.track_id, "start_s": round(g.start_s, 2),
                                   "end_s": round(g.end_s, 2), "dwell": g.dwell, "close": g.close_reason} for g in gts]
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
    return {"candidate": {**candidate, "model_sha256_12": model_sha, "op_conf": op,
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
    # provenance captured BEFORE the run (a commit made while an hour-long run is in progress must not be recorded)
    started, git_start, dirty = time.strftime("%Y-%m-%dT%H:%M:%S"), _git(), _dirty()
    zones_sha, zone_rules = _sha(CONFIG_DIR / "zones.yaml"), load_zones()["rules"]
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
                      f"measure again (it will be recorded in {_rel(LOCK)}).", file=sys.stderr)
                return 3
        else:
            lock = {"policy": "the test split is measured once; every further run must give a reason", "runs": []}
        problems = preflight(ids)
        if dirty:
            problems.append(f"uncommitted changes in tracked files: {', '.join(dirty[:5])} — commit first so the record names the code that ran")
        if problems:  # checked before the lock is written: a run that cannot finish must not use up a measurement
            print("final measurement refused before touching the lock:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 4
        lock["runs"].append({"opened": started, "git": git_start, "split_sha256_16": split["sha256_16"],
                             "zones_sha12": zones_sha, "final_sha12": _sha(CONFIG_DIR / "final.yaml"),
                             "candidates": [c["name"] for c in candidates], "reason": args.rerun_reason or "first measurement"})
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    else:
        candidates = [parse_candidate(c) for c in args.candidate] or [parse_candidate("tiny-coco:models/yolox_tiny.onnx")]
        if args.ids:
            ids = [i for i in ids if i in args.ids.split(",")]
        for problem in preflight(ids):
            print(f"[bench] WARNING {problem}", file=sys.stderr)
    results = {}
    if args.out:
        path = Path(args.out)
    elif args.split == "test":  # one file per final run: a rerun never overwrites an earlier run's result
        used = {r.get("result_file") for r in lock["runs"][:-1]}
        path = OUT_DIR / "raw" / "bench_test_final.json"
        if _rel(path) in used:
            path = OUT_DIR / "raw" / f"bench_test_final_run{len(lock['runs'])}.json"
    else:
        path = OUT_DIR / "raw" / f"bench_{args.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(complete: bool) -> dict:
        o = {"split": args.split, "started": started, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_start,
             "uncommitted_changes": dirty, "complete": complete,
             "split_sha256_16": split["sha256_16"], "zones_sha12": zones_sha,
             "zone_rules": zone_rules, "tracker": edge_cfg["tracker"],
             "matching": {"min_overlap_s": 0.5, "start_tolerance_s": 1.0, "max_foot_dist_rel": 0.5},
             "sequences": ids, "results": results}
        path.write_text(json.dumps(o, indent=1, ensure_ascii=False))
        return o

    for c in candidates:  # written after every candidate: a later crash cannot lose finished candidates
        results[c["name"]] = evaluate(c, args.split, ids, args.det_stride, edge_cfg, strict=args.split == "test")
        write(complete=False)
    out = write(complete=True)
    if args.split == "test":
        lock = json.loads(LOCK.read_text())
        lock["runs"][-1]["completed"] = out["time"]
        lock["runs"][-1]["result_file"] = _rel(path)
        LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))
    for name, r in results.items():
        a = r["events"].get("all", {})
        print(f"[bench] {name} {args.split}: P={a.get('precision')} R={a.get('recall')} F1={a.get('f1')} "
              f"FA/h={a.get('false_alarms_per_hour')} AP50={r['detection'].get('all', {}).get('ap50')} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
