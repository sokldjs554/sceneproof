"""Tracker / zone-rule experiments without re-running the detector.

The detector is the slow part (minutes per sequence); ByteTrack and the zone monitor take seconds. So the raw
detections (score >= 0.05) are cached once per (model, sequence), and tracker / zone settings are replayed on them.
A replay with the settings of a measured run reproduces that run's event counts exactly (same code path as
``run.py``: same detector output, same tracker, same zone monitor, same ground truth and matching).

    python -m sceneproof.eval.bench.replay cache --model models/finetune/yolox_tiny_bench_v2/best.onnx \\
        --ids mot16-04,mot20-01                                  # -> data/bench/cache/<model sha>/<seq>.npy
    python -m sceneproof.eval.bench.replay sweep --model ... --variants variants.json \\
        --group crowd=mot16-04,mot20-01 --group dev=@dev --out docs/eval/bench/raw/....json

variants.json: {"name": {"new_track_thresh": 0.7, "track_buffer_seconds": 3.0, "track_conf": 0.4, "rejoin": true,
"rejoin_strict": true, "rejoin_dist": 1.0, "exclude": false}, ...}; missing keys use configs/edge.yaml (tracker) and
the product zone defaults. ``@dev`` / ``@train`` / ``@test`` / ``@test2`` expand to a frozen split's sequences.
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

import numpy as np
import yaml

from sceneproof.edge.tracker import ByteTracker, TrackerParams
from sceneproof.edge.zones import CloseReason, FrameMeta, ZoneMonitor

from .data import ROOT, load_boxes, load_sequences
from .episodes import _episodes_from_monitor, gt_episodes, zone_specs
from .match import match
from .split import load_split, load_split2

CACHE = ROOT / "data" / "bench" / "cache"
TRACKER_KEYS = ("track_thresh", "low_thresh", "new_track_thresh", "match_thresh", "track_buffer_seconds", "fuse_score")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def cache_dir(model: str) -> Path:
    return CACHE / _sha(ROOT / model)


def cache(model: str, variant: str, ids: list[str], threads: int = 4) -> None:
    import cv2

    from sceneproof.edge.detector import build_detector

    edge = yaml.safe_load((ROOT / "configs" / "edge.yaml").read_text())
    det = build_detector({**edge["detector"], "model": model, "variant": variant, "conf_threshold": 0.05,
                          "intra_op_threads": threads})
    out_dir = cache_dir(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = load_sequences()
    for sid in ids:
        path = out_dir / f"{sid}.npy"
        if path.exists():
            continue
        cap = cv2.VideoCapture(str(seqs[sid].video))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            d = det.detect(img)
            frames.append(np.c_[d.boxes, d.scores].astype(np.float32))
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, np.array(frames, dtype=object), allow_pickle=True)
        tmp.replace(path)
        print(f"[replay] cached {sid}: {len(frames)} frames", flush=True)


_GT: dict[str, list] = {}


def replay_sequence(sid: str, dets: np.ndarray, tracker_cfg: dict, opts: dict) -> dict:
    s = load_sequences()[sid]
    tp = TrackerParams(tracker_cfg["track_thresh"], tracker_cfg["low_thresh"], tracker_cfg["new_track_thresh"],
                       tracker_cfg["match_thresh"], track_buffer=max(5, int(tracker_cfg["track_buffer_seconds"] * s.fps)),
                       fuse_score=tracker_cfg.get("fuse_score", True))
    trk = ByteTracker(tp)
    specs = zone_specs(s.site, exclude=bool(opts.get("exclude")), rejoin=bool(opts.get("rejoin")),
                       rejoin_dist_rel=opts.get("rejoin_dist"))
    for z in specs:
        z.rejoin_strict = bool(opts.get("rejoin_strict", True))
    mon = ZoneMonitor(specs, sid, trajectory_stride=3)
    conf = float(opts.get("track_conf", 0.4))
    evs, meta = [], None
    for i, d in enumerate(dets):
        keep = d[:, 4] >= conf
        meta = FrameMeta(i, i / s.fps, 0.0, s.width, s.height)
        evs += mon.update(trk.update(d[keep, :4], d[keep, 4]), meta)
    if meta is not None:
        evs += mon.close_all(CloseReason.STREAM_ENDED, meta)
    preds = _episodes_from_monitor(evs, "pred", s.fps, None, "pr")
    if sid not in _GT:
        _GT[sid] = gt_episodes(s, load_boxes(s), zone_specs(s.site))
    m = match(preds, _GT[sid])
    return {"gt": sum(g.label == "intrusion" for g in _GT[sid]), "tp": len(m.tp), "missed": len(m.missed),
            "fp": len(m.fp), "duplicates": len(m.duplicates), "rejoins": mon.rejoins}


def expand(group: str) -> list[str]:
    out = []
    for tok in group.split(","):
        if tok.startswith("@"):
            part = tok[1:]
            out += load_split2()["sequences"][part] if part == "test2" else load_split()["sequences"][part]
        elif tok:
            out.append(tok)
    return out


def sweep(model: str, variants: dict, groups: dict[str, list[str]]) -> dict:
    base_trk = yaml.safe_load((ROOT / "configs" / "edge.yaml").read_text())["tracker"]
    d = cache_dir(model)
    results: dict = {}
    for name, v in variants.items():
        trk = {**base_trk, **{k: v[k] for k in TRACKER_KEYS if k in v}}
        opts = {k: val for k, val in v.items() if k not in TRACKER_KEYS}
        results[name] = {"settings": {"tracker": trk, **opts}, "groups": {}}
        for g, ids in groups.items():
            tot: dict = defaultdict(int)
            for sid in ids:
                for k, val in replay_sequence(sid, np.load(d / f"{sid}.npy", allow_pickle=True), trk, opts).items():
                    tot[k] += val
            results[name]["groups"][g] = dict(tot)
        line = " | ".join(f"{g}: " + " ".join(f"{k}={val}" for k, val in r.items()) for g, r in results[name]["groups"].items())
        print(f"[replay] {name:24s} {line}", flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--model", required=True)
    c.add_argument("--variant", default="yolox_tiny")
    c.add_argument("--ids", required=True, help="comma-separated ids or @dev/@train/@test/@test2")
    c.add_argument("--threads", type=int, default=4)
    s = sub.add_parser("sweep")
    s.add_argument("--model", required=True)
    s.add_argument("--variants", required=True, help="JSON file {name: settings}")
    s.add_argument("--group", action="append", required=True, help="name=ids (comma-separated, @split allowed)")
    s.add_argument("--note", default="")
    s.add_argument("--out", default="")
    args = ap.parse_args()
    if args.cmd == "cache":
        cache(args.model, args.variant, expand(args.ids), args.threads)
        return 0
    groups = {g: expand(ids) for g, ids in (x.split("=", 1) for x in args.group)}
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    res = sweep(args.model, json.loads(Path(args.variants).read_text()), groups)
    if args.out:
        Path(args.out).write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git, "model": args.model,
                                              "model_sha256_12": _sha(ROOT / args.model), "note": args.note,
                                              "groups": groups, "results": res}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
