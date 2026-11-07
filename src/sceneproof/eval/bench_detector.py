"""Detector benchmark: PyTorch vs ONNX Runtime on identical frames, pre/post-processing and box.

Reports per-stage latency (preprocess / infer / postprocess) p50/p95, throughput, process CPU% and
RSS, and output parity (box/score differences after NMS on the same frames). Every setting is
repeated ``--repeats`` times with a warm-up; raw per-frame timings are kept in the JSON output so
the summary can be recomputed.

    python -m sceneproof.eval.bench_detector --video data/videos/campus-walkway.mp4 --frames 200 \
        --repeats 3 --variants yolox_nano,yolox_tiny,yolox_s --out docs/eval/raw/bench_detector.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from sceneproof.edge.detector import build_detector


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            return next((ln.split(":", 1)[1].strip() for ln in f if "model name" in ln), "")
    except OSError:
        return ""


def load_frames(video: str, n: int, step: int = 1) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    frames = []
    i = 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames


def run_setting(det, frames, repeats: int, warmup: int = 10) -> dict:
    proc = psutil.Process()
    det.warmup(warmup, (frames[0].shape[0], frames[0].shape[1]))
    runs = []
    outputs = None
    for r in range(repeats):
        proc.cpu_percent(None)
        rss0 = proc.memory_info().rss
        stages = {"preprocess": [], "infer": [], "postprocess": [], "total": []}
        outs = []
        t_start = time.perf_counter()
        for f in frames:
            t0 = time.perf_counter()
            d = det.detect(f)
            stages["total"].append((time.perf_counter() - t0) * 1e3)
            for k in ("preprocess", "infer", "postprocess"):
                stages[k].append(d.timings_ms[k])
            outs.append((d.boxes, d.scores))
        wall = time.perf_counter() - t_start
        cpu = proc.cpu_percent(None)
        rss1 = proc.memory_info().rss
        summ = {k: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(np.mean(v))}
                for k, v in stages.items()}
        runs.append({"repeat": r, "wall_s": wall, "fps": len(frames) / wall, "cpu_percent": cpu,
                     "rss_mb": rss1 / (1 << 20), "rss_delta_mb": (rss1 - rss0) / (1 << 20), "stages_ms": summ,
                     "raw_total_ms": [round(x, 3) for x in stages["total"]],
                     "raw_infer_ms": [round(x, 3) for x in stages["infer"]]})
        outputs = outs
    return {"runs": runs, "outputs": outputs}


def parity(a_outs, b_outs, iou_thresh: float = 0.5) -> dict:
    """Compare detections frame by frame: count diffs, max box delta for matched boxes."""
    from sceneproof.edge.tracker import iou_matrix

    count_diff = 0
    max_box = 0.0
    max_score = 0.0
    matched = 0
    unmatched = 0
    for (ba, sa), (bb, sb) in zip(a_outs, b_outs, strict=True):
        if len(sa) != len(sb):
            count_diff += 1
        if len(sa) and len(sb):
            iou = iou_matrix(ba, bb)
            for i in range(len(sa)):
                j = int(iou[i].argmax())
                if iou[i, j] >= iou_thresh:
                    matched += 1
                    max_box = max(max_box, float(np.abs(ba[i] - bb[j]).max()))
                    max_score = max(max_score, float(abs(sa[i] - sb[j])))
                else:
                    unmatched += 1
        else:
            unmatched += len(sa) + len(sb)
    return {"frames": len(a_outs), "frames_with_count_diff": count_diff, "matched_boxes": matched,
            "unmatched_boxes": unmatched, "max_box_delta_px": round(max_box, 3), "max_score_delta": round(max_score, 5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/videos/campus-walkway.mp4")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--variants", default="yolox_nano,yolox_tiny,yolox_s")
    ap.add_argument("--threads", default="4", help="comma list of intra-op thread counts")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--out", default="docs/eval/raw/bench_detector.json")
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()

    frames = load_frames(args.video, args.frames, args.step)
    env = {"platform": platform.platform(), "python": sys.version.split()[0], "cpu_count": os.cpu_count(),
           "cpu_model": _cpu_model(),
           "video": args.video, "frames": len(frames), "frame_wh": [frames[0].shape[1], frames[0].shape[0]],
           "repeats": args.repeats, "conf_threshold": args.conf, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    import onnxruntime as ort

    env["onnxruntime"] = ort.__version__
    try:
        import torch

        env["torch"] = torch.__version__
    except ImportError:
        env["torch"] = None
    results = []
    for variant in args.variants.split(","):
        for threads in [int(t) for t in args.threads.split(",")]:
            settings = [("onnxruntime", {"variant": variant, "backend": "onnxruntime", "model": f"models/{variant}.onnx",
                                         "intra_op_threads": threads, "conf_threshold": args.conf}),
                        ("onnxruntime-noopt", {"variant": variant, "backend": "onnxruntime", "model": f"models/{variant}.onnx",
                                               "intra_op_threads": threads, "conf_threshold": args.conf,
                                               "graph_optimization": "none"})]
            if not args.skip_torch and env["torch"]:
                settings.append(("torch", {"variant": variant, "backend": "torch", "model": f"models/{variant}.pth",
                                           "intra_op_threads": threads, "conf_threshold": args.conf}))
            outs = {}
            for name, cfg in settings:
                det = build_detector(cfg)
                print(f"[bench] {variant} threads={threads} {name} ({det.info.backend} {det.info.backend_version}) ...",
                      flush=True)
                r = run_setting(det, frames, args.repeats)
                outs[name] = r["outputs"]
                results.append({"variant": variant, "threads": threads, "setting": name, "backend": det.info.backend,
                                "backend_version": det.info.backend_version, "model_version": det.info.model_version,
                                "input_wh": [det.info.input_width, det.info.input_height], "runs": r["runs"]})
                del det
            par = {}
            if "torch" in outs:
                par["torch_vs_onnxruntime"] = parity(outs["torch"], outs["onnxruntime"])
            par["onnxruntime_vs_noopt"] = parity(outs["onnxruntime"], outs["onnxruntime-noopt"])
            results.append({"variant": variant, "threads": threads, "setting": "parity", "parity": par})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": env, "results": results}, indent=1))
    # short table
    print(f"\n{'variant':12s} {'thr':>3s} {'setting':18s} {'infer p50':>9s} {'infer p95':>9s} {'total p50':>9s} {'fps':>6s} {'cpu%':>6s} {'rss MB':>7s}")
    for r in results:
        if r["setting"] == "parity":
            print(f"{r['variant']:12s} {r['threads']:3d} parity: {json.dumps(r['parity'])}")
            continue
        inf50 = np.median([x["stages_ms"]["infer"]["p50"] for x in r["runs"]])
        inf95 = np.median([x["stages_ms"]["infer"]["p95"] for x in r["runs"]])
        tot50 = np.median([x["stages_ms"]["total"]["p50"] for x in r["runs"]])
        fps = np.median([x["fps"] for x in r["runs"]])
        cpu = np.median([x["cpu_percent"] for x in r["runs"]])
        rss = np.median([x["rss_mb"] for x in r["runs"]])
        print(f"{r['variant']:12s} {r['threads']:3d} {r['setting']:18s} {inf50:9.1f} {inf95:9.1f} {tot50:9.1f} {fps:6.1f} {cpu:6.0f} {rss:7.0f}")
    print(f"\nraw results -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
