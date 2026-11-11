"""GPU inference: ONNX Runtime CPU (reference) vs ONNX Runtime CUDA vs PyTorch CUDA vs TensorRT FP32 / FP16.

Every backend gets the same decoded frames and the same pre/post-processing (sceneproof.edge.detector), so the
differences are the network runtime alone. For each backend: inference and end-to-end latency (p50/p95 over the
frames, repeated), and how its detections differ from the CPU reference after NMS (same matching as
eval/bench_detector.py). A backend that cannot run here (no GPU, missing package, engine build error) is recorded
with the reason instead of stopping the run. Written for notebooks/colab_gpu_bench.ipynb (NVIDIA T4 on Colab).

    python scripts/gpu_bench.py --out results/gpu_bench.json [--variants yolox_nano,yolox_tiny,yolox_s] [--frames 300]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sceneproof.edge.detector import build_detector, letterbox  # noqa: E402
from sceneproof.eval.bench_detector import _cpu_model, load_frames, parity, run_setting  # noqa: E402


def _env() -> dict:
    env: dict = {"python": platform.python_version(), "cpu": _cpu_model(), "cpu_count": os.cpu_count()}
    try:
        q = "name,driver_version,memory.total"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        env["nvidia_smi"] = out.stdout.strip() or out.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        env["nvidia_smi"] = f"unavailable: {e}"
    for mod in ("onnxruntime", "torch", "tensorrt"):
        try:
            m = __import__(mod)
            env[mod] = m.__version__
            if mod == "onnxruntime":
                env["onnxruntime_providers"] = m.get_available_providers()
            if mod == "torch":
                env["torch_cuda"] = m.version.cuda
                env["gpu"] = m.cuda.get_device_name(0) if m.cuda.is_available() else None
        except Exception as e:  # report whatever is missing
            env[mod] = f"unavailable: {type(e).__name__}: {e}"
    return env


def _preload_ort_gpu() -> None:
    """onnxruntime-gpu >= 1.21 can load CUDA/cuDNN from the nvidia-* pip packages that PyTorch installs."""
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:  # the CUDA backend then reports its own error
        pass


def candidates(variant: str, threads: int) -> list[tuple[str, dict]]:
    onnx = str(ROOT / "models" / f"{variant}.onnx")
    pth = str(ROOT / "models" / f"{variant}.pth")
    base = {"variant": variant, "model": onnx, "conf_threshold": 0.4, "nms_threshold": 0.45}
    return [
        ("onnxruntime-cpu", {**base, "backend": "onnxruntime", "intra_op_threads": threads}),
        ("onnxruntime-cuda", {**base, "backend": "onnxruntime", "providers": ["CUDAExecutionProvider"]}),
        ("torch-cuda-fp32", {**base, "backend": "torch", "model": pth, "device": "cuda"}),
        ("tensorrt-fp32", {**base, "backend": "tensorrt", "precision": "fp32"}),
        ("tensorrt-fp16", {**base, "backend": "tensorrt", "precision": "fp16"}),
    ]


def raw_diff(ref_det, det, frames: list[np.ndarray], n: int = 20) -> dict:
    """Largest difference of the raw network output (before decoding / NMS) on the first n frames."""
    worst_box, worst_score = 0.0, 0.0
    for f in frames[:n]:
        blob, _ = letterbox(f, ref_det.input_hw)
        a = ref_det.backend.run(blob[None])
        b = det.backend.run(blob[None])
        worst_box = max(worst_box, float(np.abs(a[:, :4] - b[:, :4]).max()))
        worst_score = max(worst_score, float(np.abs(a[:, 4:] - b[:, 4:]).max()))
    return {"frames": min(n, len(frames)), "max_abs_box_output": round(worst_box, 4), "max_abs_score_output": round(worst_score, 5)}


def summarize(res: dict) -> dict:
    runs = res["runs"]
    med = lambda key, stage: float(np.median([r["stages_ms"][stage][key] for r in runs]))  # noqa: E731
    return {"infer_ms_p50": round(med("p50", "infer"), 3), "infer_ms_p95": round(med("p95", "infer"), 3),
            "total_ms_p50": round(med("p50", "total"), 3), "total_ms_p95": round(med("p95", "total"), 3),
            "fps_median": round(float(np.median([r["fps"] for r in runs])), 1),
            "per_repeat_infer_p50": [round(r["stages_ms"]["infer"]["p50"], 3) for r in runs]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "data" / "videos" / "campus-walkway.mp4"))
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--variants", default="yolox_nano,yolox_tiny,yolox_s")
    ap.add_argument("--threads", type=int, default=0, help="CPU reference threads (0 = all cores)")
    ap.add_argument("--only", default="", help="comma list of backend names to run (default: all)")
    ap.add_argument("--out", default=str(ROOT / "results" / "gpu_bench.json"))
    args = ap.parse_args()

    _preload_ort_gpu()
    threads = args.threads or os.cpu_count() or 1
    frames = load_frames(args.video, args.frames)
    if not frames:
        print(f"no frames read from {args.video}", file=sys.stderr)
        return 2
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out: dict = {"what": "same frames, same pre/post-processing; only the network runtime differs", "git": git,
                 "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "video": Path(args.video).name, "frames": len(frames),
                 "frame_hw": list(frames[0].shape[:2]), "repeats": args.repeats, "env": _env(), "results": []}
    only = {x for x in args.only.split(",") if x}
    for variant in args.variants.split(","):
        ref_det, ref_outs = None, None
        for name, cfg in candidates(variant, threads):
            if only and name not in only and name != "onnxruntime-cpu":
                continue
            row: dict = {"variant": variant, "backend": name}
            print(f"[gpu-bench] {variant} {name} ...", flush=True)
            try:
                t0 = time.perf_counter()
                det = build_detector(cfg)
                row["load_s"] = round(time.perf_counter() - t0, 2)
                b = det.backend
                if name == "onnxruntime-cuda" and "CUDAExecutionProvider" not in getattr(b, "active_providers", []):
                    raise RuntimeError(f"CUDA execution provider not active (session uses {b.active_providers})")
                row["engine_build_s"] = round(b.build_s, 1) if getattr(b, "build_s", None) else None
                row["runtime"] = f"{b.name} {b.version}"
                res = run_setting(det, frames, args.repeats)
                row.update(summarize(res))
                if ref_det is None:
                    ref_det, ref_outs = det, res["outputs"]
                    row["reference"] = True
                else:
                    row["vs_cpu_after_nms"] = parity(ref_outs, res["outputs"])
                    row["vs_cpu_raw_output"] = raw_diff(ref_det, det, frames)
                row["status"] = "ok"
            except Exception as e:  # record and continue with the next backend
                row["status"] = "failed"
                row["error"] = f"{type(e).__name__}: {e}"[:800]
                row["trace_tail"] = traceback.format_exc().strip().splitlines()[-3:]
            out["results"].append(row)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))  # keep partial results
    print(f"\n{'variant':11s} {'backend':17s} {'infer p50':>9s} {'p95':>7s} {'total p50':>9s} {'fps':>6s}  vs CPU after NMS")
    for r in out["results"]:
        if r["status"] != "ok":
            print(f"{r['variant']:11s} {r['backend']:17s} FAILED: {r['error'][:110]}")
            continue
        v = r.get("vs_cpu_after_nms")
        cmp = "reference" if r.get("reference") else (f"frames w/ count diff {v['frames_with_count_diff']}/{v['frames']}, "
                                                      f"unmatched {v['unmatched_boxes']}, max box {v['max_box_delta_px']}px, "
                                                      f"max score {v['max_score_delta']}")
        print(f"{r['variant']:11s} {r['backend']:17s} {r['infer_ms_p50']:9.2f} {r['infer_ms_p95']:7.2f} {r['total_ms_p50']:9.2f} "
              f"{r['fps_median']:6.1f}  {cmp}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
