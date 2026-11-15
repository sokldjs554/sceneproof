"""How many 30 fps camera channels one GPU keeps up with — measured with the product's own analyser.

N copies of a 30 fps sample video are replayed in real time (looped, started a few seconds apart) into ONE analyser
process (EdgeAgent): one detector shared by all cameras, and per camera a decoder thread, a bounded drop-oldest
queue, evidence ring, tracker, zone logic and events — the code a deployment runs. For each channel count the script
counts, over a fixed window, the frames every channel received and analysed, the frames the queue dropped, the
analysis lag, CPU and GPU load and the time per stage. A channel count passes when every channel analysed >= 98% of
30 fps (configs/bench/gpu_channels.yaml, written before the first GPU run). The answer is the largest passing channel
count below the first failing one.

The plan runs the sweep for more than one pipeline setting (as built, and with fewer evidence frames kept), and adds:
  * CPU reference: the as-built sweep with ONNX Runtime on the CPU.
  * Isolation: videos analysed at the same time (every frame) must give the same events as one at a time. Cameras
    share the detector, so this is the check that frames and results are not mixed up between cameras.
  * Evidence check: what keeping fewer evidence frames changes (events, key-frame timing, clips).

    python scripts/gpu_channels.py --out results/gpu_channels.json
    python scripts/gpu_channels.py --backend onnxruntime --variant yolox_nano --channels 1,2 --seconds 5   # no GPU
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpu_bench import _env, _preload_ort_gpu  # noqa: E402  (same directory)
from sceneproof.edge.agent import EdgeAgent  # noqa: E402
from sceneproof.edge.config import EdgeConfig  # noqa: E402
from sceneproof.edge.detector import PersonDetector, build_detector  # noqa: E402
from sceneproof.edge.events import JsonlSink  # noqa: E402

STAGES = ("evidence_encode", "preprocess", "infer", "postprocess", "track", "zones", "total")


# --------------------------------------------------------------------------- pure helpers (tests/test_gpu_channels.py)


def judge(per_channel: list[dict], min_ratio: float) -> bool:
    """A channel count passes when every channel analysed at least `min_ratio` of the target frame rate."""
    return bool(per_channel) and all(c["processed_ratio"] >= min_ratio for c in per_channel)


def capacity(rows: list[dict]) -> tuple[int, int | None]:
    """-> (largest passing channel count below the first failing one, first failing channel count or None)."""
    best, first_fail = 0, None
    for r in sorted(rows, key=lambda r: r["channels"]):
        if not r["pass"]:
            first_fail = r["channels"]
            break
        best = r["channels"]
    return best, first_fail


def detector_dict(variant: str, backend: str, precision: str | None) -> dict:
    d = {"conf_threshold": 0.4, "nms_threshold": 0.45, "variant": variant, "backend": backend,
         "model": f"models/{variant}.pth" if backend == "torch" else f"models/{variant}.onnx"}
    if backend == "tensorrt":
        d["precision"] = precision or "fp16"
    if backend == "onnxruntime":
        d["intra_op_threads"] = os.cpu_count() or 1
    return d


def final_events(jsonl: str | Path, camera_id: str) -> list[dict]:
    """Last revision of every event of one camera."""
    latest: dict[str, dict] = {}
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ev = json.loads(line)
                if ev.get("camera_id") == camera_id and int(ev.get("revision", 0)) >= int(latest.get(ev["event_id"], {}).get("revision", -1)):
                    latest[ev["event_id"]] = ev
    return list(latest.values())


def event_keys(events: list[dict]) -> list[list]:
    """What must not change between two runs of the same video: (start s, end s, dwell escalation, zone)."""
    out = []
    for ev in events:
        start = float(ev.get("media_entry_s", 0.0))
        end = ev.get("media_exit_s")
        end = start + float(ev.get("dwell_seconds", 0.0)) if end is None else float(end)
        out.append([round(start, 3), round(end, 3), ev.get("kind") == "EVENT_KIND_ZONE_DWELL", ev.get("zone_id", "")])
    return sorted(out)


def evidence_summary(events: list[dict]) -> dict:
    """Key frames and clips of the final events: how far the entry / exit picture is from the event's own time."""
    offsets: dict[str, list[float]] = {"entry": [], "exit": []}
    roles: dict[str, int] = {}
    clips = clip_frames = 0
    for ev in events:
        evidence = ev.get("evidence", {})
        want = {"entry": ev.get("media_entry_s"), "exit": ev.get("media_exit_s")}
        for kf in evidence.get("key_frames", []):
            role = kf.get("role", "")
            if role.endswith("_annot"):
                continue
            roles[role] = roles.get(role, 0) + 1
            if role in offsets and want[role] is not None:
                offsets[role].append(abs(float(kf.get("media_time_s", 0.0)) - float(want[role])))
        if evidence.get("clip_path"):
            clips += 1
            clip_frames += int(evidence.get("clip_frames", 0))
    out: dict = {"events": len(events), "key_frames": dict(sorted(roles.items())), "clips": clips, "clip_frames": clip_frames}
    for role, xs in offsets.items():
        out[f"{role}_picture_offset_s_max"] = round(max(xs), 3) if xs else None
        out[f"{role}_picture_offset_s_mean"] = round(float(np.mean(xs)), 3) if xs else None
    return out


# --------------------------------------------------------------------------- inputs


def sample_cameras() -> dict[str, dict]:
    """video name -> its camera entry (zones, fps_hint) from configs/edge.yaml and configs/eval.yaml."""
    out: dict[str, dict] = {}
    for cfg in ("configs/edge.yaml", "configs/eval.yaml"):
        for cam in yaml.safe_load((ROOT / cfg).read_text())["cameras"]:
            if cam["source"].startswith("file://"):
                out.setdefault(Path(cam["source"]).stem, cam)
    return out


def video_info(name: str) -> dict:
    path = ROOT / "data" / "videos" / f"{name}.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"{path} (run scripts/fetch_data.sh)")
    info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "native_fps": round(cap.get(cv2.CAP_PROP_FPS) or 25.0, 3), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    return info


def edge_config(det_cfg: dict, pipeline: dict, cameras: list[dict], evidence_root: Path) -> EdgeConfig:
    base = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text())  # tracker and pipeline defaults of the product
    data = {"device_id": "bench-01", "tracker": base["tracker"], "detector": det_cfg, "cameras": cameras,
            "evidence_root": str(evidence_root),
            # opencv: the file source that paces playback itself (and is the same on every machine)
            "pipeline": {**base["pipeline"], "source_backend": "opencv", **pipeline}}
    return EdgeConfig.model_validate(data)


# --------------------------------------------------------------------------- measuring


class GpuSampler:
    """GPU utilisation and memory twice a second from one nvidia-smi process (None when there is no GPU)."""

    def __enter__(self) -> GpuSampler:
        self.proc = None
        self.result: dict | None = None
        with contextlib.suppress(OSError):  # no nvidia-smi: no GPU numbers
            self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                          "--format=csv,noheader,nounits", "-lms", "500"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            text, _ = self.proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            text, _ = self.proc.communicate()
        rows = []
        for line in text.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and all(p.replace(".", "", 1).isdigit() for p in parts):
                rows.append((float(parts[0]), float(parts[1])))
        if rows:
            util = [r[0] for r in rows]
            self.result = {"samples": len(rows), "util_percent_mean": round(float(np.mean(util)), 1),
                           "util_percent_max": max(util), "memory_used_mb_max": max(r[1] for r in rows)}


def measure(agent: EdgeAgent, target_fps: float, measure_s: float, min_ratio: float) -> dict:
    pipes = dict(agent.pipelines)
    proc = psutil.Process()
    proc.cpu_percent(None)
    psutil.cpu_percent(None)

    def snap() -> dict:
        return {cid: (p.queue.stats(), p.stats.frames_processed) for cid, p in pipes.items()}

    lag: dict[str, list[float]] = {cid: [] for cid in pipes}
    depth: dict[str, list[int]] = {cid: [] for cid in pipes}
    with GpuSampler() as gpu:
        s0, t0 = snap(), time.perf_counter()
        while time.perf_counter() - t0 < measure_s:
            time.sleep(0.2)
            for cid, p in pipes.items():
                lag[cid].append(p.stats.analysis_lag_ms)
                depth[cid].append(len(p.queue))
        s1, dt = snap(), time.perf_counter() - t0
    cpu_process, cpu_system = proc.cpu_percent(None), psutil.cpu_percent(None)
    per_channel = []
    for cid, p in pipes.items():
        (q0, done0), (q1, done1) = s0[cid], s1[cid]
        received, processed, dropped = q1.received - q0.received, done1 - done0, q1.dropped - q0.dropped
        per_channel.append({
            "camera_id": cid, "frame_wh": [p.source.status.width, p.source.status.height],
            "received_fps": round(received / dt, 2), "processed_fps": round(processed / dt, 2),
            "processed_ratio": round(processed / dt / target_fps, 4),
            "dropped": dropped, "dropped_ratio": round(dropped / received, 4) if received else None,
            "lag_ms_p50": round(float(np.percentile(lag[cid], 50)), 1), "lag_ms_p95": round(float(np.percentile(lag[cid], 95)), 1),
            "lag_ms_max": round(max(lag[cid]), 1), "queue_depth_mean": round(float(np.mean(depth[cid])), 2),
            "stage_ms_mean": {s: round(p.timer.mean(s), 2) for s in STAGES},
            "infer_ms_p50": round(p.timer.percentile("infer", 50), 2), "last_error": p.stats.last_error})
    ratios = [c["processed_ratio"] for c in per_channel]
    return {
        "channels": len(per_channel), "pass": judge(per_channel, min_ratio), "window_s": round(dt, 2),
        "processed_fps_total": round(sum(c["processed_fps"] for c in per_channel), 1),
        "processed_ratio_min": min(ratios), "processed_ratio_mean": round(float(np.mean(ratios)), 4),
        "received_fps_min": min(c["received_fps"] for c in per_channel),
        "dropped_total": sum(c["dropped"] for c in per_channel),
        "lag_ms_p50_median": round(float(np.median([c["lag_ms_p50"] for c in per_channel])), 1),
        "lag_ms_p95_worst": max(c["lag_ms_p95"] for c in per_channel),
        # process CPU: 100 = one core fully used; system CPU: 100 = all cores fully used
        "cpu_percent_process": round(cpu_process, 1), "cpu_percent_system": round(cpu_system, 1),
        "rss_mb": round(proc.memory_info().rss / (1 << 20), 1), "gpu": gpu.result,
        # per analysed frame, averaged over the channels. infer includes waiting for the shared detector
        "stage_ms_mean": {s: round(float(np.mean([c["stage_ms_mean"][s] for c in per_channel])), 2) for s in STAGES},
        "per_channel": per_channel,
    }


def sweep(det: PersonDetector, det_cfg: dict, pipeline: dict, plan: dict, channels: list[int], info: dict, run_dir: Path,
          label: str) -> list[dict]:
    src = sample_cameras()[plan["video"]]
    rows: list[dict] = []
    failures = 0
    for n in channels:
        cams = [{"camera_id": f"ch{i + 1:02d}", "name": plan["video"], "source": src["source"], "autostart": False,
                 "realtime": True, "loop": True, "speed": plan["target_fps"] / info["native_fps"],
                 "fps_hint": src.get("fps_hint", info["native_fps"]), "zones": src["zones"]} for i in range(n)]
        cfg = edge_config(det_cfg, pipeline, cams, run_dir / f"evidence-{label}-n{n}")
        sink = JsonlSink(run_dir / f"{label}-n{n}.jsonl")
        agent = EdgeAgent(cfg, sink=sink, run_id=f"gpu-channels-{label}-n{n}", detector=det)
        try:
            for i, cam in enumerate(cams):
                if i:
                    time.sleep(plan["stagger_s"])
                agent.start_camera(cam["camera_id"])
            time.sleep(plan["warmup_s"])
            row = measure(agent, plan["target_fps"], plan["measure_s"], plan["pass"]["min_processed_ratio"])
        finally:
            agent.shutdown()
        row["events_written"] = sink.count
        rows.append(row)
        gpu = f", gpu {row['gpu']['util_percent_mean']:.0f}%" if row["gpu"] else ""
        print(f"[channels] {label} n={n:2d} {'PASS' if row['pass'] else 'FAIL'}  slowest channel "
              f"{row['processed_ratio_min'] * plan['target_fps']:.1f} fps, dropped {row['dropped_total']}, "
              f"lag p95 {row['lag_ms_p95_worst']:.0f} ms, cpu {row['cpu_percent_system']:.0f}%{gpu}", flush=True)
        failures = 0 if row["pass"] else failures + 1
        if failures >= plan["stop_after_failures"]:
            break
    return rows


# --------------------------------------------------------------------------- every-frame runs (isolation, evidence)


def run_lossless(det: PersonDetector, det_cfg: dict, pipeline: dict, videos: list[str], run_dir: Path, tag: str) -> dict:
    """Analyse every frame of the given sample videos in one agent. -> video -> {frames, events, evidence, ...}."""
    known = sample_cameras()
    cams = [{**known[v], "camera_id": v, "realtime": False, "loop": False, "autostart": True, "speed": 1.0} for v in videos]
    cfg = edge_config(det_cfg, pipeline, cams, run_dir / f"evidence-{tag}")
    jsonl = run_dir / f"{tag}.jsonl"
    agent = EdgeAgent(cfg, sink=JsonlSink(jsonl), run_id=f"gpu-channels-{tag}", detector=det)
    t0 = time.perf_counter()
    try:
        agent.autostart()
        pipes = dict(agent.pipelines)
        deadline = time.time() + 1800
        while time.time() < deadline and not all(p.ended and not p._pending_final for p in pipes.values()):
            time.sleep(0.2)
        ended = all(p.ended for p in pipes.values())
        per = {v: {"frames": p.stats.frames_processed, "evidence_encode_ms_mean": round(p.timer.mean("evidence_encode"), 2),
                   "ring_entries_at_end": p.ring.stats().entries} for v, p in pipes.items()}
    finally:
        agent.shutdown()
    for v in videos:
        events = final_events(jsonl, v)
        per[v].update({"events": event_keys(events), "evidence": evidence_summary(events)})
    return {"wall_s": round(time.perf_counter() - t0, 1), "ended": ended, "videos": per}


def isolation(det: PersonDetector, det_cfg: dict, videos: list[str], run_dir: Path, label: str) -> dict:
    together = run_lossless(det, det_cfg, {}, videos, run_dir, f"{label}-together")
    alone = {v: run_lossless(det, det_cfg, {}, [v], run_dir, f"{label}-alone-{v}") for v in videos}
    out = {}
    for v in videos:
        a, b = together["videos"][v], alone[v]["videos"][v]
        same = a["frames"] == b["frames"] and a["events"] == b["events"]
        out[v] = {"frames_together": a["frames"], "frames_alone": b["frames"], "events_together": len(a["events"]),
                  "events_alone": len(b["events"]), "same": same}
        if a["events"] != b["events"]:
            out[v]["only_together"] = [e for e in a["events"] if e not in b["events"]]
            out[v]["only_alone"] = [e for e in b["events"] if e not in a["events"]]
    return {"what": "the same videos analysed at the same time by one analyser vs one at a time; every frame, no drops",
            "compared": "per video: frames analysed and every event's (start s, end s, dwell escalation, zone)",
            "all_same": together["ended"] and all(c["same"] for c in out.values()),
            "wall_s_together": together["wall_s"], "wall_s_alone_sum": round(sum(r["wall_s"] for r in alone.values()), 1),
            "videos": out}


def evidence_check(det: PersonDetector, det_cfg: dict, videos: list[str], ring_fps: float, run_dir: Path, label: str) -> dict:
    every = run_lossless(det, det_cfg, {}, videos, run_dir, f"{label}-ring-every-frame")
    fewer = run_lossless(det, det_cfg, {"ring_fps": ring_fps}, videos, run_dir, f"{label}-ring-fps-{ring_fps:g}")
    out = {}
    for v in videos:
        a, b = every["videos"][v], fewer["videos"][v]
        out[v] = {"frames": a["frames"], "same_events": a["frames"] == b["frames"] and a["events"] == b["events"],
                  "every_frame": {k: a[k] for k in ("evidence_encode_ms_mean", "ring_entries_at_end", "evidence")},
                  f"ring_fps_{ring_fps:g}": {k: b[k] for k in ("evidence_encode_ms_mean", "ring_entries_at_end", "evidence")}}
    return {"what": f"every analysed frame kept as evidence vs pipeline.ring_fps = {ring_fps:g}; every frame analysed",
            "ring_fps": ring_fps, "all_same_events": all(c["same_events"] for c in out.values()), "videos": out}


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(ROOT / "configs" / "bench" / "gpu_channels.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results" / "gpu_channels.json"))
    ap.add_argument("--variant", default=None)
    ap.add_argument("--backend", default=None, help="onnxruntime | torch | tensorrt")
    ap.add_argument("--precision", default=None, help="tensorrt: fp32 | fp16")
    ap.add_argument("--channels", default=None, help="comma list, e.g. 1,2,4")
    ap.add_argument("--seconds", type=float, default=None, help="measuring window per channel count")
    ap.add_argument("--warmup", type=float, default=None)
    ap.add_argument("--skip-cpu-reference", action="store_true")
    ap.add_argument("--skip-isolation", action="store_true")
    ap.add_argument("--skip-evidence-check", action="store_true")
    ap.add_argument("--isolation-videos", default=None, help="comma list of sample video names")
    args = ap.parse_args()

    os.chdir(ROOT)  # configs use paths relative to the repository root
    _preload_ort_gpu()
    plan = yaml.safe_load(Path(args.plan).read_text())
    overrides: dict = {}
    for key, value in (("variant", args.variant), ("backend", args.backend), ("precision", args.precision)):
        if value is not None and plan["detector"].get(key) != value:
            overrides[f"detector.{key}"] = value
            plan["detector"][key] = value
    if args.channels:
        overrides["channels"] = plan["channels"] = [int(x) for x in args.channels.split(",")]
    if args.seconds is not None:
        overrides["measure_s"] = plan["measure_s"] = args.seconds
    if args.warmup is not None:
        overrides["warmup_s"] = plan["warmup_s"] = args.warmup
    if args.isolation_videos:
        overrides["isolation.videos"] = plan["isolation"]["videos"] = args.isolation_videos.split(",")
    for flag, name in ((args.skip_cpu_reference, "cpu_reference"), (args.skip_isolation, "isolation"),
                       (args.skip_evidence_check, "evidence_check")):
        if flag:
            overrides[name] = "skipped"

    info = video_info(plan["video"])
    run_dir = ROOT / "data" / "runs" / "gpu-channels" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out: dict = {
        "what": "channels of 30 fps video one analyser process keeps up with; a channel count passes when every "
                "channel analysed >= min_processed_ratio of target_fps over the window",
        "git": git, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": _env(), "plan": plan, "overrides": overrides,
        "video": {"name": plan["video"], **info}, "sweeps": [], "isolation": None, "evidence_check": None}

    def save() -> None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))  # keep partial results

    def failed(e: Exception) -> dict:
        return {"status": "failed", "error": f"{type(e).__name__}: {e}"[:800],
                "trace_tail": traceback.format_exc().strip().splitlines()[-3:]}

    d = plan["detector"]
    main_cfg = detector_dict(d["variant"], d["backend"], d.get("precision"))
    main_label = d["backend"] + (f"-{main_cfg['precision']}" if d["backend"] == "tensorrt" else "")
    jobs = [("main", main_label, main_cfg, v["name"], v.get("pipeline") or {}, plan["channels"]) for v in plan["variants"]]
    ref = plan.get("cpu_reference")
    if ref and not args.skip_cpu_reference and d["backend"] != ref["backend"]:
        jobs.append(("cpu_reference", ref["backend"], detector_dict(d["variant"], ref["backend"], None), "as-built", {}, ref["channels"]))
    detectors: dict[str, PersonDetector] = {}
    for role, label, det_cfg, variant, pipeline, channels in jobs:
        entry: dict = {"role": role, "label": label, "variant": variant, "pipeline": pipeline, "detector_config": det_cfg}
        print(f"[channels] {role}: {det_cfg['variant']} {label}, {variant} {pipeline or ''}, channel counts {channels}", flush=True)
        try:
            if label not in detectors:
                t0 = time.perf_counter()
                detectors[label] = build_detector(det_cfg)
                entry["load_s"] = round(time.perf_counter() - t0, 1)
            det = detectors[label]
            entry["runtime"] = f"{det.backend.name} {det.backend.version}"
            entry["rows"] = sweep(det, det_cfg, pipeline, plan, channels, info, run_dir, f"{label}-{variant}")
            entry["capacity_channels"], entry["first_failing_channels"] = capacity(entry["rows"])
            entry["largest_passing_channels"] = max((r["channels"] for r in entry["rows"] if r["pass"]), default=0)
            one = next((r for r in entry["rows"] if r["channels"] == 1), None)
            if one:  # arithmetic, not a measurement: how many channels the network alone would allow
                infer = one["per_channel"][0]["infer_ms_p50"]
                entry["network_only_ceiling_channels"] = int(1000 / (infer * plan["target_fps"])) if infer else None
            entry["status"] = "ok"
        except Exception as e:  # record and continue (no GPU, missing package, engine build error)
            entry.update(failed(e))
            print(f"[channels] {role} {variant} FAILED: {entry['error'][:200]}", flush=True)
        out["sweeps"].append(entry)
        save()

    det = detectors.get(main_label)
    for name, skip in (("isolation", args.skip_isolation), ("evidence_check", args.skip_evidence_check)):
        if skip:
            continue
        if det is None:
            out[name] = {"status": "skipped", "reason": "the measured detector could not be built"}
        else:
            print(f"[channels] {name} ({main_label}): {plan[name]['videos']}", flush=True)
            try:
                if name == "isolation":
                    res = isolation(det, main_cfg, plan[name]["videos"], run_dir, main_label)
                else:
                    res = evidence_check(det, main_cfg, plan[name]["videos"], float(plan[name]["ring_fps"]), run_dir, main_label)
                out[name] = {"status": "ok", "label": main_label, **res}
            except Exception as e:
                out[name] = failed(e)
        save()

    fps = plan["target_fps"]
    print(f"\n{'path':16s} {'setting':12s} {'ch':>3s} {'':5s} {'slowest fps':>11s} {'dropped':>8s} {'lag p95':>8s} {'cpu %':>6s} {'gpu %':>6s}")
    for s in out["sweeps"]:
        if s["status"] != "ok":
            print(f"{s['label']:16s} {s['variant']:12s} FAILED: {s['error'][:100]}")
            continue
        for r in s["rows"]:
            gpu = f"{r['gpu']['util_percent_mean']:6.0f}" if r["gpu"] else f"{'-':>6s}"
            print(f"{s['label']:16s} {s['variant']:12s} {r['channels']:3d} {'PASS' if r['pass'] else 'FAIL':5s} "
                  f"{r['processed_ratio_min'] * fps:11.1f} {r['dropped_total']:8d} {r['lag_ms_p95_worst']:8.0f} "
                  f"{r['cpu_percent_system']:6.0f} {gpu}")
        print(f"{s['label']:16s} {s['variant']:12s} -> holds {fps} fps on up to {s['capacity_channels']} channel(s)"
              + (f"; first failing count {s['first_failing_channels']}" if s["first_failing_channels"] else "; no failing count tested"))
    iso, evc = out["isolation"], out["evidence_check"]
    if iso:
        print("isolation: " + (("same frames and events together and alone" if iso["all_same"] else "DIFFERENT — see the result file")
                               if iso["status"] == "ok" else f"{iso['status']} ({iso.get('reason') or iso.get('error')})"))
    if evc:
        print("evidence check: " + ((f"events {'unchanged' if evc['all_same_events'] else 'CHANGED'} with ring_fps {evc['ring_fps']:g}")
                                    if evc["status"] == "ok" else f"{evc['status']} ({evc.get('reason') or evc.get('error')})"))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
