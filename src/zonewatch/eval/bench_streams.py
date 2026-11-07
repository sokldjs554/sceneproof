"""Multi-stream scaling: run N live RTSP streams through one analyser process and record, per
second, input/processed fps, queue depth, dropped frames and analysis lag for each camera.

    python -m zonewatch.eval.bench_streams --streams 1,2,3,4 --seconds 40 --out docs/eval/raw/bench_streams.json
Requires the RTSP simulator (python -m zonewatch.tools.rtsp_sim --port 8554) and sample videos.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import yaml

from zonewatch.edge.agent import EdgeAgent
from zonewatch.edge.config import EdgeConfig
from zonewatch.edge.events import JsonlSink

STREAMS = ["worker-zone", "one-by-one", "store-aisle", "campus-walkway", "people-indoor", "parking-lot"]
ZONE = {"zone_id": "z", "name": "z", "polygon": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.9], [0.3, 0.9]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="1,2,3,4")
    ap.add_argument("--seconds", type=int, default=40)
    ap.add_argument("--rtsp", default="rtsp://127.0.0.1:8554")
    ap.add_argument("--variant", default="yolox_tiny")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="docs/eval/raw/bench_streams.json")
    args = ap.parse_args()
    base = yaml.safe_load(Path("configs/edge.yaml").read_text())
    base["detector"]["variant"] = args.variant
    base["detector"]["model"] = f"models/{args.variant}.onnx"
    base["detector"]["intra_op_threads"] = args.threads
    base["evidence_root"] = "data/runs/bench-streams/evidence"
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    results = []
    for n in [int(x) for x in args.streams.split(",")]:
        cams = [{"camera_id": f"s{i}", "name": f"s{i}", "source": f"{args.rtsp}/{STREAMS[i % len(STREAMS)]}", "autostart": True,
                 "realtime": True, "fps_hint": 25, "zones": [ZONE]} for i in range(n)]
        cfg = EdgeConfig.model_validate({**base, "cameras": cams})
        sink = JsonlSink(f"data/runs/bench-streams/n{n}.jsonl")
        agent = EdgeAgent(cfg, sink=sink, run_id=f"bench-streams-n{n}")
        proc = psutil.Process()
        agent.autostart()
        time.sleep(8)  # warm-up / connect
        proc.cpu_percent(None)
        rows = []
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            ds = agent.device_status()
            rows.append({"t": round(time.time() - t0, 1), "cpu_percent": proc.cpu_percent(None), "rss_mb": proc.memory_info().rss / (1 << 20),
                         "cameras": [{"camera_id": c.camera_id, "state": c.state, "input_fps": c.input_fps, "processed_fps": c.processed_fps,
                                      "lag_ms": c.analysis_lag_ms, "queue_depth": c.queue_depth, "dropped": c.dropped_frames,
                                      "infer_p50": c.infer_ms_p50, "infer_p95": c.infer_ms_p95} for c in ds.cameras]})
            time.sleep(1.0)
        agent.shutdown()
        lag = [c["lag_ms"] for r in rows for c in r["cameras"]]
        proc_fps = [c["processed_fps"] for r in rows for c in r["cameras"]]
        in_fps = [c["input_fps"] for r in rows for c in r["cameras"]]
        summary = {"streams": n, "input_fps_mean_per_stream": float(np.mean(in_fps)) if in_fps else 0,
                   "processed_fps_mean_per_stream": float(np.mean(proc_fps)) if proc_fps else 0,
                   "processed_fps_total": float(np.mean(proc_fps)) * n if proc_fps else 0,
                   "lag_ms_p50": float(np.percentile(lag, 50)) if lag else 0, "lag_ms_p95": float(np.percentile(lag, 95)) if lag else 0,
                   "dropped_total": sum(c["dropped"] for c in rows[-1]["cameras"]) if rows else 0,
                   "queue_depth_mean": float(np.mean([c["queue_depth"] for r in rows for c in r["cameras"]])) if rows else 0,
                   "cpu_percent_mean": float(np.mean([r["cpu_percent"] for r in rows[1:]])) if len(rows) > 1 else 0,
                   "rss_mb_max": max(r["rss_mb"] for r in rows) if rows else 0, "events": sink.count}
        results.append({"summary": summary, "rows": rows})
        print(json.dumps(summary), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"git": git, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "variant": args.variant, "threads": args.threads,
                               "cpu_count": os.cpu_count(), "seconds": args.seconds, "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
