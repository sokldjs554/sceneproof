"""scripts/gpu_bench.py runs the CPU reference and records every GPU backend it cannot run, with the reason."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "videos" / "campus-walkway.mp4"
MODEL = ROOT / "models" / "yolox_nano.onnx"


@pytest.mark.skipif(not (VIDEO.exists() and MODEL.exists()), reason="needs scripts/fetch_data.sh and fetch_models.sh")
def test_cpu_reference_runs_and_gpu_backends_report_why_not(tmp_path):
    out = tmp_path / "gpu.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gpu_bench.py"), "--frames", "5", "--repeats", "1",
                        "--variants", "yolox_nano", "--out", str(out)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = {x["backend"]: x for x in json.loads(out.read_text())["results"]}
    assert set(rows) == {"onnxruntime-cpu", "onnxruntime-cuda", "torch-cuda-fp32", "tensorrt-fp32", "tensorrt-fp16"}
    assert rows["onnxruntime-cpu"]["status"] == "ok" and rows["onnxruntime-cpu"]["reference"]
    for name, row in rows.items():
        assert row["status"] in ("ok", "failed")
        if row["status"] == "failed":
            assert row["error"], name
        elif name != "onnxruntime-cpu":
            assert row["vs_cpu_after_nms"]["frames"] == 5


def test_tensorrt_backend_needs_tensorrt_and_a_gpu():
    from sceneproof.edge.detector import build_detector

    if importlib.util.find_spec("tensorrt") is not None:
        pytest.skip("tensorrt installed: covered by scripts/gpu_bench.py on a GPU")
    with pytest.raises(ModuleNotFoundError):
        build_detector({"variant": "yolox_nano", "backend": "tensorrt", "model": str(MODEL)})
