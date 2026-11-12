"""TensorRT >= 11 builds FP16 from an FP16 ONNX (detector.onnx_to_fp16); checked here on the CPU."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "yolox_nano.onnx"
VIDEO = ROOT / "data" / "videos" / "campus-walkway.mp4"


@pytest.mark.skipif(importlib.util.find_spec("onnxconverter_common") is None or not MODEL.exists(),
                    reason="needs onnxconverter-common (dev extra) and scripts/fetch_models.sh")
def test_fp16_onnx_keeps_float32_io_and_the_same_detections(tmp_path):
    import onnx

    from sceneproof.edge.detector import build_detector, onnx_to_fp16
    from sceneproof.eval.bench_detector import load_frames, parity

    m16 = onnx.load_from_string(onnx_to_fp16(MODEL))
    assert [i.type.tensor_type.elem_type for i in m16.graph.input] == [onnx.TensorProto.FLOAT]
    assert [o.type.tensor_type.elem_type for o in m16.graph.output] == [onnx.TensorProto.FLOAT]
    fp16_path = tmp_path / "yolox_nano.fp16.onnx"
    fp16_path.write_bytes(m16.SerializeToString())
    frames = load_frames(str(VIDEO), 10, step=15) if VIDEO.exists() else \
        [(np.random.default_rng(i).random((480, 640, 3)) * 255).astype(np.uint8) for i in range(3)]
    cfg = {"variant": "yolox_nano", "backend": "onnxruntime", "conf_threshold": 0.4}
    d32, d16 = build_detector({**cfg, "model": str(MODEL)}), build_detector({**cfg, "model": str(fp16_path)})
    outs = [[(d.boxes, d.scores) for d in map(det.detect, frames)] for det in (d32, d16)]
    p = parity(outs[0], outs[1])
    # FP16 moves raw scores by up to a few hundredths; after the 0.4 threshold and NMS the people found are the same
    assert p["frames_with_count_diff"] == 0 and p["unmatched_boxes"] == 0, p
    assert p["max_box_delta_px"] < 3.0 and p["max_score_delta"] < 0.05, p


def test_protobuf_requirement_is_not_older_than_the_generated_code():
    import re

    pb2 = (ROOT / "src" / "sceneproof" / "proto" / "sceneproof" / "v1" / "sceneproof_pb2.py").read_text()
    gencode = tuple(int(x) for x in re.search(r"Protobuf Python Version: ([\d.]+)", pb2).group(1).split("."))
    req = re.search(r'"protobuf>=([\d.]+)"', (ROOT / "pyproject.toml").read_text()).group(1)
    assert tuple(int(x) for x in req.split(".")) >= gencode, f"pyproject protobuf>={req} but the code was generated for {gencode}"
