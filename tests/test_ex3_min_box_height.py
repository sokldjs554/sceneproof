"""과제 3 — 너무 작은 사람 상자 무시하기 (src/sceneproof/edge/detector.py, src/sceneproof/edge/config.py)

PersonDetector에 min_box_height_px(기본 0 = 끔)를 추가합니다. 0보다 크면, 찾은 상자 중 높이(y2 - y1)가
이 값보다 작은 것을 결과에서 뺍니다(겹친 상자 정리가 끝난 뒤에, 순서는 그대로).
설정(DetectorConfig)과 build_detector()에서도 이 값을 받을 수 있게 합니다.
필요: scripts/fetch_models.sh(yolox_nano.onnx)와 scripts/fetch_data.sh(campus-walkway.mp4).
"""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "yolox_nano.onnx"
VIDEO = ROOT / "data" / "videos" / "campus-walkway.mp4"
needs_data = pytest.mark.skipif(not (MODEL.exists() and VIDEO.exists()), reason="모델·영상을 먼저 받으세요")


def test_config_accepts_the_new_field():
    from sceneproof.edge.config import DetectorConfig

    assert DetectorConfig().min_box_height_px == 0
    assert DetectorConfig(min_box_height_px=40).min_box_height_px == 40


@needs_data
def test_small_boxes_are_dropped_and_the_rest_is_unchanged():
    from sceneproof.edge.detector import build_detector
    from sceneproof.eval.bench_detector import load_frames

    frames = load_frames(str(VIDEO), 30, step=10)
    cfg = {"variant": "yolox_nano", "backend": "onnxruntime", "model": str(MODEL), "conf_threshold": 0.4}
    base, filt = build_detector(cfg), build_detector({**cfg, "min_box_height_px": 60})
    dropped = 0
    for f in frames:
        a, b = base.detect(f), filt.detect(f)
        keep = (a.boxes[:, 3] - a.boxes[:, 1]) >= 60
        dropped += int((~keep).sum())
        assert np.allclose(b.boxes, a.boxes[keep]) and np.allclose(b.scores, a.scores[keep])
    assert dropped > 0, "이 영상에는 60px보다 작은 사람이 있어야 합니다 — 하나도 안 빠졌다면 필터가 안 걸린 것"


@needs_data
def test_zero_means_off():
    from sceneproof.edge.detector import build_detector
    from sceneproof.eval.bench_detector import load_frames

    f = load_frames(str(VIDEO), 1)[0]
    cfg = {"variant": "yolox_nano", "backend": "onnxruntime", "model": str(MODEL), "conf_threshold": 0.4}
    a, b = build_detector(cfg).detect(f), build_detector({**cfg, "min_box_height_px": 0}).detect(f)
    assert np.array_equal(a.boxes, b.boxes)
