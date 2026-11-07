import os

import numpy as np
import pytest

from zonewatch.edge.detector import decode_raw, grid_tables, letterbox, nms_xyxy, postprocess

MODEL = "models/yolox_tiny.onnx"


def test_letterbox_shape_and_padding():
    img = np.full((300, 600, 3), 200, np.uint8)
    blob, r = letterbox(img, (416, 416))
    assert blob.shape == (3, 416, 416) and blob.dtype == np.float32
    assert abs(r - 416 / 600) < 1e-6
    assert blob[0, 415, 415] == 114  # bottom-right is padding
    assert blob[0, 0, 0] == 200


def test_grid_tables_anchor_count():
    g, s = grid_tables((416, 416))
    assert g.shape == (52 * 52 + 26 * 26 + 13 * 13, 2)
    assert set(np.unique(s)) == {8.0, 16.0, 32.0}


def test_decode_and_postprocess_single_box():
    g, s = grid_tables((416, 416))
    raw = np.zeros((g.shape[0], 85), np.float32)
    i = 100  # some stride-8 anchor
    raw[i, :2] = 0.5
    raw[i, 2:4] = np.log(5.0)  # w,h = 5 * stride
    raw[i, 4] = 0.9
    raw[i, 5] = 0.95  # person
    dec = decode_raw(raw, g, s)
    cx, cy = (g[i] + 0.5) * 8
    assert np.allclose(dec[i, :2], [cx, cy]) and np.allclose(dec[i, 2:4], [40, 40])
    boxes, scores = postprocess(dec, ratio=0.5, conf_thresh=0.5, nms_thresh=0.45)
    assert boxes.shape == (1, 4) and abs(scores[0] - 0.855) < 1e-6
    assert np.allclose(boxes[0], [(cx - 20) / 0.5, (cy - 20) / 0.5, (cx + 20) / 0.5, (cy + 20) / 0.5])


def test_nms_keeps_best_of_overlapping():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.5, 0.9, 0.7], np.float32)
    keep = nms_xyxy(boxes, scores, 0.5)
    assert list(keep) == [1, 2]


@pytest.mark.skipif(not os.path.exists(MODEL), reason="run scripts/fetch_models.sh")
def test_onnx_detector_finds_people_in_sample_frame():
    import cv2

    from zonewatch.edge.detector import build_detector

    video = "data/videos/campus-walkway.mp4"
    if not os.path.exists(video):
        pytest.skip("run scripts/fetch_data.sh")
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ok, frame = cap.read()
    assert ok
    det = build_detector({"variant": "yolox_tiny", "model": MODEL, "conf_threshold": 0.4})
    d = det.detect(frame)
    assert 3 <= len(d.scores) <= 15  # several pedestrians are visible around frame 300
    assert (d.boxes[:, 2] > d.boxes[:, 0]).all() and (d.boxes[:, 3] > d.boxes[:, 1]).all()


@pytest.mark.torch
@pytest.mark.skipif(not os.path.exists("models/yolox_tiny.pth"), reason="weights not fetched")
def test_torch_and_onnx_backends_agree():
    pytest.importorskip("torch")
    import cv2

    from zonewatch.edge.detector import build_detector

    video = "data/videos/campus-walkway.mp4"
    if not os.path.exists(video):
        pytest.skip("run scripts/fetch_data.sh")
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    _, frame = cap.read()
    a = build_detector({"variant": "yolox_tiny", "model": MODEL, "backend": "onnxruntime", "conf_threshold": 0.3})
    b = build_detector({"variant": "yolox_tiny", "model": "models/yolox_tiny.pth", "backend": "torch", "conf_threshold": 0.3})
    da, db = a.detect(frame), b.detect(frame)
    assert len(da.scores) == len(db.scores)
    assert np.abs(da.boxes - db.boxes).max() < 1.0  # sub-pixel agreement after NMS
    assert np.abs(da.scores - db.scores).max() < 1e-3
