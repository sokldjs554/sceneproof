import cv2
import numpy as np

from sceneproof.edge.detector import build_detector

model = build_detector(
    {
        "variant": "yolox_tiny",
        "backend": "onnxruntime",
        "model": "models/yolox_tiny.onnx",
        "conf_threshold": 0.4,
        "nms_threshold": 0.45,
        "intra_op_threads": 4,
    }
)
for path in (
    "data/videos/campus-walkway.mp4",
    "data/videos/people-indoor.mp4",
    "data/videos/store-aisle.mp4",
    "data/videos/parking-lot.mp4",
):
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = boxes = removed = 0
    for idx in np.linspace(0, max(0, count - 1), 30, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        result = model.detect(frame)
        heights = result.boxes[:, 3] - result.boxes[:, 1]
        frames += 1
        boxes += len(heights)
        removed += int((heights < 40).sum())
    cap.release()
    print(path, "sampled_frames=", frames, "detected_boxes=", boxes, "under_40px=", removed, flush=True)
