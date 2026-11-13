import cv2

from sceneproof.edge.detector import build_detector

cap = cv2.VideoCapture("data/videos/parking-lot.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
det = build_detector(
    {
        "variant": "yolox_tiny",
        "backend": "onnxruntime",
        "model": "models/yolox_tiny.onnx",
        "conf_threshold": 0.4,
        "nms_threshold": 0.45,
        "intra_op_threads": 4,
    }
)
for start, end in ((25, 29), (43, 47)):
    rows = []
    for frame_index in range(round(start * fps), round(end * fps), 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        d = det.detect(frame)
        for box, score in zip(d.boxes, d.scores, strict=True):
            h = float(box[3] - box[1])
            if h < 40:
                rows.append(
                    [
                        round(frame_index / fps, 3),
                        *[round(float(v), 2) for v in box],
                        round(h, 2),
                        round(float(score), 3),
                    ]
                )
    print("window", start, end, "fps", fps, "under40", rows, flush=True)
cap.release()
