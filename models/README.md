# 모델

`scripts/fetch_models.sh`가 Megvii YOLOX 공식 릴리스(COCO 학습, Apache-2.0)에서 체크포인트를 받고,
`scripts/export_onnx.py`가 ONNX로 내보냅니다. 가중치·ONNX 파일은 커밋하지 않습니다.

| 이름 | 입력 | 파라미터 | 출처 |
|---|---|---|---|
| yolox_nano | 416×416 | 0.91M | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth |
| yolox_tiny | 416×416 | 5.06M | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth |
| yolox_s | 640×640 | 9.0M | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth |

`*.onnx.json`에는 가중치 SHA-256, opset, torch/onnx/onnxruntime 버전, 내보내기 직후 PyTorch↔ONNX Runtime
원시 출력 최대 절대 오차가 기록됩니다. 사람(class 0)만 사용합니다.
