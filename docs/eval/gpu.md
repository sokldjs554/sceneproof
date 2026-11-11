# GPU(TensorRT) 측정 — Colab T4

[![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sokldjs554/sceneproof/blob/main/notebooks/colab_gpu_bench.ipynb)

개발 환경에 NVIDIA GPU가 없어, GPU 측정은 Google Colab의 NVIDIA T4에서 노트북([`notebooks/colab_gpu_bench.ipynb`](../../notebooks/colab_gpu_bench.ipynb))으로 합니다. 노트북은 저장소를 받아 설치하고, 아래 두 가지를 재서 결과 JSON을 묶어 내려받게 합니다.

## 무엇을 재나

1. **추론 속도와 결과 차이** (`scripts/gpu_bench.py`): YOLOX nano·tiny·s를 같은 영상 프레임(기본 300장), 같은 전처리·후처리(`edge/detector.py`)로 다섯 가지 방식으로 돌립니다.

   | 방식 | 내용 |
   |---|---|
   | `onnxruntime-cpu` | 기준. 지금 제품 기본값 |
   | `onnxruntime-cuda` | 같은 ONNX를 ONNX Runtime GPU로 |
   | `torch-cuda-fp32` | 원래 PyTorch 가중치를 GPU로 (최적화 전 출발점) |
   | `tensorrt-fp32` / `tensorrt-fp16` | 같은 ONNX로 TensorRT 엔진을 만들어 실행(`TensorRTBackend`) |

   방식마다 추론 시간(p50·p95, 3회 반복), 프레임당 전체 시간(전처리·후처리 포함), 엔진 만드는 시간, 그리고 **CPU 기준과의 차이**를 기록합니다. 차이는 두 단계로 봅니다: NMS 뒤 상자 수가 다른 프레임 수·짝이 안 맞는 상자·상자 좌표 최대 차이(px)·점수 최대 차이, 그리고 NMS 전 신경망 원출력의 최대 차이. GPU가 없거나 설치가 안 된 방식은 멈추지 않고 이유를 남깁니다.

2. **사건 결과가 그대로인가** (`python -m sceneproof.eval.run_eval --backend tensorrt --precision fp16`): 평가 영상 4편을 같은 Colab에서 CPU와 TensorRT FP16으로 각각 분석해 TP·잘못 울림·놓침·중복·시작/종료 오차를 비교합니다. FP16은 숫자를 덜 정밀하게 계산하므로 경계에 걸친 상자 몇 개가 달라질 수 있습니다. 그것이 사건 판정까지 바꾸는지가 실제로 궁금한 점입니다.

## 하지 않는 것

- **Jetson 같은 엣지 장치**: T4는 서버용 GPU라 엣지 장치의 숫자를 대신하지 않습니다. 같은 코드(`TensorRTBackend`)를 Jetson에서 돌리면 엔진만 그 장치에서 새로 만들어집니다(엔진 파일 이름에 GPU 이름과 TensorRT 버전이 들어감).
- **INT8 양자화**: 보정용 영상 세트와 정확도 재측정이 따로 필요해 이번 범위에서 뺐습니다.
- **DeepStream**: Colab에서는 설치가 어렵습니다.
- 여러 영상을 묶어 한 번에 넣는 배치 처리: 제품은 카메라마다 프레임을 하나씩 처리하므로 배치 1로 잽니다.

## 결과

아직 실행하지 않았습니다. 노트북을 실행해 `gpu_results.zip`을 받으면 이 절과 `docs/eval/raw/`에 반영합니다.
