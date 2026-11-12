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

### 1차 실행 (2026-11-12, Colab NVIDIA T4 · CPU 2코어 Xeon 2.0GHz, 코드 `8ea427f`, 원시 결과 `raw/gpu_bench_t4_run1.json`)

campus-walkway.mp4 300프레임(768×576), 3회 반복의 중앙값. 추론 시간에는 GPU로 보내고 받는 복사가 포함됩니다. 프레임당 전체 시간에는 CPU에서 하는 전처리(크기 맞추기)와 후처리(겹친 상자 정리)가 더해집니다.

| 모델 | 방식 | 추론 p50 (ms) | 추론 p95 (ms) | 프레임당 전체 p50 (ms) | 초당 프레임 | CPU 기준과 결과 차이 |
|---|---|---|---|---|---|---|
| nano | CPU (ONNX Runtime) | 18.1 | 50.5 | 20.9 | 39 | 기준 |
| nano | ONNX Runtime GPU | 4.4 | 6.1 | 7.2 | 138 | 없음 |
| nano | PyTorch GPU | 13.1 | 17.5 | 16.0 | 61 | 없음 |
| nano | **TensorRT FP32** | **3.2** | 4.0 | 6.2 | **159** | 없음 |
| tiny | CPU (ONNX Runtime) | 68.8 | 163.3 | 72.1 | 11.5 | 기준 |
| tiny | ONNX Runtime GPU | 4.8 | 4.9 | 7.0 | 140 | 없음 |
| tiny | PyTorch GPU | 11.7 | 19.2 | 14.6 | 60 | 없음 |
| tiny | **TensorRT FP32** | **3.6** | 3.7 | 5.8 | **169** | 상자 수·짝 모두 같음, 한 상자 최대 5.5px |
| s | CPU (ONNX Runtime) | 254.9 | 501.4 | 261.4 | 3.3 | 기준 |
| s | ONNX Runtime GPU | 9.8 | 10.1 | 14.1 | 70 | 없음 |
| s | PyTorch GPU | 12.7 | 19.3 | 17.9 | 53 | 없음 |
| s | **TensorRT FP32** | **8.4** | 8.8 | 12.9 | **76** | 없음 |

- **결과는 바뀌지 않았습니다.** 모든 GPU 방식에서 300프레임 모두 상자 수가 CPU와 같았고 모든 상자가 짝지어졌습니다(tiny 1,893개 등). 신경망 원출력의 차이는 최대 0.0002 수준입니다. tiny TensorRT FP32에서 한 상자가 5.5px 달랐는데, 같은 사람으로 짝지어진 상자이고 원출력 차이는 아주 작으므로, 점수가 거의 같은 겹친 후보 중 다른 상자가 남은 경우로 보입니다.
- **속도**: 기본 모델 tiny의 추론이 CPU(Colab 2코어) 68.8ms → TensorRT FP32 3.6ms, 프레임당 전체로는 11.5 → 169 fps입니다. 같은 GPU 안에서 TensorRT는 ONNX Runtime GPU보다 1.2~1.4배, PyTorch GPU보다 1.5~4배 빨랐습니다. 작은 모델일수록 PyTorch의 파이썬 쪽 부담이 커서 차이가 큽니다.
- **다음 병목은 CPU 쪽입니다.** tiny TensorRT에서 프레임당 5.8ms 중 추론은 3.6ms이고, 나머지 약 2.2ms(약 40%)가 CPU의 전처리·후처리입니다. 더 빠르게 하려면 크기 맞추기와 겹친 상자 정리도 GPU로 옮기거나 다음 프레임과 겹쳐 돌려야 합니다.
- TensorRT 엔진을 처음 만드는 데 모델마다 약 50초가 걸렸습니다. 한 번 만들면 파일로 저장해 다시 씁니다.
- CPU 숫자는 Colab의 2코어 기준이라, 저장소의 4코어 측정(`docs/eval/README.md` §2, tiny 21.6ms)보다 느립니다.

**1차에서 실패한 것과 고친 것**
- **TensorRT FP16**: Colab에 설치된 TensorRT 11에서는 예전의 'FP16으로 만들기' 설정(`BuilderFlag.FP16`)이 없어졌습니다. 모델 파일 자체를 FP16으로 바꿔서 넣도록 고쳤습니다(`detector.onnx_to_fp16`, 입력·출력은 FP32로 유지). CPU에서 FP16 모델을 돌려 보니 100프레임에서 찾은 사람이 FP32와 같았고, 상자 차이는 최대 2.9px(nano)·1.2px(tiny)·1.1px(s)였습니다(`tests/test_trt_fp16.py`). TensorRT 10 이하에서는 예전 방식을 그대로 씁니다.
- **사건 평가**: CPU·GPU 모두 시작하자마자 멈췄습니다. 저장소의 의존성이 `protobuf>=5.28`이었는데 생성된 코드는 7.35.1 이상을 요구해, protobuf 6.33이 깔린 Colab에서 실패했습니다. 다른 환경에서도 날 수 있는 실제 결함이라 의존성을 `protobuf>=7.35.1`로 고치고, 둘이 다시 어긋나면 실패하는 테스트를 넣었습니다.

### 2차 실행

FP16과 사건 평가를 위해 고친 노트북으로 다시 실행할 예정입니다.
