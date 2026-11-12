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

## 결과 (2차 실행, 2026-11-12)

Colab NVIDIA T4 · CPU 2코어(Xeon 2.0GHz) · TensorRT 11.4 · ONNX Runtime 1.31 · PyTorch 2.14, 코드 `b8b4426`. 원시 결과: `raw/gpu_bench_t4.json`, 사건 평가 `raw/event_eval_yolox_tiny_colab_cpu.json`·`raw/event_eval_yolox_tiny_colab_trt_fp16.json`.

### 1) 추론 속도와 결과 차이

campus-walkway.mp4 300프레임(768×576), 3회 반복의 중앙값. 추론 시간에는 GPU로 보내고 받는 복사가 포함되고, 프레임당 전체 시간에는 CPU에서 하는 전처리(크기 맞추기)와 후처리(겹친 상자 정리)가 더해집니다. '결과 차이'는 CPU 기준과 비교해, 찾은 상자 수가 다른 프레임 수 / 짝이 안 맞는 상자 수(전체 상자 수)입니다.

| 모델 | 방식 | 추론 p50 (ms) | 프레임당 전체 p50 (ms) | 초당 프레임 | 결과 차이 |
|---|---|---|---|---|---|
| nano | CPU (ONNX Runtime) | 17.7 | 20.4 | 47 | 기준 |
| nano | ONNX Runtime GPU | 4.2 | 7.1 | 145 | 0 / 0 |
| nano | PyTorch GPU | 12.3 | 15.1 | 65 | 0 / 0 |
| nano | **TensorRT FP32** | **2.7** | **4.9** | **200** | 0 / 0 |
| nano | TensorRT FP16 | 3.0 | 5.4 | 182 | 2프레임 / 1개 (2,134개 중) |
| tiny | CPU (ONNX Runtime) | 69.0 | 72.4 | 11.5 | 기준 |
| tiny | ONNX Runtime GPU | 4.8 | 7.0 | 141 | 0 / 0 |
| tiny | PyTorch GPU | 11.2 | 14.1 | 66 | 0 / 0 |
| tiny | TensorRT FP32 | 3.5 | 5.7 | 172 | 0 / 0 (한 상자 5.5px) |
| tiny | **TensorRT FP16** | **2.7** | **5.1** | **185** | 3프레임 / 2개 (1,893개 중) |
| s | CPU (ONNX Runtime) | 253.2 | 259.8 | 3.3 | 기준 |
| s | ONNX Runtime GPU | 9.8 | 14.2 | 69 | 0 / 0 |
| s | PyTorch GPU | 12.4 | 17.7 | 51 | 0 / 0 |
| s | TensorRT FP32 | 8.3 | 13.0 | 74 | 0 / 0 |
| s | **TensorRT FP16** | **4.9** | **9.3** | **106** | 1프레임 / 0개 (1,888개 중) |

- **FP32는 결과가 바뀌지 않습니다.** ONNX Runtime GPU, PyTorch GPU, TensorRT FP32 모두 300프레임에서 찾은 상자가 CPU와 같았습니다(원출력 차이 최대 0.0002). tiny TensorRT FP32의 한 상자 5.5px 차이는 같은 사람으로 짝지어진 상자입니다.
- **FP16은 큰 모델일수록 이득입니다.** s는 FP32 대비 1.7배(8.3 → 4.9ms), tiny는 1.3배(3.5 → 2.7ms) 빨라졌습니다. nano는 오히려 조금 느려졌습니다(2.7 → 3.0ms). 계산량이 작아, FP16 모델 앞뒤에 들어가는 형 변환과 FP32로 남긴 크기 조정 연산의 비용이 더 크게 보이는 것으로 짐작합니다.
- **FP16의 대가는 작습니다.** 수천 개 상자 중 0~2개가 달라졌습니다. 신경망 원출력은 상자 좌표 최대 1px, 점수 최대 0.09까지 달라지는데, 0.4 임계값과 겹친 상자 정리를 거친 뒤에는 대부분 사라집니다. 이것이 사건 판정까지 바꾸는지는 아래 2)에서 봅니다.
- **다음 병목은 CPU 쪽입니다.** tiny TensorRT FP16에서 프레임당 5.1ms 중 추론은 2.7ms이고, 나머지 약 2.3ms(약 45%)가 CPU의 전처리·후처리입니다. 더 빠르게 하려면 크기 맞추기와 겹친 상자 정리도 GPU로 옮기거나 다음 프레임과 겹쳐 돌려야 합니다.
- TensorRT 엔진을 처음 만드는 시간은 FP32 약 50초, FP16 약 2~2.5분입니다(모델마다). 한 번 만들면 파일로 저장해 다시 씁니다.
- CPU 숫자는 Colab의 2코어 기준이라 저장소의 4코어 측정(`docs/eval/README.md` §2, tiny 21.6ms)보다 느립니다.

### 2) 사건 결과: CPU vs TensorRT FP16 (평가 영상 4편, 같은 Colab)

| 영상 | 정답 침입 | CPU: 찾음 / 잘못 울림 / 놓침 / 중복 | TensorRT FP16: 찾음 / 잘못 울림 / 놓침 / 중복 | 시작 시각 오차(초) CPU → FP16 | 분석 속도(fps) CPU → FP16 |
|---|---|---|---|---|---|
| campus | 4 | 4 / 0 / 0 / 2 | 4 / 0 / 0 / 2 | 0.23 → 0.23 | 5.0 → 68.9 |
| indoor | 4 | 4 / 0 / 0 / 0 | 4 / 0 / 0 / 0 | 0.20 → 0.20 | 5.4 → 77.2 |
| store | 1 | 1 / 0 / 0 / 3 | 1 / 0 / 0 / 3 | 0.10 → 0.13 | 5.7 → 107.1 |
| parking | 3 | 3 / 0 / 0 / 1 | 3 / 0 / 0 / 1 | 0.45 → 0.51 | 5.7 → 96.2 |

- **사건 판정은 4편 모두 같았습니다.** 찾음·잘못 울림·놓침·중복이 하나도 바뀌지 않았고, 시작 시각 오차만 매장·주차장에서 0.03~0.06초 움직였습니다(종료 시각 오차는 같음). 상자 몇 개의 차이는 추적과 구역 판정(0.3초 확인)을 거치며 사건에는 드러나지 않았습니다.
- **전체 분석 속도**(영상 읽기·추적·구역 판정·근거 저장 포함)가 CPU 2코어 5~6 fps에서 TensorRT FP16 69~107 fps로 13~19배가 됐습니다. 실시간 30 fps 영상이라면 GPU 하나로 두세 대 카메라를 처리할 여유입니다(디코딩·근거 저장 부담은 카메라 수만큼 늘어나므로 실제 대수는 따로 재야 함).
- Colab CPU 결과(찾음 12/12, 중복 6)는 저장소의 검수 정답 기준 결과와 같습니다.

### 실행하며 고친 것 (1차 실행, `raw/gpu_bench_t4_run1.json`)

- **TensorRT FP16**: Colab의 TensorRT 11에서는 예전 'FP16으로 만들기' 설정(`BuilderFlag.FP16`)이 없어져 1차에서 실패했습니다. 모델 파일 자체를 FP16으로 바꿔 넣도록 고쳤습니다(`detector.onnx_to_fp16`, 입력·출력은 FP32, 크기 조정 연산은 FP32 유지). TensorRT 10 이하에서는 예전 방식을 씁니다. CPU에서 이 FP16 모델로 100프레임을 돌리면 찾은 사람이 FP32와 같습니다(`tests/test_trt_fp16.py`).
- **사건 평가**: 1차에서는 CPU·GPU 모두 시작하자마자 멈췄습니다. 저장소 의존성이 `protobuf>=5.28`이었는데 생성된 코드는 7.35.1 이상을 요구해, protobuf 6.33이 깔린 Colab에서 실패했습니다. 다른 환경에서도 날 수 있는 실제 결함이라 의존성을 고치고, 다시 어긋나면 실패하는 테스트를 넣었습니다.
- 1차의 FP32 숫자는 2차와 거의 같았습니다(예: tiny TensorRT FP32 3.6 → 3.5ms).
