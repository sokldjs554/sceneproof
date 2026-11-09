# 참고한 논문·공식 문서와 적용 범위

> 정직성 표기: 처음에는 네트워크 정책 때문에 arxiv.org·huggingface.co에 접근하지 못했고, 세션 후반에 허용된 뒤 **세 논문의 arXiv 초록 페이지(제목·저자·초록)를 직접 받아 아래 요약과 대조**했습니다(본문 PDF 전체를 다시 읽은 것은 아닙니다). "읽은 범위"는 (1) 이 세션에서 실제로 읽은 것(공식 저장소 소스 코드, 설치 패키지, arXiv 초록)과 (2) 사전 지식으로 작성한 부분을 구분합니다. 사용자가 최종 검토 시 링크를 열어 본문과 대조해 주세요.

## 1. ByteTrack — Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022
- 링크: https://arxiv.org/abs/2110.06864 · 코드: https://github.com/ifzhang/ByteTrack (MIT)
- 핵심: 탐지 점수가 낮은 상자를 버리지 않고, 1차(고점수 상자 ↔ 모든 트랙) 연계 후 남은 **추적 중** 트랙을 2차(저점수 상자)로 연계한다. 가림·블러로 점수가 떨어진 순간에도 ID가 유지된다. 칼만 필터는 (cx, cy, a, h) 등속 모델, 매칭은 IoU 비용의 헝가리안.
- 적용: `src/sceneproof/edge/tracker.py` 전체. 2단계 연계, 잠정 트랙(두 번째 프레임에서 확정), `track_buffer`(1.5초×fps) 동안 lost 유지, `fuse_score`(IoU×점수) 옵션.
- 재현한 범위: 알고리즘 구조와 기본 임계값(0.5/0.1/0.8, 2차 0.5). **MOT17 수치 재현은 하지 않았습니다**(데이터 없음). 동작은 합성 궤적 테스트(가림 중 저신뢰 상자로 ID 유지 등)와 캠퍼스 영상에서 ID 분리 확인으로 검증.
- 차이: lapjv 대신 scipy 헝가리안+비용 게이트. 카메라 모션 보정(BoT-SORT 계열) 없음.
- 읽은 범위: (1) arXiv 초록(2110.06864, ECCV 2022 판과 동일 내용: 저점수 상자를 버리지 않고 트랙릿과의 유사도로 복원, MOT17 80.3 MOTA·77.3 IDF1·63.1 HOTA @30 FPS V100). 원 코드의 `byte_tracker.py`/`kalman_filter.py` 구조는 (2) 기억에 의존해 재구성.

## 2. YOLOX — Ge et al., "YOLOX: Exceeding YOLO Series in 2021", arXiv 2021
- 링크: https://arxiv.org/abs/2107.08430 · 코드: https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0)
- 핵심: anchor-free 헤드(그리드 셀마다 1개 예측, 중심 오프셋+log 크기), decoupled head(분류/회귀 분리), SimOTA 라벨 할당, 강한 증강(Mosaic/MixUp).
- 적용: 모델 그래프 5개 파일을 벤더링(`src/sceneproof/edge/yolox_model/`, 추론 전용으로 손질)하고 공식 COCO 체크포인트(nano/tiny/s)를 사용. 디코딩(`(xy+grid)*stride`, `exp(wh)*stride`)과 NMS는 `detector.py`에 numpy로 재구현해 PyTorch/ONNX 양쪽이 같은 후처리를 쓴다. 사람 클래스(0)만 사용.
- 재현한 범위: 추론·내보내기·수치 일치, 그리고 추가 학습을 위해 원본 학습 손실(SimOTA 동적 k 할당, IoU 손실 1-IoU², 객체·분류 BCE, `reg_weight=5`)을 `yolox_model/losses.py`로 복원. 공식 학습기·Mosaic/MixUp은 쓰지 않고 크기·위치·좌우·색상 증강만 직접 구현(`sceneproof.train.finetune`).
- 읽은 범위: (1) 공식 저장소의 `yolox/models/*.py`, `exps/default/yolox_{nano,tiny,s}.py`, `yolox/exp/yolox_base.py`를 이 세션에서 직접 내려받아 읽음(벤더링 과정); arXiv 초록(2107.08430: anchor-free, decoupled head, SimOTA; YOLOX-Nano 0.91M 파라미터 25.3% AP). 본문 세부는 (2).

## 2-1. 평가 데이터 — CAVIAR (EC Funded CAVIAR project/IST 2001 37540) · MOT16 (Milan et al., arXiv:1603.00831) · MOT20 (Dendorfer et al., arXiv:2003.09003)
- 적용: 연구용 벤치마크(`docs/eval/bench/README.md`). CAVIAR 데이터 페이지에서 정답이 "이미지에 손으로 표시한 것"이고 움직이지 않는 사람은 상자가 없다는 점, 쇼핑몰 두 시점이 같은 촬영이라는 점을 확인하고 분할·구역 설계에 반영. MOT 데이터 페이지에서 CC BY-NC-SA 3.0, 시퀀스별 정지/이동 카메라와 MOT 클래스 체계(보행자·정지한 사람·방해물·반사)를 확인.
- 확인한 것과 다른 점: CHAD(주차장, Apache-2.0)는 논문 §4에서 상자를 YOLOv4+HRNet이 만들었다고 밝혀 사람 정답이 아니므로 제외.

## 3. ONNX Runtime — 공식 문서(Python API, Graph Optimizations, Execution Providers)
- 링크: https://onnxruntime.ai/docs/ · https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html · https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- 적용: `SessionOptions.intra_op_num_threads`, `graph_optimization_level`(none/basic/extended/all 비교를 벤치마크 항목으로), `providers` 목록(CPU 기본, TensorRT/CUDA EP는 설정으로 지정 가능하되 이 세션에서 실행 안 함).
- 읽은 범위: (1) 설치된 onnxruntime 1.31.1 패키지의 API를 코드로 확인. 문서 본문은 (2).

## 4. PyTorch ONNX export — `torch.onnx.export` (dynamo=False 경로)
- 링크: https://pytorch.org/docs/stable/onnx.html
- 적용: `scripts/export_onnx.py` (opset 17, 상수 접기, 내보내기 직후 무작위 입력으로 PyTorch↔ORT 원시 출력 최대 오차 기록). torch 2.15에서 `dynamo=False`를 명시해 TorchScript 기반 내보내기를 사용.
- 읽은 범위: (1) 설치 패키지 시그니처 확인.

## 5. GStreamer — `rtspsrc`, `appsink`, `decodebin` 공식 문서
- 링크: https://gstreamer.freedesktop.org/documentation/rtsp/rtspsrc.html · https://gstreamer.freedesktop.org/documentation/app/appsink.html · https://gstreamer.freedesktop.org/documentation/rtsp-server/
- 적용: `sources.GStreamerSource`(`rtspsrc latency/protocols/tcp-timeout` → `decodebin` → `videoconvert` → `appsink emit-signals max-buffers drop`), 버스 메시지(EOS/ERROR)로 재연결 상태머신, `src/sceneproof/tools/rtsp_sim.py`(gst-rtsp-server, H.264 패스스루).
- 배운 점(이 세션 실측): 파일 소스에 `sync=true`와 `drop=true`를 같이 쓰면 디코더가 앞서 가며 렌더 직전 버퍼가 계속 교체돼 4초에 2프레임만 전달됨(`sources.py` 주석). 라이브는 `sync=false drop=true`, 파일은 `sync=true drop=false`.
- 읽은 범위: (1) `gst-inspect-1.0`로 요소 속성 확인, (2) 문서 본문.

## 6. gRPC / Protocol Buffers — 공식 가이드(proto3 언어 가이드, Python asyncio API, Go 생성 코드)
- 링크: https://protobuf.dev/programming-guides/proto3/ · https://grpc.io/docs/languages/python/ · https://grpc.io/docs/languages/go/
- 적용: `proto/sceneproof/v1/sceneproof.proto`(단방향 unary + 서버 스트리밍 + 클라이언트 스트리밍 업로드), `grpc.aio` 서버, Go 클라이언트(연결 상태 감시·백오프는 애플리케이션 레벨).
- 읽은 범위: (1) 생성 코드와 설치 패키지, (2) 가이드 본문.

## 7. Anthropic Claude API — Vision, Structured Outputs, Pricing
- 링크: https://platform.claude.com/docs/en/build-with-claude/vision · https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- 적용: `server/vlm/anthropic_provider.py` — 이미지 base64 블록 + `output_config.format=json_schema`(모든 속성 required, additionalProperties false), 시스템 프롬프트 캐시 블록, `stop_reason=="refusal"` 처리, 토큰 사용량→추정 비용→예산 상한.
- 읽은 범위: (1) 이 세션에 제공된 Claude API 스킬 문서(Python README, structured outputs 예시, 가격표 2026-10 기준)를 읽고 작성. (2) 실제 호출: 사용자 승인(상한 $5) 후 GitHub Actions `vlm-eval`에서 Claude Opus 5.5로 평가 19건·데모 15건 해석(https://github.com/sokldjs554/sceneproof/actions/runs/42460838313).
- 적용하며 확인한 사실: Opus 5.5는 thinking을 끌 수 없고 thinking 토큰이 `max_tokens`에 포함된다(1024에서 JSON이 잘림 → 하한 4096, dev-log L12); `output_config.format=json_schema`(strict)로 34건 모두 스키마 위반 0; 시스템 프롬프트 `cache_control`은 두 번째 호출부터 `cache_read_input_tokens`(≈1.3k/건)로 집계됨; 해석 1건 지연 p50 ≈ 20초(동시성 2).

## 8. SmolVLM — Marafioti et al., "SmolVLM: Redefining small and efficient multimodal models", 2025 · vLLM 멀티모달 지원
- 링크: https://arxiv.org/abs/2504.05299 · https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct · https://docs.vllm.ai/en/latest/models/supported_models.html (Multimodal)
- 적용: `server/vlm/transformers_local.py`(CPU 로컬), `server/vlm/openai_compat.py`(vLLM `vllm serve` OpenAI 호환 엔드포인트). vLLM 지원 모델 표의 Qwen2-VL·SmolVLM(Idefics3 계열) 항목을 근거로 OpenAI 호환 경로를 설계.
- 읽은 범위: (1) arXiv 초록(2504.05299: 256M 모델이 1GB 미만 GPU 메모리, 토큰화·데이터 큐레이션 최적화). 네트워크 허용 후 SmolVLM-500M-Instruct 가중치를 받아 **CPU에서 실제 실행**(결과는 `docs/eval/README.md` §5).

## 9. Intel OpenVINO sample videos / OpenCV samples — 데이터 출처
- `data/README.md` 참조(CC BY 4.0, Apache-2.0).
