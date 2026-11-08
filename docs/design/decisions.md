# 설계 결정과 대안

각 항목은 "결정 — 대안 — 이유 — 확인 방법" 순서입니다. 코드 위치를 같이 적었습니다.

## D1. 탐지기: YOLOX(Apache-2.0) + 자체 ONNX 내보내기
- 대안: Ultralytics YOLOv8/11(AGPL-3.0, 내장 추적기·export), torchvision SSDLite(BSD, 정확도 낮음), RT-DETR(무거움).
- 이유: 상용 회사 지원용 포트폴리오에서 라이선스는 설계 판단의 일부입니다. YOLOX는 Apache-2.0이고, 모델 코드 5개 파일만 벤더링(`src/sceneproof/edge/yolox_model/`)하면 PyTorch 경로와 ONNX 경로를 **같은 전·후처리**로 비교할 수 있습니다. CPU 4코어에서 tiny가 416 입력 기준 p50 ≈ 21 ms로 30 fps 입력을 처리합니다(`docs/eval/README.md`).
- 확인: `scripts/export_onnx.py`가 내보내기 직후 무작위 입력으로 원시 출력 최대 오차(1e-5 수준)를 기록, `tests/test_detector.py::test_torch_and_onnx_backends_agree`가 실제 프레임에서 NMS 후 상자 일치를 검사.

## D2. 추적기: ByteTrack 자체 구현 (`src/sceneproof/edge/tracker.py`)
- 대안: `supervision`/`boxmot` 라이브러리, 간단한 IoU 추적.
- 이유: 구역 체류 시간은 ID가 안정적일 때만 의미가 있습니다. ByteTrack의 핵심(저신뢰 상자를 2차 연계에 사용)은 ~250줄로 구현 가능하고, 직접 구현해야 `track_buffer`·2단계 임계값·가림 시 ID 유지 같은 동작을 테스트로 고정할 수 있습니다(`tests/test_tracker.py`: 가림 중 저신뢰 상자로 ID 유지, 버퍼 내 재출현 시 ID 복원, 교차 시 ID 구분).
- 차이: 헝가리안(scipy)에 비용 게이트를 둔 방식이라 원 구현의 lapjv와 동률 처리에서 드물게 다를 수 있습니다.

## D3. 구역 판정 앵커 = 상자 하단 중앙(발) (`zones.anchor_point`)
- 대안: 상자 중심, 상자-다각형 겹침 비율.
- 이유: 구역은 바닥에 그린 다각형입니다. 상체가 구역 위에 겹쳐도 발이 밖이면 침입이 아닙니다.
- 한계(평가에서 확인): 선반·물건에 발이 가리면 상자 하단이 발이 아닙니다(`eval-store`, `occluded` 라벨). 상자가 프레임 하단에 걸리면 앵커가 프레임 가장자리가 됩니다 → **상태 유지**로 처리(`box_clipped_at_bottom`, `tests/test_zones.py::test_box_clipped_at_frame_bottom_holds_state`). 이 버그는 캠퍼스 영상에서 한 사람의 체류가 4개 사건으로 쪼개지는 것을 보고 찾았습니다(`docs/dev-log.md`).

## D4. 디바운스는 프레임 수가 아니라 초 단위 (`ZoneSpec.entry/exit_confirm_seconds`)
- 처음에는 프레임 수(진입 3, 이탈 5)로 구현했습니다. 30 fps 영상에서 이탈 확정 8프레임은 0.27초라 깜빡임마다 사건이 쪼개졌습니다(`eval-store` 초안 #2/#3, #5/#6). 미디어 시간 기준으로 바꾸고 프레임 수는 하위 호환으로 남겼습니다.

## D5. 사건은 스냅샷 + revision; 큐는 병합 가능 (`proto`, `gateway/internal/queue`)
- 대안: 증분 이벤트(opened/dwell/closed 각각 별도 메시지).
- 이유: 증분 이벤트는 순서·누락에 취약합니다. 스냅샷이면 서버는 `(event_id, revision)` 비교만으로 멱등하고, 게이트웨이는 오프라인 동안 쌓인 같은 사건의 오래된 revision을 버려도 됩니다. 중복 전달·재시작·재전송이 모두 같은 규칙으로 처리됩니다(`scripts/failure_scenarios.py::sc_duplicates`).

## D6. 게이트웨이를 Go로, 분석기와 분리
- 대안: Python 분석기 안에서 직접 서버로 전송.
- 이유: 내구 큐·재연결·업로드는 영상 처리와 수명이 다릅니다(분석기 재시작·모델 교체와 무관하게 전송은 계속돼야 함). Go는 정적 바이너리·goroutine으로 이런 데몬에 맞고, 공고의 "Python 외 C/Go" 요건을 실제 책임이 있는 모듈로 채웁니다. 경계는 gRPC 계약 하나(`EdgeIngress`)입니다. 상세: [`gateway/README.md`](../../gateway/README.md).

## D7. VLM은 서버에서, 비동기로, 범주형 출력만 (`server/vlm/*`)
- 대안: 엣지에서 VLM 실행, 동기 호출, 자유 텍스트 출력.
- 이유: VLM은 느리고(수 초~수십 초) 실패합니다. 사건 기록·CV 근거는 VLM과 무관하게 먼저 저장되고, 해석은 큐에서 처리됩니다(타임아웃·재시도·예산 상한). 출력은 JSON 스키마로 검증하고 확률·신뢰도 필드를 두지 않습니다. 모의(mock) 제공자는 결과에 `[MOCK]`을 박고 품질 수치로 보고하지 않습니다.
- 제공자 추상화: Anthropic(Claude) / OpenAI 호환(vLLM·llama.cpp) / transformers 로컬(SmolVLM) / mock. 어떤 것을 실제로 돌렸는지는 `docs/eval/README.md`에 적습니다.
- 실측으로 확인(2026-11-08, Claude Opus 5.5, 평가 19건): 정답 침입 10건 중 3건을 모델이 `likely_benign`으로 판단했습니다. VLM 판단을 자동 종결 게이트로 썼다면 이 3건은 사람 눈에 띄지 않았을 것입니다. 보조 표시로만 쓰고 기본 상태를 '확인 필요'로 두는 이 결정은 유지합니다(`docs/eval/vlm-findings.md`).

## D8. 링버퍼와 프레임 큐의 정책 분리 (`frames.py`)
- 프레임 큐: 분석이 늦으면 **가장 오래된 프레임 폐기**(실시간성 우선). 오프라인 평가는 무손실 모드(재현성 우선).
- 링버퍼: 근거를 위해 최근 N초를 보관, 열린 사건의 pre-roll은 **고정(pin)**, 다만 바이트 상한이 최종 우선. 두 버퍼의 목적이 달라 정책도 다릅니다.

## D9. 저장소: SQLite(WAL)
- 대안: PostgreSQL.
- 이유: 단일 서버·단일 작성자 데모에 충분하고 재시작 복구·멱등 upsert를 한 파일로 검증할 수 있습니다. k8s에서는 `Recreate` 전략 + RWO PVC로 작성자 하나를 보장합니다. 다중 서버는 범위 밖(한계 문서).

## D10. 콘솔은 빌드 없는 정적 파일
- 대안: React/Vite.
- 이유: 공개 데모·오프라인 실행·Pages 정적 재생을 같은 파일로 하기 위해서입니다. 상태는 서버가 가지고 콘솔은 2초 폴링으로 동기화합니다.

## 채택하지 않은 기술

| 기술 | 결정 | 이유 | 남은 검증 |
|---|---|---|---|
| MQTT | 미적용 | 사건은 ack·멱등이 필요해 gRPC 단일 계약으로 충분. 상태·명령도 같은 연결(WatchCommands)로 해결 | 브로커 기반 다수 장치 팬인 |
| DeepStream / Jetson / TensorRT | 설정·경로만 준비 | 이 세션에 NVIDIA 장비 없음. ORT 백엔드의 `providers` 설정으로 TensorRT EP를 지정할 수 있게 해 두었음 | 장비에서 실행·측정 |
| vLLM | 클라이언트 경로만 | CPU 환경. OpenAI 호환 제공자로 `vllm serve <VLM>`에 바로 연결 가능 | GPU에서 실행·지연 측정 |
| Kubernetes(로컬) | CI(kind)에서 검증 | 이 세션 샌드박스에서는 중첩 런타임 제약으로 파드 기동 실패(`docs/evidence/k8s-local-attempt/`) | CI 실행 결과 확인 |
