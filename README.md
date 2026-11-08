# SceneProof — 산업현장 제한구역 접근·체류 영상 사건 검토

**어느 카메라에서 어떤 일이 있었고, 어떤 장면을 근거로 확인이 필요하다고 판단했으며, 담당자가 어떻게 처리했는가** — 현장 관리자가 이 질문에 답할 수 있도록, CV가 사건을 만들고 VLM이 장면을 해석하고 사람이 검토하는 흐름을 하나로 연결한 서비스입니다.

[![ci](https://github.com/sokldjs554/sceneproof/actions/workflows/ci.yml/badge.svg)](https://github.com/sokldjs554/sceneproof/actions/workflows/ci.yml) [![k8s-runtime](https://github.com/sokldjs554/sceneproof/actions/workflows/k8s-runtime.yml/badge.svg)](https://github.com/sokldjs554/sceneproof/actions/workflows/k8s-runtime.yml)

> 노타 AI **[Solution] AI Software Engineer Intern (전환형)** 지원용 프로젝트입니다. 공고 항목별 대응은 [`docs/job-mapping.md`](docs/job-mapping.md)에 있습니다.
> 샘플 영상은 공개 데이터(Intel IoT DevKit CC BY 4.0, OpenCV Apache-2.0)이며 신원 식별 기능은 없습니다.

![콘솔 첫 화면](docs/images/01_overview.png)

위 화면은 실시간 모드(mock 제공자)입니다. Claude Opus 5.5가 실제로 생성한 해석이 표시된 재생 모드 화면은 [`docs/images/replay-vlm/03_event_vlm.png`](docs/images/replay-vlm/03_event_vlm.png).

## 무엇을 하는가

1. 샘플 영상 또는 RTSP 스트림을 고른다 (콘솔 → 장치로 명령 전달).
2. 확인할 구역(다각형)과 조건(진입 확정 시간, 체류 시간)을 설정한다.
3. CV가 사람을 탐지·추적하고, 발 위치가 구역 안에 머무는 시간으로 **진입/체류 사건**을 만든다.
4. 사건 전후 3초의 클립과 근거 프레임(진입 직전·진입·체류·이탈·이탈 직후)을 자동으로 확보한다.
5. VLM이 근거 프레임을 보고 **보이는 것만** 구조화된 형식으로 설명하고, 판단의 한계를 적는다.
6. 담당자가 원본 장면과 설명을 보고 `확인 필요 → 검토 중 → 처리 완료`로 상태를 바꾼다.
7. 사건·처리 결과·실행 조건(모델·설정 버전)을 조회하고 CSV/JSON으로 내보낸다.

```
RTSP/파일 → [엣지 분석기: GStreamer → YOLOX(ONNX Runtime) → ByteTrack → 구역 상태머신 → 근거 클립]
          → [엣지 게이트웨이(Go): 내구 큐·재연결·업로드·하트비트] ──gRPC──▶ [검토 서버: SQLite·VLM 워커·API] → 관제 콘솔
```

상세 구조: [`docs/architecture.md`](docs/architecture.md) · 설계 결정과 대안: [`docs/design/decisions.md`](docs/design/decisions.md)

## 바로 실행

```bash
git clone https://github.com/sokldjs554/sceneproof.git && cd sceneproof
python -m venv .venv && . .venv/bin/activate
pip install -e ".[server,dev,torch]"          # PyGObject(GStreamer)는 선택: pip install -e ".[gst]"
bash scripts/fetch_data.sh && bash scripts/fetch_models.sh
(cd gateway && make build)                    # Go 1.24

# 터미널 1: 서버(콘솔 http://127.0.0.1:8000)   터미널 2: 게이트웨이   터미널 3: 분석기
SP_SERVER_VLM_PROVIDER=mock sceneproof-server run
gateway/bin/sceneproof-gateway --listen :50051 --server 127.0.0.1:50052 --device-id edge-01 --queue-dir data/gateway --evidence-root data/evidence --http :9100 --insecure
sceneproof-edge --config configs/edge.yaml run --camera cam-bench
```

콘솔에서 카메라 행의 **분석 시작**을 누르면 샘플이 실시간 속도로 분석되고 사건이 쌓입니다. 분석기 없이 결과만 보려면 `sceneproof-server run --mode replay --replay-bundle demo/bundle` (화면에 "저장된 실행 결과 재생" 표시).
docker compose(`docker compose up --build`)와 Kubernetes(`scripts/k8s_verify.sh`)는 [`docs/ops/runbook.md`](docs/ops/runbook.md).

VLM 제공자: `mock`(형식 검증, 결과에 `[MOCK]` 표시) / `anthropic`(Claude, 구조화 JSON) / `openai_compat`(vLLM 등 자체 호스팅) / `transformers`(SmolVLM 로컬). 선택과 비용 상한은 runbook 참조.

## 결과 요약 (실측; 상세는 [`docs/eval/README.md`](docs/eval/README.md))

- **사건 평가(held-out 4편, 정답 draft)**: yolox_tiny(ONNX, CPU) precision 1.0 / recall 1.0 / 오탐 0건·중복 5건 — 중복은 추적 ID 전환으로 한 사건이 쪼개진 것. nano는 원거리 자전거 탑승자 2건을 놓쳐 recall 0.8. 정답은 Claude가 초안을 만든 **draft** 상태이며 사용자의 최종 검수 항목이 [`docs/eval/ground-truth.md`](docs/eval/ground-truth.md)에 있습니다.
- **추론 성능(4 vCPU, 3회 반복 중앙값)**: yolox_tiny 추론 p50 — PyTorch 44.6 ms → ONNX Runtime 21.6 ms(그래프 최적화 끔 32.1 ms); nano 26.8 → 8.6 ms; s 123.3 → 69.8 ms. 변환 전후 NMS 결과 상자·점수 차이 0(150프레임). 표와 반복 변동은 docs/eval/README.md §2.
- **스트림 수 증가(tiny, 4 vCPU)**: 1스트림 28.6 fps(입력 30), 4스트림 합계 34 fps·스트림당 8.5 fps, 지연 p95 180 → 560 ms, 큐 깊이 상한(4) 유지·폐기 프레임 증가 — §3.
- **실시간·장애 대응**: 과입력(입력 30 fps·1스레드 → 처리 19 fps, 폐기 417, 지연 최대 202 ms), 스트림 단절·복구(RTSP 14초 단절 → 4초 만에 재연결), VLM 지연·실패(수신 ack p95 2.7 ms 유지, 타임아웃·재시도·재요청), 장치–서버 단절(큐잉 → 재전송, 중복 0), 중복 전달, 재시작 — 6가지 모두 스크립트로 재현·통과. 이 과정에서 VLM 워커의 실제 버그 2개를 찾아 고침(`docs/dev-log.md` L9–L10). §4와 [`docs/evidence/failure-scenarios/`](docs/evidence/failure-scenarios/).
- **CV만 vs CV+VLM(Claude Opus 5.5, held-out 19건, CI 실행)**: 해석 19/19 완료(스키마 위반 0, p50 19.8초, 비용 $1.01). CV 결과와 대조하면 정답 침입 10건 중 VLM이 '확인 필요'로 유지한 것은 4건, 3건은 '정상 가능성 높음', 3건은 '불확실'. CV 오탐 1건은 VLM이 걸러내지 못했고 오히려 '구역 안'으로 봤습니다(정답 초안 재검수 대상). 경계 통과 3건은 2건 '정상 가능성 높음'·1건 '불확실'로 구분. **결론: 현재 프롬프트(v1.2)의 assessment는 '위험 판단'으로 해석되어 기하학적 침입 정답과 어긋납니다. 그래서 제품은 VLM 판단을 보조 표시로만 쓰고 모든 사건을 '확인 필요'로 유지합니다(D7).** 사람 채점 루브릭은 미채점. 사례·개선안은 [`docs/eval/vlm-findings.md`](docs/eval/vlm-findings.md), 원시 결과는 `docs/eval/raw/vlm_eval*.json`.

## 기술 선택 (왜)

| 영역 | 선택 | 이유 (대안·한계는 decisions.md) |
|---|---|---|
| 탐지 | YOLOX nano/tiny/s (Apache-2.0) → 자체 ONNX 내보내기 | 라이선스가 명확하고, PyTorch/ONNX를 같은 전후처리로 비교 가능. CPU 4코어에서 tiny가 30fps 입력을 처리 |
| 추적 | ByteTrack 자체 구현 | 체류 시간은 ID 안정성에 달림. 저신뢰 상자 2차 연계를 테스트로 고정 |
| 구역 판정 | 발 위치 앵커 + 시간 기반 디바운스 + 추적 끊김 유예 + 하단 클리핑 보정 | 평가 데이터에서 발견한 실패(사건 쪼개짐)를 고친 결과 |
| 입력 | GStreamer(rtspsrc/appsink), OpenCV 대체 경로 | RTSP 재연결·백오프, 파일은 실시간 페이싱 |
| 엣지↔서버 | Protobuf/gRPC 단일 계약, Go 게이트웨이 | 사건은 스냅샷+revision → 멱등·병합·재전송이 한 규칙 |
| VLM | 서버 비동기 워커, 범주형 구조화 출력, 예산 상한 | 분석·기록은 VLM과 무관하게 진행; 확률·신뢰도 미표시 |
| 저장·배포 | SQLite(WAL), Docker/compose/k8s 매니페스트, CI | 단일 서버 데모에 충분; 재시작 복구 검증 |

## 범위와 한계

- NVIDIA 장비가 없는 환경에서 만들었습니다. TensorRT·DeepStream·Jetson·vLLM은 설정 경로와 클라이언트만 준비했고 **실행·측정하지 않았습니다**.
- Kubernetes는 GitHub Actions의 kind 클러스터에서 배포·서버 파드 삭제 복구·엣지 파드 삭제 복구·중복 없음 검사를 통과했습니다([k8s-runtime 실행](https://github.com/sokldjs554/sceneproof/actions/runs/42402112637)). 개발 샌드박스 자체에서는 중첩 컨테이너 런타임 제약으로 kind/k3s 파드가 뜨지 않았습니다(증적 보존).
- 정답 라벨은 같은 계열 탐지기 초안에서 출발했고 사람이 최종 검수하지 않았습니다.
- 선반·물건에 발이 가리면 앵커가 지면 위치가 아니어서 구역 판정이 흔들립니다(매장 영상). 구조적 한계로 기록했습니다.
- 인증·TLS·다중 테넌트·다중 서버 확장은 범위 밖입니다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/job-mapping.md](docs/job-mapping.md) | 공고 전체 항목 대응표(검증 상태 포함) |
| [docs/architecture.md](docs/architecture.md) | 구조, 사건 생애주기, 스레드·큐, 시간의 종류 |
| [docs/design/decisions.md](docs/design/decisions.md) | 설계 결정·대안·미채택 기술 |
| [docs/eval/README.md](docs/eval/README.md) | 평가·성능·장애 시나리오 결과(자동 생성) |
| [docs/eval/ground-truth.md](docs/eval/ground-truth.md) | 정답 기준·작성 방법·남은 검수 |
| [data/README.md](data/README.md) | 데이터 출처·라이선스·분할 |
| [docs/ops/runbook.md](docs/ops/runbook.md) · [failure-modes.md](docs/ops/failure-modes.md) | 실행·상태 확인·복구 |
| [docs/dev-log.md](docs/dev-log.md) | 발견한 실패·원인·수정 전후, 리팩토링, AI 활용 |
| [docs/code-review-guide.md](docs/code-review-guide.md) | 읽을 순서·입출력·판단 지점·직접 해 볼 항목 |
| [docs/papers-and-docs.md](docs/papers-and-docs.md) | 논문·공식 문서 요약과 적용·재현 범위 |
| [docs/handoff.md](docs/handoff.md) | 인계: 완료/검증/미검증 구분, 본인·AI 수행 범위, 남은 일 |
| [gateway/README.md](gateway/README.md) | Go 게이트웨이 설계 |

## 라이선스

코드: Apache-2.0. 벤더링한 YOLOX 모델 코드: Apache-2.0(Megvii). 샘플 영상: CC BY 4.0(Intel IoT DevKit), Apache-2.0(OpenCV). 자세한 출처는 [`data/README.md`](data/README.md), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
