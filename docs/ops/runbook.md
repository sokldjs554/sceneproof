# 운영 안내

## 로컬 실행 (가장 빠른 길)

```bash
# 0) 준비 (한 번)
python -m venv .venv && . .venv/bin/activate
pip install -e ".[server,dev,gst]"      # gst는 PyGObject(선택). 없으면 OpenCV 백엔드로 동작
bash scripts/fetch_data.sh               # 샘플 영상 (CC BY 4.0 / Apache-2.0)
bash scripts/fetch_models.sh             # YOLOX 가중치 + ONNX 내보내기 (torch 필요: pip install -e ".[torch]")
(cd gateway && make build)               # Go 1.24

# 1) 서버 (콘솔 http://127.0.0.1:8000, gRPC :50052)
SP_SERVER_VLM_PROVIDER=mock sceneproof-server run

# 2) 게이트웨이
gateway/bin/sceneproof-gateway --listen :50051 --server 127.0.0.1:50052 --device-id edge-01 \
  --queue-dir data/gateway --evidence-root data/evidence --http :9100 --insecure

# 3) 분석기 (샘플 파일을 실시간 페이싱으로 분석; RTSP는 configs/edge.yaml의 cam-rtsp-worker)
sceneproof-edge --config configs/edge.yaml run --camera cam-bench
#    RTSP 시뮬레이터가 필요하면: python -m sceneproof.tools.rtsp_sim --dir data/videos --port 8554
```

콘솔에서 카메라 행의 **분석 시작**을 누르면 서버 → 게이트웨이 → 분석기로 명령이 전달됩니다.

### VLM 제공자 선택

| `SP_SERVER_VLM_PROVIDER` | 필요한 것 | 비고 |
|---|---|---|
| `mock` | 없음 | 형식 검증용. 결과에 `[MOCK]` 표시 |
| `anthropic` | `ANTHROPIC_API_KEY`, `SP_SERVER_VLM_MODEL`(기본 `claude-opus-5-5`), `SP_SERVER_VLM_BUDGET_USD` | 구조화 JSON 출력, 예산 상한 초과 시 `skipped` |
| `openai_compat` | `SP_SERVER_VLM_OPENAI_BASE_URL`(예: vLLM `http://host:8001/v1`), `SP_SERVER_VLM_MODEL` | 자체 호스팅 VLM |
| `transformers` | `pip install -e ".[vlm-local]"`, huggingface.co 접근 | SmolVLM 등 CPU 소형 모델, 느림 |
| `none` | | 해석 없음(재생 데모) |

## docker compose

```bash
docker compose up --build        # rtsp-sim + server + gateway + edge
# 콘솔 http://localhost:8000 · 게이트웨이 상태 http://localhost:9100/status
```

## Kubernetes (kind)

`scripts/k8s_verify.sh`가 이미지 빌드 → kind 클러스터 → 배포 → 장치 온라인·사건 수신 확인 → 서버 파드 삭제 후 복구 → 엣지 파드 삭제 후 복구 → 중복 없음 검사까지 수행하고 `docs/evidence/k8s/`에 증적을 남깁니다. CI(`.github/workflows/k8s-runtime.yml`)에서 같은 스크립트가 돕니다.

```bash
kubectl -n sceneproof get pods -o wide
kubectl -n sceneproof logs deploy/edge -c gateway --tail=50
kubectl -n sceneproof port-forward svc/server 8000:8000
```

## 상태 확인 지점

| 무엇 | 어디 |
|---|---|
| 서버 생존 | `GET /api/health` |
| 장치·카메라·업링크·VLM 큐 | `GET /api/status` (콘솔 왼쪽 패널과 같은 데이터) |
| 게이트웨이 | `GET :9100/healthz` `/readyz` `/status` `/metrics`(Prometheus 텍스트) |
| 분석기 | 게이트웨이를 통해 하트비트(2s). 로그는 stdout(JSON 아님, 사람용) |

## 복구 절차

| 증상 | 확인 | 조치 |
|---|---|---|
| 콘솔에 장치 `오프라인` | `/api/status.devices[].age_s` 커짐 | 게이트웨이 프로세스/파드 상태 → `:9100/status`의 `uplink.state`. 서버 주소·포트 확인 |
| 업링크 `대기 사건` 증가 | 게이트웨이 `/status.queue.pending` | 서버 gRPC(:50052) 접근성. 복구되면 자동 재전송, 서버에서 중복 무시 |
| 카메라 `재연결 중` 지속 | `last_error`에 rtspsrc 오류 | 카메라/RTSP 주소, 네트워크. 분석기는 백오프로 계속 재시도 |
| `폐기 프레임` 급증 | 입력 fps ≫ 처리 fps | 정상 동작(실시간 우선). 모델을 nano로, `intra_op_threads` 조정, 카메라 수 축소 |
| VLM `timeout`/`failed` 다수 | `/api/status.vlm.db` | 제공자 응답성·키·예산. 사건 기록은 영향 없음. 콘솔 `다시 해석` |
| 근거 클립 `업로드 대기` | `uploads_pending` | 분석기 finalizer가 클립을 쓰는 중(post-roll 3s + 인코딩) 또는 업로드 재시도 중 |

## 데이터 내보내기·백업

- CSV/JSON: 콘솔 상단 버튼 또는 `GET /api/export?format=csv`. 모델·백엔드·설정 해시·VLM 버전이 열에 포함됩니다.
- 재생 번들: `sceneproof-server export <dir>` → `events.jsonl`, `explanations.jsonl`, `evidence/`. 다른 서버에서 `sceneproof-server import <dir>`.
- SQLite 파일(`data/server/sceneproof.db`)과 `data/server/evidence/`가 전부입니다.
