# 구조

## 한 장 요약

```
카메라(RTSP) / 샘플 영상
      │  GStreamer: rtspsrc → decodebin → appsink  (파일: filesrc, 실시간 페이싱)
      ▼
┌───────────────── 엣지 분석기 (Python, sceneproof.edge) ─────────────────┐
│ FrameQueue(용량 4, 가장 오래된 프레임 폐기 / 오프라인은 무손실)          │
│   → YOLOX(ONNX Runtime CPU, 사람 클래스) → ByteTrack(자체 구현)        │
│   → ZoneMonitor(앵커=발 위치, 시간 기반 디바운스, 체류, 추적 끊김 유예)  │
│   → FrameRing(최근 12초 JPEG, 바이트 상한, 열린 사건 구간 고정)         │
│   → EvidenceWriter(키프레임 5종 + 전후 클립 mp4)                       │
│   → Event(snapshot, revision↑) ──gRPC EdgeIngress──▶                   │
└────────────────────────────────────────────────────────────────────────┘
      │ localhost:50051                                 ▲ SubscribeCommands
      ▼                                                 │
┌───────────── 엣지 게이트웨이 (Go, gateway/) ──────────┴──────────────┐
│ WAL 기반 내구 큐(event_id별 최신 revision으로 병합, 바이트 상한, 퇴거) │
│ 업링크 상태머신(DISCONNECTED→CONNECTING→CONNECTED, 지수 백오프)       │
│ 근거 파일 업로드(256KiB 청크, sha256, 완료 장부) · 하트비트(2s)         │
│ /healthz /readyz /status /metrics                                     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ gRPC Ingest (network) :50052
                                ▼
┌───────────── 검토 서버 (Python, sceneproof.server) ────────────────────┐
│ SQLite(WAL): events(멱등 upsert) · evidence_files · explanations      │
│   · review_log · devices · commands                                   │
│ VLM 워커(비동기, 동시성·타임아웃·재시도·예산 상한, 구조화 출력 검증)      │
│ HTTP API + 정적 콘솔(/) · 명령 허브(WatchCommands)                     │
└───────────────────────────────────────────────────────────────────────┘
                                │ :8000
                                ▼
                    관제 콘솔 (console/, vanilla JS)
```

## 책임 경계

| 구성요소 | 언어 | 책임 | 하지 않는 것 |
|---|---|---|---|
| 엣지 분석기 | Python | 디코딩, 탐지·추적·구역 판정, 근거 생성, 상태 보고, 명령 실행 | 네트워크 재전송·내구 저장(게이트웨이 역할), 장면 해석(VLM) |
| 엣지 게이트웨이 | Go | 사건의 내구 큐·중복 제거·병합, 서버 연결 복구, 근거 업로드, 하트비트, 명령 중계 | 영상 처리, 사건 판단 |
| 검토 서버 | Python | 저장·멱등 수신, VLM 해석(비동기), 검토 상태·감사 로그, 내보내기, 콘솔 | 영상 분석 |
| 콘솔 | JS | 현장 관리자의 검토 업무 화면 | 모델 실행 |

분석기와 게이트웨이는 같은 장치(같은 pod/compose 네트워크)에서 동작하고 근거 디렉터리를 공유합니다.
서버는 어디에 있어도 됩니다.

## 사건(Event)의 생애주기

1. 트랙의 앵커(상자 하단 중앙)가 구역 다각형 안에 `entry_confirm_seconds` 이상 연속으로 있으면 **사건 열림** (`revision=1`, `status=OPEN`, 키프레임 `context_before`, `entry`).
2. 구역 안 체류 시간(미디어 시간 기준)이 `dwell_seconds`를 넘으면 **체류 승격** (`kind=DWELL`, `revision=2`, 키프레임 `dwell`).
3. 앵커가 `exit_confirm_seconds` 이상 밖에 있거나(`EXITED`), 추적이 `lost_grace_seconds` 넘게 끊기면(`TRACK_LOST`) **사건 닫힘** (`status=CLOSED`, 키프레임 `exit`).
4. post-roll이 지나면 **클립 완성** (`revision`+1, `context_after`, `clip.mp4`). 게이트웨이가 파일을 업로드하고 서버는 모든 참조 파일이 도착하면 `evidence_ready=1`로 표시합니다.
5. 서버 VLM 워커가 닫힌 사건(기본)을 해석합니다. 해석 상태는 `queued → running → done | invalid | failed | timeout | skipped`.
6. 담당자가 `확인 필요 → 검토 중 → 처리 완료(침입 확인/오탐 처리/상위 보고/조치 불필요)`로 상태를 바꾸고, 모든 전이는 `review_log`에 남습니다.

Event는 항상 **완전한 스냅샷**입니다. 소비자는 `(event_id, revision)`이 더 크면 갱신하고 아니면 무시합니다. 이 성질 덕분에 게이트웨이 큐는 같은 사건의 오래된 revision을 최신 것으로 병합해도 정보를 잃지 않습니다.

## 스레드와 큐 (분석기, 카메라당)

| 스레드 | 하는 일 | 경계 |
|---|---|---|
| source | GStreamer appsink 콜백 / OpenCV read → `FrameQueue.put` | 실시간: 가득 차면 가장 오래된 프레임 폐기·카운트. 오프라인: 생산자 블록(무손실) |
| worker | `FrameQueue.get` → 링버퍼 인코딩 → 탐지 → 추적 → 구역 → 사건 전이 → 키프레임 | 단계별 시간 기록, 지연 = now − 캡처시각 |
| finalizer | post-roll 경과한 닫힌 사건의 `context_after`·클립 작성 → revision 발행 | ffmpeg 인코딩(0.7–1.2s)이 분석 루프를 막지 않음 |
| gateway-sink | 작은 병합 아웃박스(512) → gRPC SubmitEvent, 실패 시 백오프 | 게이트웨이가 죽어 있는 짧은 시간만 가림 |
| status | 2초마다 DeviceStatus 보고 | |
| commands | 게이트웨이의 SubscribeCommands 스트림 유지, START/STOP 실행 | 재시작 멱등(같은 소스면 no-op) |

메모리 상한: `FrameQueue` 용량(프레임 수), `FrameRing`(초·바이트 모두), 아웃박스(건수), 게이트웨이 WAL(바이트), 서버 VLM 동시성(건수).

## 시간의 종류

- `media_s`: 소스 미디어 시간(파일 오프셋 / 스트림 PTS 기준). 체류·디바운스·클립 구간·정답 매칭은 모두 이 시간으로 계산합니다. 분석기가 느려져도 체류 시간이 부풀지 않습니다.
- `wall_ts`: 장치 시계. 지연 측정(캡처→분석→서버 수신→최초 표시)에 씁니다.

## 계약

- gRPC/Protobuf: [`proto/sceneproof/v1/sceneproof.proto`](../proto/sceneproof/v1/sceneproof.proto) — 메시지 주석이 의미론을 정의합니다. Python 코드는 `src/sceneproof/proto`, Go 코드는 `gateway/gen/sceneproofv1`에 생성됩니다(`scripts/gen_proto.sh`).
- HTTP API: [`src/sceneproof/server/api.py`](../src/sceneproof/server/api.py) (FastAPI, `/docs`에서 OpenAPI 확인).
- VLM 출력 스키마: [`src/sceneproof/server/vlm/schema.py`](../src/sceneproof/server/vlm/schema.py) (schema_version 1).
- 설정: [`configs/edge.yaml`](../configs/edge.yaml) (주석 포함).

## 평가·학습 경로 (오프라인, 제품 실행과 분리)

```
연구용 데이터(사람이 그린 상자) ──▶ eval/bench/episodes.gt_episodes ──┐  (제품의 ZoneMonitor를 그대로 사용)
같은 영상 ──▶ detector → ByteTrack → ZoneMonitor (eval/bench/episodes.run_analyzer) ──┴─▶ eval/bench/match (시간 + 같은 사람)
                                                                                         └─▶ eval/bench/detection (사람 찾기 AP)
학습 분할 상자 ──▶ train/finetune (YOLOX 원본 손실, CPU) ──▶ export_onnx ──▶ 위와 같은 경로로 조정·최종 분할 평가
```

평가 실행기는 제품의 탐지기·추적기·구역 판정 클래스를 프레임마다 직접 호출하고, 근거 영상 저장만 생략합니다. 분할·구역·후보 모델은 측정 전에 `configs/bench/`에 고정하고, 최종 분할은 `docs/eval/bench/test_lock.json`으로 한 번만 잽니다(D11). 첫 측정 뒤 바꾼 것은 새로 고정한 두 번째 최종 분할(`split2.yaml`, `test2_lock.json`)로 잽니다. 추적기·구역 규칙만 바꾸는 실험은 `eval/bench/replay`가 탐지 결과를 한 번 저장해 두고 그 뒤 단계만 다시 돌립니다(같은 설정이면 측정값을 그대로 재현).
