# 코드 검토 가이드 — 읽는 순서, 입력·출력, 판단 지점, 확인 방법

면접 전에 직접 읽고 실행해 볼 파일을 순서대로 골랐습니다. 각 항목에 "바꿔 보면 무엇이 달라지는지"를 적었습니다.

## 0. 먼저 돌려 보기 (10분)

```bash
. .venv/bin/activate
pytest -q                                   # 30+ 테스트
sceneproof-edge offline --camera cam-bench --out /tmp/bench.jsonl --summary /tmp/bench.json
python - <<'PY'
import json; rows=[json.loads(l) for l in open('/tmp/bench.jsonl')]
print(len({r['event_id'] for r in rows}), '사건,', len(rows), '리비전')
PY
```

## 1. 계약: `proto/sceneproof/v1/sceneproof.proto`
- 입력/출력: 메시지 정의 자체. `Event`는 스냅샷, `revision` 단조 증가.
- 판단 지점: 왜 증분 이벤트가 아니라 스냅샷인가(주석 참조). 왜 `media_*_s`와 `*_ts`가 둘 다 있나.
- 확인: `scripts/gen_proto.sh` 실행 후 `git diff`가 비어 있어야 함(생성물 최신).

## 2. 탐지: `src/sceneproof/edge/detector.py`
- 입력: BGR 프레임. 출력: `Detection(boxes xyxy px, scores, timings_ms)`.
- 판단 지점: `letterbox`(114 패딩, 정규화 없음 — YOLOX 규약), `decode_raw`(그리드/스트라이드), `postprocess`(사람 클래스만, NMS), 백엔드가 전·후처리를 공유.
- 해 볼 것: `conf_threshold`를 0.25로 낮추고 `python -m sceneproof.eval.run_eval --variant yolox_tiny --conf 0.25 --out /tmp/e.json` → FP/h와 recall이 어떻게 움직이는지.
- 테스트: `tests/test_detector.py` (단일 상자 디코딩 수식, PyTorch↔ONNX 일치).

## 3. 추적: `src/sceneproof/edge/tracker.py`
- 입력: 상자·점수. 출력: `TrackOutput(track_id, xyxy, score, hits, age)`.
- 판단 지점: `update()`의 3단 연계(고점수→추적+lost, 저점수→남은 추적, 미확정 트랙), `track_buffer`, `fuse_score`.
- 해 볼 것: `TrackerParams(low_thresh=0.5)`로 2차 연계를 사실상 끄고 `tests/test_tracker.py::test_low_score_detection_bridges_occlusion`이 실패하는지 확인.

## 4. 구역 상태머신: `src/sceneproof/edge/zones.py`
- 입력: 트랙 목록 + `FrameMeta`. 출력: `EpisodeEvent(OPENED|DWELL|CLOSED)`.
- 판단 지점: 앵커=발, 시간 기반 디바운스, `box_clipped_at_bottom` 상태 유지, 추적 끊김 유예, 결정적 `episode_id`(uuid5).
- 해 볼 것: `exit_confirm_seconds`를 0.1로 줄이고 `cam-worker`를 오프라인 실행 → 사건 수가 늘고 짧아지는지.
- 테스트: `tests/test_zones.py`.

## 5. 버퍼: `src/sceneproof/edge/frames.py`
- `FrameQueue`(폐기 vs 무손실), `FrameRing`(초·바이트 상한, pin), `StageTimer`.
- 해 볼 것: `ring_max_mb: 2`로 줄이고 실행 → 클립 길이가 줄거나 `context_before`가 빠지는지(바이트 상한이 pin보다 우선).

## 6. 파이프라인·에이전트: `pipeline.py`, `agent.py`, `events.py`
- `CameraPipeline._worker`가 한 프레임을 처리하는 순서와 단계별 시간 기록, `_finalizer`가 post-roll 후 클립을 만드는 이유.
- `GatewaySink`의 병합 아웃박스(512)와 `JsonlSink`가 같은 Event를 받는 구조.
- `EdgeAgent._handle_command`의 멱등 START.
- 해 볼 것: 게이트웨이를 끈 채 `sceneproof-edge run --camera cam-bench` → 로그의 `gateway-sink` 실패·재시도, 게이트웨이를 켜면 밀린 사건이 흘러가는지.

## 7. 게이트웨이: `gateway/internal/queue/wal.go`, `queue.go`, `gateway/internal/uplink/sender.go`
- WAL 레코드 형식, 병합 인덱스, 퇴거, 재시작 복구. 업링크 상태머신과 백오프.
- 해 볼 것: `go test ./internal/queue -run Recovery -v`; `--queue-max-bytes 65536`으로 바이너리를 띄우고 서버를 끈 채 사건을 흘려 `dropped_events`가 증가하는지(`:9100/status`).

## 8. 서버: `src/sceneproof/server/db.py`(upsert), `ingest.py`, `vlm/worker.py`, `api.py`
- `upsert_event`의 revision 비교가 멱등성의 전부. `record_evidence` → `evidence_ready`.
- 워커: 큐 → 실행 중 → 완료/실패/타임아웃/형식오류/예산초과, 재시작 시 `requeue_running`+`backfill`.
- `api.py`의 상태 전이 표(`ALLOWED_TRANSITIONS`)와 CSV 열(실행 조건 포함).
- 해 볼 것: `SP_SERVER_VLM_MOCK_FAIL_EVERY=2 SP_SERVER_VLM_TIMEOUT_S=1 SP_SERVER_VLM_MOCK_DELAY_S=3` 으로 서버를 띄우고 콘솔에서 상태 칩과 "다시 해석" 동작 확인.

## 9. 평가: `src/sceneproof/eval/metrics.py`, `run_eval.py`, `bench_detector.py`
- 매칭 규칙과 분모(문서 `docs/eval/ground-truth.md`). `rescore`로 정답만 바꿔 재채점.
- 해 볼 것: `data/ground_truth/eval-campus.yaml`의 e01 라벨을 `intrusion`으로 바꾸고 rescore → precision 변화.

## 10. 콘솔: `console/app.js`
- 폴링·diff 렌더, 상태 전이 버튼, 정적 재생 모드(`window.SP_STATIC_BASE`).

## 검토 체크리스트 (직접 수행 후 표시)

- [ ] 오프라인 실행으로 사건·클립 생성 확인
- [ ] 단위 테스트 전부 통과
- [ ] 서버+게이트웨이+분석기 로컬 실행, 콘솔에서 사건 확인·검토 완료까지
- [ ] 게이트웨이 끊김/복구 시 사건 중복 없음 확인
- [ ] 정답 YAML 검수 후 재채점
- [ ] (승인 시) 실제 VLM 제공자로 해석 실행, 루브릭 CSV 채점
