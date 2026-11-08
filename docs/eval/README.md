# 평가·실험 결과

모든 표는 `scripts/eval_report.py`가 `docs/eval/raw/*.json`과 `docs/evidence/`에서 생성합니다(손으로 적은 숫자 없음). 목표치가 아니라 **실측**이며, 측정 환경은 각 절에 적습니다. 데이터·분할·정답: `data/README.md`, `docs/eval/ground-truth.md`.

## 1. 사건 단위 평가 (held-out 영상 4편, 정답은 draft)

매칭 규칙·분모: `docs/eval/ground-truth.md`. 실행: `python -m sceneproof.eval.run_eval --variant <v>`; 재채점: `rescore.py`.

| 모델(ONNX, conf 0.4) | precision | recall | F1 | FP/h | TP | FP | 놓침 | 중복 | 경계·가림 겹침 | 오프라인 처리 fps(평균) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolox_nano | 1.0 | 0.8 | 0.889 | 0.00 | 8 | 0 | 2 | 3 | 3 | 70.2 |
| yolox_tiny | 1.0 | 1.0 | 1.0 | 0.00 | 10 | 0 | 0 | 5 | 3 | 18.1 |
| yolox_s | 1.0 | 1.0 | 1.0 | 0.00 | 10 | 0 | 0 | 2 | 3 | 12.1 |

오프라인 처리 fps는 평가 실행 중 다른 작업(다른 모델 평가·벤치마크)과 CPU를 나눠 쓴 값이라 참고용입니다. 추론 성능 비교는 §2를 보세요.

정답 양성(intrusion) 사건 수: eval-campus 2, eval-indoor 4, eval-store 2, eval-parking 2

### 영상별 (yolox_tiny — 기본 설정)

| 영상 | P | R | TP | FP | 놓침 | 중복 | 경계·가림 | 시작 MAE(s) | 종료 MAE(s) | 체류 판정 일치/불일치 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval-campus | 1.0 | 1.0 | 2 | 0 | 0 | 1 | 2 | 0.15 | 2.35 | 1/1 |
| eval-indoor | 1.0 | 1.0 | 4 | 0 | 0 | 0 | 0 | 0.24 | 0.24 | 4/0 |
| eval-store | 1.0 | 1.0 | 2 | 0 | 0 | 3 | 0 | 0.12 | 0.38 | 2/0 |
| eval-parking | 1.0 | 1.0 | 2 | 0 | 0 | 1 | 1 | 0.43 | 2.52 | 2/0 |

### 모델별 차이가 드러난 곳

- **yolox_nano**: eval-campus: 중복 1건; eval-store: 중복 2건; eval-parking: 놓침 ['e01', 'e03']
- **yolox_tiny**: eval-campus: 중복 1건; eval-store: 중복 3건; eval-parking: 중복 1건
- **yolox_s**: eval-store: 중복 2건

해석: 중복(duplicates)은 같은 정답 사건이 추적 ID 전환·추적 끊김으로 둘 이상의 사건으로 쪼개진 횟수입니다. 놓침은 소형 원거리 객체(주차장 자전거)에서 nano가 사람을 탐지하지 못한 경우입니다. 경계·가림 겹침은 정답에서 `boundary_pass`/`occluded`로 표시한 구간과 겹친 예측으로, 오탐으로 세지 않되 숨기지도 않습니다.

## 2. 추론 성능 — PyTorch vs ONNX Runtime (같은 프레임·같은 전후처리·같은 상자 기준)

환경: Intel(R) Xeon(R) Processor @ 2.10GHz ×4 vCPU, Linux-6.18.44-fc-v310-x86_64-with-glibc2.39, Python 3.13.18, torch 2.15.0+cu132, onnxruntime 1.31.1. 입력 data/videos/campus-walkway.mp4 앞 150프레임(768×576), 워밍업 10프레임, **3회 반복의 중앙값**. 원시 per-frame 시간은 `raw/bench_detector.json`.

| 모델 | 스레드 | 설정 | infer p50 (ms) | infer p95 (ms) | 전체 p50 (ms) | 전체 p95 (ms) | 전처리 p50 | 후처리 p50 | fps | CPU % | RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yolox_nano | 4 | onnxruntime | 8.6 | 16.8 | 10.4 | 18.5 | 0.88 | 0.84 | 82.8 | 367 | 791 |
| yolox_nano | 4 | onnxruntime-noopt | 18.1 | 45.8 | 20.1 | 51.6 | 0.94 | 0.95 | 42.7 | 359 | 794 |
| yolox_nano | 4 | torch | 26.8 | 38.6 | 29.2 | 41.4 | 1.00 | 1.19 | 32.0 | 394 | 813 |
| yolox_nano | 1 | onnxruntime | 12.7 | 16.2 | 14.2 | 18.1 | 0.70 | 0.78 | 70.1 | 105 | 845 |
| yolox_nano | 1 | onnxruntime-noopt | 26.7 | 33.2 | 28.4 | 35.2 | 0.77 | 0.88 | 35.6 | 102 | 848 |
| yolox_nano | 1 | torch | 40.0 | 46.3 | 41.7 | 48.4 | 0.78 | 0.99 | 24.3 | 101 | 850 |
| yolox_tiny | 4 | onnxruntime | 21.6 | 29.3 | 23.5 | 31.3 | 0.91 | 0.85 | 40.9 | 392 | 887 |
| yolox_tiny | 4 | onnxruntime-noopt | 32.1 | 46.7 | 34.0 | 49.1 | 0.95 | 0.92 | 27.0 | 384 | 880 |
| yolox_tiny | 4 | torch | 44.6 | 59.6 | 47.0 | 62.0 | 1.03 | 1.25 | 20.4 | 392 | 884 |
| yolox_tiny | 1 | onnxruntime | 57.0 | 67.9 | 58.9 | 70.3 | 0.88 | 0.98 | 17.5 | 101 | 896 |
| yolox_tiny | 1 | onnxruntime-noopt | 75.3 | 95.6 | 77.3 | 97.9 | 0.96 | 1.07 | 12.5 | 100 | 902 |
| yolox_tiny | 1 | torch | 90.8 | 108.8 | 93.1 | 111.0 | 0.92 | 1.05 | 10.7 | 100 | 902 |
| yolox_s | 4 | onnxruntime | 69.8 | 92.3 | 74.2 | 97.3 | 2.15 | 2.33 | 12.8 | 391 | 980 |
| yolox_s | 4 | onnxruntime-noopt | 100.3 | 124.5 | 104.6 | 129.7 | 2.03 | 2.31 | 9.4 | 393 | 1006 |
| yolox_s | 4 | torch | 123.3 | 164.0 | 128.1 | 168.7 | 1.69 | 2.24 | 7.4 | 389 | 910 |
| yolox_s | 1 | onnxruntime | 210.3 | 243.3 | 214.1 | 247.6 | 1.69 | 2.18 | 4.8 | 100 | 989 |
| yolox_s | 1 | onnxruntime-noopt | 306.2 | 352.8 | 309.9 | 357.5 | 1.66 | 2.23 | 3.3 | 100 | 1014 |
| yolox_s | 1 | torch | 316.1 | 353.0 | 320.2 | 357.1 | 1.56 | 2.50 | 3.2 | 100 | 945 |

반복 간 변동(infer p50, 3회의 최소–최대):

- yolox_nano thr4 onnxruntime: 8.2–8.9 ms
- yolox_nano thr4 onnxruntime-noopt: 17.4–20.5 ms
- yolox_nano thr4 torch: 24.5–27.3 ms
- yolox_nano thr1 onnxruntime: 12.3–14.9 ms
- yolox_nano thr1 onnxruntime-noopt: 24.7–27.1 ms
- yolox_nano thr1 torch: 33.3–40.8 ms
- yolox_tiny thr4 onnxruntime: 20.8–21.8 ms
- yolox_tiny thr4 onnxruntime-noopt: 31.8–32.7 ms
- yolox_tiny thr4 torch: 44.2–46.8 ms
- yolox_tiny thr1 onnxruntime: 56.1–59.6 ms
- yolox_tiny thr1 onnxruntime-noopt: 75.2–81.6 ms
- yolox_tiny thr1 torch: 83.0–96.1 ms
- yolox_s thr4 onnxruntime: 69.2–70.2 ms
- yolox_s thr4 onnxruntime-noopt: 97.8–103.1 ms
- yolox_s thr4 torch: 113.3–124.3 ms
- yolox_s thr1 onnxruntime: 197.7–214.3 ms
- yolox_s thr1 onnxruntime-noopt: 302.8–314.9 ms
- yolox_s thr1 torch: 305.1–316.6 ms

### 변환 전후 출력 일치 (NMS 이후 상자·점수 비교)

| 모델 | 스레드 | 비교 | 프레임 수 | 상자 수가 다른 프레임 | 매칭 상자 | 비매칭 상자 | 최대 좌표 차(px) | 최대 점수 차 |
|---|---|---|---|---|---|---|---|---|
| yolox_nano | 4 | torch_vs_onnxruntime | 150 | 0 | 991 | 0 | 0.0 | 0.0 |
| yolox_nano | 4 | onnxruntime_vs_noopt | 150 | 0 | 991 | 0 | 0.0 | 0.0 |
| yolox_nano | 1 | torch_vs_onnxruntime | 150 | 0 | 991 | 0 | 0.0 | 0.0 |
| yolox_nano | 1 | onnxruntime_vs_noopt | 150 | 0 | 991 | 0 | 0.0 | 0.0 |
| yolox_tiny | 4 | torch_vs_onnxruntime | 150 | 0 | 857 | 0 | 0.0 | 0.0 |
| yolox_tiny | 4 | onnxruntime_vs_noopt | 150 | 0 | 857 | 0 | 0.0 | 0.0 |
| yolox_tiny | 1 | torch_vs_onnxruntime | 150 | 0 | 857 | 0 | 0.0 | 0.0 |
| yolox_tiny | 1 | onnxruntime_vs_noopt | 150 | 0 | 857 | 0 | 0.0 | 0.0 |
| yolox_s | 4 | torch_vs_onnxruntime | 150 | 0 | 869 | 0 | 0.0 | 0.0 |
| yolox_s | 4 | onnxruntime_vs_noopt | 150 | 0 | 869 | 0 | 0.0 | 0.0 |
| yolox_s | 1 | torch_vs_onnxruntime | 150 | 0 | 869 | 0 | 0.0 | 0.0 |
| yolox_s | 1 | onnxruntime_vs_noopt | 150 | 0 | 869 | 0 | 0.0 | 0.0 |

## 3. 스트림 수 증가 — 한 분석기 프로세스, RTSP 시뮬레이터 입력

yolox_tiny, intra-op 4스레드, 4 vCPU, 각 설정 40초 관측(워밍업 8초 제외). 입력은 샘플 영상의 실제 fps(10–30).

| 스트림 수 | 입력 fps/스트림 | 처리 fps/스트림 | 처리 fps 합계 | 지연 p50 (ms) | 지연 p95 (ms) | 평균 큐 깊이 | 폐기 프레임(합) | CPU % | RSS 최대 MB | 사건 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 30.0 | 28.6 | 28.6 | 76 | 180 | 1.7 | 75 | 381 | 344 | 12 |
| 2 | 20.0 | 15.7 | 31.4 | 121 | 204 | 1.8 | 428 | 382 | 467 | 20 |
| 3 | 23.3 | 12.0 | 36.0 | 186 | 338 | 2.7 | 1617 | 374 | 597 | 46 |
| 4 | 20.0 | 8.5 | 34.0 | 301 | 560 | 3.5 | 2195 | 342 | 782 | 126 |

## 4. 장애 시나리오 (`scripts/failure_scenarios.py`, git 6c6de58, 2026-11-07T13:13:10)

| 시나리오 | 결과 | 확인 항목 | 요약 |
|---|---|---|---|
| overload | PASS | input_faster_than_processing=✓, frames_dropped_by_policy=✓, queue_depth_bounded=✓, lag_bounded_under_2s=✓, analysis_kept_running=✓ | {"max_lag_ms": 201.6582489013672, "median_input_fps": 30.0, "median_processed_fps": 19.0, "dropped_frames": 417, "frames_processed": 658} |
| stream_loss | PASS | streaming_before_loss=✓, reconnecting_observed=✓, streaming_after_recovery=✓, reconnect_count_incremented=✓, frames_resumed=✓ | {"rtsp_down_s": 13.8, "recovered_after_s": 4.1, "reconnect_count": 5, "last_error_seen": ["Could not open resource for reading and writing. (../gst/rtsp/gstrtspsrc.c(8442): gst_rtspsrc_retrieve_sdp (): /GstPipeline:pipeline2/GstRTSPSrc:rtspsrc2:\nFailed to connect. (Generic error))"]} |
| vlm_faults | PASS | ingest_ack_p95_under_500ms_while_vlm_stuck=✓, timeouts_or_failures_recorded=✓, no_job_left_running_or_queued_forever=✓, retry_after_recovery_succeeds=✓ | {"ack_p50_ms": 2.0, "ack_p95_ms": 2.7} |
| uplink_loss | PASS | gateway_disconnected_during_hang=✓, events_queued_while_down=✓, reconnected_after_restart=✓, queue_drained=✓, no_duplicate_event_rows=✓, events_survived_restart=✓, review_survived_restart=✓ | {"events_before": 2, "events_after": 4, "ingest_counters": {"events": 6, "duplicates": 0, "uploads": 24, "upload_bytes": 1131419, "heartbeats": 9}, "gateway_final": {"uplink_state": "CONNECTED", "pending": 0, "sent": 12, "acked": 11, "dup_acks": 0, "reconnects": 7, "dropped": 0, "queue_bytes": 0, "u |
| duplicates | PASS | first_delivery_accepted=✓, repeat_same_revision_is_duplicate=✓, newer_revision_accepted=✓, older_revision_after_newer_ignored=✓, single_row_with_max_revision=✓ | null |
| restart | PASS | events_persisted=✓, no_duplicate_rows=✓, review_persisted=✓, vlm_queue_resumed_after_restart=✓, gateway_reconnected=✓ | null |

## 5. CV만 vs CV+VLM (anthropic / claude-opus-5-5 / prompt v1.3-ko)

- 사건–정답 매칭: 19/19건 매칭 (CV 결과별 {"tp": 10, "fp": 1, "duplicate": 5, "boundary": 3, "unmatched": 0}, CV가 놓친 정답 침입 0건 — VLM은 CV가 만든 사건만 볼 수 있음)
- 해석 상태 분포: {"done": 19}
- 해석 지연 p50/p95: 20120 / 26527 ms (n=19), 추정 비용 0.9847 USD
- CV만: 모든 사건이 '확인 필요' (19건; 그중 정답 침입 10, CV 오탐 1)
- CV+VLM: 정답 침입을 '확인 필요'로 유지 8/10, CV 오탐을 '정상 가능성 높음'으로 표시 0/1
- assessment × CV 결과: `{"duplicate": {"needs_attention": 5}, "tp": {"needs_attention": 8, "unclear": 2}, "boundary": {"needs_attention": 2, "unclear": 1}, "fp": {"needs_attention": 1}}`
- 사람 채점 루브릭: `docs/eval/raw/vlm_eval-rubric.csv` (채점 전)
- 읽는 법: VLM의 '정상 가능성 높음'을 자동 종결에 썼다면 정답 침입 0/10건이 사람 눈에 띄지 않았을 것입니다. 그래서 제품은 VLM 판단을 보조 표시로만 쓰고 모든 사건을 '확인 필요'로 유지합니다(decisions D7). 사례별 소견과 프롬프트 개선안: `docs/eval/vlm-findings.md`.

프롬프트 버전 비교 (같은 사건 집합 · 같은 모델; 뒤 버전은 앞 버전의 결과를 보고 assessment 규칙을 명시한 것이므로 같은 사건에 대한 재측정은 뒤 버전에 유리한 평가입니다 — `docs/eval/vlm-findings.md` §4):

| 프롬프트 | 해석 완료 | 정답 침입→확인 필요 | 정답 침입→정상 가능성 높음 | CV 오탐→정상 가능성 높음 | 경계 통과→정상/불확실/확인 필요 | 비용 USD | 지연 p50 ms |
|---|---|---|---|---|---|---|---|
| v1.2-ko | 19/19 | 4/10 | 3/10 | 0/1 | 2/1/0 | 1.0091 | 19764 |
| v1.3-ko | 19/19 | 8/10 | 0/10 | 0/1 | 0/1/2 | 0.9847 | 20120 |

## 재현 절차

```bash
bash scripts/fetch_data.sh && bash scripts/fetch_models.sh
# 1 사건 평가
for v in yolox_nano yolox_tiny yolox_s; do python -m sceneproof.eval.run_eval --variant $v --out docs/eval/raw/event_eval_$v.json; done
# 2 추론 성능 (조용한 CPU에서)
python -m sceneproof.eval.bench_detector --video data/videos/campus-walkway.mp4 --frames 150 --repeats 3 --threads 4,1 --out docs/eval/raw/bench_detector.json
# 3 스트림 수
python -m sceneproof.tools.rtsp_sim --dir data/videos --port 8554 &  python -m sceneproof.eval.bench_streams --streams 1,2,3,4 --out docs/eval/raw/bench_streams.json
# 4 장애 시나리오
python scripts/failure_scenarios.py --out docs/evidence/failure-scenarios
# 5 CV vs CV+VLM: 평가 영상을 서버에 재생 → 실제 VLM 제공자로 해석 → 비교
sceneproof-edge --config configs/eval.yaml offline --camera eval-campus --out /tmp/eval-bundle/events.jsonl   # (4편 반복, evidence 복사)
SP_SERVER_VLM_PROVIDER=anthropic SP_SERVER_DATA_DIR=data/server-eval sceneproof-server run   # 해석 완료까지 대기
python -m sceneproof.eval.vlm_eval --data-dir data/server-eval --events-jsonl /tmp/eval-bundle/events.jsonl --out docs/eval/raw/vlm_eval.json   # 같은 실행의 사건을 정답과 매칭
python scripts/eval_report.py
```
