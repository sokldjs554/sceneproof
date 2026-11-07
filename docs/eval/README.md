# 평가·실험 결과

모든 표는 `scripts/eval_report.py`가 `docs/eval/raw/*.json`과 `docs/evidence/`에서 생성합니다(손으로 적은 숫자 없음). 목표치가 아니라 **실측**이며, 측정 환경은 각 절에 적습니다. 데이터·분할·정답: `data/README.md`, `docs/eval/ground-truth.md`.

## 1. 사건 단위 평가 (held-out 영상 4편, 정답은 draft)

매칭 규칙·분모: `docs/eval/ground-truth.md`. 실행: `python -m zonewatch.eval.run_eval --variant <v>`; 재채점: `rescore.py`.

| 모델(ONNX, conf 0.4) | precision | recall | F1 | FP/h | TP | FP | 놓침 | 중복 | 경계·가림 겹침 | 오프라인 처리 fps(평균) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolox_nano | 1.0 | 0.8 | 0.889 | 0.00 | 8 | 0 | 2 | 3 | 3 | 70.2 |
| yolox_tiny | 1.0 | 1.0 | 1.0 | 0.00 | 10 | 0 | 0 | 5 | 3 | 18.1 |
| yolox_s | 1.0 | 1.0 | 1.0 | 0.00 | 10 | 0 | 0 | 2 | 3 | 12.1 |

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

## 2. 추론 성능 (PyTorch vs ONNX Runtime)

(아직 실행 전)

## 3. 스트림 수 증가 (RTSP 시뮬레이터)

(아직 실행 전)

## 4. 장애 시나리오

(아직 실행 전)

## 5. CV만 vs CV+VLM

실제 VLM 제공자 실행은 사용자 승인(유료 API) 또는 네트워크 허용(로컬 모델) 후 진행합니다. 실행 명령: `docs/eval/README.md` 하단 재현 절차 §5. 모의(mock) 제공자 결과는 품질 수치로 보고하지 않습니다.

## 재현 절차

```bash
bash scripts/fetch_data.sh && bash scripts/fetch_models.sh
# 1 사건 평가
for v in yolox_nano yolox_tiny yolox_s; do python -m zonewatch.eval.run_eval --variant $v --out docs/eval/raw/event_eval_$v.json; done
# 2 추론 성능 (조용한 CPU에서)
python -m zonewatch.eval.bench_detector --video data/videos/campus-walkway.mp4 --frames 150 --repeats 3 --threads 4,1 --out docs/eval/raw/bench_detector.json
# 3 스트림 수
python -m zonewatch.tools.rtsp_sim --dir data/videos --port 8554 &  python -m zonewatch.eval.bench_streams --streams 1,2,3,4 --out docs/eval/raw/bench_streams.json
# 4 장애 시나리오
python scripts/failure_scenarios.py --out docs/evidence/failure-scenarios
# 5 CV vs CV+VLM: 평가 영상을 서버에 재생 → 실제 VLM 제공자로 해석 → 비교
zonewatch-edge --config configs/eval.yaml offline --camera eval-campus --out /tmp/eval-bundle/events.jsonl   # (4편 반복, evidence 복사)
ZW_SERVER_VLM_PROVIDER=anthropic ZW_SERVER_DATA_DIR=data/server-eval zonewatch-server run   # 해석 완료까지 대기
python -m zonewatch.eval.vlm_eval --data-dir data/server-eval --event-eval docs/eval/raw/event_eval_yolox_tiny.json --out docs/eval/raw/vlm_eval.json
python scripts/eval_report.py
```
