#!/usr/bin/env python
"""Render docs/eval/README.md tables from the raw JSON results so numbers in the docs are never typed by hand.

    python scripts/eval_report.py   # reads docs/eval/raw/*.json and docs/evidence/**, writes docs/eval/README.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "docs/eval/raw"


def load(name: str):
    p = RAW / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=1):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def event_section() -> str:
    out = ["## 1. 사건 단위 평가 (held-out 영상 4편, 정답은 draft)", "",
           "매칭 규칙·분모: `docs/eval/ground-truth.md`. 실행: `python -m sceneproof.eval.run_eval --variant <v>`; 재채점: `rescore.py`.", "",
           "| 모델(ONNX, conf 0.4) | precision | recall | F1 | FP/h | TP | FP | 놓침 | 중복 | 경계·가림 겹침 | 오프라인 처리 fps(평균) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for v in ("yolox_nano", "yolox_tiny", "yolox_s"):
        d = load(f"event_eval_{v}.json")
        if not d:
            continue
        a = d["aggregate"]
        fps = np.mean([s["run"]["fps"] for s in d["per_video"].values()])
        out.append(f"| {v} | {a['precision']} | {a['recall']} | {a['f1']} | {fmt(a['false_alarms_per_hour'], 2)} | {a['tp']} | {a['fp']} | {a['missed']} | {a['duplicates']} | {a['boundary_hits']} | {fps:.1f} |")
        rows.append((v, d))
    out += ["", "오프라인 처리 fps는 평가 실행 중 다른 작업(다른 모델 평가·벤치마크)과 CPU를 나눠 쓴 값이라 참고용입니다. 추론 성능 비교는 §2를 보세요.", "", "정답 양성(intrusion) 사건 수: " + ", ".join(f"{cam} {s['gt_positives']}" for cam, s in rows[0][1]["per_video"].items()) if rows else "", "",
            "### 영상별 (yolox_tiny — 기본 설정)", "",
            "| 영상 | P | R | TP | FP | 놓침 | 중복 | 경계·가림 | 시작 MAE(s) | 종료 MAE(s) | 체류 판정 일치/불일치 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    tiny = load("event_eval_yolox_tiny.json")
    if tiny:
        for cam, s in tiny["per_video"].items():
            dw = s["dwell"]
            out.append(f"| {cam} | {s['precision']} | {s['recall']} | {s['tp']} | {s['fp']} | {s['missed']} | {s['duplicates']} | {s['boundary_hits']} | {fmt(s['start_mae_s'], 2)} | {fmt(s['end_mae_s'], 2)} | {dw['tp_dwell_correct']}/{dw['tp_dwell_wrong']} |")
    out += ["", "### 모델별 차이가 드러난 곳", ""]
    for v, d in rows:
        notes = []
        for cam, s in d["per_video"].items():
            if s["missed"]:
                notes.append(f"{cam}: 놓침 {s['details']['missed']}")
            if s["duplicates"]:
                notes.append(f"{cam}: 중복 {s['duplicates']}건")
        out.append(f"- **{v}**: " + ("; ".join(notes) if notes else "놓침·중복 없음"))
    out += ["", "해석: 중복(duplicates)은 같은 정답 사건이 추적 ID 전환·추적 끊김으로 둘 이상의 사건으로 쪼개진 횟수입니다. 놓침은 소형 원거리 객체(주차장 자전거)에서 nano가 사람을 탐지하지 못한 경우입니다. 경계·가림 겹침은 정답에서 `boundary_pass`/`occluded`로 표시한 구간과 겹친 예측으로, 오탐으로 세지 않되 숨기지도 않습니다.", ""]
    return "\n".join(out)


def bench_section() -> str:
    d = load("bench_detector.json")
    if not d:
        return "## 2. 추론 성능 (PyTorch vs ONNX Runtime)\n\n(아직 실행 전)\n"
    env = d["env"]
    out = ["## 2. 추론 성능 — PyTorch vs ONNX Runtime (같은 프레임·같은 전후처리·같은 상자 기준)", "",
           f"환경: {env['cpu_model']} ×{env['cpu_count']} vCPU, {env['platform']}, Python {env['python']}, torch {env['torch']}, onnxruntime {env['onnxruntime']}. "
           f"입력 {env['video']} 앞 {env['frames']}프레임({env['frame_wh'][0]}×{env['frame_wh'][1]}), 워밍업 10프레임, **{env['repeats']}회 반복의 중앙값**. 원시 per-frame 시간은 `raw/bench_detector.json`.", "",
           "| 모델 | 스레드 | 설정 | infer p50 (ms) | infer p95 (ms) | 전체 p50 (ms) | 전체 p95 (ms) | 전처리 p50 | 후처리 p50 | fps | CPU % | RSS MB |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def med(r, k, q):
        return np.median([x["stages_ms"][k][q] for x in r["runs"]])

    for r in d["results"]:
        if r["setting"] == "parity":
            continue
        out.append(f"| {r['variant']} | {r['threads']} | {r['setting']} | {med(r, 'infer', 'p50'):.1f} | {med(r, 'infer', 'p95'):.1f} | {med(r, 'total', 'p50'):.1f} | {med(r, 'total', 'p95'):.1f} | {med(r, 'preprocess', 'p50'):.2f} | {med(r, 'postprocess', 'p50'):.2f} | {np.median([x['fps'] for x in r['runs']]):.1f} | {np.median([x['cpu_percent'] for x in r['runs']]):.0f} | {np.median([x['rss_mb'] for x in r['runs']]):.0f} |")
    out += ["", "반복 간 변동(infer p50, 3회의 최소–최대):", ""]
    for r in d["results"]:
        if r["setting"] == "parity":
            continue
        vals = [x["stages_ms"]["infer"]["p50"] for x in r["runs"]]
        out.append(f"- {r['variant']} thr{r['threads']} {r['setting']}: {min(vals):.1f}–{max(vals):.1f} ms")
    out += ["", "### 변환 전후 출력 일치 (NMS 이후 상자·점수 비교)", "", "| 모델 | 스레드 | 비교 | 프레임 수 | 상자 수가 다른 프레임 | 매칭 상자 | 비매칭 상자 | 최대 좌표 차(px) | 최대 점수 차 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in d["results"]:
        if r["setting"] != "parity":
            continue
        for k, p in r["parity"].items():
            out.append(f"| {r['variant']} | {r['threads']} | {k} | {p['frames']} | {p['frames_with_count_diff']} | {p['matched_boxes']} | {p['unmatched_boxes']} | {p['max_box_delta_px']} | {p['max_score_delta']} |")
    out.append("")
    return "\n".join(out)


def streams_section() -> str:
    d = load("bench_streams.json")
    if not d:
        return "## 3. 스트림 수 증가 (RTSP 시뮬레이터)\n\n(아직 실행 전)\n"
    out = ["## 3. 스트림 수 증가 — 한 분석기 프로세스, RTSP 시뮬레이터 입력", "",
           f"{d['variant']}, intra-op {d['threads']}스레드, {d['cpu_count']} vCPU, 각 설정 {d['seconds']}초 관측(워밍업 8초 제외). 입력은 샘플 영상의 실제 fps(10–30).", "",
           "| 스트림 수 | 입력 fps/스트림 | 처리 fps/스트림 | 처리 fps 합계 | 지연 p50 (ms) | 지연 p95 (ms) | 평균 큐 깊이 | 폐기 프레임(합) | CPU % | RSS 최대 MB | 사건 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d["results"]:
        s = r["summary"]
        out.append(f"| {s['streams']} | {s['input_fps_mean_per_stream']:.1f} | {s['processed_fps_mean_per_stream']:.1f} | {s['processed_fps_total']:.1f} | {s['lag_ms_p50']:.0f} | {s['lag_ms_p95']:.0f} | {s['queue_depth_mean']:.1f} | {s['dropped_total']} | {s['cpu_percent_mean']:.0f} | {s['rss_mb_max']:.0f} | {s['events']} |")
    out.append("")
    return "\n".join(out)


def scenarios_section() -> str:
    p = ROOT / "docs/evidence/failure-scenarios/summary.json"
    if not p.exists():
        return "## 4. 장애 시나리오\n\n(아직 실행 전)\n"
    d = json.loads(p.read_text())
    out = [f"## 4. 장애 시나리오 (`scripts/failure_scenarios.py`, git {d['git']}, {d['time']})", "", "| 시나리오 | 결과 | 확인 항목 | 요약 |", "|---|---|---|---|"]
    for name, s in d["scenarios"].items():
        checks = ", ".join(f"{k}={'✓' if v else ('—' if v is None else '✗')}" for k, v in s["checks"].items())
        out.append(f"| {name} | {'PASS' if s['ok'] else 'FAIL'} | {checks} | {json.dumps(s.get('summary'), ensure_ascii=False)[:300]} |")
    out.append("")
    return "\n".join(out)


def vlm_section() -> str:
    d = load("vlm_eval.json")
    if not d:
        return ("## 5. CV만 vs CV+VLM\n\n"
                "- **로컬 소형 VLM(SmolVLM-500M-Instruct, CPU)**: 네트워크 허용 후 실제로 실행했으나 구조화 출력·장면 기술 모두 신뢰할 수 없어 "
                "비교 수치로 쓰지 않았습니다(이미지 분할 기본값에서는 1건에 9분 초과, 분할을 끄면 25초지만 스키마 위반·예시 복제). "
                "시도별 원시 출력: `docs/evidence/vlm-local/`.\n"
                "- **API 모델(Claude)**: 유료이므로 사용자 승인 후 `bash scripts/run_vlm_eval.sh`(제공자 anthropic, 예산 상한 환경변수)로 실행하면 "
                "이 절이 자동으로 채워집니다. 추정 비용은 `docs/handoff.md` §4.\n"
                "- 모의(mock) 제공자 결과는 품질 수치로 보고하지 않습니다.\n")
    out = [f"## 5. CV만 vs CV+VLM ({d['provider']} / {d['model']} / prompt {d['prompt_version']})", "",
           (f"- 사건–정답 매칭: {d['matching']['matched']}/{d['matching']['events']}건 매칭 (CV 결과별 {json.dumps(d['matching']['by_cv_outcome'])}, "
            f"CV가 놓친 정답 침입 {d['matching']['gt_intrusions_missed_by_cv']}건 — VLM은 CV가 만든 사건만 볼 수 있음)" if d.get("matching") else
            "- 사건–정답 매칭: (이 결과 파일에는 매칭 정보가 없음 — 이전 형식)"),
           f"- 해석 상태 분포: {json.dumps(d['explanations_by_status'])}", f"- 해석 지연 p50/p95: {fmt(d['latency_ms']['p50'],0)} / {fmt(d['latency_ms']['p95'],0)} ms (n={d['latency_ms']['n']}), 추정 비용 {d['estimated_cost_usd']} USD",
           f"- CV만: 모든 사건이 '확인 필요' ({d['cv_only']['events_needing_review']}건; 그중 정답 침입 {d['cv_only']['true_intrusions_flagged']}, CV 오탐 {d['cv_only']['cv_false_alarms_flagged']})",
           f"- CV+VLM: 정답 침입을 '확인 필요'로 유지 {d['cv_plus_vlm']['true_intrusions_kept_as_needs_attention']}, CV 오탐을 '정상 가능성 높음'으로 표시 {d['cv_plus_vlm']['cv_false_alarms_marked_likely_benign']}",
           f"- assessment × CV 결과: `{json.dumps(d['cv_plus_vlm']['assessment_by_cv_outcome'], ensure_ascii=False)}`",
           f"- 사람 채점 루브릭: `{d['human_rubric_csv']}` (채점 전)"]
    a = d["cv_plus_vlm"]["assessment_by_cv_outcome"]
    tp = a.get("tp", {})
    tp_total = sum(tp.values())
    tp_benign = tp.get("likely_benign", 0)
    if tp_total:
        out.append(f"- 읽는 법: VLM의 '정상 가능성 높음'을 자동 종결에 썼다면 정답 침입 {tp_benign}/{tp_total}건이 사람 눈에 띄지 않았을 것입니다. "
                   "그래서 제품은 VLM 판단을 보조 표시로만 쓰고 모든 사건을 '확인 필요'로 유지합니다(decisions D7). 사례별 소견과 프롬프트 개선안: `docs/eval/vlm-findings.md`.")
    out.append("")
    return "\n".join(out)


def main() -> int:
    head = ["# 평가·실험 결과", "", "모든 표는 `scripts/eval_report.py`가 `docs/eval/raw/*.json`과 `docs/evidence/`에서 생성합니다(손으로 적은 숫자 없음). "
            "목표치가 아니라 **실측**이며, 측정 환경은 각 절에 적습니다. 데이터·분할·정답: `data/README.md`, `docs/eval/ground-truth.md`.", ""]
    body = [event_section(), bench_section(), streams_section(), scenarios_section(), vlm_section()]
    tail = ["## 재현 절차", "", "```bash", "bash scripts/fetch_data.sh && bash scripts/fetch_models.sh",
            "# 1 사건 평가", "for v in yolox_nano yolox_tiny yolox_s; do python -m sceneproof.eval.run_eval --variant $v --out docs/eval/raw/event_eval_$v.json; done",
            "# 2 추론 성능 (조용한 CPU에서)", "python -m sceneproof.eval.bench_detector --video data/videos/campus-walkway.mp4 --frames 150 --repeats 3 --threads 4,1 --out docs/eval/raw/bench_detector.json",
            "# 3 스트림 수", "python -m sceneproof.tools.rtsp_sim --dir data/videos --port 8554 &  python -m sceneproof.eval.bench_streams --streams 1,2,3,4 --out docs/eval/raw/bench_streams.json",
            "# 4 장애 시나리오", "python scripts/failure_scenarios.py --out docs/evidence/failure-scenarios",
            "# 5 CV vs CV+VLM: 평가 영상을 서버에 재생 → 실제 VLM 제공자로 해석 → 비교",
            "sceneproof-edge --config configs/eval.yaml offline --camera eval-campus --out /tmp/eval-bundle/events.jsonl   # (4편 반복, evidence 복사)",
            "SP_SERVER_VLM_PROVIDER=anthropic SP_SERVER_DATA_DIR=data/server-eval sceneproof-server run   # 해석 완료까지 대기",
            "python -m sceneproof.eval.vlm_eval --data-dir data/server-eval --events-jsonl /tmp/eval-bundle/events.jsonl --out docs/eval/raw/vlm_eval.json   # 같은 실행의 사건을 정답과 매칭",
            "python scripts/eval_report.py", "```", ""]
    (ROOT / "docs/eval/README.md").write_text("\n".join(head + body + tail))
    print("docs/eval/README.md written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
