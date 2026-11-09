"""Renders docs/eval/bench/README.md from docs/eval/bench/raw/*.json (no hand-written numbers).

Sections: data & protocol, dev baseline (error analysis that motivated fine-tuning), fine-tuning log,
final test (measured once; per site, seen vs unseen sites, detection AP), human agreement status.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "docs" / "eval" / "bench"
RAW = BENCH / "raw"
SITE_KO = {"caviar-lobby": "실내 로비(위에서 내려다봄)", "caviar-mall-cor": "쇼핑몰 복도(복도 방향)",
           "caviar-mall-front": "쇼핑몰 매장 정면", "mot16-02-square": "광장(사람 눈높이) — 처음 보는 장소",
           "mot16-09-lowangle": "거리(낮은 각도) — 처음 보는 장소", "mot20-station": "기차역 실내(혼잡) — 처음 보는 장소",
           "mot16-04-street": "거리(높은 시점, 밤)", "seen_sites": "본 적 있는 장소 합계", "unseen_sites": "처음 보는 장소 합계",
           "all": "전체"}


def load(name: str):
    p = RAW / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def events_table(results: dict, keys: list[str]) -> list[str]:
    names = list(results)
    out = ["| 장소 | 영상 길이(분) | 정답 사건 | " + " | ".join(f"{n}: 찾음 / 놓침 / 잘못 울림 / 중복" for n in names) + " | "
           + " | ".join(f"{n}: 정밀도·재현율·F1 / 시간당 잘못 울림" for n in names) + " |",
           "|---|---|---|" + "---|" * (2 * len(names))]
    for k in keys:
        first = results[names[0]]["events"].get(k)
        if not first:
            continue
        cells = [SITE_KO.get(k, k), f(first["duration_s"] / 60, 1), str(first["gt_intrusions"])]
        a, b = [], []
        for n in names:
            e = results[n]["events"].get(k, {})
            a.append(f"{e.get('tp')} / {e.get('missed')} / {e.get('fp')} / {e.get('duplicates')}")
            b.append(f"{f(e.get('precision'))} · {f(e.get('recall'))} · {f(e.get('f1'))} / {f(e.get('false_alarms_per_hour'), 1)}")
        out.append("| " + " | ".join(cells + a + b) + " |")
    return out


def detection_table(results: dict, keys: list[str]) -> list[str]:
    names = list(results)
    out = ["| 장소 | 평가 프레임 / 정답 상자 | " + " | ".join(f"{n}: AP50 / AP50:95" for n in names) + " | "
           + " | ".join(f"{n}: 0.4 기준 재현율 (작음 <50px · 중간 · 큼)" for n in names) + " |",
           "|---|---|" + "---|" * (2 * len(names))]
    for k in keys:
        d0 = results[names[0]]["detection"].get(k)
        if not d0 or not d0.get("gt_boxes"):
            continue
        cells = [SITE_KO.get(k, k), f"{d0['frames']} / {d0['gt_boxes']}"]
        a, b = [], []
        for n in names:
            d = results[n]["detection"].get(k, {})
            a.append(f"{f(d.get('ap50'))} / {f(d.get('ap50_95'))}")
            r = d.get("recall_at_op_by_height", {})
            b.append(" · ".join(f(r.get(h)) for h in ("h<50px", "50-100px", "h>=100px")))
        out.append("| " + " | ".join(cells + a + b) + " |")
    return out


def timing_by_close(results: dict) -> list[str]:
    """End-time error of matched events, split by how the predicted event was closed (computed from per-sequence
    records, so it works for result files written before this breakdown existed)."""
    import statistics

    close_ko = {"exited": "사람이 구역을 나가서 닫힘", "track_lost": "사람을 놓쳐서 닫힘", "stream_ended": "영상이 끝나서 닫힘"}
    names = list(results)
    out = ["| 사건이 닫힌 이유 | " + " | ".join(f"{n}: 건수 / 끝 시각 차이 중앙값(초) / 90% 범위(초)" for n in names) + " |",
           "|---|" + "---|" * len(names)]
    rows: dict[str, list[str]] = {}
    for n in names:
        by: dict[str, list[float]] = {}
        for e in results[n]["per_sequence"].values():
            if "outcome" not in e:
                continue
            pred = {x["id"]: x for x in e["predictions"]}
            gt = {x["id"]: x for x in e["gt_events"]}
            for pid, gid in e["outcome"]["tp"]:
                by.setdefault(pred[pid]["close"], []).append(pred[pid]["end_s"] - gt[gid]["end_s"])
        for k in close_ko:
            v = by.get(k, [])
            if not v:
                rows.setdefault(k, []).append("—")
                continue
            a = sorted(abs(x) for x in v)
            rows.setdefault(k, []).append(f"{len(v)} / {statistics.median(v):+.2f} / {a[min(len(a) - 1, int(0.9 * len(a)))]:.2f}")
    for k, cells in rows.items():
        out.append(f"| {close_ko[k]} | " + " | ".join(cells) + " |")
    return out


def followup_section() -> list[str]:
    """Section 5-2..: what was fixed after the first final measurement, the DEV evidence for each choice, and the
    second held-out measurement (MOT20-03/05) that compares the product before and after."""
    L: list[str] = []
    trk = load("bench_dev_tracking.json")
    if trk:
        merged = {}
        for r in trk["runs"]:
            merged.update(r["results"])
        names = {"base": "현재 제품 (점수 0.4 이상 상자만 추적)", "base-ex": "+ 마네킹 제외 영역",
                 "low": "낮은 점수 상자(0.1 이상)도 추적기에 넣기", "low-rj10": "낮은 점수 + 이어받기 1.0",
                 "low-rj05": "낮은 점수 + 이어받기 0.5", "rj05": "이어받기 0.5", "rj075": "이어받기 0.75",
                 "rj10": "이어받기 1.0", "ex-rj10": "제외 영역 + 이어받기 1.0"}
        L += ["## 8. 측정 뒤 보완 (개발 일지 L19·L20)", "",
              "### 8-1. 추적 끊김·마네킹 대응: 조정 분할에서 고르기", "",
              "기본 작은 모델(tiny-coco)로 조정 분할 12편(정답 사건 70건)을 돌려 비교했습니다. 이어받기 = 추적기가 같은 사람에게 새 번호를 "
              "붙였을 때, 사람이 사라진 뒤 예상 위치(걷던 속도로 옮긴 위치)에서 상자 높이 × 숫자 안에 나타난 새 추적이 열린 사건을 이어받음. "
              "제외 영역 = 매장 정면의 마네킹 자리(조정 분할 잘못 울림 4건이 모두 같은 자리). 둘 다 프로그램 쪽에만 적용하고 정답에는 적용하지 않습니다.", "",
              "| 설정 | 찾음 | 놓침 | 잘못 울림 | 중복(쪼개짐) |", "|---|---|---|---|---|"]
        for k in ("base", "base-ex", "low", "low-rj10", "low-rj05", "rj05", "rj075", "rj10", "ex-rj10"):
            e = merged.get(k, {}).get("events", {}).get("all")
            if e:
                L.append(f"| {names[k]} | {e['tp']} | {e['missed']} | {e['fp']} | {e['duplicates']} |")
        L += ["", "고른 것: **이어받기 1.0 + 제외 영역**(설치 때 사람 모양 물체가 있으면 그림). 낮은 점수 상자를 추적기에 넣는 방식은 "
              "잘못 울림이 늘어(매장 정면 4→7) 쓰지 않습니다. 1회째 비교는 커밋 `50a826c`로 "
              "올린 코드를 커밋하기 직전에 돌려 결과 파일에 '커밋 안 된 변경'이 적혀 있습니다(같은 내용).", ""]
    ft2 = load("finetune_log_v2.json")
    if ft2:
        ft1 = load("finetune_log.json")
        L += ["### 8-2. 추가 학습 다시 하기 (학습 정답 상자를 사진 범위로 자르도록 고친 뒤)", "",
              "같은 데이터·설정으로 다시 학습했습니다. 조정 분할 사람 찾기 점수(AP50:95):", "",
              "| 회차 | 처음 학습 | 다시 학습 |", "|---|---|---|"]
        e1 = {e["epoch"]: e for e in (ft1 or {}).get("epochs", [])}
        for e in ft2["epochs"]:
            mark = " ← 선택" if e["epoch"] == ft2.get("best_epoch") else ""
            L.append(f"| {e['epoch']}{mark} | {f(e1.get(e['epoch'], {}).get('dev', {}).get('ap50_95'))} | {f(e['dev'].get('ap50_95'))} |")
        L += [""]
    dv2 = load("bench_dev_v2.json")
    if dv2:
        L += ["조정 분할에서 다시 학습한 모델(tiny-ft2)과 이어받기·제외 영역을 함께 쓴 결과:", ""]
        L += events_table(dv2["results"], ["caviar-lobby", "caviar-mall-cor", "caviar-mall-front", "all"])
        L += [""]
        L += detection_table(dv2["results"], ["caviar-lobby", "caviar-mall-cor", "caviar-mall-front", "all"])
        L += [""]
    t2 = load("bench_test2_final.json")
    if t2:
        lock2 = json.loads((BENCH / "test2_lock.json").read_text()) if (BENCH / "test2_lock.json").exists() else {}
        run = (lock2.get("runs") or [{}])[0]
        L += ["### 8-3. 두 번째 최종 측정 (새 장면 2곳, 한 번만)", "",
              "첫 최종 측정 결과를 본 뒤에 바꾼 것(이어받기, 다시 학습)을 첫 최종 분할로 다시 재면 공정하지 않습니다. 그래서 어디에도 쓰지 않은 "
              "MOT20-03(밤 광장)과 MOT20-05(아주 붐비는 광장)를 `configs/bench/split2.yaml`로 먼저 고정하고, 구역과 후보"
              "(`configs/bench/final2.yaml`)도 모델을 돌리기 전에 커밋한 뒤 한 번 쟀습니다.", "",
              f"측정 {t2.get('started', '?')} ~ {t2['time']}, 시작 커밋 `{t2['git']}`, 분할 해시 `{t2['split_sha256_16']}`, "
              f"구역 파일 해시 `{t2['zones_sha12']}`, 측정 기록 `docs/eval/bench/test2_lock.json` (측정 시작 {len(lock2.get('runs', []))}회, "
              f"이유: {run.get('reason', '?')}).", "", "#### 구역 사건", ""]
        keys2 = ["mot20-03-plaza", "mot20-05-square", "all"]
        L += events_table(t2["results"], keys2)
        L += ["", "#### 사건이 끝난 시각의 정확도 (맞힌 사건만)", ""]
        L += timing_by_close(t2["results"])
        L += ["", "#### 사람 찾기 (상자 단위, 정답 상자는 화면 안으로 자름)", ""]
        L += detection_table(t2["results"], keys2)
        L += [""]
    return L


def final_provenance(final: dict, lock: dict | None) -> str:
    """Which code ran the final measurement. Result files written before `started` was recorded stamp the HEAD at save
    time (a later commit can land during an hour-long run); the lock entry records the commit at the start."""
    run = next((r for r in (lock or {}).get("runs", []) if r.get("result_file", "").endswith("bench_test_final.json")), None)
    n = len(lock["runs"]) if lock else "?"
    if "started" in final:
        when = f"측정 {final['started']} ~ {final['time']}, 시작 커밋 `{final['git']}`"
        if final.get("uncommitted_changes"):
            when += f" (커밋 안 된 변경 {len(final['uncommitted_changes'])}개 있음)"
    elif run:
        when = (f"측정 {run['opened']} ~ {run.get('completed', final['time'])}, 시작 커밋 `{run['git']}` "
                f"(결과 파일의 `git: {final['git']}` 값은 저장 시점의 최신 커밋입니다. 측정 중에 들어간 커밋은 결과 저장 방식과 "
                f"문서만 바꿨고 계산 코드는 그대로였습니다)")
    else:
        when = f"측정 {final['time']}, 커밋 `{final['git']}`"
    return (f"{when}, 분할 해시 `{final['split_sha256_16']}`, 구역 파일 해시 `{final['zones_sha12']}`. "
            f"측정 기록: `docs/eval/bench/test_lock.json` (측정 시작 {n}회, 이유는 파일에 기록).")


def main() -> int:
    src = yaml.safe_load((ROOT / "configs" / "bench" / "sources.yaml").read_text())
    split = yaml.safe_load((ROOT / "configs" / "bench" / "split.yaml").read_text())
    man_p = ROOT / "data" / "bench" / "manifest.json"
    L = ["# 연구용 벤치마크: 사람이 표시한 정답으로 잰 구역 사건 성능과 추가 학습 효과", "",
         "이 문서는 `scripts/bench_report.py`가 `docs/eval/bench/raw/*.json`에서 만듭니다. 손으로 적은 숫자는 없습니다.", "",
         "## 1. 데이터와 정답", "",
         "| 데이터 | 라이선스 | 정답을 만든 방법 | 출처 표기 |", "|---|---|---|---|"]
    for k, d in src["datasets"].items():
        L.append(f"| {k} | {d['license']} | {d['ground_truth']} | {d['attribution']} |")
    L += ["", "원본 영상은 저장소에 올리지 않습니다. `python scripts/fetch_bench.py`가 원래 사이트에서 내려받아 다시 인코딩합니다"
          "(영상 프레임 번호 = 정답 프레임 번호가 되도록). 제외한 데이터와 이유는 `configs/bench/sources.yaml`의 `excluded`에 있습니다.", "",
          "**사건 정답은 사람이 그린 상자에서 만듭니다.** 데이터셋 작성자들이 프레임마다 손으로 그린 사람 상자를, 제품과 똑같은 구역 판정 코드"
          "(발 위치가 구역 안에 0.3초 이상 → 들어감, 0.8초 이상 밖 → 나감, 5초 이상 → 체류)에 넣습니다. 그래서 정답과 프로그램의 차이는 "
          "\"상자를 누가 그렸는가(사람 vs 탐지기+추적기)\" 하나뿐입니다. 구역 선 근처를 스친 경우는 '경계 통과'로 따로 두어 맞음도 틀림도 아닌 것으로 셉니다.", "",
          "**프로그램이 낸 사건이 정답과 같은 사건인지**는 시간(겹침 0.5초 이상 또는 시작 1초 이내)과 사람(같은 순간 발 위치 차이가 그 사람 키의 절반 이하)을 "
          "모두 봅니다. 사람이 많은 장면에서 시간만 보면 다른 사람의 사건으로 점수를 받는 일을 막기 위해서입니다.", "",
          "## 2. 분할 — 결과를 보기 전에 고정", "",
          f"`configs/bench/split.yaml` (시드 {split['seed']}, 해시 `{split['sha256_16']}`, 생성 {split['created']}). 이 파일은 구역을 그리거나 "
          "어떤 측정도 하기 전에 커밋했습니다(git 기록으로 확인 가능). 같은 장면의 두 시점(CAVIAR 쇼핑몰 복도/정면)은 항상 같은 분할에 둡니다.", "",
          "| 분할 | 용도 | 영상 수 | 장소 |", "|---|---|---|---|"]
    use = {"train": "추가 학습에만 사용", "dev": "오류 분석, 학습 설정·회차 선택", "test": "최종 측정 1회 (잠금 파일로 강제)"}
    for part, ids in split["sequences"].items():
        sites = sorted({s["site"] for s in src["sequences"] if s["id"] in ids})
        L.append(f"| {part} | {use[part]} | {len(ids)} | {', '.join(SITE_KO.get(s, s) for s in sites)} |")
    L += ["", f"처음 보는 장소(학습·조정 어디에도 없음): {', '.join(SITE_KO.get(s, s) for s in split['unseen_sites'])}.", ""]
    if man_p.exists():
        man = json.loads(man_p.read_text())["sequences"]
        tot = {p: sum(man.get(i, {}).get("duration_s", 0) for i in ids) / 60 for p, ids in split["sequences"].items()}
        L += [f"영상 길이 합계: 학습 {tot['train']:.1f}분 · 조정 {tot['dev']:.1f}분 · 최종 {tot['test']:.1f}분.", ""]

    dev = load("bench_dev_baseline.json")
    if dev:
        L += ["## 3. 조정 영상에서 찾은 문제 (추가 학습을 한 이유)", "",
              f"측정 시각 {dev['time']}, 커밋 `{dev['git']}`. 사람 찾기 정확도(상자 단위, 5프레임마다):", ""]
        L += detection_table(dev["results"], ["caviar-lobby", "caviar-mall-cor", "caviar-mall-front", "all"])
        L += ["", "구역 사건 단위:", ""]
        L += events_table(dev["results"], ["caviar-lobby", "caviar-mall-cor", "caviar-mall-front", "all"])
        L += [""]
    ft = None
    for p in sorted((ROOT / "models" / "finetune").glob("*/train_log.json")) if (ROOT / "models" / "finetune").exists() else []:
        ft = json.loads(p.read_text())
    ftc = load("finetune_log.json") or ft
    if ftc:
        d = ftc["data"]
        L += ["## 4. 추가 학습", "",
              f"학습 분할 영상 {len(d['train_sequences'])}편에서 사진 {d['train_images']}장(사람 상자 {d['train_boxes']}개), "
              f"조정 분할 {d['dev_frames']}프레임으로 회차 선택. 설정: `{json.dumps({k: ftc['args'][k] for k in ('variant', 'epochs', 'batch', 'lr', 'backbone_lr_mult', 'stride', 'mot_stride')})}`. "
              "CPU 4코어, GPU 없음.", "",
              "| 회차 | 학습 손실 | 조정 AP50 | 조정 AP50:95 | 걸린 시간(초) |", "|---|---|---|---|---|",
              f"| 학습 전 | — | {f(ftc['dev_before'].get('ap50'))} | {f(ftc['dev_before'].get('ap50_95'))} | — |"]
        for e in ftc["epochs"]:
            mark = " ← 선택" if e["epoch"] == ftc.get("best_epoch") else ""
            L.append(f"| {e['epoch']}{mark} | {f(e['total_loss'])} | {f(e['dev'].get('ap50'))} | {f(e['dev'].get('ap50_95'))} | {e['train_s']} |")
        L += [""]
    dev_ft = load("bench_dev_finetuned.json")
    if dev_ft:
        merged = {**(dev["results"] if dev else {}), **dev_ft["results"]}
        L += ["조정 영상에서 학습 전·후 비교(선택한 회차를 제품과 같은 ONNX 경로로 실행). tiny-coco = 제품 기본 모델, "
              "s-coco = 약 3배 큰 COCO 모델(추가 학습 없음), tiny-ft = 제품 기본 모델을 추가 학습한 것:", ""]
        L += detection_table(merged, ["caviar-lobby", "caviar-mall-cor", "caviar-mall-front", "all"])
        L += [""]
        L += events_table(merged, ["caviar-lobby", "caviar-mall-cor", "caviar-mall-front", "all"])
        L += [""]
    final = load("bench_test_final.json")
    lock = json.loads((BENCH / "test_lock.json").read_text()) if (BENCH / "test_lock.json").exists() else None
    if final:
        L += ["## 5. 최종 측정 (한 번만)", "",
              final_provenance(final, lock), "",
              "결과를 읽는 법과 소견(용어 설명, 장소 묶음별 해석, 한계, 다음 단계): [`findings.md`](findings.md).", "",
              "### 구역 사건", ""]
        keys = ["caviar-lobby", "caviar-mall-cor", "caviar-mall-front", "mot16-02-square", "mot16-09-lowangle", "mot20-station",
                "seen_sites", "unseen_sites", "all"]
        L += events_table(final["results"], keys)
        L += ["", "### 사건이 끝난 시각의 정확도 (맞힌 사건만)", "",
              "끝 시각 차이 = 프로그램이 정한 끝 − 정답의 끝(음수면 일찍 닫음). 사람이 나가서 닫힌 사건은 거의 정확하고, "
              "사람을 놓쳐서 닫힌 사건은 일찍 닫히는 쪽으로 몇 초까지 어긋납니다. 결과 파일의 `end_mae_s`(평균 끝 시각 오차)는 이 두 경우를 섞은 값입니다.", ""]
        L += timing_by_close(final["results"])
        L += ["", "### 사람 찾기 (상자 단위)", ""]
        L += detection_table(final["results"], [*keys[:6], "all"])
        L += ["", "정답 상자 일부는 화면 밖까지 그려져 있습니다(화면 가장자리에서 잘린 사람의 몸 전체). 프로그램의 상자는 화면 안으로 "
              "잘리므로, 화면 안쪽 부분과 겹침이 0.5 미만이라 어떤 상자로도 맞힐 수 없는 정답 상자가 최종 분할 40,503개 중 29개(0.07%)이고, "
              "그중 22개가 저각도 거리 영상(766개 중 2.9%)입니다. 이 영상의 AP50은 최대 약 0.03 낮게 나왔을 수 있습니다. "
              "점수 계산은 측정 당시 그대로 둡니다(다시 재지 않음).", ""]
    L += followup_section()
    L += ["## 6. 두 사람 표시 일치도", "",
          "도구: `tools/event_labeler/index.html`(브라우저로 열고 영상 선택), 계산: `python -m sceneproof.eval.bench.agreement A.json B.json --with-gt`.", ""]
    agr = sorted((BENCH / "agreement").glob("*.json")) if (BENCH / "agreement").exists() else []
    if agr:
        L += ["| 영상 | 비교 | 사건 F1 | 프레임 카파 | 들어감 시각 차이(중앙값, 초) |", "|---|---|---|---|---|"]
        for p in agr:
            r = json.loads(p.read_text())
            for pr in r["pairs"]:
                L.append(f"| {r['sequence']} | {pr['a']} vs {pr['b']} | {pr['event_f1']} | {pr['frame_kappa']} | {pr['median_entry_diff_s']} |")
    else:
        L += ["**아직 측정하지 않았습니다.** 사람 두 명이 같은 영상을 따로 표시해야 하므로 사용자가 해야 하는 작업입니다. 결과 JSON을 "
              "`docs/eval/bench/agreement/`에 넣고 이 스크립트를 다시 돌리면 이 절이 채워집니다."]
    L += ["", "## 7. 재현", "", "```bash",
          "python scripts/fetch_bench.py                      # 원 사이트에서 84편 다운로드·재인코딩 (약 2.3GB, 프레임 수 = 정답 프레임 수 검사)",
          "python -m sceneproof.eval.bench.run --split dev --candidate tiny-coco:models/yolox_tiny.onnx \\",
          "    --candidate s-coco:models/yolox_s.onnx:yolox_s --out docs/eval/bench/raw/bench_dev_baseline.json",
          "python -m sceneproof.train.finetune --variant yolox_tiny --epochs 10   # CPU 4코어 약 1.7시간, 조정 분할로 회차 선택",
          "python scripts/export_onnx.py --variant yolox_tiny --weights models/finetune/yolox_tiny_bench/best.pth \\",
          "    --out models/finetune/yolox_tiny_bench/best.onnx",
          "python -m sceneproof.eval.bench.run --split dev --candidate tiny-ft:models/finetune/yolox_tiny_bench/best.onnx:yolox_tiny \\",
          "    --out docs/eval/bench/raw/bench_dev_finetuned.json",
          "python -m sceneproof.eval.bench.run --split test --final   # configs/bench/final.yaml의 후보 전부, 한 번만 (이미 잠겨 있음)",
          "python scripts/bench_report.py",
          "```", "",
          "추가 학습은 스레드 순서 때문에 비트 단위로 같지 않을 수 있습니다. 최종 측정에 쓴 가중치는 `configs/bench/final.yaml`에 ONNX 해시로 적어 두었습니다.", ""]
    (BENCH / "README.md").write_text("\n".join(L) + "\n")
    print(f"wrote {BENCH / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
