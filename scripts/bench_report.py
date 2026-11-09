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
              f"측정 {final['time']}, 커밋 `{final['git']}`, 분할 해시 `{final['split_sha256_16']}`, 구역 파일 해시 `{final['zones_sha12']}`. "
              f"측정 기록: `docs/eval/bench/test_lock.json` (측정 {len(lock['runs']) if lock else '?'}회).", "",
              "결과를 읽는 법과 소견(용어 설명, 장소 묶음별 해석, 한계, 다음 단계): [`findings.md`](findings.md).", "",
              "### 구역 사건", ""]
        keys = ["caviar-lobby", "caviar-mall-cor", "caviar-mall-front", "mot16-02-square", "mot16-09-lowangle", "mot20-station",
                "seen_sites", "unseen_sites", "all"]
        L += events_table(final["results"], keys)
        L += ["", "### 사람 찾기 (상자 단위)", ""]
        L += detection_table(final["results"], [*keys[:6], "all"])
        L += [""]
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
