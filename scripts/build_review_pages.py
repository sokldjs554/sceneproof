"""Builds the two self-contained review pages a person fills in by clicking (no install, no server):

  tools/review/gt_review.html   — the open questions on the draft ground truth of the 4 product-eval clips
                                  (docs/eval/ground-truth.md "남은 검수 작업"), each with the moments to look at,
                                  the zone drawn on the frame, and answer buttons.
  tools/review/vlm_rubric.html  — the 19 Claude explanations (docs/eval/raw/vlm_eval-rubric.csv, prompt v1.3) next
                                  to the frames of the same moments, with the four rubric scores as buttons.

Both pages keep the answers in the browser and save them with a 내보내기 button (a JSON file). Rubric scores are written
into the rubric CSV with `python scripts/apply_rubric_scores.py <file>`; ground-truth answers are applied to
data/ground_truth/*.yaml by hand (each answer means a different edit), then `python -m sceneproof.eval.rescore`.

Where the rubric frames come from: the evidence images of the CI run that produced the explanations were not kept,
but event ids are deterministic (uuid5 of camera, zone, track id, entry frame). The (track id, entry frame) of every
rubric event was recovered by trying all values; the frames are then cut from the same video at the same frame
numbers (context_before = entry - pre-roll, entry, exit, context_after = exit + pre-roll, as the evidence writer
does). Exit frames come from re-running the same pipeline here (`--rerun`); for the two events that the re-run does
not reproduce (CPU differences), the page says that the exit moment is approximate.

    python scripts/build_review_pages.py --rerun data/runs/review-rerun   # runs the eval pipeline first if missing
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import subprocess
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tools" / "review"
GT_DIR = ROOT / "data" / "ground_truth"
CAMS = {"eval-campus": "campus-walkway.mp4", "eval-indoor": "people-indoor.mp4", "eval-store": "store-aisle.mp4",
        "eval-parking": "parking-lot.mp4"}

# --------------------------------------------------------------------------- ground-truth questions
NOW = " — 지금 정답"
GT_ITEMS = [
    ("campus-e01", "eval-campus", [0.2, 0.5, 0.9], "검은 재킷 보행자(e01)의 발이 잔디(빨간 구역) 안으로 들어갔나요?",
     ["선 위나 밖을 따라 걸음(경계 통과)" + NOW, "안으로 들어감(침입)", "판단 불가"]),
    ("campus-e04", "eval-campus", [74.7, 76.5, 78.5], "빨간 재킷 보행자(e04)의 발이 잔디 안으로 들어갔나요?",
     ["선 위나 밖을 따라 걸음(경계 통과)" + NOW, "안으로 들어감(침입)", "판단 불가"]),
    ("campus-e03-gap", "eval-campus", [57.0, 59.0, 61.0, 62.5],
     "자주색 재킷 보행자(e03)는 발이 화면 아래로 잘린 동안(58~62초)에도 잔디 위에 계속 있었다고 봐도 되나요?",
     ["계속 잔디 위(한 사건)" + NOW, "잔디 밖으로 나갔다 들어옴(두 사건)", "판단 불가"]),
    ("campus-third", "eval-campus", [52.5, 55.0, 57.5, 60.0],
     "이 시간대에 빨간 재킷·자주색 재킷 두 사람 말고 잔디에 발을 디딘 사람이 있나요?",
     ["없음" + NOW, "있음(사건을 더해야 함)", "판단 불가"]),
    ("indoor-e04-start", "eval-indoor", [44.9, 45.2, 45.5, 45.8, 46.1],
     "사람(e04)의 발이 빨간 구역 안에 처음 들어간 사진은 몇 번인가요? (지금 정답: ②, 45.2초)",
     ["①", "②" + NOW, "③", "④", "⑤", "판단 불가"]),
    ("store-gap", "eval-store", [42.0, 43.5, 45.0, 46.5, 48.0],
     "42.7~47.9초 사이에 그 사람이 구역(통로 안쪽) 밖으로 나갔나요?",
     ["나갔다 다시 들어옴(두 사건)" + NOW, "계속 구역 안(한 사건으로 합침)", "판단 불가"]),
    ("store-e03", "eval-store", [52.0, 56.0, 60.0, 64.0], "선반에 가려진 사람(e03)의 발이 구역 안에 있나요?",
     ["안(침입)", "밖", "가려서 판단 불가" + NOW]),
    ("store-green", "eval-store", [30.0, 38.0, 46.0, 54.0, 62.0],
     "오른쪽 선반 쪽 초록 상의 사람의 발이 구역에 닿은 적이 있나요?", ["없음" + NOW, "있음(사건을 더해야 함)", "판단 불가"]),
    ("store-zone-meaning", "eval-store", [35.0],
     "이 구역(통로 안쪽 재고 구역)을 '직원만 들어가는 곳'으로 볼까요? Claude는 구역 안의 고객을 보고 '정상일 가능성 높음'으로 판단했습니다.",
     ["직원 전용(고객이 들어가도 알림)" + NOW, "고객도 다니는 통로(이 구역은 평가에서 뺌)", "잘 모르겠음"]),
    ("parking-e02", "eval-parking", [44.4, 45.0, 45.6], "보행자(e02)의 발이 빨간 구역 안에 들어갔나요? (사진을 누르면 크게 보입니다)",
     ["선 위나 밖(경계 통과)" + NOW, "안으로 들어감(침입)", "판단 불가"]),
    ("parking-e01-start", "eval-parking", [26.4, 26.7, 27.0, 27.3, 27.6],
     "자전거 탄 사람(e01)이 구역 안에 처음 들어간 사진은? (지금 정답: ③, 27.0초)", ["①", "②", "③" + NOW, "④", "⑤", "판단 불가"]),
    ("parking-e01-end", "eval-parking", [28.5, 29.0, 29.5, 30.0, 30.5],
     "자전거 탄 사람(e01)이 구역 밖으로 나간(또는 화면에서 사라진) 첫 사진은? (지금 정답: ②, 29.0초) "
     "— CV가 28.8초부터 낸 사건을 '잘못 울림'으로 채점했는데, Claude는 이때 구역 안에 사람이 있다고 봤습니다.",
     ["①", "②" + NOW, "③", "④", "⑤", "판단 불가"]),
    ("parking-e03-start", "eval-parking", [45.2, 45.5, 45.8, 46.1, 46.4],
     "사람(e03)이 구역 안에 처음 들어간 사진은? (지금 정답: ③, 45.8초)", ["①", "②", "③" + NOW, "④", "⑤", "판단 불가"]),
    ("parking-e03-end", "eval-parking", [48.6, 48.9, 49.2, 49.5, 49.8],
     "사람(e03)이 구역 밖으로 나간 첫 사진은? (지금 정답: ③, 49.2초)", ["①", "②", "③" + NOW, "④", "⑤", "판단 불가"]),
]


def _zone(cam: str) -> list[tuple[float, float]]:
    gt = yaml.safe_load((GT_DIR / f"{cam}.yaml").read_text())
    return [tuple(p) for p in gt["zones"][0]["polygon"]]


def _frame(video: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx} of {video}")
    return img


def _draw(img: np.ndarray, poly, box=None, label: str = "") -> np.ndarray:
    h, w = img.shape[:2]
    pts = np.array([[x * w, y * h] for x, y in poly], np.int32)
    over = img.copy()
    cv2.fillPoly(over, [pts], (48, 59, 255))
    img = cv2.addWeighted(over, 0.15, img, 0.85, 0)
    cv2.polylines(img, [pts], True, (48, 59, 255), 3)
    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
        if label:
            cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    return img


def _b64(img: np.ndarray) -> str:
    _ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


PAGE_CSS = """
:root { --fg:#1d1d1f; --muted:#666; --line:#ddd; --bg:#fafafa; --card:#fff; --accent:#0a66c2; }
@media (prefers-color-scheme: dark) { :root { --fg:#eee; --muted:#aaa; --line:#444; --bg:#161616; --card:#222; --accent:#5aa0ff; } }
body { margin:0; font:15px/1.5 -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif; color:var(--fg); background:var(--bg); }
header { position:sticky; top:0; z-index:2; background:var(--card); border-bottom:1px solid var(--line); padding:10px 16px;
         display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
main { max-width:1100px; margin:0 auto; padding:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin-bottom:18px; }
.card.done { border-color:#2e7d32; }
.imgs { display:flex; gap:8px; overflow-x:auto; padding-bottom:6px; }
.imgs figure { margin:0; flex:0 0 auto; }
.imgs img { height:200px; border-radius:6px; cursor:zoom-in; display:block; }
.imgs img.big { height:auto; width:min(1000px, 92vw); cursor:zoom-out; }
figcaption { font-size:13px; color:var(--muted); text-align:center; }
.q { font-weight:600; margin:10px 0 6px; }
.opts { display:flex; gap:6px; flex-wrap:wrap; }
.opts button { border:1px solid var(--line); background:var(--bg); color:var(--fg); border-radius:18px; padding:6px 12px; cursor:pointer; font:inherit; }
.opts button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
textarea, input[type=text] { font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:6px; }
textarea { width:100%; box-sizing:border-box; margin-top:8px; min-height:38px; }
.muted { color:var(--muted); font-size:13px; }
.primary { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:8px 14px; font:inherit; cursor:pointer; }
.obs { white-space:pre-wrap; background:var(--bg); border-radius:6px; padding:8px; font-size:14px; }
"""

PAGE_JS = """
const KEY = document.body.dataset.key;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} progress(); };
document.querySelectorAll('.imgs img').forEach((im) => im.addEventListener('click', () => im.classList.toggle('big')));
document.querySelectorAll('.opts').forEach((g) => {
  const id = g.dataset.id, field = g.dataset.field;
  g.querySelectorAll('button').forEach((b) => {
    if ((state[id] || {})[field] === b.dataset.v) b.classList.add('on');
    b.addEventListener('click', () => {
      state[id] = state[id] || {}; state[id][field] = b.dataset.v;
      g.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b)); save();
    });
  });
});
document.querySelectorAll('textarea[data-id]').forEach((t) => {
  t.value = (state[t.dataset.id] || {})[t.dataset.field] || '';
  t.addEventListener('input', () => { state[t.dataset.id] = state[t.dataset.id] || {}; state[t.dataset.id][t.dataset.field] = t.value; save(); });
});
const nameBox = document.getElementById('who');
nameBox.value = state.__who || ''; nameBox.addEventListener('input', () => { state.__who = nameBox.value; save(); });
function progress() {
  let done = 0;
  document.querySelectorAll('.card[data-id]').forEach((c) => {
    const need = c.querySelectorAll('.opts').length, have = c.querySelectorAll('.opts button.on').length;
    c.classList.toggle('done', have === need); if (have === need) done++;
  });
  document.getElementById('progress').textContent = `${done} / ${document.querySelectorAll('.card[data-id]').length} 완료`;
}
progress();
document.getElementById('export').addEventListener('click', () => {
  const total = document.querySelectorAll('.card[data-id]').length;
  const left = total - document.querySelectorAll('.card.done').length;
  if (left && !confirm(`아직 ${left}개를 답하지 않았습니다. 지금까지만 저장할까요?`)) return;
  const { __who, ...answers } = state;
  const out = { page: KEY, who: __who || '', saved: new Date().toISOString(), answers };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: 'application/json' }));
  a.download = `${KEY}.${(__who || 'reviewer').replace(/[^\\w가-힣-]/g, '_')}.json`; a.click();
});
"""


def _page(title: str, key: str, intro: str, cards: list[str]) -> str:
    return (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{PAGE_CSS}</style></head><body data-key='{key}'>"
            f"<header><b>{html.escape(title)}</b><label>이름 <input type='text' id='who' size='10' placeholder='예: 홍길동'></label>"
            f"<span id='progress' class='muted'></span><button class='primary' id='export'>내보내기 (JSON)</button></header>"
            f"<main><div class='card'>{intro}</div>{''.join(cards)}</main><script>{PAGE_JS}</script></body></html>")


def _opts(item: str, field: str, choices: list[str]) -> str:
    return (f"<div class='opts' data-id='{item}' data-field='{field}'>"
            + "".join(f"<button data-v='{html.escape(c)}'>{html.escape(c)}</button>" for c in choices) + "</div>")


def build_gt_page() -> str:
    cards = []
    for n, (item, cam, times, q, choices) in enumerate(GT_ITEMS, 1):
        fps = yaml.safe_load((GT_DIR / f"{cam}.yaml").read_text())["fps"]
        poly = _zone(cam)
        figs = []
        for k, t in enumerate(times, 1):
            img = _draw(_frame(ROOT / "data" / "videos" / CAMS[cam], round(t * fps)), poly)
            num = "①②③④⑤⑥"[k - 1] if len(times) > 1 else ""
            figs.append(f"<figure><img src='{_b64(img)}' alt='{cam} {t}s'><figcaption>{num} {t:.1f}초</figcaption></figure>")
        cards.append(f"<div class='card' data-id='{item}'><div class='muted'>{n}. {cam} ({CAMS[cam]})</div>"
                     f"<div class='imgs'>{''.join(figs)}</div><div class='q'>{html.escape(q)}</div>{_opts(item, 'choice', choices)}"
                     f"<textarea data-id='{item}' data-field='note' placeholder='메모(선택)'></textarea></div>")
    intro = ("처음 만든 평가 영상 4편의 정답 초안(Claude가 만든 것) 중 사람이 확인해야 할 14곳입니다. 빨간 칸이 구역이고, "
             "<b>사람의 발(신발 밑)</b>이 빨간 칸 안에 있으면 '들어감'입니다. 사진을 누르면 크게 보입니다. 답은 이 브라우저에 자동으로 남고, "
             "다 끝나면 위의 <b>내보내기</b>를 눌러 파일을 저장해 올려 주세요. '— 지금 정답'이 붙은 답이 현재 초안입니다.")
    return _page("평가 영상 정답 검수", "gt_review", intro, cards)


# --------------------------------------------------------------------------- rubric page
def _recover_ids(rubric: list[dict], cfg: dict) -> dict[str, tuple[int, int]]:
    """event id -> (track id, entry frame), by trying all values of the deterministic uuid5."""
    zones = {c["camera_id"]: c["zones"][0]["zone_id"] for c in cfg["cameras"]}
    want = {r["event_id"]: r["camera_id"] for r in rubric}
    nframes = {cam: int(cv2.VideoCapture(str(ROOT / "data" / "videos" / v)).get(cv2.CAP_PROP_FRAME_COUNT)) for cam, v in CAMS.items()}
    out = {}
    for eid, cam in want.items():
        for f in range(nframes[cam] + 1):
            hit = next((t for t in range(1, 4000)
                        if str(uuid.uuid5(uuid.NAMESPACE_URL, f"sceneproof:{cam}:{zones[cam]}:{t}:{f}")) == eid), None)
            if hit:
                out[eid] = (hit, f)
                break
    return out


def _rerun_events(rerun: Path) -> list[dict]:
    if not list(rerun.glob("**/*.jsonl")):
        subprocess.run([sys.executable, "-m", "sceneproof.eval.run_eval", "--variant", "yolox_tiny", "--runs-dir", str(rerun),
                        "--out", str(rerun / "eval.json")], check=True, cwd=ROOT)
    evs = {}
    for f in rerun.glob("**/*.jsonl"):
        for line in f.read_text().splitlines():
            e = json.loads(line)
            evs[e["event_id"]] = e  # last revision wins (closed, with exit and trajectory)
    return list(evs.values())


def build_rubric_page(rerun: Path) -> str:
    rubric = list(csv.DictReader((ROOT / "docs" / "eval" / "raw" / "vlm_eval-rubric.csv").open(encoding="utf-8")))
    cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text())
    ctx_s = 1.5  # sceneproof.edge.evidence.CONTEXT_OFFSET_S: context_before/after are 1.5 s before entry / after exit
    dwell_s = float(yaml.safe_load((GT_DIR / "eval-campus.yaml").read_text())["rules"]["dwell_seconds"])
    ids = _recover_ids(rubric, cfg)
    local = _rerun_events(rerun)
    cards = []
    outcome_ko = {"tp": "정답 침입(CV가 맞힘)", "fp": "CV 잘못 울림", "duplicate": "같은 침입의 중복 사건", "boundary": "정답 경계 통과"}
    for n, r in enumerate(rubric, 1):
        cam, eid = r["camera_id"], r["event_id"]
        fps = yaml.safe_load((GT_DIR / f"{cam}.yaml").read_text())["fps"]
        poly = _zone(cam)
        tid, f0 = ids.get(eid, (None, None))
        # the re-run event of the same person: same camera, entry frame within 6 frames
        twin = min((e for e in local if e["camera_id"] == cam and abs(int(e["entry_frame_index"]) - f0) <= 6),
                   key=lambda e: abs(int(e["entry_frame_index"]) - f0), default=None) if f0 is not None else None
        if twin is not None and twin.get("exit_frame_index") is not None:
            f1, approx = int(twin["exit_frame_index"]) + (f0 - int(twin["entry_frame_index"])), False
        else:
            f1, approx = f0 + round(1.0 * fps), True
        traj = {int(p["frame_index"]): p["box"] for p in (twin or {}).get("trajectory", [])}
        # the frames the VLM received (server setting vlm_frames): context_before, entry, dwell (only when the
        # event lasted >= dwell_seconds), exit, context_after; the tracked box is drawn on entry / dwell / exit
        frames = [("들어가기 전", f0 - round(ctx_s * fps), False), ("들어감", f0, True)]
        if (f1 - f0) / fps >= dwell_s:
            frames.append((f"{dwell_s:.0f}초 머묾", f0 + round(dwell_s * fps), True))
        frames += [("나감" + (" (추정)" if approx else ""), f1, True), ("나간 뒤" + (" (추정)" if approx else ""), f1 + round(ctx_s * fps), False)]
        obs = [o.strip() for o in r["observations"].split(" | ")]
        figs = []
        nfr = int(cv2.VideoCapture(str(ROOT / "data" / "videos" / CAMS[cam])).get(cv2.CAP_PROP_FRAME_COUNT))
        frames = [f for f in frames if 0 <= f[1] < nfr]  # before the first / after the last frame: the VLM got no image either
        for k, (role, idx, with_box) in enumerate(frames):
            box = None
            if traj and with_box:
                near = min(traj, key=lambda q: abs(q - idx))
                if abs(near - idx) <= 3:
                    b = traj[near]
                    box = (b["x1"], b["y1"], b["x2"], b["y2"])
            img = _draw(_frame(ROOT / "data" / "videos" / CAMS[cam], idx), poly, box, f"track {tid}" if box is not None else "")
            figs.append(f"<figure><img src='{_b64(img)}' alt='{role}'><figcaption>{k + 1}. {role} · {idx / fps:.1f}초</figcaption></figure>")
        obs_html = "".join(f"<div class='obs'><b>관찰 {k + 1}</b> {html.escape(o)}</div>" for k, o in enumerate(obs))
        cards.append(
            f"<div class='card' data-id='{eid}'><div class='muted'>{n}. {cam} · 사건 {eid[:8]} · {outcome_ko.get(r['cv_outcome_vs_gt'], r['cv_outcome_vs_gt'])}"
            f" · Claude 판단: {html.escape(r['assessment'])} / {html.escape(r['zone_relation'])}</div>"
            f"<div class='imgs'>{''.join(figs)}</div>"
            f"<div class='q'>Claude의 요약</div><div class='obs'>{html.escape(r['scene_summary'])}</div>"
            f"<div class='q'>Claude의 관찰 (보통 사진 순서대로)</div>{obs_html}"
            f"<div class='q'>Claude가 적은 '불확실한 점'</div><div class='obs'>{html.escape(r['uncertainties'])}</div>"
            f"<div class='q'>1) 관찰 내용이 사진과 맞나요?</div>{_opts(eid, 'observations_match_frames', ['1 맞음', '0 틀린 내용 있음'])}"
            f"<div class='q'>2) 사진에 없는 내용을 지어냈나요?</div>{_opts(eid, 'hallucination', ['0 없음', '1 있음'])}"
            f"<div class='q'>3) '불확실한 점'이 타당한가요?</div>{_opts(eid, 'uncertainties_sensible', ['1 타당', '0 아님'])}"
            f"<div class='q'>4) 한국어가 읽기 쉬운가요?</div>{_opts(eid, 'readability', ['1 어려움', '2 보통', '3 쉬움'])}"
            f"<textarea data-id='{eid}' data-field='note' placeholder='지어낸 내용이 있으면 무엇인지 적어 주세요(선택)'></textarea></div>")
    intro = ("Claude가 사건마다 근거 사진(3~5장)을 보고 쓴 설명 19건입니다. 사진은 Claude가 본 것과 <b>같은 영상의 같은 순간</b>을 다시 뽑은 것입니다"
             "(빨간 칸 = 구역, 노란 상자 = 추적 상자. 상자 위치는 같은 분석을 다시 돌린 결과라 조금 다를 수 있고, '추정'이 붙은 사진은 나간 시각을 "
             "정확히 모르는 경우입니다). 사진을 누르면 크게 보입니다. 네 가지를 버튼으로 고르고, 다 끝나면 <b>내보내기</b>를 눌러 파일을 올려 주세요.")
    return _page("Claude 해석 채점 (19건)", "vlm_rubric", intro, cards)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", default="data/runs/review-rerun", help="where the eval pipeline re-run is kept")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gt_review.html").write_text(build_gt_page())
    (OUT / "vlm_rubric.html").write_text(build_rubric_page(ROOT / args.rerun))
    for p in ("gt_review.html", "vlm_rubric.html"):
        print(f"wrote {OUT / p} ({(OUT / p).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
