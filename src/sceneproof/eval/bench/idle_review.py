"""False alarms per hour on LONG footage, checked by a person — the way alarm quality is measured on site.

Frame-by-frame ground truth for hours of video is impractical; but every alarm the system raises can be looked
at. Run the product's detector -> tracker -> zone rule over a long video (e.g. an hour of an empty corridor that
you recorded yourself), then a person marks each raised event as "someone really entered" or "false alarm".
False alarms / hours of video is the number that matters on site. (It does not measure missed entries — use the
benchmark for that.)

    # 1) run: writes <out>/review.html (one card per event: entry frame with the box and the zone) + events.json
    python -m sceneproof.eval.bench.idle_review run --video my_corridor.mp4 --zone "0.3,0.5 0.7,0.5 0.7,0.9 0.3,0.9" --out data/idle/corridor
    # 2) open review.html, click 진짜 / 잘못 울림 for every card, press 내보내기 -> verdicts.json
    # 3) score
    python -m sceneproof.eval.bench.idle_review score data/idle/corridor/events.json verdicts.json
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from sceneproof.edge.detector import build_detector
from sceneproof.edge.zones import ZoneSpec

from .data import ROOT, Sequence
from .episodes import run_analyzer


def _thumb(video: Path, frame: int, box, poly, max_w: int = 480) -> str:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return ""
    h, w = img.shape[:2]
    pts = np.array([[x * w, y * h] for x, y in poly], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 255), max(1, w // 400))
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), max(1, w // 400))
    s = min(1.0, max_w / w)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def run(args) -> int:
    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    poly = [tuple(float(v) for v in p.split(",")) for p in args.zone.split()]
    edge = yaml.safe_load((ROOT / "configs" / "edge.yaml").read_text())
    rules = yaml.safe_load((ROOT / "configs" / "bench" / "zones.yaml").read_text())["rules"]
    exclude = [[tuple(float(v) for v in p.split(",")) for p in poly_s.split()] for poly_s in args.exclude]
    spec = ZoneSpec("z-review", args.zone_name, poly, dwell_seconds=float(rules["dwell_seconds"]),
                    lost_grace_seconds=float(rules["lost_grace_seconds"]), entry_confirm_seconds=float(rules["entry_confirm_seconds"]),
                    exit_confirm_seconds=float(rules["exit_confirm_seconds"]),
                    exclude=exclude, rejoin=args.rejoin)  # same as the product defaults
    det_cfg = {**edge["detector"], "model": args.model, "variant": args.variant, "conf_threshold": 0.05}
    detector = build_detector(det_cfg)
    seq = Sequence(video.stem, "user", "user", video.stem, fps, video, video, n, w, h)
    t0 = time.time()
    r = run_analyzer(seq, [spec], detector, edge["tracker"], float(edge["detector"]["conf_threshold"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events = []
    for e in r.episodes:
        f0 = min(e.boxes) if e.boxes else int(e.start_s * fps)
        events.append({"id": e.id, "start_s": round(e.start_s, 2), "end_s": round(e.end_s, 2), "dwell": e.dwell,
                       "frame": f0, "box": e.boxes.get(f0)})
    meta = {"video": str(video), "frames": r.frames, "fps": fps, "hours": round(r.frames / fps / 3600, 4), "zone": poly,
            "model": args.model, "conf_threshold": edge["detector"]["conf_threshold"], "rules": rules,
            "wall_s": round(time.time() - t0, 1), "events": events}
    (out / "events.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    cards = "\n".join(
        f'<div class="card" data-id="{e["id"]}"><img src="{_thumb(video, e["frame"], e["box"], poly)}" alt="">'
        f'<div>{html.escape(e["id"])} · {e["start_s"]:.1f}–{e["end_s"]:.1f}초{" · 체류" if e["dwell"] else ""}</div>'
        f'<button data-v="real">진짜 들어감</button> <button data-v="false">잘못 울림</button> <span class="v"></span></div>'
        for e in events)
    page = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>울림 확인</title><style>body{{font:14px system-ui,sans-serif;margin:16px;background:#f6f7f9;color:#1d2330}}
@media (prefers-color-scheme:dark){{body{{background:#14171c;color:#e8ebf0}}.card{{background:#1c2027!important}}}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #ccc;border-radius:8px;padding:8px}}img{{width:100%}}button{{font:inherit;margin-top:6px}}
.v{{font-weight:700}}</style>
<h1>울림 확인 — {html.escape(video.name)}</h1>
<p>영상 {meta['hours'] * 60:.1f}분, 울린 사건 {len(events)}건. 사진은 들어간 것으로 판단한 순간(빨강 = 구역, 노랑 = 사람으로 본 상자)입니다.
필요하면 원본 영상의 해당 시각을 직접 확인하세요. 모두 고른 뒤 <button id="exp">내보내기</button></p>
<div class="grid">{cards}</div>
<script>
const v = {{}};
document.querySelectorAll('.card button').forEach(b => b.onclick = () => {{
  const c = b.closest('.card'); v[c.dataset.id] = b.dataset.v; c.querySelector('.v').textContent = b.dataset.v === 'real' ? '진짜' : '잘못 울림'; }});
document.getElementById('exp').onclick = () => {{
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify({{video: {json.dumps(video.name)}, verdicts: v}}, null, 1)], {{type: 'application/json'}}));
  a.download = 'verdicts.json'; a.click(); }};
</script></html>"""
    (out / "review.html").write_text(page)
    print(f"{len(events)} events in {meta['hours'] * 60:.1f} min of video -> {out / 'review.html'}")
    return 0


def score(args) -> int:
    meta = json.loads(Path(args.events).read_text())
    verdicts = json.loads(Path(args.verdicts).read_text())["verdicts"]
    ids = [e["id"] for e in meta["events"]]
    missing = [i for i in ids if i not in verdicts]
    false = sum(1 for i in ids if verdicts.get(i) == "false")
    real = sum(1 for i in ids if verdicts.get(i) == "real")
    res = {"video": meta["video"], "hours": meta["hours"], "events": len(ids), "real": real, "false_alarms": false,
           "unreviewed": len(missing), "false_alarms_per_hour": round(false / meta["hours"], 2) if meta["hours"] else None,
           "note": "checked by a person from the raised events only; missed entries are not measured here"}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--video", required=True)
    r.add_argument("--zone", required=True, help='normalised polygon "x,y x,y x,y ..."')
    r.add_argument("--zone-name", default="통제 구역")
    r.add_argument("--exclude", action="append", default=[], help='exclusion polygon inside the zone (repeatable), same format')
    r.add_argument("--rejoin", action="store_true", help="turn on strict event rejoin (off by default, as in the product)")
    r.add_argument("--model", default="models/yolox_tiny.onnx")
    r.add_argument("--variant", default="yolox_tiny")
    r.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("events")
    s.add_argument("verdicts")
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else score(args)


if __name__ == "__main__":
    sys.exit(main())
