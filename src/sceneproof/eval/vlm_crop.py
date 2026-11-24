"""Where should a small VLM look? Whole frame vs region, measured on the demo evidence.

Question (after ERGO, ICLR 2026: choose where to look in a high-resolution image instead of paying for
all of it): for a small local VLM on a CPU, does sending only the region around the tracked person
(one 512-px tile, 64 visual tokens) judge the person's position relative to the restricted zone as well
as, or better than, the whole frame — and at what cost in visual tokens and latency?

Strategies (one image per request):
  full   whole frame, one tile                       — the product's current setting (cheap, little detail)
  split  whole frame cut into several tiles + a global view (image splitting) — detail, many tokens
  crop   the region chosen by sceneproof.server.vlm.crop.crop_for_vlm, one tile

Queries. Every key frame of the demo events that has the tracked person's box (entry, dwell, exit) is
drawn twice from the raw frame, the way the analyser draws evidence (red zone, yellow box):
  zone_true     the real zone — the person's feet are inside it                       → label "inside"
  zone_shifted  the same zone moved sideways until the feet are clearly outside it   → label "outside"
The shifted zone is a controlled counterpart, not a real scene: it tells whether the model reads the
relation at all, which a set of only "inside" answers cannot.

    python -m sceneproof.eval.vlm_crop --strategies full,crop --out docs/experiments/vlm-crop/result.json
    python -m sceneproof.eval.vlm_crop --dry-run --save-images /tmp/vlm-crop   # no model: just the images

A model answer counts as right when zone_relation is clearly_inside for "inside" and outside for "outside";
near_boundary and unclear are counted as abstentions. Small models often answer with the bare value
("near_boundary") instead of the JSON object; that is accepted when exactly one of the four values appears.
Anything else is unparsed.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from sceneproof.edge.evidence import draw_annotations
from sceneproof.edge.zones import point_in_polygon
from sceneproof.server.vlm.schema import extract_json

BUNDLE = Path("demo/bundle")
ROLES = ("entry", "dwell", "exit")
STRATEGIES = ("full", "split", "crop")
SHORT_PROMPT = (
    "The red polygon marks a restricted zone on the floor. The yellow box marks one tracked person. "
    "Where are this person's feet relative to the red polygon? Answer with ONE single-line JSON object and "
    'nothing else: {"zone_relation": "clearly_inside" | "near_boundary" | "outside" | "unclear"}')
RIGHT = {"inside": "clearly_inside", "outside": "outside"}


@dataclass
class Query:
    event_id: str
    camera_id: str
    role: str
    variant: str                    # zone_true | zone_shifted
    label: str                      # inside | outside
    image: np.ndarray = field(repr=False)
    box: tuple[float, float, float, float]
    zone: list[tuple[float, float]]  # pixels, as drawn


def feet(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, box[3])


def _boundary_distance(p, poly) -> float:
    best = float("inf")
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1], strict=True):
        dx, dy = bx - ax, by - ay
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
        best = min(best, float(np.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))))
    return best


def shifted_zone(zone_norm: list[tuple[float, float]], foot_norm: tuple[float, float],
                 min_gap: float = 0.06) -> list[tuple[float, float]] | None:
    """The zone moved sideways (then, if needed, up) by the smallest step that leaves the feet at least
    ``min_gap`` (fraction of the frame) outside it, with the zone still inside the frame."""
    xs, ys = [p[0] for p in zone_norm], [p[1] for p in zone_norm]
    candidates = []
    for step in np.arange(0.02, 1.0, 0.01):
        candidates += [(float(step), 0.0), (-float(step), 0.0)]
    for step in np.arange(0.02, 1.0, 0.01):
        candidates += [(0.0, -float(step))]
    for dx, dy in candidates:
        if min(xs) + dx < 0 or max(xs) + dx > 1 or min(ys) + dy < 0 or max(ys) + dy > 1:
            continue
        moved = [(x + dx, y + dy) for x, y in zone_norm]
        if not point_in_polygon(foot_norm[0], foot_norm[1], moved) and _boundary_distance(foot_norm, moved) >= min_gap:
            return moved
    return None


def load_queries(bundle: Path = BUNDLE, roles=ROLES) -> list[Query]:
    out: list[Query] = []
    for line in (bundle / "events.jsonl").read_text().splitlines():
        ev = json.loads(line)
        w, h = int(ev["frame_width"]), int(ev["frame_height"])
        zone_norm = [(float(p["x"]), float(p["y"])) for p in ev.get("zone_polygon") or []]
        if not zone_norm:
            continue
        for kf in ev["evidence"]["key_frames"]:
            if kf["role"] not in roles or not kf.get("box"):
                continue
            raw = cv2.imread(str(bundle / "evidence" / kf["path"]))
            if raw is None:
                continue
            ih, iw = raw.shape[:2]
            b = kf["box"]
            box = (b["x1"] * iw / w, b["y1"] * ih / h, b["x2"] * iw / w, b["y2"] * ih / h)   # into image pixels
            foot_norm = (feet(box)[0] / iw, feet(box)[1] / ih)
            variants = [("zone_true", zone_norm, "inside" if point_in_polygon(*foot_norm, zone_norm) else "outside")]
            moved = shifted_zone(zone_norm, foot_norm)
            if moved is not None:
                variants.append(("zone_shifted", moved, "outside"))
            for variant, zn, label in variants:
                img = draw_annotations(raw, SimpleNamespace(polygon=zn, name=ev.get("zone_name", "")), box)
                out.append(Query(ev["event_id"], ev["camera_id"], kf["role"], variant, label, img, box,
                                 [(x * iw, y * ih) for x, y in zn]))
    return out


def region(q: Query, strategy: str, crop_fn: Callable | None = None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = q.image.shape[:2]
    if strategy != "crop":
        return q.image, (0, 0, w, h)
    if crop_fn is None:
        from sceneproof.server.vlm.crop import crop_for_vlm as crop_fn
    x1, y1, x2, y2 = crop_fn(w, h, q.box, q.zone)
    return q.image[y1:y2, x1:x2], (x1, y1, x2, y2)


def judge(label: str, text: str) -> tuple[str | None, str]:
    """(zone_relation or None, outcome) where outcome is right | wrong | abstain | unparsed."""
    obj = extract_json(text)
    rel = obj.get("zone_relation") if isinstance(obj, dict) else None
    if rel is None and not isinstance(obj, dict):
        found = set(re.findall(r"\b(clearly_inside|near_boundary|outside|unclear)\b", text or ""))
        rel = found.pop() if len(found) == 1 else None
    if rel not in ("clearly_inside", "near_boundary", "outside", "unclear"):
        return None, "unparsed"
    if rel in ("near_boundary", "unclear"):
        return rel, "abstain"
    return rel, "right" if rel == RIGHT[label] else "wrong"


def summarize(rows: list[dict]) -> dict:
    out = {}
    for strategy in sorted({r["strategy"] for r in rows}):
        rs = [r for r in rows if r["strategy"] == strategy]
        outcomes = {k: sum(r["outcome"] == k for r in rs) for k in ("right", "wrong", "abstain", "unparsed")}
        by_label = {lab: {k: sum(r["outcome"] == k for r in rs if r["label"] == lab) for k in ("right", "wrong", "abstain", "unparsed")}
                    for lab in ("inside", "outside")}
        lat = [r["latency_ms"] for r in rs]
        out[strategy] = {
            "queries": len(rs), **outcomes,
            "accuracy": round(outcomes["right"] / len(rs), 3) if rs else None,
            "by_label": by_label,
            "visual_tokens_mean": round(statistics.mean(r["visual_tokens"] for r in rs), 1) if rs else None,
            "input_tokens_mean": round(statistics.mean(r["input_tokens"] for r in rs), 1) if rs else None,
            "latency_ms_median": round(statistics.median(lat), 1) if rs else None,
            "latency_ms_max": round(max(lat), 1) if rs else None,
            "region_area_share_mean": round(statistics.mean(r["region_area_share"] for r in rs), 3) if rs else None,
        }
    return out


class SmolVLM:
    """One model, several ways of feeding it an image. Same settings as the product's local provider."""

    def __init__(self, model: str, threads: int = 0, longest_edge: int = 512, split_edge: int = 1024,
                 max_new_tokens: int = 48):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if threads:
            torch.set_num_threads(threads)
        self.torch, self.model_name = torch, model
        self.processor = AutoProcessor.from_pretrained(model)
        self.net = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.float32).eval()
        self.longest_edge, self.split_edge, self.max_new_tokens = longest_edge, split_edge, max_new_tokens
        self.image_token_id = getattr(self.net.config, "image_token_id", None)

    def _configure(self, split: bool) -> None:
        ip = self.processor.image_processor
        ip.do_image_splitting = split
        edge = self.split_edge if split else self.longest_edge
        ip.size = {"longest_edge": edge}
        if hasattr(ip, "max_image_size"):
            ip.max_image_size = {"longest_edge": 512}

    def __call__(self, image_bgr: np.ndarray, prompt: str, split: bool) -> dict:
        from PIL import Image

        self._configure(split)
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        t0 = time.perf_counter()
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        with self.torch.inference_mode():
            out = self.net.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode([gen], skip_special_tokens=True)[0]
        ms = (time.perf_counter() - t0) * 1e3
        ids = inputs["input_ids"][0]
        visual = int((ids == self.image_token_id).sum()) if self.image_token_id is not None else -1
        return {"text": answer, "latency_ms": round(ms, 1), "input_tokens": int(ids.shape[0]), "visual_tokens": visual,
                "output_tokens": int(gen.shape[0])}


def run(queries: list[Query], strategies, answer: Callable[[np.ndarray, str, bool], dict], prompt: str = SHORT_PROMPT,
        crop_fn: Callable | None = None, save_images: Path | None = None, log: Callable[[str], None] = print) -> list[dict]:
    rows = []
    total = len(queries) * len(strategies)
    for q in queries:
        for strategy in strategies:
            img, rect = region(q, strategy, crop_fn)
            if save_images is not None:
                save_images.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_images / f"{q.event_id[:8]}_{q.role}_{q.variant}_{strategy}.jpg"), img)
            res = answer(img, prompt, strategy == "split")
            rel, outcome = judge(q.label, res["text"])
            h, w = q.image.shape[:2]
            rows.append({"event_id": q.event_id, "camera_id": q.camera_id, "role": q.role, "variant": q.variant,
                         "label": q.label, "strategy": strategy, "region": list(rect),
                         "region_area_share": round((rect[2] - rect[0]) * (rect[3] - rect[1]) / (w * h), 4),
                         "zone_relation": rel, "outcome": outcome, **res})
            log(f"[{len(rows)}/{total}] {q.event_id[:8]} {q.role:5} {q.variant:12} {strategy:5} label={q.label:7} "
                f"answer={rel} {outcome} {res['latency_ms']:.0f}ms tokens={res['visual_tokens']}")
    return rows


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strategies", default="full,crop", help="comma list of " + ", ".join(STRATEGIES))
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--split-edge", type=int, default=1024, help="longest edge before splitting into 512-px tiles")
    ap.add_argument("--roles", default=",".join(ROLES))
    ap.add_argument("--limit", type=int, default=0, help="use only the first N queries (0 = all)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-images", type=Path, default=None, help="write every image that is sent to the model here")
    ap.add_argument("--dry-run", action="store_true", help="build the queries and images without loading a model")
    args = ap.parse_args(argv)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    bad = [s for s in strategies if s not in STRATEGIES]
    if bad:
        ap.error(f"unknown strategies {bad}; choose from {STRATEGIES}")
    queries = load_queries(roles=tuple(r.strip() for r in args.roles.split(",")))
    if "crop" in strategies and queries:
        try:
            region(queries[0], "crop")
        except NotImplementedError:
            print("crop_for_vlm (src/sceneproof/server/vlm/crop.py) is not written yet: run without 'crop', "
                  "e.g. --strategies full,split")
            return 2
    if args.limit:
        queries = queries[: args.limit]
    labels = {lab: sum(q.label == lab for q in queries) for lab in ("inside", "outside")}
    print(f"{len(queries)} queries ({labels}), strategies {strategies}")
    if args.dry_run:
        dummy = lambda img, prompt, split: {"text": "", "latency_ms": 0.0, "input_tokens": 0, "visual_tokens": 0, "output_tokens": 0}  # noqa: E731
        run(queries, strategies, dummy, save_images=args.save_images or Path("/tmp/vlm-crop-images"), log=lambda s: None)
        print(f"images written to {args.save_images or '/tmp/vlm-crop-images'}")
        return 0
    model = SmolVLM(args.model, threads=args.threads, split_edge=args.split_edge)
    rows = run(queries, strategies, model, save_images=args.save_images)
    summary = summarize(rows)
    for s, v in summary.items():
        print(f"{s:6} right {v['right']}/{v['queries']} (acc {v['accuracy']})  wrong {v['wrong']}  abstain {v['abstain']}  "
              f"unparsed {v['unparsed']}  visual tokens {v['visual_tokens_mean']}  median {v['latency_ms_median']}ms  "
              f"region {v['region_area_share_mean']:.0%} of the frame")
    if args.out:
        import torch
        import transformers

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "what": "where a small VLM should look: whole frame vs region (sceneproof.eval.vlm_crop)",
            "git": _git(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": args.model,
            "settings": {"strategies": strategies, "threads": args.threads, "longest_edge": 512, "split_edge": args.split_edge,
                         "max_new_tokens": model.max_new_tokens, "prompt": SHORT_PROMPT, "roles": args.roles, "limit": args.limit},
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                            "transformers": transformers.__version__, "cpu_count": __import__("os").cpu_count()},
            "queries": labels, "summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
