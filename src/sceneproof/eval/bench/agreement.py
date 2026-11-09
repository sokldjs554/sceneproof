"""Agreement between people who marked zone events independently (tools/event_labeler), and between each person
and the ground truth derived from the dataset's hand-drawn boxes.

Two measures, both standard for time-interval annotations:
  * event agreement — the same time-overlap rule as the benchmark (overlap >= min(0.5 s, half the shorter event)
    or starts within 1 s), one-to-one; reported as matched / only-A / only-B and an F1, plus the median
    difference of the entry and exit times of matched events;
  * per-frame agreement — at each 0.1 s step, "is anyone inside the zone?" for each annotator, then Cohen's kappa
    (1 = identical, 0 = no better than chance).
Only events labelled intrusion are compared; boundary_pass / unsure events are listed but not scored.

    python -m sceneproof.eval.bench.agreement labels/caviar-lobby-walk1.A.json labels/caviar-lobby-walk1.B.json [--with-gt]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


def _match(a: list[tuple[float, float]], b: list[tuple[float, float]], min_overlap_s=0.5, tol_s=1.0):
    cands = []
    for i, (s1, e1) in enumerate(a):
        for j, (s2, e2) in enumerate(b):
            ov = max(0.0, min(e1, e2) - max(s1, s2))
            need = min(min_overlap_s, 0.5 * min(e1 - s1, e2 - s2))
            if (ov >= need and ov > 0) or abs(s1 - s2) <= tol_s:
                cands.append((ov, -abs(s1 - s2), i, j))
    cands.sort(reverse=True)
    ua, ub, pairs = set(), set(), []
    for _, _, i, j in cands:
        if i not in ua and j not in ub:
            ua.add(i)
            ub.add(j)
            pairs.append((i, j))
    return pairs


def _occupancy(events: list[tuple[float, float]], duration: float, step: float = 0.1) -> np.ndarray:
    t = np.arange(0, duration, step)
    occ = np.zeros(len(t), bool)
    for s, e in events:
        occ |= (t >= s) & (t <= e)
    return occ


def cohen_kappa(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) == 0:
        return None
    po = float(np.mean(x == y))
    pe = float(np.mean(x) * np.mean(y) + np.mean(~x) * np.mean(~y))
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 3)


def compare(a: dict, b: dict) -> dict:
    ia = [(e["start_s"], e["end_s"]) for e in a["events"] if e.get("label", "intrusion") == "intrusion"]
    ib = [(e["start_s"], e["end_s"]) for e in b["events"] if e.get("label", "intrusion") == "intrusion"]
    pairs = _match(ia, ib)
    n = len(pairs)
    dur = max(a.get("duration_s") or 0, b.get("duration_s") or 0, *(e for _, e in ia + ib), 0.1)
    f1 = 2 * n / (len(ia) + len(ib)) if ia or ib else 1.0
    ds = [abs(ia[i][0] - ib[j][0]) for i, j in pairs]
    de = [abs(ia[i][1] - ib[j][1]) for i, j in pairs]
    return {"a": a.get("annotator"), "b": b.get("annotator"), "events_a": len(ia), "events_b": len(ib), "matched": n,
            "only_a": len(ia) - n, "only_b": len(ib) - n, "event_f1": round(f1, 3),
            "median_entry_diff_s": round(float(np.median(ds)), 2) if ds else None,
            "median_exit_diff_s": round(float(np.median(de)), 2) if de else None,
            "frame_kappa": cohen_kappa(_occupancy(ia, dur), _occupancy(ib, dur))}


def gt_as_annotation(sequence: str) -> dict:
    """The benchmark ground truth for one sequence, in the labeler's JSON shape."""
    from .data import load_boxes, load_sequences
    from .episodes import gt_episodes, zone_specs

    s = load_sequences()[sequence]
    eps = gt_episodes(s, load_boxes(s), zone_specs(s.site))
    return {"sequence": sequence, "annotator": "dataset boxes -> zone rule", "duration_s": s.duration_s,
            "events": [{"start_s": e.start_s, "end_s": e.end_s, "label": e.label} for e in eps if e.label in ("intrusion", "boundary_pass")]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="labeler JSON exports for the SAME sequence")
    ap.add_argument("--with-gt", action="store_true", help="also compare each person with the dataset-derived ground truth")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    anns = [json.loads(Path(f).read_text()) for f in args.files]
    seqs = {a["sequence"] for a in anns}
    if len(seqs) != 1:
        print(f"files are for different sequences: {sorted(seqs)}", file=sys.stderr)
        return 2
    if args.with_gt:
        anns.append(gt_as_annotation(next(iter(seqs))))
    res = {"sequence": next(iter(seqs)), "pairs": [compare(a, b) for a, b in combinations(anns, 2)]}
    text = json.dumps(res, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
