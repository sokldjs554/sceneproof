"""Spatio-temporal matching of predicted episodes to ground-truth episodes.

A prediction can match a GT episode only if (1) same zone, (2) they overlap in time by at least
min(min_overlap_s, half the shorter one) or start within start_tolerance_s of each other — the rule used by
the original event evaluation (metrics.py) — and (3) they are the same person: at the frames where both have
a box, the median distance between the two feet points is at most ``max_foot_dist_rel`` x the GT box height.
Condition (3) is new: in crowded scenes two different people are often in the zone at the same time, and a
time-only rule would credit a detection of person A to the ground truth of person B.

Outcome per prediction: tp (first match of a GT intrusion), duplicate (another prediction of an already
matched GT intrusion — usually a track split), neutral (matches a boundary_pass or ignore episode), fp.
GT intrusions without a match are misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .episodes import Ep


def _foot(b) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, b[3]


def same_person(pred: Ep, gt: Ep, max_foot_dist_rel: float = 0.5, frame_tol: int = 2) -> float | None:
    """Median normalised feet distance over common frames (None when they never co-occur)."""
    if not pred.boxes or not gt.boxes:
        return None
    gframes = np.array(sorted(gt.boxes))
    dists = []
    for f, pb in pred.boxes.items():
        j = int(np.searchsorted(gframes, f))
        cand = [g for g in (gframes[j - 1] if j > 0 else None, gframes[j] if j < len(gframes) else None)
                if g is not None and abs(g - f) <= frame_tol]
        if not cand:
            continue
        g = min(cand, key=lambda g: abs(g - f))
        gb = gt.boxes[int(g)]
        gh = max(1.0, gb[3] - gb[1])
        (px, py), (gx, gy) = _foot(pb), _foot(gb)
        dists.append(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5 / gh)
    if not dists:
        return None
    return float(np.median(dists))


def _time_ok(p: Ep, g: Ep, min_overlap_s: float, tol_s: float) -> float | None:
    ov = max(0.0, min(p.end_s, g.end_s) - max(p.start_s, g.start_s))
    need = min(min_overlap_s, 0.5 * min(p.duration, g.duration))
    if ov >= need and ov > 0:
        return ov
    if abs(p.start_s - g.start_s) <= tol_s:
        return ov
    return None


@dataclass
class Match:
    tp: list[tuple[str, str]] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    neutral: list[tuple[str, str]] = field(default_factory=list)
    fp: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    start_err_s: list[float] = field(default_factory=list)
    end_err_s: list[float] = field(default_factory=list)
    dwell_agree: int = 0
    dwell_total: int = 0


def match(preds: list[Ep], gts: list[Ep], min_overlap_s: float = 0.5, tol_s: float = 1.0,
          max_foot_dist_rel: float = 0.5) -> Match:
    m = Match()
    intr = [g for g in gts if g.label == "intrusion"]
    other = [g for g in gts if g.label != "intrusion"]
    # candidate pairs with a score: (time overlap, -feet distance)
    cands: list[tuple[float, float, int, int]] = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(intr):
            if g.zone_id != p.zone_id:
                continue
            ov = _time_ok(p, g, min_overlap_s, tol_s)
            if ov is None:
                continue
            d = same_person(p, g, max_foot_dist_rel)
            if d is None or d > max_foot_dist_rel:
                continue
            cands.append((ov, -d, pi, gi))
    cands.sort(reverse=True)
    pred_done: dict[int, str] = {}
    gt_hit: dict[int, int] = {}
    for _ov, _nd, pi, gi in cands:  # greedy: best overlap first, one TP per GT
        if pi in pred_done or gi in gt_hit:
            continue
        gt_hit[gi] = pi
        pred_done[pi] = "tp"
        p, g = preds[pi], intr[gi]
        m.tp.append((p.id, g.id))
        m.start_err_s.append(p.start_s - g.start_s)
        m.end_err_s.append(p.end_s - g.end_s)
        m.dwell_total += 1
        m.dwell_agree += int(p.dwell == g.dwell)
    for _ov, _nd, pi, gi in cands:  # remaining predictions on an already matched GT intrusion
        if pi in pred_done:
            continue
        pred_done[pi] = "dup"
        m.duplicates.append((preds[pi].id, intr[gi].id))
    for pi, p in enumerate(preds):
        if pi in pred_done:
            continue
        hit = None
        for g in other:
            if g.zone_id == p.zone_id and _time_ok(p, g, min_overlap_s, tol_s) is not None:
                d = same_person(p, g, max_foot_dist_rel)
                if d is not None and d <= max_foot_dist_rel:
                    hit = g
                    break
        if hit is not None:
            pred_done[pi] = "neutral"
            m.neutral.append((p.id, hit.id))
        else:
            pred_done[pi] = "fp"
            m.fp.append(p.id)
    m.missed = [g.id for gi, g in enumerate(intr) if gi not in gt_hit]
    return m


def summarize(m: Match, duration_s: float, n_gt: int) -> dict:
    tp, fp, miss = len(m.tp), len(m.fp), len(m.missed)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + miss) if tp + miss else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    def mae(xs):
        return (round(float(np.mean(np.abs(xs))), 2) if xs else None)
    return {"gt_intrusions": n_gt, "tp": tp, "fp": fp, "missed": miss, "duplicates": len(m.duplicates),
            "neutral": len(m.neutral), "precision": None if p is None else round(p, 3),
            "recall": None if r is None else round(r, 3), "f1": None if f1 is None else round(f1, 3),
            "duration_s": round(duration_s, 1),
            "false_alarms_per_hour": round(fp / (duration_s / 3600), 1) if duration_s else None,
            "start_mae_s": mae(m.start_err_s), "end_mae_s": mae(m.end_err_s),
            "dwell_agreement": f"{m.dwell_agree}/{m.dwell_total}"}
