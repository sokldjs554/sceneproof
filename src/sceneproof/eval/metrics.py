"""Event-level metrics.

Matching rule (documented so the numbers mean something):
  * a predicted episode [p_start, p_end] is a candidate for a GT episode [g_start, g_end] when
    their temporal overlap >= min(``min_overlap_s``, half of the shorter episode) OR the predicted
    start lies within ``tolerance_s`` of the GT start (short events). Among all GT events
    (intrusions AND boundary/occluded ones) the one with the largest overlap wins; ties go to the
    intrusion (conservative: a prediction is then a duplicate rather than hidden under 'occluded'). Each GT event
    can be matched by at most one prediction; extra predictions overlapping an already-matched GT
    intrusion are counted as **duplicates** (not FPs).
  * predictions that match nothing are false positives; GT positives without a match are misses.
  * precision = TP / (TP + FP), recall = TP / (TP + miss), F1 the harmonic mean.
  * false alarms per hour = FP / (video duration in hours).
  * labels: GT ``boundary_pass`` / ``occluded`` events are reported separately (a prediction
    matching one is neither TP nor FP: it is listed under ``boundary_hits``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .gt import GroundTruth, GTEvent


@dataclass
class PredEvent:
    event_id: str
    start_s: float
    end_s: float
    dwell: bool
    track_id: int = 0
    zone_id: str = ""
    extra: dict = field(default_factory=dict)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


@dataclass
class MatchResult:
    tp: list[tuple[str, str]]  # (pred_id, gt_id)
    fp: list[str]
    missed: list[str]
    duplicates: list[tuple[str, str]]
    boundary_hits: list[tuple[str, str]]
    start_errors_s: list[float]
    end_errors_s: list[float]
    dwell_confusion: dict


def match_events(preds: list[PredEvent], gt: GroundTruth, min_overlap_s: float = 0.5, tolerance_s: float = 1.0) -> MatchResult:
    positives = {e.id: e for e in gt.positives()}
    others = {e.id: e for e in gt.events if e.label != "intrusion"}
    matched_gt: dict[str, str] = {}
    tp, fp, dup, boundary, start_err, end_err = [], [], [], [], [], []
    dwell_conf = {"tp_dwell_correct": 0, "tp_dwell_wrong": 0}

    def candidates(p: PredEvent, pool: dict[str, GTEvent]) -> list[tuple[float, GTEvent]]:
        out = []
        p_dur = max(1e-6, p.end_s - p.start_s)
        for g in pool.values():
            ov = _overlap(p.start_s, p.end_s, g.start_s, g.end_s)
            need = min(min_overlap_s, 0.5 * min(p_dur, max(1e-6, g.duration)))
            if ov >= need or abs(p.start_s - g.start_s) <= tolerance_s:
                out.append((ov, g))
        return sorted(out, key=lambda t: -t[0])

    for p in sorted(preds, key=lambda x: x.start_s):
        pos = candidates(p, positives)
        oth = candidates(p, others)
        best_pos = pos[0][0] if pos else -1.0
        best_oth = oth[0][0] if oth else -1.0
        if oth and best_oth > best_pos:
            boundary.append((p.event_id, oth[0][1].id))
            continue
        if pos:
            # prefer an unmatched GT; otherwise it's a duplicate of the best match
            free = [g for _, g in pos if g.id not in matched_gt]
            if free:
                g = free[0]
                matched_gt[g.id] = p.event_id
                tp.append((p.event_id, g.id))
                start_err.append(p.start_s - g.start_s)
                end_err.append(p.end_s - g.end_s)
                if p.dwell == g.dwell:
                    dwell_conf["tp_dwell_correct"] += 1
                else:
                    dwell_conf["tp_dwell_wrong"] += 1
            else:
                dup.append((p.event_id, pos[0][1].id))
            continue
        fp.append(p.event_id)
    missed = [gid for gid in positives if gid not in matched_gt]
    return MatchResult(tp, fp, missed, dup, boundary, start_err, end_err, dwell_conf)


def summarize(m: MatchResult, gt: GroundTruth, duration_s: float) -> dict:
    n_tp, n_fp, n_miss = len(m.tp), len(m.fp), len(m.missed)
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / (n_tp + n_miss) if (n_tp + n_miss) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    hours = duration_s / 3600 if duration_s else 0.0
    import statistics as st

    def mae(xs: list[float]) -> float | None:
        return round(st.mean(abs(x) for x in xs), 2) if xs else None

    return {
        "gt_positives": len(gt.positives()), "gt_other": len(gt.events) - len(gt.positives()),
        "predicted": n_tp + n_fp + len(m.duplicates) + len(m.boundary_hits),
        "tp": n_tp, "fp": n_fp, "missed": n_miss, "duplicates": len(m.duplicates), "boundary_hits": len(m.boundary_hits),
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "false_alarms_per_hour": round(n_fp / hours, 2) if hours else None,
        "start_mae_s": mae(m.start_errors_s), "end_mae_s": mae(m.end_errors_s),
        "dwell": m.dwell_confusion, "duration_s": duration_s,
        "denominators": {"precision": "TP+FP (duplicates and boundary hits excluded)", "recall": "TP+missed (GT intrusion events)",
                         "false_alarms_per_hour": "FP / video hours"},
        "details": {"tp": m.tp, "fp": m.fp, "missed": m.missed, "duplicates": m.duplicates, "boundary_hits": m.boundary_hits},
    }


def preds_from_jsonl(path: str, camera_id: str | None = None) -> list[PredEvent]:
    """Final revision per event_id from an edge events.jsonl (optionally one camera only)."""
    import json

    latest: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ev = json.loads(line)
                if camera_id is not None and ev.get("camera_id") != camera_id:
                    continue
                if ev["event_id"] not in latest or int(ev.get("revision", 0)) > int(latest[ev["event_id"]]["revision"]):
                    latest[ev["event_id"]] = ev
    out = []
    for ev in latest.values():
        end = ev.get("media_exit_s")
        if end is None:
            end = ev.get("media_entry_s", 0.0) + ev.get("dwell_seconds", 0.0)
        out.append(PredEvent(ev["event_id"], float(ev.get("media_entry_s", 0.0)), float(end),
                             ev.get("kind") == "EVENT_KIND_ZONE_DWELL", int(ev.get("track_id", 0)), ev.get("zone_id", ""),
                             {"close_reason": ev.get("close_reason"), "revision": ev.get("revision")}))
    return out


def to_dict(m: MatchResult) -> dict:
    return asdict(m)
