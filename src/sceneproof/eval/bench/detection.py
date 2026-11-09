"""Person-detection accuracy on hand-labelled frames: COCO-style AP at IoU 0.5 and 0.5:0.95, plus recall at the
product's operating threshold by person height (small people are what a wide camera loses first).

Ignore handling (a detection there is neither right nor wrong): MOT distractors/reflections, MOT boxes the
benchmark itself excludes from detection scoring (consider=0, e.g. static people behind glass), and people with
visibility < 0.2 (almost fully hidden — the box is a guess of where the body is).
"""

from __future__ import annotations

import numpy as np

from .data import GTBox

HEIGHT_BUCKETS = ((0, 50, "h<50px"), (50, 100, "50-100px"), (100, 1e9, "h>=100px"))


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _ioa(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the area of a (detection inside an ignore region)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / np.maximum(aa[:, None], 1e-9)


def split_gt(boxes: list[GTBox]) -> tuple[np.ndarray, np.ndarray]:
    pos, ign = [], []
    for b in boxes:
        if b.label == "person" and b.consider and b.visibility >= 0.2:
            pos.append(b.xyxy)
        else:
            ign.append(b.xyxy)
    return np.array(pos, np.float32).reshape(-1, 4), np.array(ign, np.float32).reshape(-1, 4)


class APAccumulator:
    """Accumulates (score, is_tp) per IoU threshold over frames; one per evaluated group (site, split...)."""

    THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)

    def __init__(self) -> None:
        # per IoU threshold: (score, is_tp) of every detection that counts at that threshold. A detection that
        # matches no person but lies in an ignore region is dropped — and that can differ between thresholds
        # (matched at 0.5, unmatched at 0.75), so scores are kept per threshold, as COCO does.
        self.scores: dict[float, list[float]] = {t: [] for t in self.THRESHOLDS}
        self.tp = {t: [] for t in self.THRESHOLDS}
        self.n_gt = 0
        self.frames = 0
        self.bucket_gt = {name: 0 for *_, name in HEIGHT_BUCKETS}
        self.bucket_hit = {name: 0 for *_, name in HEIGHT_BUCKETS}

    def add(self, det_boxes: np.ndarray, det_scores: np.ndarray, gt: list[GTBox], op_thresh: float = 0.4) -> None:
        pos, ign = split_gt(gt)
        self.frames += 1
        self.n_gt += len(pos)
        order = np.argsort(-det_scores)
        d, s = det_boxes[order], det_scores[order]
        iou = _iou(d, pos) if len(d) and len(pos) else np.zeros((len(d), len(pos)))
        ioa = _ioa(d, ign).max(1) if len(d) and len(ign) else np.zeros(len(d))
        for t in self.THRESHOLDS:
            taken = np.zeros(len(pos), bool)
            flags = []
            for i in range(len(d)):
                j = -1
                if len(pos):
                    cand = np.where(~taken & (iou[i] >= t))[0]
                    if len(cand):
                        j = int(cand[np.argmax(iou[i, cand])])
                if j >= 0:
                    taken[j] = True
                    flags.append(1)
                elif ioa[i] >= 0.5:
                    flags.append(-1)  # inside an ignore region: dropped
                else:
                    flags.append(0)
            if t == self.THRESHOLDS[0]:
                # recall by height at the operating threshold (IoU 0.5)
                hit_at_op = np.zeros(len(pos), bool)
                taken2 = np.zeros(len(pos), bool)
                for i in np.where(s >= op_thresh)[0]:
                    cand = np.where(~taken2 & (iou[i] >= 0.5))[0] if len(pos) else []
                    if len(cand):
                        j = int(cand[np.argmax(iou[i, cand])])
                        taken2[j] = hit_at_op[j] = True
                for k, b in enumerate(pos):
                    h = b[3] - b[1]
                    for lo, hi, name in HEIGHT_BUCKETS:
                        if lo <= h < hi:
                            self.bucket_gt[name] += 1
                            self.bucket_hit[name] += int(hit_at_op[k])
            self.tp[t] += [f for f in flags if f != -1]
            self.scores[t] += [float(x) for x, f in zip(s, flags, strict=True) if f != -1]

    def result(self) -> dict:
        if not self.n_gt:
            return {"frames": self.frames, "gt_boxes": 0}
        aps = {}
        for t in self.THRESHOLDS:
            order = np.argsort(-np.array(self.scores[t]), kind="stable") if self.scores[t] else np.array([], int)
            tp = np.array(self.tp[t], float)[order] if len(order) else np.zeros(0)
            ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
            rec = ctp / self.n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            for i in range(len(prec) - 2, -1, -1):  # monotone precision envelope
                prec[i] = max(prec[i], prec[i + 1])
            ap = 0.0
            for r in np.linspace(0, 1, 101):  # COCO 101-point interpolation
                idx = np.searchsorted(rec, r, side="left")
                ap += prec[idx] if idx < len(prec) else 0.0
            aps[float(t)] = ap / 101
        return {"frames": self.frames, "gt_boxes": self.n_gt, "ap50": round(aps[0.5], 4),
                "ap50_95": round(float(np.mean(list(aps.values()))), 4),
                "recall_at_op_by_height": {k: (round(self.bucket_hit[k] / v, 3) if v else None) for k, v in self.bucket_gt.items()},
                "gt_by_height": dict(self.bucket_gt)}
