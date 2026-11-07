"""ByteTrack multi-object tracker (own implementation).

Paper: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
ECCV 2022 (arXiv:2110.06864). The idea that matters here: low-score detections are not thrown
away; they are matched in a *second* association round against tracks that were not matched
by high-score boxes. That keeps IDs stable through partial occlusion and motion blur, which is
exactly what makes zone dwell timing trustworthy.

What this file implements (and what tests/test_tracker.py pins down):
  * constant-velocity Kalman filter in (cx, cy, aspect, h) measurement space, as in ByteTrack/SORT
  * two-stage association (high boxes → tracked+lost, then low boxes → remaining *tracked*)
  * tentative tracks: a new track is only reported after it is matched on a second frame
  * lost tracks are kept for ``track_buffer`` frames so a re-appearance keeps its id
Hungarian assignment uses scipy's linear_sum_assignment with a cost gate instead of lapjv;
the gate makes the result equivalent for all but degenerate ties.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- kalman


class KalmanFilter:
    """8-dim state (cx, cy, a, h, vcx, vcy, va, vh), 4-dim measurement (cx, cy, a, h)."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion[i, ndim + i] = dt
        self._update = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[m, np.zeros_like(m)]
        std = [2 * self._std_weight_position * m[3], 2 * self._std_weight_position * m[3], 1e-2,
               2 * self._std_weight_position * m[3], 10 * self._std_weight_velocity * m[3],
               10 * self._std_weight_velocity * m[3], 1e-5, 10 * self._std_weight_velocity * m[3]]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [self._std_weight_position * mean[3], self._std_weight_position * mean[3], 1e-2,
                   self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3], self._std_weight_velocity * mean[3], 1e-5,
                   self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion @ mean
        cov = self._motion @ cov @ self._motion.T + motion_cov
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = [self._std_weight_position * mean[3], self._std_weight_position * mean[3], 1e-1,
               self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        return self._update @ mean, self._update @ cov @ self._update.T + innovation_cov

    def update(self, mean: np.ndarray, cov: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pm, pc = self.project(mean, cov)
        kalman_gain = np.linalg.solve(pc, (cov @ self._update.T).T).T
        innovation = m - pm
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = cov - kalman_gain @ pc @ kalman_gain.T
        return new_mean, new_cov


# --------------------------------------------------------------------------- tracks


class TrackState(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class STrack:
    _next_id = 0

    def __init__(self, xyxy: np.ndarray, score: float) -> None:
        self._det_xyxy = xyxy.astype(np.float32)
        self.score = float(score)
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.state = TrackState.NEW
        self.is_activated = False
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.hits = 0

    @classmethod
    def next_id(cls) -> int:
        cls._next_id += 1
        return cls._next_id

    @classmethod
    def reset_ids(cls) -> None:
        cls._next_id = 0

    @property
    def xyxy(self) -> np.ndarray:
        if self.mean is None:
            return self._det_xyxy
        return xyah_to_xyxy(self.mean)

    def activate(self, kf: KalmanFilter, frame_id: int) -> None:
        self.track_id = self.next_id()
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(self._det_xyxy))
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = frame_id == 1  # only the very first frame activates immediately
        self.frame_id = self.start_frame = frame_id
        self.hits = 1

    def re_activate(self, kf: KalmanFilter, new: STrack, frame_id: int) -> None:
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_xyah(new._det_xyxy))
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new.score
        self.hits += 1

    def update(self, kf: KalmanFilter, new: STrack, frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_xyah(new._det_xyxy))
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new.score
        self.hits += 1

    def predict(self, kf: KalmanFilter) -> None:
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[7] = 0  # lost tracks: freeze height velocity (ByteTrack convention)
        self.mean, self.cov = kf.predict(mean, self.cov)

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


# --------------------------------------------------------------------------- matching


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a[n,4] and b[m,4] (xyxy)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)).astype(np.float32)


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching with a cost gate. Returns (matches, unmatched_rows, unmatched_cols)."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    gated = cost.copy()
    gated[gated > thresh] = thresh + 1e3
    rows, cols = linear_sum_assignment(gated)
    matches, ur, uc = [], set(range(cost.shape[0])), set(range(cost.shape[1]))
    for r, c in zip(rows, cols, strict=True):
        if cost[r, c] <= thresh:
            matches.append((int(r), int(c)))
            ur.discard(int(r))
            uc.discard(int(c))
    return matches, sorted(ur), sorted(uc)


# --------------------------------------------------------------------------- tracker


@dataclass
class TrackOutput:
    track_id: int
    xyxy: np.ndarray
    score: float
    hits: int
    age_frames: int


@dataclass
class TrackerParams:
    track_thresh: float = 0.5   # high-score boxes
    low_thresh: float = 0.1     # low-score boxes used in the second association
    new_track_thresh: float = 0.6  # minimum score to start a new track
    match_thresh: float = 0.8   # 1-IoU gate for the first association
    second_match_thresh: float = 0.5
    unconfirmed_match_thresh: float = 0.7
    track_buffer: int = 30      # frames a lost track is kept
    fuse_score: bool = True     # multiply IoU similarity by detection score (ByteTrack MOT17 default)

    def hash(self) -> str:
        import hashlib
        import json

        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()[:8]


class ByteTracker:
    def __init__(self, params: TrackerParams | None = None) -> None:
        self.p = params or TrackerParams()
        self.kf = KalmanFilter()
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.removed: list[STrack] = []
        self.frame_id = 0

    @staticmethod
    def _sub(a: list[STrack], b: list[STrack]) -> list[STrack]:
        ids = {id(t) for t in b}
        return [t for t in a if id(t) not in ids]

    def _cost(self, tracks: list[STrack], dets: list[STrack], fuse: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=np.float32)
        iou = iou_matrix(np.stack([t.xyxy for t in tracks]), np.stack([d.xyxy for d in dets]))
        if fuse and self.p.fuse_score:
            iou = iou * np.array([d.score for d in dets], dtype=np.float32)[None, :]
        return 1.0 - iou

    def update(self, boxes_xyxy: np.ndarray, scores: np.ndarray) -> list[TrackOutput]:
        self.frame_id += 1
        p = self.p
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        high = scores >= p.track_thresh
        low = (scores >= p.low_thresh) & ~high
        dets_high = [STrack(b, s) for b, s in zip(boxes_xyxy[high], scores[high], strict=True)]
        dets_low = [STrack(b, s) for b, s in zip(boxes_xyxy[low], scores[low], strict=True)]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]

        # ---- 1st association: (tracked + lost) x high-score boxes
        pool = tracked + self.lost
        for t in pool:
            t.predict(self.kf)
        matches, u_track, u_det = linear_assignment(self._cost(pool, dets_high, fuse=True), p.match_thresh)
        activated, refind, lost_now, removed_now = [], [], [], []
        for ti, di in matches:
            t, d = pool[ti], dets_high[di]
            if t.state == TrackState.TRACKED:
                t.update(self.kf, d, self.frame_id)
                activated.append(t)
            else:
                t.re_activate(self.kf, d, self.frame_id)
                refind.append(t)

        # ---- 2nd association: remaining *tracked* x low-score boxes (the ByteTrack idea)
        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.TRACKED]
        matches2, u_track2, _ = linear_assignment(self._cost(r_tracked, dets_low, fuse=False), p.second_match_thresh)
        for ti, di in matches2:
            t, d = r_tracked[ti], dets_low[di]
            t.update(self.kf, d, self.frame_id)
            activated.append(t)
        for i in u_track2:
            t = r_tracked[i]
            if t.state != TrackState.LOST:
                t.mark_lost()
                lost_now.append(t)

        # ---- unconfirmed (tentative) tracks x remaining high boxes
        dets_left = [dets_high[i] for i in u_det]
        matches3, u_unconf, u_det3 = linear_assignment(self._cost(unconfirmed, dets_left, fuse=True),
                                                        p.unconfirmed_match_thresh)
        for ti, di in matches3:
            unconfirmed[ti].update(self.kf, dets_left[di], self.frame_id)
            activated.append(unconfirmed[ti])
        for i in u_unconf:
            unconfirmed[i].mark_removed()
            removed_now.append(unconfirmed[i])

        # ---- new tracks
        for i in u_det3:
            d = dets_left[i]
            if d.score < p.new_track_thresh:
                continue
            d.activate(self.kf, self.frame_id)
            activated.append(d)

        # ---- expire lost tracks
        for t in self.lost:
            if self.frame_id - t.frame_id > p.track_buffer:
                t.mark_removed()
                removed_now.append(t)

        self.tracked = [t for t in self.tracked if t.state == TrackState.TRACKED]
        self.tracked = self._merge(self.tracked, activated)
        self.tracked = self._merge(self.tracked, refind)
        self.lost = self._sub(self.lost, self.tracked)
        self.lost.extend(lost_now)
        self.lost = self._sub(self.lost, removed_now)
        self.removed.extend(removed_now)
        self.removed = self.removed[-200:]

        return [TrackOutput(t.track_id, t.xyxy.copy(), t.score, t.hits, self.frame_id - t.start_frame)
                for t in self.tracked if t.is_activated]

    @staticmethod
    def _merge(a: list[STrack], b: list[STrack]) -> list[STrack]:
        seen = {id(t) for t in a}
        out = list(a)
        for t in b:
            if id(t) not in seen:
                out.append(t)
                seen.add(id(t))
        return out
