"""Zone episodes from hand-labelled boxes (ground truth) and from the detector + tracker (prediction).

Both go through the product's ZoneMonitor with the same ZoneSpecs, so "inside the zone" means the same
thing on both sides: bottom-centre of the box inside the polygon for at least entry_confirm_seconds,
closed after exit_confirm_seconds outside or lost_grace_seconds without a box, boxes touching the frame
bottom hold the state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from sceneproof.edge.tracker import ByteTracker, TrackerParams
from sceneproof.edge.zones import CloseReason, FrameMeta, Transition, ZoneMonitor, ZoneSpec, anchor_point

from .data import CONFIG_DIR, GTBox, Sequence


@dataclass
class Ep:
    id: str
    zone_id: str
    track_id: int
    start_s: float
    end_s: float
    dwell: bool
    label: str  # GT: intrusion | boundary_pass | ignore ; prediction: pred
    close_reason: str = ""
    boxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)  # frame -> xyxy (for matching)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def load_zones() -> dict:
    return yaml.safe_load((CONFIG_DIR / "zones.yaml").read_text())


def zone_specs(site: str, zones_cfg: dict | None = None, *, exclude: bool = False, rejoin: bool = False,
               rejoin_dist_rel: float | None = None) -> list[ZoneSpec]:
    """exclude/rejoin are corrections for the PREDICTION side only. Ground truth always uses the plain zone: a real
    person standing in an exclusion area is still an intrusion (the product missing it counts as a miss), and
    hand-labelled track ids do not break, so there is nothing to rejoin."""
    z = zones_cfg or load_zones()
    rules = z["rules"]
    out = []
    for zd in z["sites"][site]["zones"]:
        out.append(ZoneSpec(zd["zone_id"], zd["name"], [tuple(p) for p in zd["polygon"]],
                            dwell_seconds=float(zd.get("dwell_seconds", rules["dwell_seconds"])),
                            lost_grace_seconds=float(rules["lost_grace_seconds"]),
                            entry_confirm_seconds=float(rules["entry_confirm_seconds"]),
                            exit_confirm_seconds=float(rules["exit_confirm_seconds"]),
                            exclude=[[tuple(p) for p in poly] for poly in zd.get("exclude", [])] if exclude else [],
                            rejoin=rejoin, rejoin_dist_rel=float(rejoin_dist_rel or rules.get("rejoin_dist_rel", 1.0))))
    return out


def _poly_dist_px(x: float, y: float, poly_px: list[tuple[float, float]]) -> float:
    """Distance from a point to the polygon outline (0 when inside is handled by the caller)."""
    best = float("inf")
    n = len(poly_px)
    for i in range(n):
        (x1, y1), (x2, y2) = poly_px[i], poly_px[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
        best = min(best, ((x - x1 - t * dx) ** 2 + (y - y1 - t * dy) ** 2) ** 0.5)
    return best


def _episodes_from_monitor(events, label: str, fps: float, track_boxes: dict | None, prefix: str) -> list[Ep]:
    out = []
    for ev in events:
        if ev.transition != Transition.CLOSED:
            continue
        e = ev.episode
        end = e.exit_media_s if e.exit_media_s is not None else e.last_media_s
        if track_boxes is not None:  # GT: the track's own boxes around the episode
            lo, hi = e.entry_frame - int(fps), (e.exit_frame or e.last_frame) + int(fps)
            boxes = {f: b for f, b in track_boxes.get(e.track_id, {}).items() if lo <= f <= hi}
        else:  # prediction: trajectory samples (every trajectory_stride frames)
            boxes = {p.frame_index: tuple(p.xyxy) for p in e.trajectory}
        out.append(Ep(f"{prefix}{len(out):03d}", e.zone.zone_id, e.track_id, e.entry_media_s, end, e.dwell_escalated, label,
                      e.close_reason.value if e.close_reason else "", boxes))
    return out


def _run_monitor(frames: dict[int, list], n_frames: int, fps: float, w: int, h: int, specs: list[ZoneSpec], cam: str):
    mon = ZoneMonitor(specs, cam, trajectory_stride=1)
    evs = []
    meta = FrameMeta(0, 0.0, 0.0, w, h)
    for i in range(n_frames):
        meta = FrameMeta(i, i / fps, 0.0, w, h)
        evs += mon.update(frames.get(i, []), meta)
    evs += mon.close_all(CloseReason.STREAM_ENDED, meta)
    return evs


def gt_episodes(seq: Sequence, boxes: dict[int, list[GTBox]], specs: list[ZoneSpec],
                near_margin_rel: float = 0.15) -> list[Ep]:
    """Intrusions (and 'ignore' episodes for distractors/reflections) from the hand-labelled boxes, plus
    boundary_pass spans: a person whose feet came inside or within near_margin_rel * box height of a zone
    without a confirmed episode — a detection there is neither a hit nor a false alarm."""
    w, h, fps = seq.width, seq.height, seq.fps
    n = seq.frames or (max(boxes) + 1 if boxes else 0)
    track_boxes: dict[str, dict[int, dict[int, tuple]]] = {"person": {}, "ignore": {}}
    per_label: dict[str, dict[int, list]] = {"person": {}, "ignore": {}}
    for f, bl in boxes.items():
        for b in bl:
            track_boxes[b.label].setdefault(b.track_id, {})[f] = b.xyxy
            per_label[b.label].setdefault(f, []).append(SimpleNamespace(track_id=b.track_id, xyxy=np.array(b.xyxy, np.float32), score=1.0))
    eps = _episodes_from_monitor(_run_monitor(per_label["person"], n, fps, w, h, specs, seq.id), "intrusion", fps,
                                 track_boxes["person"], "gt")
    eps += _episodes_from_monitor(_run_monitor(per_label["ignore"], n, fps, w, h, specs, seq.id), "ignore", fps,
                                  track_boxes["ignore"], "ig")
    # boundary passes
    for z in specs:
        poly = z.polygon_px(w, h)
        for tid, tb in track_boxes["person"].items():
            near = []
            for f in sorted(tb):
                _x1, y1, _x2, y2 = tb[f]
                if y2 >= h - 2:
                    continue  # feet out of view: no ground position
                ax, ay = anchor_point(np.array(tb[f]))
                if z.contains_px(ax, ay, w, h) or _poly_dist_px(ax, ay, poly) <= near_margin_rel * (y2 - y1):
                    near.append(f)
            if not near:
                continue
            spans, s0, prev = [], near[0], near[0]
            for f in near[1:]:
                if f - prev > fps * 0.5:
                    spans.append((s0, prev))
                    s0 = f
                prev = f
            spans.append((s0, prev))
            mine = [e for e in eps if e.label == "intrusion" and e.track_id == tid and e.zone_id == z.zone_id]
            for a, b in spans:
                if any(e.start_s - 1.0 <= b / fps and a / fps <= e.end_s + 1.0 for e in mine):
                    continue  # part of a real intrusion
                lo, hi = a - int(fps), b + int(fps)
                eps.append(Ep(f"bp{len(eps):03d}", z.zone_id, tid, a / fps, b / fps, False, "boundary_pass", "",
                              {f: bb for f, bb in tb.items() if lo <= f <= hi}))
    return eps


@dataclass
class RunResult:
    episodes: list[Ep]
    frames: int
    wall_s: float
    det_samples: dict[int, tuple[np.ndarray, np.ndarray]]  # frame -> (boxes, scores) at the low AP threshold


def run_analyzer(seq: Sequence, specs: list[ZoneSpec], detector, tracker_cfg: dict, conf_threshold: float,
                 det_stride: int = 0) -> RunResult:
    """The product's detector -> ByteTrack -> ZoneMonitor on every frame (evidence writing is not needed here).

    The detector should be built with a LOW conf threshold (e.g. 0.05) so that det_samples serve the AP
    evaluation; boxes below ``conf_threshold`` are dropped before tracking, which is exactly what the product
    does with its detector threshold (greedy NMS on boxes above a threshold does not depend on boxes below it).
    With conf_threshold <= the tracker's low_thresh, ByteTrack's second association (low-score boxes extend
    existing tracks, never start one) sees every box it was designed for."""
    cap = cv2.VideoCapture(str(seq.video))
    fps = seq.fps
    tp = TrackerParams(tracker_cfg["track_thresh"], tracker_cfg["low_thresh"], tracker_cfg["new_track_thresh"],
                       tracker_cfg["match_thresh"], track_buffer=max(5, int(tracker_cfg["track_buffer_seconds"] * fps)),
                       fuse_score=tracker_cfg.get("fuse_score", True))
    trk = ByteTracker(tp)
    mon = ZoneMonitor(specs, seq.id, trajectory_stride=3)
    evs, samples = [], {}
    i = 0
    t0 = time.perf_counter()
    meta = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        d = detector.detect(frame)
        keep = d.scores >= conf_threshold
        tracks = trk.update(d.boxes[keep], d.scores[keep])
        meta = FrameMeta(i, i / fps, time.time(), w, h)
        evs += mon.update(tracks, meta)
        if det_stride and i % det_stride == 0:
            samples[i] = (d.boxes.copy(), d.scores.copy())
        i += 1
    cap.release()
    if meta is not None:
        evs += mon.close_all(CloseReason.STREAM_ENDED, meta)
    return RunResult(_episodes_from_monitor(evs, "pred", fps, None, "pr"), i, time.perf_counter() - t0, samples)
