"""Zone geometry and the per-track zone state machine.

Why a state machine instead of "inside = event": a detector flickers at polygon edges, people
step on the line, and the tracker loses boxes for a few frames behind pillars. Entry/exit are
therefore *debounced* (N consecutive frames), dwell is measured on media time (not wall clock,
so a slow analyser does not inflate it), and a lost track gets a grace period before its
episode is closed. Every transition is explicit so tests can pin the behaviour.

States per (track_id, zone_id):
    OUTSIDE --inside xN--> INSIDE(episode) --outside xM--> OUTSIDE   (close: EXITED)
    INSIDE --track missing > grace--> OUTSIDE                        (close: TRACK_LOST)
    INSIDE --dwell >= T--> INSIDE (dwell escalated once)

Two corrections for what the tracker and detector get wrong:
  * rejoin: in a crowd the tracker often loses a person and starts a NEW track id for the same person a moment
    later. While the old episode is still in its grace period, a track that first appeared after the loss, close
    to where the person should be by now and of similar size, continues that episode instead of opening a second
    one. Strict mode (default) does it only when the choice is unambiguous: exactly one visible person near the
    expected spot, and that person is the new track, inside the zone. In a dense crowd a guess hands one person's
    event to another (measured: more false alarms than the duplicates it removed), so no guess is made there.
  * exclude: polygons inside a zone where a person-shaped object stands (mannequin, signboard). A box whose feet
    are there cannot OPEN an episode; an already open episode is held (a real person walking past the spot is
    neither cut nor closed).
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray casting (even-odd). Points exactly on an edge count as inside-ish; fine for our use."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_int = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside


def anchor_point(xyxy: np.ndarray) -> tuple[float, float]:
    """Bottom-centre of the box = where the feet touch the ground. Using the box centre would
    flag a person whose upper body overlaps a zone drawn on the floor while standing outside it."""
    return float((xyxy[0] + xyxy[2]) / 2), float(xyxy[3])


@dataclass
class ZoneSpec:
    zone_id: str
    name: str
    polygon: list[tuple[float, float]]  # normalized (0..1)
    entry_confirm_frames: int = 3
    exit_confirm_frames: int = 5
    dwell_seconds: float = 5.0
    lost_grace_seconds: float = 1.5
    # Time-based debounce (preferred). Frame counts depend on the camera fps: 8 frames is 0.27 s at
    # 30 fps but 0.8 s at 10 fps, which produced split episodes on 30 fps footage. When set, these
    # override the frame counts.
    entry_confirm_seconds: float | None = None
    exit_confirm_seconds: float | None = None
    exclude: list[list[tuple[float, float]]] = field(default_factory=list)  # normalized polygons (see module doc)
    rejoin: bool = False
    rejoin_dist_rel: float = 1.0  # max feet distance for a rejoin, in units of the person's box height
    # strict rejoin (crowds): continue only when exactly ONE visible person is near where the lost one should be,
    # and that person is a new track standing inside the zone; otherwise leave the episode to close as track lost
    rejoin_strict: bool = True

    def polygon_px(self, w: int, h: int) -> list[tuple[float, float]]:
        return [(x * w, y * h) for x, y in self.polygon]

    def contains_px(self, x: float, y: float, w: int, h: int) -> bool:
        return point_in_polygon(x / w, y / h, self.polygon)

    def excluded_px(self, x: float, y: float, w: int, h: int) -> bool:
        return any(point_in_polygon(x / w, y / h, poly) for poly in self.exclude)


class CloseReason(Enum):
    EXITED = "exited"
    TRACK_LOST = "track_lost"
    STREAM_ENDED = "stream_ended"
    SHUTDOWN = "shutdown"


@dataclass
class TrajPoint:
    frame_index: int
    media_s: float
    wall_ts: float
    xyxy: tuple[float, float, float, float]
    score: float
    in_zone: bool


@dataclass
class Episode:
    episode_id: str
    zone: ZoneSpec
    track_id: int
    entry_frame: int
    entry_media_s: float
    entry_wall_ts: float
    entry_xyxy: tuple[float, float, float, float]
    last_frame: int
    last_media_s: float
    last_wall_ts: float
    last_xyxy: tuple[float, float, float, float]
    dwell_escalated: bool = False
    dwell_frame: int | None = None
    dwell_media_s: float | None = None
    closed: bool = False
    close_reason: CloseReason | None = None
    exit_frame: int | None = None
    exit_media_s: float | None = None
    exit_wall_ts: float | None = None
    trajectory: list[TrajPoint] = field(default_factory=list)
    revision: int = 0
    rejoined_track_ids: list[int] = field(default_factory=list)  # later track ids that continued this episode

    @property
    def dwell_seconds(self) -> float:
        end = self.exit_media_s if self.exit_media_s is not None else self.last_media_s
        return max(0.0, end - self.entry_media_s)


class Transition(Enum):
    OPENED = "opened"
    DWELL = "dwell"
    CLOSED = "closed"


@dataclass
class EpisodeEvent:
    transition: Transition
    episode: Episode


@dataclass
class _TrackZoneState:
    inside_count: int = 0
    outside_count: int = 0
    inside_since: float | None = None   # media time of the first consecutive inside frame
    outside_since: float | None = None  # media time of the first consecutive outside frame
    episode: Episode | None = None
    last_seen_media_s: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=12))  # (media_s, feet x, feet y): walking velocity


def box_clipped_at_bottom(xyxy, height: int, margin: int = 2) -> bool:
    """True when the box touches the frame bottom: the feet are out of view, so the anchor point is
    not the ground position. The zone state is held (neither inside nor outside is counted)."""
    return float(xyxy[3]) >= height - margin


@dataclass
class FrameMeta:
    frame_index: int
    media_s: float
    wall_ts: float
    width: int
    height: int


class ZoneMonitor:
    """Feeds per-frame tracks, emits episode transitions. One instance per camera."""

    def __init__(self, zones: list[ZoneSpec], camera_id: str, trajectory_stride: int = 3) -> None:
        self.zones = zones
        self.camera_id = camera_id
        self.trajectory_stride = max(1, trajectory_stride)
        self._state: dict[tuple[int, str], _TrackZoneState] = {}
        self._first_seen: dict[int, float] = {}  # track id -> media time of its first frame
        self.open_episodes: dict[str, Episode] = {}
        self.rejoins = 0

    # ------------------------------------------------------------------ helpers
    def _new_episode(self, zone: ZoneSpec, track_id: int, meta: FrameMeta, xyxy: np.ndarray) -> Episode:
        # deterministic id: same camera/zone/track/entry-frame -> same id (safe to re-emit after restart)
        eid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"sceneproof:{self.camera_id}:{zone.zone_id}:{track_id}:{meta.frame_index}"))
        box = tuple(float(v) for v in xyxy)
        return Episode(eid, zone, track_id, meta.frame_index, meta.media_s, meta.wall_ts, box,
                       meta.frame_index, meta.media_s, meta.wall_ts, box)

    def _close(self, st: _TrackZoneState, reason: CloseReason, meta: FrameMeta, out: list[EpisodeEvent]) -> None:
        ep = st.episode
        if ep is None:
            return
        ep.closed = True
        ep.close_reason = reason
        if reason == CloseReason.EXITED:
            ep.exit_frame, ep.exit_media_s, ep.exit_wall_ts = meta.frame_index, meta.media_s, meta.wall_ts
        else:  # lost/ended: the last time we actually saw the person inside
            ep.exit_frame, ep.exit_media_s, ep.exit_wall_ts = ep.last_frame, ep.last_media_s, ep.last_wall_ts
        ep.revision += 1
        out.append(EpisodeEvent(Transition.CLOSED, ep))
        self.open_episodes.pop(ep.episode_id, None)
        st.episode = None
        st.inside_count = st.outside_count = 0
        st.inside_since = st.outside_since = None

    # ------------------------------------------------------------------ main step
    def update(self, tracks: list, meta: FrameMeta) -> list[EpisodeEvent]:
        """tracks: objects with .track_id, .xyxy, .score (TrackOutput)."""
        out: list[EpisodeEvent] = []
        seen: set[tuple[int, str]] = set()
        for t in tracks:
            self._first_seen.setdefault(t.track_id, meta.media_s)
        for zone in self.zones:
            if zone.rejoin:
                self._rejoin(zone, tracks, meta)
            for t in tracks:
                key = (t.track_id, zone.zone_id)
                seen.add(key)
                st = self._state.setdefault(key, _TrackZoneState())
                st.last_seen_media_s = meta.media_s
                ax, ay = anchor_point(t.xyxy)
                st.recent.append((meta.media_s, ax, ay))
                clipped = box_clipped_at_bottom(t.xyxy, meta.height)
                inside = zone.contains_px(ax, ay, meta.width, meta.height)
                if inside and zone.exclude and zone.excluded_px(ax, ay, meta.width, meta.height):
                    if st.episode is None:
                        inside = False  # a person-shaped object here never opens an episode
                    else:
                        clipped = True  # hold an open episode (same as feet not visible)
                if clipped:
                    # feet not visible: hold the current state, but keep an open episode alive
                    if st.episode is not None:
                        ep = st.episode
                        ep.last_frame, ep.last_media_s, ep.last_wall_ts = meta.frame_index, meta.media_s, meta.wall_ts
                        ep.last_xyxy = tuple(float(v) for v in t.xyxy)
                        st.outside_count, st.outside_since = 0, None
                    continue
                if st.episode is None:
                    if inside:
                        st.inside_count += 1
                        st.outside_count = 0
                        if st.inside_since is None:
                            st.inside_since = meta.media_s
                        confirmed = (meta.media_s - st.inside_since >= zone.entry_confirm_seconds
                                     if zone.entry_confirm_seconds is not None else st.inside_count >= zone.entry_confirm_frames)
                        if confirmed:
                            ep = self._new_episode(zone, t.track_id, meta, t.xyxy)
                            ep.revision = 1
                            st.episode = ep
                            self.open_episodes[ep.episode_id] = ep
                            ep.trajectory.append(TrajPoint(meta.frame_index, meta.media_s, meta.wall_ts,
                                                           tuple(float(v) for v in t.xyxy), float(t.score), True))
                            out.append(EpisodeEvent(Transition.OPENED, ep))
                    else:
                        st.inside_count, st.inside_since = 0, None
                else:
                    ep = st.episode
                    if inside:
                        st.outside_count, st.outside_since = 0, None
                        ep.last_frame, ep.last_media_s, ep.last_wall_ts = meta.frame_index, meta.media_s, meta.wall_ts
                        ep.last_xyxy = tuple(float(v) for v in t.xyxy)
                        if not ep.dwell_escalated and ep.dwell_seconds >= zone.dwell_seconds:
                            ep.dwell_escalated = True
                            ep.dwell_frame, ep.dwell_media_s = meta.frame_index, meta.media_s
                            ep.revision += 1
                            out.append(EpisodeEvent(Transition.DWELL, ep))
                    else:
                        st.outside_count += 1
                        if st.outside_since is None:
                            st.outside_since = meta.media_s
                    if (meta.frame_index - ep.entry_frame) % self.trajectory_stride == 0 and len(ep.trajectory) < 600:
                        ep.trajectory.append(TrajPoint(meta.frame_index, meta.media_s, meta.wall_ts,
                                                       tuple(float(v) for v in t.xyxy), float(t.score), inside))
                    exited = (st.outside_since is not None and meta.media_s - st.outside_since >= zone.exit_confirm_seconds
                              if zone.exit_confirm_seconds is not None else st.outside_count >= zone.exit_confirm_frames)
                    if exited:
                        self._close(st, CloseReason.EXITED, meta, out)
        # tracks that vanished this frame
        for key, st in list(self._state.items()):
            if key in seen:
                continue
            if st.episode is not None:
                zone = st.episode.zone
                if meta.media_s - st.episode.last_media_s > zone.lost_grace_seconds:
                    self._close(st, CloseReason.TRACK_LOST, meta, out)
            elif meta.media_s - st.last_seen_media_s > 30:
                del self._state[key]  # forget stale (track_id, zone) pairs
        if len(self._first_seen) > 4 * len(self._state) + 1000:
            live = {k[0] for k in self._state}
            self._first_seen = {tid: ts for tid, ts in self._first_seen.items() if tid in live}
        return out

    def _rejoin(self, zone: ZoneSpec, tracks: list, meta: FrameMeta) -> None:
        """Hand open episodes whose track vanished to a track that appeared after the loss, nearby (see module doc)."""
        present = {t.track_id for t in tracks}
        orphans = [(key, st) for key, st in self._state.items()
                   if key[1] == zone.zone_id and st.episode is not None and key[0] not in present]
        if not orphans:
            return
        pairs = []
        for key, st in orphans:
            ep = st.episode
            h_old = ep.last_xyxy[3] - ep.last_xyxy[1]
            if h_old <= 0:
                continue
            px, py = self._expected_feet(st, ep, meta.media_s)
            near, cands = 0, []
            for t in tracks:
                ax, ay = anchor_point(t.xyxy)
                h_new = float(t.xyxy[3] - t.xyxy[1])
                if h_new <= 0 or math.hypot(ax - px, ay - py) / max(h_old, h_new) > zone.rejoin_dist_rel:
                    continue
                near += 1
                st_new = self._state.get((t.track_id, zone.zone_id))
                if st_new is not None and st_new.episode is not None:
                    continue  # already has its own episode here
                if self._first_seen.get(t.track_id, -1.0) <= ep.last_media_s:
                    continue  # this track existed while the person was still visible: a different person
                if not 0.5 <= h_new / h_old <= 2.0:
                    continue
                if zone.rejoin_strict and not zone.contains_px(ax, ay, meta.width, meta.height):
                    continue  # continuing an intrusion means standing inside the zone
                cands.append((math.hypot(ax - px, ay - py) / max(h_old, h_new), t.track_id))
            if zone.rejoin_strict and (near != 1 or len(cands) != 1):
                continue  # ambiguous: someone else is right there too
            pairs += [(d, tid, key) for d, tid in cands]
        if zone.rejoin_strict:  # a new track claimed by two vanished people is ambiguous as well
            claims: dict[int, int] = {}
            for _d, tid, _key in pairs:
                claims[tid] = claims.get(tid, 0) + 1
            pairs = [p for p in pairs if claims[p[1]] == 1]
        used_new, used_old = set(), set()
        for _d, tid, key in sorted(pairs):
            if tid in used_new or key in used_old:
                continue
            used_new.add(tid)
            used_old.add(key)
            old = self._state[key]
            ep = old.episode
            old.episode = None
            old.inside_count = old.outside_count = 0
            old.inside_since = old.outside_since = None
            new = self._state.setdefault((tid, zone.zone_id), _TrackZoneState())
            new.episode = ep
            new.inside_count = new.outside_count = 0
            new.inside_since = new.outside_since = None
            new.last_seen_media_s = meta.media_s
            ep.rejoined_track_ids.append(tid)
            self.rejoins += 1

    @staticmethod
    def _expected_feet(st: _TrackZoneState, ep: Episode, now: float) -> tuple[float, float]:
        """Where the vanished person's feet should be now: last position moved on at the walking velocity of the
        last second (a person walking behind a pillar comes out further along, not where they disappeared)."""
        lx, ly = (ep.last_xyxy[0] + ep.last_xyxy[2]) / 2, ep.last_xyxy[3]
        pts = [p for p in st.recent if ep.last_media_s - p[0] <= 1.0]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 0.2:
            return lx, ly
        dt = pts[-1][0] - pts[0][0]
        vx, vy = (pts[-1][1] - pts[0][1]) / dt, (pts[-1][2] - pts[0][2]) / dt
        gap = min(max(0.0, now - ep.last_media_s), ep.zone.lost_grace_seconds)
        return lx + vx * gap, ly + vy * gap

    def close_all(self, reason: CloseReason, meta: FrameMeta) -> list[EpisodeEvent]:
        out: list[EpisodeEvent] = []
        for st in self._state.values():
            if st.episode is not None:
                self._close(st, reason, meta, out)
        return out
