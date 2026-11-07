"""Per-camera analysis pipeline: source → bounded queue → detect → track → zones → evidence → events.

Threads per camera:
  * source thread (inside VideoSource) pushes frames into FrameQueue (drop-oldest when full)
  * worker thread: the analysis loop below
  * finalizer thread: waits for post-roll frames after an episode closes, writes clip + context_after,
    then emits one more revision. Keeps the worker loop free of ffmpeg time.
The pipeline never blocks on the sink; events go through EventSink.submit (non-blocking).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import psutil

from .config import CameraConfig, EdgeConfig
from .detector import PersonDetector
from .events import EvidenceState, RunContext, episode_to_event, EventSink
from .evidence import EvidenceWriter
from .frames import FrameQueue, FrameRing, StageTimer
from .sources import SourceState, VideoSource, open_source
from .tracker import ByteTracker, TrackerParams
from .zones import CloseReason, Episode, FrameMeta, Transition, ZoneMonitor

log = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    frames_processed: int = 0
    processed_fps: float = 0.0
    analysis_lag_ms: float = 0.0
    media_time_s: float = 0.0
    open_episodes: int = 0
    events_emitted: int = 0
    last_error: str = ""
    _fps_window: list[float] = field(default_factory=list)


class CameraPipeline:
    def __init__(self, cfg: EdgeConfig, cam: CameraConfig, detector: PersonDetector, sink: EventSink,
                 run_id: str, edge_version: str, source_override: str | None = None,
                 zones_override: list | None = None) -> None:
        self.cfg = cfg
        self.cam = cam
        self.source_uri = source_override or cam.source
        self.detector = detector
        self.sink = sink
        pc = cfg.pipeline
        # realtime=False means 'analyse every frame' (offline/eval): block the decoder instead of dropping
        self.queue = FrameQueue(pc.queue_capacity, lossless=not cam.realtime)
        self.ring = FrameRing(pc.ring_seconds, pc.ring_max_mb << 20, pc.ring_max_side, pc.ring_jpeg_quality)
        self.evidence = EvidenceWriter(cfg.evidence_root, pc.pre_roll_s, pc.post_roll_s, pc.clip_fps, pc.clip_max_frames)
        tp = TrackerParams(cfg.tracker.track_thresh, cfg.tracker.low_thresh, cfg.tracker.new_track_thresh,
                           cfg.tracker.match_thresh, track_buffer=max(5, int(cfg.tracker.track_buffer_seconds * cam.fps_hint)),
                           fuse_score=cfg.tracker.fuse_score)
        self.tracker = ByteTracker(tp)
        zones = zones_override if zones_override is not None else [z.to_spec() for z in cam.zones]
        self.monitor = ZoneMonitor(zones, cam.camera_id, pc.trajectory_stride)
        self.ctx = RunContext(run_id, cfg.device_id, cam.camera_id, self.source_uri, edge_version,
                              detector.info.model_name, detector.info.model_version, detector.info.backend,
                              f"bytetrack@{tp.hash()}", cfg.config_hash(),
                              (detector.info.input_width, detector.info.input_height))
        self.source: VideoSource = open_source(self.source_uri, self.queue, pc.source_backend, realtime=cam.realtime,
                                               loop=cam.loop, speed=cam.speed, rtsp_latency_ms=pc.rtsp_latency_ms,
                                               rtsp_protocols=pc.rtsp_protocols, max_width=cam.max_width)
        self.timer = StageTimer()
        self.stats = PipelineStats()
        self._evidence_state: dict[str, EvidenceState] = {}
        self._pending_final: list[tuple[Episode, float]] = []  # (episode, finalize_after_media_s)
        self._final_cv = threading.Condition()
        self._stop = threading.Event()
        self._frame_wh = (0, 0)
        self._threads: list[threading.Thread] = []
        self.ended = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.source.start()
        for name, target in (("worker", self._worker), ("finalizer", self._finalizer)):
            t = threading.Thread(target=target, name=f"{name}:{self.cam.camera_id}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("camera %s started: %s", self.cam.camera_id, self.source_uri)

    def stop(self, reason: CloseReason = CloseReason.SHUTDOWN) -> None:
        self._stop.set()
        self.source.stop()
        self.queue.close()
        with self._final_cv:
            self._final_cv.notify_all()
        for t in self._threads:
            t.join(10.0)
        if self.monitor.open_episodes:
            meta = self._last_meta()
            for ev in self.monitor.close_all(reason, meta):
                self._on_transition(ev.transition, ev.episode)
            self._finalize_pending(force=True)
        log.info("camera %s stopped", self.cam.camera_id)

    def _last_meta(self) -> FrameMeta:
        return FrameMeta(self.stats.frames_processed, self.stats.media_time_s, time.time(), *self._frame_wh)

    # ------------------------------------------------------------------ worker loop
    def _worker(self) -> None:
        det, trk, mon, ring, timer = self.detector, self.tracker, self.monitor, self.ring, self.timer
        while not self._stop.is_set():
            frame = self.queue.get(0.5)
            if frame is None:
                if self.source.status.state in (SourceState.ENDED, SourceState.ERROR) and len(self.queue) == 0:
                    if mon.open_episodes:
                        for ev in mon.close_all(CloseReason.STREAM_ENDED, self._last_meta()):
                            self._on_transition(ev.transition, ev.episode)
                    self.ended = True
                    with self._final_cv:
                        self._final_cv.notify_all()
                    return
                continue
            try:
                t0 = time.perf_counter()
                ring.append(frame)
                t1 = time.perf_counter()
                d = det.detect(frame.image)
                t2 = time.perf_counter()
                tracks = trk.update(d.boxes, d.scores)
                t3 = time.perf_counter()
                self._frame_wh = (frame.width, frame.height)
                meta = FrameMeta(frame.index, frame.media_s, frame.wall_ts, frame.width, frame.height)
                transitions = mon.update(tracks, meta)
                for ev in transitions:
                    self._on_transition(ev.transition, ev.episode)
                t4 = time.perf_counter()
                # pin the ring so pre-roll of the oldest open episode survives
                if mon.open_episodes:
                    ring.pinned_from = min(e.entry_media_s for e in mon.open_episodes.values()) - self.cfg.pipeline.pre_roll_s
                else:
                    ring.pinned_from = None
                timer.add("evidence_encode", (t1 - t0) * 1e3)
                timer.add("preprocess", d.timings_ms["preprocess"])
                timer.add("infer", d.timings_ms["infer"])
                timer.add("postprocess", d.timings_ms["postprocess"])
                timer.add("track", (t3 - t2) * 1e3)
                timer.add("zones", (t4 - t3) * 1e3)
                timer.add("total", (t4 - t0) * 1e3)
                now = time.time()
                st = self.stats
                st.frames_processed += 1
                st.media_time_s = frame.media_s
                st.analysis_lag_ms = (now - frame.wall_ts) * 1e3
                st.open_episodes = len(mon.open_episodes)
                st._fps_window.append(now)
                st._fps_window = [t for t in st._fps_window if now - t <= 2.0]
                st.processed_fps = len(st._fps_window) / 2.0
                with self._final_cv:
                    self._final_cv.notify_all()
            except Exception as e:  # noqa: BLE001 - keep analysing; report the error
                log.exception("frame %s failed", frame.index)
                self.stats.last_error = str(e)[:200]

    # ------------------------------------------------------------------ transitions
    def _on_transition(self, tr: Transition, ep: Episode) -> None:
        ev = self._evidence_state.setdefault(ep.episode_id, EvidenceState())
        try:
            if tr == Transition.OPENED:
                for kf in self.evidence.keyframes_for(ep, self.ring, self._frame_wh, ["context_before", "entry"]):
                    ev.key_frames[kf.role] = kf
            elif tr == Transition.DWELL:
                for kf in self.evidence.keyframes_for(ep, self.ring, self._frame_wh, ["dwell"]):
                    ev.key_frames[kf.role] = kf
            elif tr == Transition.CLOSED:
                for kf in self.evidence.keyframes_for(ep, self.ring, self._frame_wh, ["exit"]):
                    ev.key_frames[kf.role] = kf
                with self._final_cv:
                    end = ep.exit_media_s if ep.exit_media_s is not None else ep.last_media_s
                    self._pending_final.append((ep, end + self.cfg.pipeline.post_roll_s))
                    self._final_cv.notify_all()
        except Exception as e:  # noqa: BLE001
            log.exception("evidence for %s failed", ep.episode_id)
            self.stats.last_error = f"evidence: {e}"[:200]
        self._emit(ep)

    def _emit(self, ep: Episode) -> None:
        ev = self._evidence_state.setdefault(ep.episode_id, EvidenceState())
        self.sink.submit(episode_to_event(ep, self.ctx, ev, self._frame_wh))
        self.stats.events_emitted += 1

    # ------------------------------------------------------------------ finalizer
    def _finalizer(self) -> None:
        while not self._stop.is_set():
            with self._final_cv:
                self._final_cv.wait(0.5)
            self._finalize_pending(force=self.ended)
            if self.ended and not self._pending_final:
                return

    def _finalize_pending(self, force: bool = False) -> None:
        with self._final_cv:
            ready = [(ep, t) for ep, t in self._pending_final if force or self.stats.media_time_s >= t]
            self._pending_final = [(ep, t) for ep, t in self._pending_final if (ep, t) not in ready]
        for ep, _ in ready:
            ev = self._evidence_state.setdefault(ep.episode_id, EvidenceState())
            try:
                t0 = time.perf_counter()
                for kf in self.evidence.keyframes_for(ep, self.ring, self._frame_wh, ["context_after"]):
                    ev.key_frames[kf.role] = kf
                clip = self.evidence.write_clip(ep, self.ring)
                if clip is not None:
                    ev.clip = clip
                self.timer.add("clip", (time.perf_counter() - t0) * 1e3)
            except Exception as e:  # noqa: BLE001
                log.exception("clip for %s failed", ep.episode_id)
                self.stats.last_error = f"clip: {e}"[:200]
            ep.revision += 1
            self._emit(ep)
            self._evidence_state.pop(ep.episode_id, None)

    # ------------------------------------------------------------------ status
    def camera_status(self):
        from zonewatch.proto.zonewatch.v1 import zonewatch_pb2 as pb

        from .events import ts

        st, src, q = self.stats, self.source.status, self.queue.stats()
        state = {
            SourceState.CONNECTING: pb.CAMERA_STATE_CONNECTING, SourceState.STREAMING: pb.CAMERA_STATE_STREAMING,
            SourceState.RECONNECTING: pb.CAMERA_STATE_RECONNECTING, SourceState.ENDED: pb.CAMERA_STATE_ENDED,
            SourceState.STOPPED: pb.CAMERA_STATE_STOPPED, SourceState.ERROR: pb.CAMERA_STATE_ERROR,
        }[src.state]
        cs = pb.CameraStatus(
            camera_id=self.cam.camera_id, source=self.source_uri, name=self.cam.name, state=state,
            input_fps=src.input_fps(), processed_fps=st.processed_fps, analysis_lag_ms=st.analysis_lag_ms,
            queue_depth=q.depth, queue_capacity=q.capacity, dropped_frames=q.dropped, frames_received=q.received,
            frames_processed=st.frames_processed, infer_ms_p50=self.timer.percentile("infer", 50),
            infer_ms_p95=self.timer.percentile("infer", 95), stage_ms_decode=0.0,
            stage_ms_preprocess=self.timer.mean("preprocess"), stage_ms_infer=self.timer.mean("infer"),
            stage_ms_postprocess=self.timer.mean("postprocess"), stage_ms_track=self.timer.mean("track"),
            stage_ms_evidence=self.timer.mean("evidence_encode"), reconnect_count=src.reconnects,
            last_error=(src.last_error or st.last_error)[:200], open_episodes=st.open_episodes,
            ring_buffer_bytes=self.ring.stats().bytes, media_time_s=st.media_time_s,
        )
        if src.last_frame_wall_ts:
            cs.last_frame_ts.CopyFrom(ts(src.last_frame_wall_ts))
        for z in self.monitor.zones:
            zs = cs.zones.add(zone_id=z.zone_id, name=z.name, entry_confirm_frames=z.entry_confirm_frames,
                              exit_confirm_frames=z.exit_confirm_frames, dwell_seconds=z.dwell_seconds,
                              lost_grace_seconds=z.lost_grace_seconds)
            for x, y in z.polygon:
                zs.polygon.add(x=x, y=y)
        return cs


def process_cpu_rss() -> tuple[float, float]:
    p = psutil.Process()
    return p.cpu_percent(interval=None), p.memory_info().rss / (1 << 20)
