"""EdgeAgent: owns the detector, the camera pipelines, the status reporter and the command subscriber."""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading
import time
import uuid
from pathlib import Path

from sceneproof import __version__
from sceneproof.proto.sceneproof.v1 import sceneproof_pb2 as pb

from .config import CameraConfig, EdgeConfig
from .detector import build_detector
from .events import EventSink, GatewaySink, JsonlSink, ts
from .pipeline import CameraPipeline, process_cpu_rss
from .zone_proto import fill_pb, from_pb
from .zones import CloseReason, ZoneSpec

log = logging.getLogger(__name__)


def edge_version() -> str:
    sha = ""
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2,
                             cwd=Path(__file__).resolve().parents[3]).stdout.strip()
    return f"{__version__}+{sha}" if sha else __version__


class EdgeAgent:
    def __init__(self, cfg: EdgeConfig, sink: EventSink | None = None, run_id: str | None = None) -> None:
        self.cfg = cfg
        self.run_id = run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.version = edge_version()
        self.detector = build_detector(cfg.detector.model_dump())
        self.detector.warmup()
        self.sink = sink or GatewaySink(cfg.gateway)
        self.pipelines: dict[str, CameraPipeline] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        log.info("edge agent %s run=%s detector=%s/%s backend=%s", cfg.device_id, self.run_id,
                 self.detector.info.model_name, self.detector.info.model_version, self.detector.info.backend)

    # ------------------------------------------------------------------ cameras
    def start_camera(self, camera_id: str, source: str | None = None, zones: list[ZoneSpec] | None = None) -> str:
        cam = self.cfg.camera(camera_id)
        if cam is None:
            if source is None:
                raise KeyError(f"unknown camera {camera_id}")
            cam = CameraConfig(camera_id=camera_id, name=camera_id, source=source)
        with self._lock:
            old = self.pipelines.get(camera_id)
            if old is not None and not old.ended and old.source_uri == (source or cam.source):
                return f"{camera_id} already running ({old.source_uri})"  # idempotent: repeated START is a no-op
            old = self.pipelines.pop(camera_id, None)
        if old is not None:
            old.stop(CloseReason.SHUTDOWN)
        p = CameraPipeline(self.cfg, cam, self.detector, self.sink, self.run_id, self.version, source, zones)
        p.start()
        with self._lock:
            self.pipelines[camera_id] = p
        return f"started {camera_id} ({p.source_uri})"

    def stop_camera(self, camera_id: str) -> str:
        with self._lock:
            p = self.pipelines.pop(camera_id, None)
        if p is None:
            return f"{camera_id} not running"
        p.stop(CloseReason.SHUTDOWN)
        return f"stopped {camera_id}"

    def autostart(self) -> None:
        for cam in self.cfg.cameras:
            if cam.autostart:
                self.start_camera(cam.camera_id)

    # ------------------------------------------------------------------ status
    def device_status(self) -> pb.DeviceStatus:
        cpu, rss = process_cpu_rss()
        ds = pb.DeviceStatus(device_id=self.cfg.device_id, analyzer_run_id=self.run_id, cpu_percent=cpu, rss_mb=rss)
        ds.ts.CopyFrom(ts(time.time()))
        with self._lock:
            running = dict(self.pipelines)
        for cam in self.cfg.cameras:  # configured but idle cameras are reported so the console can start them
            p = running.get(cam.camera_id)
            if p is not None:
                ds.cameras.append(p.camera_status())
            else:
                cs = ds.cameras.add(camera_id=cam.camera_id, source=cam.source, name=cam.name,
                                    state=pb.CAMERA_STATE_STOPPED)
                for z in cam.zones:
                    fill_pb(z.to_spec(), cs.zones.add())
        for cid, p in running.items():
            if self.cfg.camera(cid) is None:
                ds.cameras.append(p.camera_status())
        return ds

    def _status_loop(self) -> None:
        sink = self.sink
        while not self._stop.is_set():
            if isinstance(sink, GatewaySink):
                sink.report_status(self.device_status())
            self._stop.wait(self.cfg.status_interval_s)

    # ------------------------------------------------------------------ commands
    def _command_loop(self) -> None:
        import grpc

        sink = self.sink
        if not isinstance(sink, GatewaySink):
            return
        backoff = 0.5
        while not self._stop.is_set():
            try:
                stream = sink.stub().SubscribeCommands(pb.WatchRequest(device_id=self.cfg.device_id))
                backoff = 0.5
                for cmd in stream:
                    if self._stop.is_set():
                        break
                    self._handle_command(cmd)
            except grpc.RpcError as e:
                if self._stop.is_set():
                    return
                log.debug("command stream closed: %s", e.code().name)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 10.0)

    def _handle_command(self, cmd: pb.Command) -> None:
        res = pb.CommandResult(command_id=cmd.command_id, device_id=self.cfg.device_id)
        try:
            if cmd.type == pb.COMMAND_TYPE_START_CAMERA:
                zones = None
                if cmd.zones:
                    zones = [from_pb(z) for z in cmd.zones]
                res.message = self.start_camera(cmd.camera_id, cmd.source or None, zones)
                res.ok = True
            elif cmd.type == pb.COMMAND_TYPE_STOP_CAMERA:
                res.message = self.stop_camera(cmd.camera_id)
                res.ok = True
            else:
                res.message = f"unsupported command type {cmd.type}"
        except Exception as e:
            res.message = f"{type(e).__name__}: {e}"[:200]
            log.exception("command %s failed", cmd.command_id)
        res.completed_at.CopyFrom(ts(time.time()))
        if isinstance(self.sink, GatewaySink):
            try:
                self.sink.stub().ReportCommandResult(res, timeout=2.0)
            except Exception:
                log.warning("could not report command result %s", cmd.command_id)
        log.info("command %s %s: %s", cmd.command_id, "ok" if res.ok else "FAILED", res.message)

    # ------------------------------------------------------------------ lifecycle
    def run_forever(self) -> None:
        for name, target in (("status", self._status_loop), ("commands", self._command_loop)):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        try:
            while not self._stop.is_set():
                self._stop.wait(1.0)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        with self._lock:
            ps = list(self.pipelines.values())
            self.pipelines.clear()
        for p in ps:
            p.stop(CloseReason.SHUTDOWN)
        self.sink.close()

    def all_ended(self) -> bool:
        with self._lock:
            return bool(self.pipelines) and all(p.ended for p in self.pipelines.values())


def run_offline(cfg: EdgeConfig, camera_id: str, out_jsonl: str, source: str | None = None,
                realtime: bool = False, run_id: str | None = None) -> dict:
    """Process one camera's source to completion without a gateway; events go to a JSONL file."""
    cam = cfg.camera(camera_id)
    if cam is None:
        raise KeyError(camera_id)
    cam = cam.model_copy(update={"realtime": realtime, "loop": False})
    cfg = cfg.model_copy(update={"cameras": [cam if c.camera_id == camera_id else c for c in cfg.cameras]})
    sink = JsonlSink(out_jsonl)
    agent = EdgeAgent(cfg, sink=sink, run_id=run_id)
    t0 = time.perf_counter()
    agent.start_camera(camera_id, source)
    p = agent.pipelines[camera_id]
    while not p.ended:
        time.sleep(0.2)
    # finalizer drains pending clips when ended; give it a moment
    deadline = time.time() + 30
    while p._pending_final and time.time() < deadline:
        time.sleep(0.2)
    wall = time.perf_counter() - t0
    status = p.camera_status()
    agent.shutdown()
    return {"run_id": agent.run_id, "frames": status.frames_processed, "wall_s": round(wall, 2),
            "fps": round(status.frames_processed / wall, 2) if wall else 0, "events_written": sink.count,
            "timings": p.timer.snapshot(), "dropped": status.dropped_frames, "detector": agent.detector.info.__dict__}
