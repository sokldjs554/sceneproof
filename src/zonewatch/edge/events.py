"""Episode -> protobuf Event conversion and event sinks (gateway gRPC / JSONL file)."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from google.protobuf import json_format
from google.protobuf.timestamp_pb2 import Timestamp

from zonewatch.proto.zonewatch.v1 import zonewatch_pb2 as pb
from zonewatch.proto.zonewatch.v1 import zonewatch_pb2_grpc as pbg

from .evidence import ClipOut, KeyFrameOut
from .zones import CloseReason, Episode

log = logging.getLogger(__name__)


def ts(t: float) -> Timestamp:
    out = Timestamp()
    out.FromNanoseconds(int(t * 1e9))
    return out


_CLOSE = {
    CloseReason.EXITED: pb.CLOSE_REASON_EXITED,
    CloseReason.TRACK_LOST: pb.CLOSE_REASON_TRACK_LOST,
    CloseReason.STREAM_ENDED: pb.CLOSE_REASON_STREAM_ENDED,
    CloseReason.SHUTDOWN: pb.CLOSE_REASON_SHUTDOWN,
}


@dataclass
class RunContext:
    run_id: str
    device_id: str
    camera_id: str
    source: str
    camera_name: str
    edge_version: str
    model_name: str
    model_version: str
    backend: str
    tracker: str
    config_hash: str
    input_wh: tuple[int, int]


@dataclass
class EvidenceState:
    key_frames: dict[str, KeyFrameOut] = field(default_factory=dict)
    clip: ClipOut | None = None


def episode_to_event(ep: Episode, ctx: RunContext, ev: EvidenceState, frame_wh: tuple[int, int]) -> pb.Event:
    e = pb.Event(
        event_id=ep.episode_id, revision=ep.revision, device_id=ctx.device_id, camera_id=ctx.camera_id,
        zone_id=ep.zone.zone_id, zone_name=ep.zone.name,
        kind=pb.EVENT_KIND_ZONE_DWELL if ep.dwell_escalated else pb.EVENT_KIND_ZONE_ENTRY,
        status=pb.EVENT_STATUS_CLOSED if ep.closed else pb.EVENT_STATUS_OPEN,
        close_reason=_CLOSE.get(ep.close_reason, pb.CLOSE_REASON_UNSPECIFIED) if ep.closed else pb.CLOSE_REASON_UNSPECIFIED,
        track_id=ep.track_id, entry_frame_index=ep.entry_frame, media_entry_s=ep.entry_media_s,
        dwell_seconds=ep.dwell_seconds, frame_width=frame_wh[0], frame_height=frame_wh[1], source=ctx.source,
        camera_name=ctx.camera_name,
    )
    e.entry_ts.CopyFrom(ts(ep.entry_wall_ts))
    e.detected_at.CopyFrom(ts(time.time()))
    if ep.closed and ep.exit_wall_ts is not None:
        e.exit_ts.CopyFrom(ts(ep.exit_wall_ts))
        e.exit_frame_index = ep.exit_frame or 0
        e.media_exit_s = ep.exit_media_s or 0.0
    for x, y in ep.zone.polygon:
        e.zone_polygon.add(x=x, y=y)
    traj = ep.trajectory
    if len(traj) > 60:  # keep the wire message small; the stride is uniform
        step = len(traj) / 60
        traj = [traj[int(i * step)] for i in range(60)]
    for p in traj:
        tp = e.trajectory.add(frame_index=p.frame_index, media_time_s=p.media_s, score=p.score, in_zone=p.in_zone)
        tp.ts.CopyFrom(ts(p.wall_ts))
        tp.box.CopyFrom(pb.BBox(x1=p.xyxy[0], y1=p.xyxy[1], x2=p.xyxy[2], y2=p.xyxy[3]))
    for role in ("context_before", "entry", "dwell", "exit", "context_after"):
        kf = ev.key_frames.get(role)
        if kf is None:
            continue
        k = e.evidence.key_frames.add(role=role, frame_index=kf.index, media_time_s=kf.media_s, path=kf.path,
                                      width=kf.width, height=kf.height)
        k.ts.CopyFrom(ts(kf.wall_ts))
        if kf.box is not None:
            k.box.CopyFrom(pb.BBox(x1=kf.box[0], y1=kf.box[1], x2=kf.box[2], y2=kf.box[3]))
        # annotated twin travels as a second key frame so the server can show it without re-drawing
        k2 = e.evidence.key_frames.add(role=f"{role}_annot", frame_index=kf.index, media_time_s=kf.media_s,
                                       path=kf.annot_path, width=kf.width, height=kf.height)
        k2.ts.CopyFrom(ts(kf.wall_ts))
    if ev.clip is not None:
        e.evidence.clip_path = ev.clip.path
        e.evidence.clip_start_media_s = ev.clip.start_media_s
        e.evidence.clip_end_media_s = ev.clip.end_media_s
        e.evidence.clip_fps = ev.clip.fps
        e.evidence.clip_frames = ev.clip.frames
    e.run.CopyFrom(pb.RunInfo(run_id=ctx.run_id, edge_version=ctx.edge_version, model_name=ctx.model_name,
                              model_version=ctx.model_version, backend=ctx.backend, tracker=ctx.tracker,
                              config_hash=ctx.config_hash, input_width=ctx.input_wh[0], input_height=ctx.input_wh[1]))
    return e


class EventSink:
    def submit(self, event: pb.Event) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def stats(self) -> dict:
        return {}


class JsonlSink(EventSink):
    """Appends every revision as one JSON line (offline runs, evaluation, recorded demos)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.count = 0

    def submit(self, event: pb.Event) -> None:
        line = json_format.MessageToJson(event, preserving_proto_field_name=True, indent=None)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()
            self.count += 1

    def close(self) -> None:
        with self._lock:
            self._f.close()

    def stats(self) -> dict:
        return {"written": self.count}


class GatewaySink(EventSink):
    """Non-blocking gRPC submission to the local gateway with a tiny coalescing retry outbox.

    The gateway is the durable store; this outbox only bridges the moments when the gateway
    process itself is down (restart, crash). It coalesces by event_id so a backlog never
    carries stale revisions, and it is bounded (oldest dropped, counted) so the analyser's
    memory stays flat if the gateway is gone for a long time.
    """

    def __init__(self, address: str, max_outbox: int = 512, timeout_s: float = 2.0) -> None:
        import grpc

        self.address = address
        self.timeout_s = timeout_s
        self.max_outbox = max_outbox
        self._channel = grpc.insecure_channel(address)
        self._stub = pbg.EdgeIngressStub(self._channel)
        self._outbox: OrderedDict[str, pb.Event] = OrderedDict()
        self._cv = threading.Condition()
        self._stop = False
        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self.connected = False
        self.last_error = ""
        self._thread = threading.Thread(target=self._loop, name="gateway-sink", daemon=True)
        self._thread.start()

    def submit(self, event: pb.Event) -> None:
        with self._cv:
            if event.event_id in self._outbox:
                if self._outbox[event.event_id].revision > event.revision:
                    return
                del self._outbox[event.event_id]
            while len(self._outbox) >= self.max_outbox:
                self._outbox.popitem(last=False)
                self.dropped += 1
            self._outbox[event.event_id] = event
            self._cv.notify()

    def _loop(self) -> None:
        import grpc

        backoff = 0.5
        while True:
            with self._cv:
                while not self._outbox and not self._stop:
                    self._cv.wait(1.0)
                if self._stop and not self._outbox:
                    return
                eid, ev = next(iter(self._outbox.items()))
            try:
                ack = self._stub.SubmitEvent(ev, timeout=self.timeout_s)
                self.connected = True
                backoff = 0.5
                if not ack.accepted and not ack.duplicate:
                    self.last_error = f"gateway rejected {eid}: {ack.message}"
                with self._cv:
                    cur = self._outbox.get(eid)
                    if cur is not None and cur.revision == ev.revision:
                        del self._outbox[eid]
                self.sent += 1
            except grpc.RpcError as e:
                self.connected = False
                self.failed += 1
                self.last_error = f"{e.code().name}: {e.details()}"[:200]
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

    def report_status(self, status: pb.DeviceStatus) -> bool:
        import grpc

        try:
            self._stub.ReportStatus(status, timeout=self.timeout_s)
            self.connected = True
            return True
        except grpc.RpcError as e:
            self.connected = False
            self.last_error = f"{e.code().name}: {e.details()}"[:200]
            return False

    def stub(self) -> pbg.EdgeIngressStub:
        return self._stub

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(3.0)
        self._channel.close()

    def stats(self) -> dict:
        with self._cv:
            pending = len(self._outbox)
        return {"sent": self.sent, "failed": self.failed, "dropped": self.dropped, "pending": pending,
                "connected": self.connected, "last_error": self.last_error}


def event_to_dict(e: pb.Event) -> dict:
    return json.loads(json_format.MessageToJson(e, preserving_proto_field_name=True))
