"""gRPC Ingest service (gateway → server) and the command hub (server → gateway)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from pathlib import Path

import grpc
from google.protobuf import json_format
from google.protobuf.timestamp_pb2 import Timestamp

from zonewatch.proto.zonewatch.v1 import zonewatch_pb2 as pb
from zonewatch.proto.zonewatch.v1 import zonewatch_pb2_grpc as pbg

from .db import Store
from .vlm.worker import VLMWorker

log = logging.getLogger(__name__)


def _ts(t: float) -> Timestamp:
    out = Timestamp()
    out.FromNanoseconds(int(t * 1e9))
    return out


class CommandHub:
    """Per-device asyncio queues feeding WatchCommands streams; undelivered commands are kept in SQLite
    so a device that connects later still receives them (bounded by age)."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def issue(self, device_id: str, type_: str, camera_id: str = "", source: str = "", zones: list[dict] | None = None,
              issued_by: str = "console") -> dict:
        cmd = {"command_id": uuid.uuid4().hex, "device_id": device_id, "type": type_, "camera_id": camera_id,
               "source": source, "zones": zones or [], "issued_at": time.time(), "issued_by": issued_by}
        self.store.add_command(cmd)
        for q in self._queues.get(device_id, []):
            q.put_nowait(cmd)
        return cmd

    def subscribe(self, device_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._queues[device_id].append(q)
        for cmd in self.store.undelivered_commands(device_id):
            q.put_nowait(json.loads(cmd["payload_json"]))
        return q

    def unsubscribe(self, device_id: str, q: asyncio.Queue) -> None:
        if q in self._queues.get(device_id, []):
            self._queues[device_id].remove(q)

    def subscribers(self, device_id: str) -> int:
        return len(self._queues.get(device_id, []))


def command_to_pb(cmd: dict) -> pb.Command:
    c = pb.Command(command_id=cmd["command_id"], device_id=cmd["device_id"], camera_id=cmd.get("camera_id", ""),
                   source=cmd.get("source", ""), issued_by=cmd.get("issued_by", ""),
                   type=pb.COMMAND_TYPE_START_CAMERA if cmd["type"] == "START_CAMERA" else pb.COMMAND_TYPE_STOP_CAMERA)
    c.issued_at.CopyFrom(_ts(cmd["issued_at"]))
    for z in cmd.get("zones", []) or []:
        zs = c.zones.add(zone_id=z["zone_id"], name=z.get("name", z["zone_id"]),
                         entry_confirm_frames=int(z.get("entry_confirm_frames", 3)),
                         exit_confirm_frames=int(z.get("exit_confirm_frames", 5)),
                         dwell_seconds=float(z.get("dwell_seconds", 5.0)),
                         lost_grace_seconds=float(z.get("lost_grace_seconds", 1.5)))
        for x, y in z["polygon"]:
            zs.polygon.add(x=float(x), y=float(y))
    return c


class IngestService(pbg.IngestServicer):
    def __init__(self, store: Store, evidence_dir: str | Path, worker: VLMWorker, hub: CommandHub,
                 ingest_source: str = "live") -> None:
        self.store = store
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.worker = worker
        self.hub = hub
        self.ingest_source = ingest_source
        self.counters = {"events": 0, "duplicates": 0, "uploads": 0, "upload_bytes": 0, "heartbeats": 0}

    async def PublishEvent(self, request: pb.Event, context) -> pb.Ack:
        ev = json.loads(json_format.MessageToJson(request, preserving_proto_field_name=True))
        if not ev.get("event_id"):
            return pb.Ack(accepted=False, message="event_id required")
        accepted, duplicate, rev = self.store.upsert_event(ev, self.ingest_source)
        self.counters["events" if accepted else "duplicates"] += 1
        if accepted:
            self.worker.maybe_enqueue_for_event(ev, accepted=True)
        return pb.Ack(event_id=ev["event_id"], revision=rev, accepted=accepted, duplicate=duplicate,
                      message="stored" if accepted else f"have revision {rev}")

    async def UploadEvidence(self, request_iterator, context) -> pb.UploadResult:
        event_id = path = ""
        h = hashlib.sha256()
        tmp = None
        size = 0
        expected_sha = ""
        try:
            async for chunk in request_iterator:
                if tmp is None:
                    event_id, path = chunk.event_id, chunk.path
                    if not event_id or not path or ".." in path or path.startswith("/"):
                        return pb.UploadResult(event_id=event_id, path=path, ok=False, message="bad path")
                    existing = self.store.has_evidence(event_id, path)
                    dest = self.evidence_dir / path
                    if existing and dest.exists() and existing.get("sha256") == (chunk.sha256 or existing.get("sha256")):
                        # drain and report already present (gateway treats this as success)
                        async for _ in request_iterator:
                            pass
                        return pb.UploadResult(event_id=event_id, path=path, ok=True, already_present=True,
                                               size=int(existing.get("size") or 0))
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    tmp.open("wb").close()
                with tmp.open("ab") as f:
                    f.write(chunk.data)
                h.update(chunk.data)
                size += len(chunk.data)
                if chunk.last:
                    expected_sha = chunk.sha256
            if tmp is None:
                return pb.UploadResult(ok=False, message="empty upload")
            digest = h.hexdigest()
            if expected_sha and expected_sha != digest:
                tmp.unlink(missing_ok=True)
                return pb.UploadResult(event_id=event_id, path=path, ok=False, message="sha256 mismatch")
            dest = self.evidence_dir / path
            tmp.replace(dest)
            self.store.record_evidence(event_id, path, digest, size)
            self.counters["uploads"] += 1
            self.counters["upload_bytes"] += size
            row = self.store.get_event(event_id)
            if row:
                self.worker.maybe_enqueue_for_event(json.loads(row["event_json"]), accepted=True)
            return pb.UploadResult(event_id=event_id, path=path, ok=True, size=size)
        except Exception as e:
            log.exception("upload failed")
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            return pb.UploadResult(event_id=event_id, path=path, ok=False, message=str(e)[:200])

    async def Heartbeat(self, request: pb.DeviceStatus, context) -> pb.HeartbeatAck:
        status = json.loads(json_format.MessageToJson(request, preserving_proto_field_name=True))
        self.store.upsert_device(request.device_id, status, request.gateway_version, request.analyzer_run_id)
        self.counters["heartbeats"] += 1
        ack = pb.HeartbeatAck()
        ack.server_ts.CopyFrom(_ts(time.time()))
        return ack

    async def WatchCommands(self, request: pb.WatchRequest, context):
        q = self.hub.subscribe(request.device_id)
        log.info("device %s watching commands", request.device_id)
        try:
            while True:
                try:
                    cmd = await asyncio.wait_for(q.get(), timeout=15.0)
                except TimeoutError:
                    if context.cancelled() if hasattr(context, "cancelled") else False:
                        break
                    continue
                self.store.mark_command_delivered(cmd["command_id"])
                yield command_to_pb(cmd)
        finally:
            self.hub.unsubscribe(request.device_id, q)
            log.info("device %s command stream closed", request.device_id)

    async def PublishCommandResult(self, request: pb.CommandResult, context) -> pb.Empty:
        self.store.complete_command(request.command_id, request.ok, request.message)
        return pb.Empty()


async def serve_grpc(service: IngestService, listen: str) -> grpc.aio.Server:
    server = grpc.aio.server(options=[("grpc.max_receive_message_length", 32 << 20)])
    pbg.add_IngestServicer_to_server(service, server)
    server.add_insecure_port(listen)
    await server.start()
    log.info("gRPC Ingest listening on %s", listen)
    return server
