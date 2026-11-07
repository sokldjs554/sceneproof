"""HTTP API for the review console. Plain JSON; the console is static files served from /."""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from zonewatch import __version__

from .db import Store

REVIEW_STATES = ("needs_review", "in_review", "resolved")
ALLOWED_TRANSITIONS = {
    "needs_review": {"in_review", "resolved"},
    "in_review": {"resolved", "needs_review"},
    "resolved": {"in_review"},  # reopen
}
RESOLUTIONS = ("confirmed", "dismissed", "escalated", "no_action")


class ReviewBody(BaseModel):
    state: str
    actor: str = Field(default="담당자", max_length=80)
    note: str = Field(default="", max_length=2000)
    resolution: str | None = None


class CommandBody(BaseModel):
    type: str  # START_CAMERA | STOP_CAMERA
    camera_id: str = ""
    source: str = ""
    zones: list[dict] | None = None
    issued_by: str = "console"


def _event_view(store: Store, row: dict, expl: dict | None, devices: dict[str, dict]) -> dict:
    ev = json.loads(row["event_json"])
    cam_name = row.get("camera_name") or _camera_name(devices, row.get("device_id"), row.get("camera_id")) or row.get("camera_id")
    evidence = ev.get("evidence", {})
    files = {f["path"]: f for f in store.evidence_files(row["event_id"])}
    key_frames = []
    for k in evidence.get("key_frames", []):
        key_frames.append({**k, "available": k["path"] in files, "url": f"/api/evidence/{k['path']}"})
    clip = evidence.get("clip_path")
    out = {
        "event_id": row["event_id"], "revision": row["revision"], "device_id": row["device_id"],
        "camera_id": row["camera_id"], "camera_name": cam_name, "zone_id": row["zone_id"], "zone_name": row["zone_name"],
        "kind": row["kind"], "status": row["status"], "close_reason": row["close_reason"], "track_id": row["track_id"],
        "entry_ts": row["entry_ts"], "exit_ts": row["exit_ts"], "detected_at": row["detected_at"],
        "first_received_at": row["first_received_at"], "received_at": row["received_at"],
        "first_displayed_at": row.get("first_displayed_at"),
        "media_entry_s": row["media_entry_s"], "media_exit_s": row["media_exit_s"], "dwell_seconds": row["dwell_seconds"],
        "source": row["source"], "ingest_source": row["ingest_source"], "evidence_ready": bool(row["evidence_ready"]),
        "review": {"state": row["review_state"], "resolution": row.get("review_resolution"), "note": row.get("review_note"),
                   "actor": row.get("review_actor"), "updated_at": row.get("review_updated_at")},
        "evidence": {"key_frames": key_frames, "clip_path": clip, "clip_available": bool(clip and clip in files),
                     "clip_url": f"/api/evidence/{clip}" if clip else None,
                     "clip_start_media_s": evidence.get("clip_start_media_s"), "clip_end_media_s": evidence.get("clip_end_media_s"),
                     "clip_fps": evidence.get("clip_fps")},
        "zone_polygon": ev.get("zone_polygon", []), "frame_width": ev.get("frame_width"), "frame_height": ev.get("frame_height"),
        "trajectory": ev.get("trajectory", []), "run": ev.get("run", {}),
        "explanation": _expl_view(expl),
        "latency": {
            # ingest latency only makes sense for live ingestion; replayed bundles carry old detected_at values
            "ingest_ms": (row["first_received_at"] - row["detected_at"]) * 1e3
            if row.get("detected_at") and row.get("ingest_source") == "live" else None,
            "first_display_ms": (row["first_displayed_at"] - row["entry_ts"]) * 1e3 if row.get("first_displayed_at") and row.get("entry_ts") else None,
        },
    }
    return out


def _expl_view(x: dict | None) -> dict | None:
    if not x:
        return None
    return {"id": x["id"], "status": x["status"], "provider": x["provider"], "model": x["model"],
            "prompt_version": x["prompt_version"], "schema_version": x["schema_version"], "attempts": x["attempts"],
            "requested_at": x["requested_at"], "started_at": x.get("started_at"), "completed_at": x.get("completed_at"),
            "latency_ms": x.get("latency_ms"), "error": x.get("error"),
            "result": json.loads(x["result_json"]) if x.get("result_json") else None,
            "usage": json.loads(x["usage_json"]) if x.get("usage_json") else {},
            "input": json.loads(x["input_json"]) if x.get("input_json") else {}, "raw_text": x.get("raw_text"),
            "event_revision": x["event_revision"]}


def _camera_name(devices: dict[str, dict], device_id: str | None, camera_id: str | None) -> str | None:
    d = devices.get(device_id or "")
    if not d:
        return None
    for c in d.get("status", {}).get("cameras", []):
        if c.get("camera_id") == camera_id:
            return c.get("name") or None
    return None


def build_router(app_state: Any) -> APIRouter:
    r = APIRouter(prefix="/api")
    store: Store = app_state.store
    s = app_state.settings

    def devices_view() -> dict[str, dict]:
        now = time.time()
        out = {}
        for d in store.devices():
            status = json.loads(d["status_json"])
            age = now - d["last_seen_ts"]
            out[d["device_id"]] = {"device_id": d["device_id"], "last_seen_ts": d["last_seen_ts"], "age_s": round(age, 1),
                                   "online": age <= s.device_offline_after_s, "gateway_version": d["gateway_version"],
                                   "analyzer_run_id": d["analyzer_run_id"], "status": status,
                                   "command_subscribers": app_state.hub.subscribers(d["device_id"])}
        return out

    @r.get("/health")
    def health():
        return {"ok": True, "version": __version__, "mode": s.mode, "time": time.time()}

    @r.get("/status")
    def status():
        return {"mode": s.mode, "version": __version__, "public_url": s.public_url, "devices": devices_view(),
                "events": store.count_events(), "vlm": app_state.worker.snapshot(),
                "ingest": app_state.ingest.counters if app_state.ingest else {}, "server_time": time.time(),
                "grpc_listen": s.grpc_listen}

    @r.get("/events")
    def list_events(camera_id: str | None = None, state: str | None = None, kind: str | None = None,
                    device_id: str | None = None, since: float | None = None, until: float | None = None,
                    limit: int = Query(200, le=2000), offset: int = 0):
        rows = store.list_events(camera_id, state, kind, since, until, limit, offset, device_id)
        expl = store.latest_explanation_map([x["event_id"] for x in rows])
        devs = devices_view()
        items = [_event_view(store, x, expl.get(x["event_id"]), devs) for x in rows]
        for it in items:  # list view: keep payload small
            it.pop("trajectory", None)
            if it["explanation"]:
                it["explanation"].pop("raw_text", None)
        return {"items": items, "count": len(items), "counts": store.count_events()}

    @r.get("/events/{event_id}")
    def get_event(event_id: str):
        row = store.get_event(event_id)
        if not row:
            raise HTTPException(404, "event not found")
        expls = store.explanations_for(event_id)
        view = _event_view(store, row, expls[0] if expls else None, devices_view())
        view["explanations"] = [_expl_view(x) for x in expls]
        view["review_log"] = store.review_log(event_id)
        view["evidence_files"] = store.evidence_files(event_id)
        return view

    @r.post("/events/{event_id}/review")
    def review(event_id: str, body: ReviewBody):
        row = store.get_event(event_id)
        if not row:
            raise HTTPException(404, "event not found")
        if body.state not in REVIEW_STATES:
            raise HTTPException(400, f"state must be one of {REVIEW_STATES}")
        cur = row["review_state"]
        if body.state != cur and body.state not in ALLOWED_TRANSITIONS[cur]:
            raise HTTPException(409, f"transition {cur} -> {body.state} not allowed")
        if body.state == "resolved" and body.resolution not in RESOLUTIONS:
            raise HTTPException(400, f"resolution must be one of {RESOLUTIONS} when resolving")
        updated = store.set_review(event_id, body.state, body.actor, body.note, body.resolution if body.state == "resolved" else None)
        expls = store.explanations_for(event_id)
        return _event_view(store, updated, expls[0] if expls else None, devices_view())

    @r.post("/events/{event_id}/explain")
    def explain(event_id: str):
        row = store.get_event(event_id)
        if not row:
            raise HTTPException(404, "event not found")
        if app_state.worker.provider is None:
            raise HTTPException(409, "VLM provider is disabled on this server")
        if not row["evidence_ready"]:
            raise HTTPException(409, "evidence not complete yet; retry later")
        job = app_state.worker.enqueue(event_id, int(row["revision"]), force=True)
        return {"queued": job is not None, "job_id": job}

    @r.post("/events/{event_id}/displayed")
    def displayed(event_id: str, body: dict | None = None):
        when = float((body or {}).get("client_ts") or time.time())
        store.mark_displayed(event_id, min(when, time.time()))
        return {"ok": True}

    @r.get("/evidence/{path:path}")
    def evidence(path: str):
        if ".." in path or path.startswith("/"):
            raise HTTPException(400, "bad path")
        p = Path(s.resolved_evidence()) / path
        if not p.exists():
            raise HTTPException(404, "evidence not found")
        media = "video/mp4" if p.suffix == ".mp4" else "image/jpeg" if p.suffix in (".jpg", ".jpeg") else None
        return FileResponse(str(p), media_type=media)

    @r.get("/videos")
    def videos():
        d = Path(s.videos_dir)
        out = []
        if d.exists():
            for p in sorted(d.glob("*.mp4")):
                out.append({"name": p.name, "size": p.stat().st_size, "url": f"/api/videos/{p.name}"})
        return {"items": out}

    @r.get("/videos/{name}")
    def video(name: str):
        if "/" in name or ".." in name:
            raise HTTPException(400, "bad name")
        p = Path(s.videos_dir) / name
        if not p.exists():
            raise HTTPException(404, "video not found")
        return FileResponse(str(p), media_type="video/mp4")

    @r.get("/devices")
    def devices():
        return {"items": list(devices_view().values())}

    @r.post("/devices/{device_id}/commands")
    def issue_command(device_id: str, body: CommandBody):
        if s.mode == "replay":
            raise HTTPException(409, "replay mode: no device attached; commands are disabled")
        if body.type not in ("START_CAMERA", "STOP_CAMERA"):
            raise HTTPException(400, "type must be START_CAMERA or STOP_CAMERA")
        cmd = app_state.hub.issue(device_id, body.type, body.camera_id, body.source, body.zones, body.issued_by)
        return {"command": cmd, "subscribers": app_state.hub.subscribers(device_id)}

    @r.get("/commands/{command_id}")
    def get_command(command_id: str):
        c = store.get_command(command_id)
        if not c:
            raise HTTPException(404, "command not found")
        return c

    @r.get("/timeline")
    def timeline(camera_id: str | None = None, limit: int = 2000):
        rows = store.list_events(camera_id=camera_id, limit=limit)
        return {"items": [{"event_id": x["event_id"], "camera_id": x["camera_id"], "zone_id": x["zone_id"],
                           "kind": x["kind"], "status": x["status"], "review_state": x["review_state"],
                           "entry_ts": x["entry_ts"], "exit_ts": x["exit_ts"], "media_entry_s": x["media_entry_s"],
                           "media_exit_s": x["media_exit_s"], "dwell_seconds": x["dwell_seconds"]} for x in rows]}

    @r.get("/export")
    def export(format: str = "csv", camera_id: str | None = None, state: str | None = None):
        rows = store.list_events(camera_id, state, limit=100000)
        expl = store.latest_explanation_map([x["event_id"] for x in rows])
        devs = devices_view()
        items = [_event_view(store, x, expl.get(x["event_id"]), devs) for x in rows]
        meta = {"exported_at": time.time(), "server_version": __version__, "mode": s.mode,
                "vlm": app_state.worker.snapshot(), "count": len(items)}
        if format == "json":
            for it in items:
                it.pop("trajectory", None)
            return JSONResponse({"meta": meta, "items": items})
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["event_id", "camera_id", "camera_name", "zone_name", "kind", "status", "close_reason", "track_id",
                    "entry_ts", "exit_ts", "media_entry_s", "media_exit_s", "dwell_seconds", "review_state",
                    "review_resolution", "review_actor", "review_note", "review_updated_at", "ingest_source",
                    "model_name", "model_version", "backend", "config_hash", "run_id", "edge_version",
                    "vlm_status", "vlm_provider", "vlm_model", "vlm_prompt_version", "vlm_assessment", "vlm_summary",
                    "vlm_latency_ms", "ingest_latency_ms", "first_display_ms"])
        for it in items:
            run, x = it["run"], it["explanation"] or {}
            res = x.get("result") or {}
            w.writerow([it["event_id"], it["camera_id"], it["camera_name"], it["zone_name"], it["kind"], it["status"],
                        it["close_reason"], it["track_id"], it["entry_ts"], it["exit_ts"], it["media_entry_s"],
                        it["media_exit_s"], it["dwell_seconds"], it["review"]["state"], it["review"]["resolution"],
                        it["review"]["actor"], it["review"]["note"], it["review"]["updated_at"], it["ingest_source"],
                        run.get("model_name"), run.get("model_version"), run.get("backend"), run.get("config_hash"),
                        run.get("run_id"), run.get("edge_version"), x.get("status"), x.get("provider"), x.get("model"),
                        x.get("prompt_version"), res.get("assessment"), res.get("scene_summary"), x.get("latency_ms"),
                        it["latency"]["ingest_ms"], it["latency"]["first_display_ms"]])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="zonewatch-events-{int(time.time())}.csv"'})

    @r.get("/audit")
    def audit(limit: int = 500):
        rows = store.q("SELECT * FROM review_log ORDER BY id DESC LIMIT ?", (limit,))
        return {"items": [dict(x) for x in rows]}

    return r
