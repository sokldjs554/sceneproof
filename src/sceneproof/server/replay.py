"""Import a recorded run (events.jsonl + evidence/) as a replay bundle.

Used for the public demo (no device attached) and for evaluation (feed offline results through
the exact same storage and VLM path as live ingestion). Imports are idempotent: the same
(event_id, revision) pairs are no-ops.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

from .db import Store

log = logging.getLogger(__name__)


def import_bundle(store: Store, bundle_dir: str | Path, evidence_dir: str | Path, worker=None,
                  ingest_source: str = "replay", camera_names: dict[str, str] | None = None) -> dict:
    bundle = Path(bundle_dir)
    evidence_dir = Path(evidence_dir)
    events_file = bundle / "events.jsonl"
    if not events_file.exists():
        raise FileNotFoundError(events_file)
    latest: dict[str, dict] = {}
    for line in events_file.read_text().splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev["event_id"] not in latest or int(ev.get("revision", 0)) > int(latest[ev["event_id"]].get("revision", 0)):
            latest[ev["event_id"]] = ev
    src_evidence = bundle / "evidence"
    stats = {"events": 0, "duplicates": 0, "files": 0, "explanations": 0}
    for ev in latest.values():
        if camera_names and ev.get("camera_id") in camera_names:
            ev["camera_name"] = camera_names[ev["camera_id"]]
        accepted, _duplicate, _ = store.upsert_event(ev, ingest_source)
        stats["events" if accepted else "duplicates"] += 1
        paths = [k["path"] for k in ev.get("evidence", {}).get("key_frames", [])]
        if ev.get("evidence", {}).get("clip_path"):
            paths.append(ev["evidence"]["clip_path"])
        for rel in paths:
            src = src_evidence / rel
            if not src.exists():
                continue
            dst = evidence_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(src, dst)
            digest = hashlib.sha256(dst.read_bytes()).hexdigest()
            store.record_evidence(ev["event_id"], rel, digest, dst.stat().st_size)
            stats["files"] += 1
    expl_file = bundle / "explanations.jsonl"
    if expl_file.exists():
        for line in expl_file.read_text().splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            job = store.enqueue_explanation(x["event_id"], int(x["event_revision"]), x.get("provider", "replay"),
                                            x.get("model", ""), x.get("prompt_version", ""), x.get("schema_version", "1"))
            if job is not None:
                store.update_explanation(job, status=x.get("status", "done"), completed_at=x.get("completed_at"),
                                         started_at=x.get("started_at"), latency_ms=x.get("latency_ms"),
                                         result_json=json.dumps(x.get("result"), ensure_ascii=False) if x.get("result") else None,
                                         raw_text=x.get("raw_text"), error=x.get("error"), attempts=x.get("attempts", 1),
                                         usage_json=json.dumps(x.get("usage", {})), input_json=json.dumps(x.get("input", {})))
                stats["explanations"] += 1
    elif worker is not None:
        for ev in latest.values():
            worker.maybe_enqueue_for_event(ev, accepted=True)
    log.info("replay import %s: %s", bundle, stats)
    return stats


def export_bundle(store: Store, evidence_dir: str | Path, out_dir: str | Path, camera_id: str | None = None) -> dict:
    """Writes events.jsonl / explanations.jsonl / evidence/ for the current DB (used to build demo bundles)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = store.list_events(camera_id=camera_id, limit=100000)
    n_files = 0
    with (out / "events.jsonl").open("w") as f:
        for row in events:
            ev = json.loads(row["event_json"])
            ev["camera_name"] = row.get("camera_name") or ""
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            for file in store.evidence_files(row["event_id"]):
                src = Path(evidence_dir) / file["path"]
                if src.exists():
                    dst = out / "evidence" / file["path"]
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    n_files += 1
    with (out / "explanations.jsonl").open("w") as f:
        n_expl = 0
        for row in events:
            for x in store.explanations_for(row["event_id"]):
                rec = {k: x.get(k) for k in ("event_id", "event_revision", "provider", "model", "prompt_version",
                                               "schema_version", "status", "attempts", "started_at", "completed_at",
                                               "latency_ms", "raw_text", "error")}
                rec["result"] = json.loads(x["result_json"]) if x.get("result_json") else None
                rec["usage"] = json.loads(x["usage_json"]) if x.get("usage_json") else {}
                rec["input"] = json.loads(x["input_json"]) if x.get("input_json") else {}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_expl += 1
    return {"events": len(events), "files": n_files, "explanations": n_expl, "dir": str(out)}
