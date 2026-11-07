"""SQLite store for events, evidence, explanations, reviews, devices, commands.

Idempotency lives here: ``upsert_event`` only replaces a row when the incoming revision is
higher, so resends from the gateway (after a reconnect) or a re-imported replay bundle are
no-ops. Everything the console shows is read from this file; restarting the server loses
nothing, including queued VLM jobs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
  device_id TEXT PRIMARY KEY,
  last_seen_ts REAL NOT NULL,
  status_json TEXT NOT NULL,
  gateway_version TEXT,
  analyzer_run_id TEXT
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  revision INTEGER NOT NULL,
  device_id TEXT, camera_id TEXT, camera_name TEXT, zone_id TEXT, zone_name TEXT,
  kind TEXT, status TEXT, close_reason TEXT, track_id INTEGER,
  entry_ts REAL, exit_ts REAL, detected_at REAL,
  first_received_at REAL NOT NULL, received_at REAL NOT NULL,
  first_displayed_at REAL,
  media_entry_s REAL, media_exit_s REAL, dwell_seconds REAL,
  source TEXT, run_json TEXT, event_json TEXT NOT NULL,
  review_state TEXT NOT NULL DEFAULT 'needs_review',
  review_resolution TEXT, review_note TEXT, review_actor TEXT, review_updated_at REAL,
  evidence_ready INTEGER NOT NULL DEFAULT 0,
  ingest_source TEXT NOT NULL DEFAULT 'live'
);
CREATE INDEX IF NOT EXISTS idx_events_camera_entry ON events(camera_id, entry_ts);
CREATE INDEX IF NOT EXISTS idx_events_state ON events(review_state);
CREATE TABLE IF NOT EXISTS evidence_files (
  event_id TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT, size INTEGER, uploaded_at REAL,
  PRIMARY KEY (event_id, path)
);
CREATE TABLE IF NOT EXISTS explanations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL, event_revision INTEGER NOT NULL,
  provider TEXT, model TEXT, prompt_version TEXT, schema_version TEXT,
  status TEXT NOT NULL,  -- queued | running | done | failed | timeout | invalid | skipped
  attempts INTEGER NOT NULL DEFAULT 0,
  requested_at REAL NOT NULL, started_at REAL, completed_at REAL, latency_ms REAL,
  input_json TEXT, result_json TEXT, raw_text TEXT, error TEXT, usage_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_expl_event ON explanations(event_id, id);
CREATE TABLE IF NOT EXISTS review_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL, from_state TEXT, to_state TEXT NOT NULL, resolution TEXT,
  actor TEXT, note TEXT, ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS commands (
  command_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, camera_id TEXT, type TEXT NOT NULL,
  payload_json TEXT, issued_at REAL NOT NULL, issued_by TEXT,
  delivered_at REAL, result_ok INTEGER, result_message TEXT, completed_at REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        with self.tx() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def q(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def q1(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        rows = self.q(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ events
    def upsert_event(self, ev: dict[str, Any], ingest_source: str = "live") -> tuple[bool, bool, int]:
        """Returns (accepted, duplicate, stored_revision). ``ev`` is the protobuf JSON dict."""
        now = time.time()
        eid, rev = ev["event_id"], int(ev.get("revision", 0))
        with self.tx() as c:
            row = c.execute("SELECT revision, first_received_at FROM events WHERE event_id=?", (eid,)).fetchone()
            if row is not None and row["revision"] >= rev:
                return False, True, int(row["revision"])
            first = row["first_received_at"] if row is not None else now
            evidence = ev.get("evidence", {})
            fields = dict(
                event_id=eid, revision=rev, device_id=ev.get("device_id"), camera_id=ev.get("camera_id"),
                camera_name=ev.get("camera_name") or "", zone_id=ev.get("zone_id"), zone_name=ev.get("zone_name"),
                kind=ev.get("kind"), status=ev.get("status"), close_reason=ev.get("close_reason"),
                track_id=ev.get("track_id"), entry_ts=_ts(ev.get("entry_ts")), exit_ts=_ts(ev.get("exit_ts")),
                detected_at=_ts(ev.get("detected_at")), first_received_at=first, received_at=now,
                media_entry_s=ev.get("media_entry_s", 0.0), media_exit_s=ev.get("media_exit_s"),
                dwell_seconds=ev.get("dwell_seconds", 0.0), source=ev.get("source"),
                run_json=json.dumps(ev.get("run", {})), event_json=json.dumps(ev), ingest_source=ingest_source,
            )
            cols = ", ".join(fields)
            placeholders = ", ".join("?" for _ in fields)
            updates = ", ".join(f"{k}=excluded.{k}" for k in fields if k not in ("event_id", "first_received_at"))
            c.execute(f"INSERT INTO events ({cols}) VALUES ({placeholders}) ON CONFLICT(event_id) DO UPDATE SET {updates}",
                      tuple(fields.values()))
            # evidence readiness: all referenced files present?
            paths = [k["path"] for k in evidence.get("key_frames", [])]
            if evidence.get("clip_path"):
                paths.append(evidence["clip_path"])
            have = {r["path"] for r in c.execute("SELECT path FROM evidence_files WHERE event_id=?", (eid,))}
            ready = 1 if paths and all(p in have for p in paths) else 0
            c.execute("UPDATE events SET evidence_ready=? WHERE event_id=?", (ready, eid))
        return True, False, rev

    def get_event(self, event_id: str) -> dict | None:
        row = self.q1("SELECT * FROM events WHERE event_id=?", (event_id,))
        return dict(row) if row else None

    def list_events(self, camera_id: str | None = None, state: str | None = None, kind: str | None = None,
                    since: float | None = None, until: float | None = None, limit: int = 200, offset: int = 0,
                    device_id: str | None = None) -> list[dict]:
        where, params = [], []
        if camera_id:
            where.append("camera_id=?")
            params.append(camera_id)
        if device_id:
            where.append("device_id=?")
            params.append(device_id)
        if state:
            where.append("review_state=?")
            params.append(state)
        if kind:
            where.append("kind=?")
            params.append(kind)
        if since is not None:
            where.append("entry_ts>=?")
            params.append(since)
        if until is not None:
            where.append("entry_ts<=?")
            params.append(until)
        sql = "SELECT * FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY entry_ts DESC, event_id LIMIT ? OFFSET ?"
        params += [limit, offset]
        return [dict(r) for r in self.q(sql, tuple(params))]

    def count_events(self) -> dict[str, int]:
        rows = self.q("SELECT review_state, COUNT(*) AS n FROM events GROUP BY review_state")
        out = {r["review_state"]: r["n"] for r in rows}
        out["total"] = sum(out.values())
        return out

    def mark_displayed(self, event_id: str, when: float) -> None:
        with self.tx() as c:
            c.execute("UPDATE events SET first_displayed_at=? WHERE event_id=? AND first_displayed_at IS NULL", (when, event_id))

    # ------------------------------------------------------------------ evidence
    def record_evidence(self, event_id: str, path: str, sha256: str, size: int) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO evidence_files(event_id, path, sha256, size, uploaded_at) VALUES (?,?,?,?,?)",
                      (event_id, path, sha256, size, time.time()))
            row = c.execute("SELECT event_json FROM events WHERE event_id=?", (event_id,)).fetchone()
            if row:
                ev = json.loads(row["event_json"])
                evidence = ev.get("evidence", {})
                paths = [k["path"] for k in evidence.get("key_frames", [])]
                if evidence.get("clip_path"):
                    paths.append(evidence["clip_path"])
                have = {r["path"] for r in c.execute("SELECT path FROM evidence_files WHERE event_id=?", (event_id,))}
                ready = 1 if paths and all(p in have for p in paths) else 0
                c.execute("UPDATE events SET evidence_ready=? WHERE event_id=?", (ready, event_id))

    def evidence_files(self, event_id: str) -> list[dict]:
        return [dict(r) for r in self.q("SELECT * FROM evidence_files WHERE event_id=? ORDER BY path", (event_id,))]

    def has_evidence(self, event_id: str, path: str) -> dict | None:
        r = self.q1("SELECT * FROM evidence_files WHERE event_id=? AND path=?", (event_id, path))
        return dict(r) if r else None

    # ------------------------------------------------------------------ explanations
    def enqueue_explanation(self, event_id: str, revision: int, provider: str, model: str, prompt_version: str,
                            schema_version: str, force: bool = False) -> int | None:
        """One job per (event, revision) unless forced. Returns the job id or None if already present."""
        with self.tx() as c:
            if not force:
                row = c.execute("SELECT id, status FROM explanations WHERE event_id=? AND event_revision=? "
                                "AND status IN ('queued','running','done') ORDER BY id DESC LIMIT 1",
                                (event_id, revision)).fetchone()
                if row:
                    return None
            cur = c.execute("INSERT INTO explanations(event_id, event_revision, provider, model, prompt_version, "
                            "schema_version, status, requested_at) VALUES (?,?,?,?,?,?,'queued',?)",
                            (event_id, revision, provider, model, prompt_version, schema_version, time.time()))
            return int(cur.lastrowid)

    def next_queued_explanations(self, limit: int = 10) -> list[dict]:
        return [dict(r) for r in self.q("SELECT * FROM explanations WHERE status='queued' ORDER BY id LIMIT ?", (limit,))]

    def requeue_running(self) -> int:
        """On startup: jobs left 'running' by a crash go back to the queue."""
        with self.tx() as c:
            cur = c.execute("UPDATE explanations SET status='queued' WHERE status='running'")
            return cur.rowcount

    def update_explanation(self, job_id: int, **fields: Any) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE explanations SET {sets} WHERE id=?", (*fields.values(), job_id))

    def explanations_for(self, event_id: str) -> list[dict]:
        return [dict(r) for r in self.q("SELECT * FROM explanations WHERE event_id=? ORDER BY id DESC", (event_id,))]

    def latest_explanation_map(self, event_ids: list[str]) -> dict[str, dict]:
        if not event_ids:
            return {}
        out: dict[str, dict] = {}
        for i in range(0, len(event_ids), 500):
            chunk = event_ids[i:i + 500]
            marks = ",".join("?" for _ in chunk)
            for r in self.q(f"SELECT * FROM explanations WHERE id IN (SELECT MAX(id) FROM explanations WHERE event_id IN ({marks}) GROUP BY event_id)", tuple(chunk)):
                out[r["event_id"]] = dict(r)
        return out

    def explanation_stats(self) -> dict[str, int]:
        return {r["status"]: r["n"] for r in self.q("SELECT status, COUNT(*) AS n FROM explanations GROUP BY status")}

    # ------------------------------------------------------------------ review
    def set_review(self, event_id: str, to_state: str, actor: str, note: str, resolution: str | None) -> dict:
        with self.tx() as c:
            row = c.execute("SELECT review_state FROM events WHERE event_id=?", (event_id,)).fetchone()
            if row is None:
                raise KeyError(event_id)
            now = time.time()
            c.execute("UPDATE events SET review_state=?, review_resolution=?, review_note=?, review_actor=?, review_updated_at=? "
                      "WHERE event_id=?", (to_state, resolution, note, actor, now, event_id))
            c.execute("INSERT INTO review_log(event_id, from_state, to_state, resolution, actor, note, ts) VALUES (?,?,?,?,?,?,?)",
                      (event_id, row["review_state"], to_state, resolution, actor, note, now))
        return self.get_event(event_id) or {}

    def review_log(self, event_id: str) -> list[dict]:
        return [dict(r) for r in self.q("SELECT * FROM review_log WHERE event_id=? ORDER BY id", (event_id,))]

    # ------------------------------------------------------------------ devices
    def upsert_device(self, device_id: str, status: dict, gateway_version: str, run_id: str) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO devices(device_id, last_seen_ts, status_json, gateway_version, analyzer_run_id) VALUES (?,?,?,?,?) "
                      "ON CONFLICT(device_id) DO UPDATE SET last_seen_ts=excluded.last_seen_ts, status_json=excluded.status_json, "
                      "gateway_version=excluded.gateway_version, analyzer_run_id=excluded.analyzer_run_id",
                      (device_id, time.time(), json.dumps(status), gateway_version, run_id))

    def devices(self) -> list[dict]:
        return [dict(r) for r in self.q("SELECT * FROM devices ORDER BY device_id")]

    # ------------------------------------------------------------------ commands
    def add_command(self, cmd: dict) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO commands(command_id, device_id, camera_id, type, payload_json, issued_at, issued_by) "
                      "VALUES (?,?,?,?,?,?,?)", (cmd["command_id"], cmd["device_id"], cmd.get("camera_id"), cmd["type"],
                                                json.dumps(cmd), cmd["issued_at"], cmd.get("issued_by")))

    def mark_command_delivered(self, command_id: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE commands SET delivered_at=? WHERE command_id=? AND delivered_at IS NULL", (time.time(), command_id))

    def complete_command(self, command_id: str, ok: bool, message: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE commands SET result_ok=?, result_message=?, completed_at=? WHERE command_id=?",
                      (1 if ok else 0, message, time.time(), command_id))

    def get_command(self, command_id: str) -> dict | None:
        r = self.q1("SELECT * FROM commands WHERE command_id=?", (command_id,))
        return dict(r) if r else None

    def undelivered_commands(self, device_id: str, max_age_s: float = 300) -> list[dict]:
        rows = self.q("SELECT * FROM commands WHERE device_id=? AND delivered_at IS NULL AND issued_at>=? ORDER BY issued_at",
                      (device_id, time.time() - max_age_s))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ meta
    def set_meta(self, key: str, value: str) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value))

    def get_meta(self, key: str) -> str | None:
        r = self.q1("SELECT value FROM meta WHERE key=?", (key,))
        return r["value"] if r else None


def _ts(v: Any) -> float | None:
    """protobuf JSON timestamps are RFC3339 strings; protobuf dicts may already carry floats."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    from datetime import datetime

    s = str(v).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None
