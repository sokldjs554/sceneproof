"""Review server: idempotent ingest, evidence readiness, review transitions, VLM worker with the mock provider."""

import time

import pytest
from fastapi.testclient import TestClient

from zonewatch.server.app import create_app
from zonewatch.server.config import ServerSettings
from zonewatch.server.db import Store


def event(eid="ev-1", rev=1, status="EVENT_STATUS_OPEN", kind="EVENT_KIND_ZONE_ENTRY", paths=None):
    ev = {"event_id": eid, "revision": rev, "device_id": "d", "camera_id": "c", "zone_id": "z", "zone_name": "Z",
          "kind": kind, "status": status, "track_id": 1, "entry_ts": 1000.0, "detected_at": 1000.5, "media_entry_s": 3.0,
          "dwell_seconds": 1.0, "source": "file://x.mp4", "run": {"model_name": "yolox_tiny"}, "evidence": {"key_frames": []}}
    for p in paths or []:
        ev["evidence"]["key_frames"].append({"role": p.split("/")[-1].split(".")[0], "path": p, "media_time_s": 3.0})
    return ev


def test_upsert_is_idempotent_and_monotonic(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    assert s.upsert_event(event(rev=1)) == (True, False, 1)
    assert s.upsert_event(event(rev=1)) == (False, True, 1)
    assert s.upsert_event(event(rev=3, status="EVENT_STATUS_CLOSED")) == (True, False, 3)
    assert s.upsert_event(event(rev=2)) == (False, True, 3)
    row = s.get_event("ev-1")
    assert row["revision"] == 3 and row["status"] == "EVENT_STATUS_CLOSED"
    assert row["first_received_at"] <= row["received_at"]


def test_evidence_ready_only_when_all_files_present(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.upsert_event(event(paths=["ev-1/entry.jpg", "ev-1/exit.jpg"]))
    assert s.get_event("ev-1")["evidence_ready"] == 0
    s.record_evidence("ev-1", "ev-1/entry.jpg", "aa", 10)
    assert s.get_event("ev-1")["evidence_ready"] == 0
    s.record_evidence("ev-1", "ev-1/exit.jpg", "bb", 10)
    assert s.get_event("ev-1")["evidence_ready"] == 1


@pytest.fixture
def client(tmp_path):
    settings = ServerSettings(data_dir=str(tmp_path / "srv"), mode="replay", vlm_provider="mock", vlm_mock_delay_s=0.01,
                              console_dir=str(tmp_path / "no-console"))
    app = create_app(settings)
    with TestClient(app) as c:
        yield c, app.state.zw


def _seed(state, tmp_dir, eid="ev-1", closed=True):
    paths = [f"{eid}/entry.jpg", f"{eid}/entry_annot.jpg"]
    ev = event(eid, rev=3 if closed else 1, status="EVENT_STATUS_CLOSED" if closed else "EVENT_STATUS_OPEN", paths=paths)
    state.store.upsert_event(ev)
    import cv2
    import numpy as np

    for p in paths:
        f = state.worker.evidence_dir / p
        f.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(f), np.zeros((60, 80, 3), np.uint8))
        state.store.record_evidence(eid, p, "x", 10)
    state.worker.maybe_enqueue_for_event(ev, accepted=True)


def test_review_transitions_and_audit(client):
    c, state = client
    _seed(state, None)
    r = c.post("/api/events/ev-1/review", json={"state": "resolved", "actor": "a", "note": "n"})
    assert r.status_code == 400  # resolution required
    r = c.post("/api/events/ev-1/review", json={"state": "in_review", "actor": "a", "note": "검토 시작"})
    assert r.status_code == 200 and r.json()["review"]["state"] == "in_review"
    r = c.post("/api/events/ev-1/review", json={"state": "resolved", "resolution": "confirmed", "actor": "a", "note": "done"})
    assert r.json()["review"]["resolution"] == "confirmed"
    r = c.post("/api/events/ev-1/review", json={"state": "needs_review", "actor": "a"})
    assert r.status_code == 409  # resolved -> needs_review not allowed (reopen goes to in_review)
    log = c.get("/api/events/ev-1").json()["review_log"]
    assert [x["to_state"] for x in log] == ["in_review", "resolved"]
    csv = c.get("/api/export?format=csv").text
    assert "confirmed" in csv and "yolox_tiny" in csv


def test_mock_vlm_worker_produces_validated_explanation(client):
    c, state = client
    _seed(state, None)
    for _ in range(100):
        x = c.get("/api/events/ev-1").json()["explanation"]
        if x and x["status"] == "done":
            break
        time.sleep(0.05)
    assert x and x["status"] == "done" and x["provider"] == "mock"
    assert x["result"]["assessment"] in ("needs_attention", "likely_benign", "unclear")
    assert x["result"]["scene_summary"].startswith("[MOCK]")
    assert x["prompt_version"] and x["schema_version"] == "1"


def test_replay_mode_rejects_commands_and_lists_status(client):
    c, _state = client
    r = c.post("/api/devices/d/commands", json={"type": "START_CAMERA", "camera_id": "c"})
    assert r.status_code == 409
    st = c.get("/api/status").json()
    assert st["mode"] == "replay" and st["vlm"]["provider"] == "mock"


def test_displayed_timestamp_recorded_once(client):
    c, state = client
    _seed(state, None)
    c.post("/api/events/ev-1/displayed", json={"client_ts": 1001.0})
    c.post("/api/events/ev-1/displayed", json={"client_ts": 2000.0})
    assert state.store.get_event("ev-1")["first_displayed_at"] == 1001.0
    assert c.get("/api/events/ev-1").json()["latency"]["first_display_ms"] == 1000.0
