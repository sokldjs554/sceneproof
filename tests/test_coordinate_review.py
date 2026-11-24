from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from google.protobuf.json_format import MessageToDict

from sceneproof.edge.events import EvidenceState, RunContext, episode_to_event
from sceneproof.edge.zones import FrameMeta, Transition, ZoneMonitor, ZoneSpec
from sceneproof.server.app import create_app
from sceneproof.server.config import ServerSettings


def track_step(monitor, index, x, y):
    meta = FrameMeta(index, index / 10, 1000 + index / 10, 200, 200)
    track = SimpleNamespace(track_id=1, score=.9, xyxy=np.array([x-6, y-30, x+6, y]))
    return monitor.update([track], meta)


def monitor():
    zone = ZoneSpec("z", "floor", [(.25,.25), (.75,.25), (.75,.75), (.25,.75)], dwell_seconds=.5)
    return ZoneMonitor([zone], "cam", review_uncertain=True)


def test_uncertain_position_does_not_confirm_entry_or_dwell():
    m = monitor()
    assert not track_step(m, 0, 10, 100)  # confidently outside
    assert not track_step(m, 1, 100, 100)
    assert not track_step(m, 2, 100, 100)
    edge, = track_step(m, 3, 50, 100)
    assert (edge.transition, edge.episode.coordinate_review_reason) == (Transition.REVIEW, "zone_boundary")
    assert not track_step(m, 4, 50, 100)
    assert not track_step(m, 5, 100, 100)
    assert not track_step(m, 6, 100, 100)
    opened, = track_step(m, 7, 100, 100)
    assert opened.transition == Transition.OPENED
    clipped, = track_step(m, 8, 100, 191)
    assert clipped.episode.coordinate_review_reason == "frame_bottom"
    for i in range(9, 15):
        assert not track_step(m, i, 100, 195)
    for i in range(15, 20):
        assert not track_step(m, i, 100, 100)
    dwell, = track_step(m, 20, 100, 100)
    assert dwell.transition == Transition.DWELL


def test_uncertain_observation_reaches_live_review_without_vlm(tmp_path):
    review, = track_step(monitor(), 0, 100, 200)
    ctx = RunContext("run", "dev", "cam", "file", "Cam", "v1", "model", "v1", "cpu", "tracker", "abc", (200, 200))
    wire = episode_to_event(review.episode, ctx, EvidenceState(), (200, 200))
    data = MessageToDict(wire, preserving_proto_field_name=True)
    settings = ServerSettings(data_dir=str(tmp_path), grpc_listen="127.0.0.1:0", vlm_provider="mock")
    app = create_app(settings)
    with TestClient(app) as c:
        sp = app.state.sp
        sp.store.upsert_event(data)
        assert sp.worker.maybe_enqueue_for_event(data, accepted=True) is None
        path = f"/api/events/{wire.event_id}"
        assert c.post(path+"/explain").status_code == 409
        row = c.get(path).json()
        assert row["kind"] == "EVENT_KIND_COORDINATE_REVIEW"
        assert row["coordinate_review_reason"] == "frame_bottom"
        assert row["review"]["state"] == "needs_review"
        assert c.get("/api/events?state=needs_review").json()["items"][0]["event_id"] == wire.event_id
        assert row["explanation"] is None
