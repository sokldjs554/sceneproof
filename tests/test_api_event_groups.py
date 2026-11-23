"""GET /api/events gives every row the same group information, whatever the filter or the page.

Until now the default listing grouped only the rows it returned, so a review-state filter or a page boundary
could hide that an event had company. `concurrent_only=true` grouped over all events. Now both use
the groups of all events (kept by a GroupIndex), and `group_gap_s` sets how far apart events may be.
"""

import pytest
from fastapi.testclient import TestClient

import sceneproof.server.api as api_mod
import sceneproof.server.groups as groups_mod
from sceneproof.server.app import create_app
from sceneproof.server.config import ServerSettings


def event(eid, m0, m1):
    return {"event_id": eid, "revision": 3, "device_id": "d", "camera_id": "c", "zone_id": "z", "zone_name": "Z",
            "kind": "EVENT_KIND_ZONE_ENTRY", "status": "EVENT_STATUS_CLOSED", "track_id": 1,
            "entry_ts": 1000.0 + m0, "exit_ts": 1000.0 + m1, "detected_at": 1000.0 + m0,
            "media_entry_s": m0, "media_exit_s": m1, "dwell_seconds": m1 - m0, "source": "file://x.mp4",
            "run": {"model_name": "yolox_tiny"}, "evidence": {"key_frames": []}}


EVENTS = (("g-1", 10.0, 16.0), ("g-2", 12.0, 18.0), ("solo", 40.0, 42.0), ("near-1", 60.0, 65.0), ("near-2", 66.5, 70.0))


@pytest.fixture
def served(tmp_path):
    s = ServerSettings(data_dir=str(tmp_path / "srv"), mode="replay", vlm_provider="none", console_dir=str(tmp_path / "x"))
    app = create_app(s)
    with TestClient(app) as c:
        for eid, m0, m1 in EVENTS:
            app.state.sp.store.upsert_event(event(eid, m0, m1))
        yield c, app.state.sp.store


def by_id(body):
    return {x["event_id"]: x for x in body["items"]}


def test_list_a_review_state_filter_keeps_the_whole_group(served):
    client, _ = served
    assert client.post("/api/events/g-2/review", json={"state": "in_review", "actor": "reviewer"}).status_code == 200
    items = by_id(client.get("/api/events", params={"state": "needs_review"}).json())
    assert "g-2" not in items                                           # filtered out of the rows...
    assert items["g-1"]["concurrent"]["event_ids"] == ["g-1", "g-2"]    # ...but g-1 still shows it had company
    assert items["g-1"]["concurrent"]["size"] == 2
    assert items["solo"]["concurrent"] is None


def test_list_a_page_boundary_keeps_the_whole_group(served):
    client, _ = served
    everything = by_id(client.get("/api/events").json())
    for offset in range(len(EVENTS)):
        page = client.get("/api/events", params={"limit": 1, "offset": offset}).json()
        assert page["count"] == 1
        row = page["items"][0]
        assert row["concurrent"] == everything[row["event_id"]]["concurrent"]
    assert everything["g-2"]["concurrent"]["event_ids"] == ["g-1", "g-2"]


def test_list_default_and_concurrent_only_agree(served):
    client, _ = served
    default = by_id(client.get("/api/events").json())
    only = by_id(client.get("/api/events", params={"concurrent_only": "true"}).json())
    assert set(only) == {"g-1", "g-2"}
    assert all(only[k]["concurrent"] == default[k]["concurrent"] for k in only)
    assert {k for k, v in default.items() if v["concurrent"]} == set(only)


def test_list_group_gap_s_joins_events_that_almost_touch(served):
    client, _ = served
    assert by_id(client.get("/api/events").json())["near-1"]["concurrent"] is None
    wide = by_id(client.get("/api/events", params={"group_gap_s": 2}).json())
    assert wide["near-1"]["concurrent"]["event_ids"] == ["near-1", "near-2"]
    assert wide["g-1"]["concurrent"]["event_ids"] == ["g-1", "g-2"] and wide["solo"]["concurrent"] is None
    only = client.get("/api/events", params={"group_gap_s": 2, "concurrent_only": "true"}).json()
    assert {x["event_id"] for x in only["items"]} == {"g-1", "g-2", "near-1", "near-2"}
    assert by_id(client.get("/api/events", params={"group_gap_s": 1}).json())["near-1"]["concurrent"] is None


def test_list_group_gap_s_is_limited_to_zero_through_sixty_seconds(served):
    client, _ = served
    assert client.get("/api/events", params={"group_gap_s": 0}).status_code == 200
    assert client.get("/api/events", params={"group_gap_s": 60}).status_code == 200
    for bad in (-0.1, 60.5, "abc"):
        assert client.get("/api/events", params={"group_gap_s": bad}).status_code == 422, bad


def test_list_all_events_are_grouped_once_until_an_event_is_stored(served, monkeypatch):
    client, store = served
    calls = {"n": 0}
    real = groups_mod.concurrent_groups

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(groups_mod, "concurrent_groups", counted)
    monkeypatch.setattr(api_mod, "concurrent_groups", counted, raising=False)   # in case api.py still calls it itself
    for params in ({}, {"state": "needs_review"}, {"limit": 1}, {"concurrent_only": "true"}, {"concurrent_only": "true", "limit": 1}):
        assert client.get("/api/events", params=params).status_code == 200
    assert calls["n"] == 1
    assert client.post("/api/events/g-1/review", json={"state": "in_review", "actor": "reviewer"}).status_code == 200
    assert client.get("/api/events").status_code == 200
    assert calls["n"] == 1                     # a review does not change who was in the zone together
    store.upsert_event(event("joins-solo", 41.0, 45.0))
    assert by_id(client.get("/api/events").json())["solo"]["concurrent"]["event_ids"] == ["solo", "joins-solo"]
    assert calls["n"] == 2


def test_list_counts_and_rows_are_as_before(served):
    client, _ = served
    body = client.get("/api/events").json()
    assert body["count"] == len(EVENTS) and body["counts"]["total"] == len(EVENTS)
    assert [x["event_id"] for x in body["items"]] == [x["event_id"] for x in client.get("/api/events", params={"limit": 2000}).json()["items"]]
    assert all("trajectory" not in x for x in body["items"])
