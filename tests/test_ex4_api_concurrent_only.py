"""과제 4 — 사건 목록 API에 '묶인 사건만' 조건 추가 (src/sceneproof/server/api.py)

GET /api/events?concurrent_only=true 이면 동시에 구역 안에 있던 사건(concurrent가 null이 아닌 것)만 돌려줍니다.
묶음 계산은 지금처럼 '거르기 전' 목록 전체로 해야 합니다(거른 뒤에 계산하면 묶음이 깨짐).
응답의 counts(처리 상태별 개수)는 지금처럼 전체 기준 그대로 둡니다. 기본값은 false(지금과 같음).
"""

import pytest
from fastapi.testclient import TestClient

from sceneproof.server.app import create_app
from sceneproof.server.config import ServerSettings


def event(eid, m0, m1):
    return {"event_id": eid, "revision": 3, "device_id": "d", "camera_id": "c", "zone_id": "z", "zone_name": "Z",
            "kind": "EVENT_KIND_ZONE_ENTRY", "status": "EVENT_STATUS_CLOSED", "track_id": 1,
            "entry_ts": 1000.0 + m0, "exit_ts": 1000.0 + m1, "detected_at": 1000.0 + m0,
            "media_entry_s": m0, "media_exit_s": m1, "dwell_seconds": m1 - m0, "source": "file://x.mp4",
            "run": {"model_name": "yolox_tiny"}, "evidence": {"key_frames": []}}


@pytest.fixture
def client(tmp_path):
    s = ServerSettings(data_dir=str(tmp_path / "srv"), mode="replay", vlm_provider="none", console_dir=str(tmp_path / "x"))
    app = create_app(s)
    with TestClient(app) as c:
        for eid, m0, m1 in (("g-1", 10.0, 16.0), ("g-2", 12.0, 18.0), ("solo", 40.0, 42.0)):
            app.state.sp.store.upsert_event(event(eid, m0, m1))
        yield c


def test_default_lists_everything(client):
    ids = {x["event_id"] for x in client.get("/api/events").json()["items"]}
    assert ids == {"g-1", "g-2", "solo"}


def test_concurrent_only(client):
    r = client.get("/api/events", params={"concurrent_only": "true"}).json()
    assert {x["event_id"] for x in r["items"]} == {"g-1", "g-2"}
    assert r["count"] == 2
    assert all(x["concurrent"]["size"] == 2 for x in r["items"])
    assert r["counts"]["total"] == 3  # 처리 상태별 개수는 전체 기준


def test_concurrent_only_keeps_group_members_in_other_review_states(client):
    review = client.post("/api/events/g-2/review", json={"state": "in_review", "actor": "reviewer"})
    assert review.status_code == 200
    r = client.get("/api/events", params={"state": "needs_review", "concurrent_only": "true"}).json()
    assert [item["event_id"] for item in r["items"]] == ["g-1"]
    assert r["items"][0]["concurrent"]["event_ids"] == ["g-1", "g-2"]
    assert r["counts"]["total"] == 3


def test_concurrent_only_filters_before_pagination(client):
    first = client.get("/api/events", params={"concurrent_only": "true", "limit": 1}).json()
    second = client.get("/api/events", params={"concurrent_only": "true", "limit": 1, "offset": 1}).json()
    assert [item["event_id"] for item in first["items"]] == ["g-2"]
    assert [item["event_id"] for item in second["items"]] == ["g-1"]
    assert first["count"] == second["count"] == 1
