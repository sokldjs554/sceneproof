"""Regression cases for event times from camera devices."""

from sceneproof.server.groups import concurrent_groups


def test_missing_wall_entry_uses_zero():
    events = [
        {"event_id": "a", "camera_id": "c", "zone_id": "z", "media_entry_s": 0,
         "media_exit_s": 3, "entry_ts": None, "exit_ts": 3},
        {"event_id": "b", "camera_id": "c", "zone_id": "z", "media_entry_s": 1,
         "media_exit_s": 2, "entry_ts": 1, "exit_ts": 2},
    ]
    assert concurrent_groups(events, now=10)["a"]["event_ids"] == ["a", "b"]


def test_open_event_end_is_not_earlier_than_its_start():
    events = [
        {"event_id": "a", "camera_id": "c", "zone_id": "z", "media_entry_s": 0,
         "media_exit_s": None, "entry_ts": 101, "exit_ts": None},
        {"event_id": "b", "camera_id": "c", "zone_id": "z", "media_entry_s": 1,
         "media_exit_s": 3, "entry_ts": 102, "exit_ts": 103},
    ]
    assert concurrent_groups(events, now=100, gap_s=1)["a"]["event_ids"] == ["a", "b"]
