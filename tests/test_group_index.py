"""Read and group all events only when the stored events have changed.

Store.events_version() changes when an event is stored (new, or a newer revision) and at no other time.
GroupIndex(store).groups(gap_s) returns what concurrent_groups(store.list_events(limit=-1), gap_s=gap_s) returns,
but reads the events again only after the version has changed.
"""

import threading
import time

import pytest

from sceneproof.server.db import Store
from sceneproof.server.groups import GroupIndex, concurrent_groups


def event(eid, m0, m1, rev=3):
    return {"event_id": eid, "revision": rev, "device_id": "d", "camera_id": "c", "zone_id": "z", "zone_name": "Z",
            "kind": "EVENT_KIND_ZONE_ENTRY", "status": "EVENT_STATUS_CLOSED", "track_id": 1,
            "entry_ts": 1000.0 + m0, "exit_ts": 1000.0 + m1, "detected_at": 1000.0 + m0,
            "media_entry_s": m0, "media_exit_s": m1, "dwell_seconds": m1 - m0, "source": "file://x.mp4",
            "run": {"model_name": "yolox_tiny"}, "evidence": {"key_frames": []}}


@pytest.fixture
def store(tmp_path):
    st = Store(tmp_path / "events.db")
    for eid, m0, m1 in (("g-1", 10.0, 16.0), ("g-2", 12.0, 18.0), ("solo", 40.0, 42.0)):
        st.upsert_event(event(eid, m0, m1))
    yield st
    st.close()


def count_full_reads(store, delay_s=0.0):
    """Wrap store.list_events and count the calls that read every event (limit=-1, no filter)."""
    calls = {"n": 0}
    real = store.list_events

    def counted(*args, **kwargs):
        limit = kwargs.get("limit", args[5] if len(args) > 5 else 200)
        if limit == -1 and not any(args[:5]) and not any(v for k, v in kwargs.items() if k not in ("limit", "offset")):
            calls["n"] += 1
            if delay_s:
                time.sleep(delay_s)
        return real(*args, **kwargs)

    store.list_events = counted
    return calls


def test_index_version_changes_only_when_an_event_is_stored(store):
    v0 = store.events_version()
    assert isinstance(v0, int)
    store.upsert_event(event("new", 100.0, 101.0))
    v1 = store.events_version()
    assert v1 > v0
    accepted, duplicate, _ = store.upsert_event(event("new", 100.0, 101.0))   # same revision again: not stored
    assert (accepted, duplicate) == (False, True) and store.events_version() == v1
    store.upsert_event(event("new", 100.0, 105.0, rev=4))                      # a newer revision: stored
    v2 = store.events_version()
    assert v2 > v1
    store.set_review("g-1", "in_review", "reviewer", "", None)
    store.mark_displayed("g-1", 2000.0)
    store.record_evidence("g-1", "g-1/entry.jpg", "0" * 64, 10)
    assert store.events_version() == v2   # none of these changes when the event was in the zone


def test_index_index_gives_the_same_groups_as_grouping_all_events(store):
    index = GroupIndex(store)
    want = concurrent_groups(store.list_events(limit=-1))
    assert index.groups() == want
    assert index.groups()["g-1"]["event_ids"] == ["g-1", "g-2"] and "solo" not in index.groups()


def test_index_index_reads_the_events_once_until_they_change(store):
    reads = count_full_reads(store)
    index = GroupIndex(store)
    first = index.groups()
    for _ in range(5):
        assert index.groups() == first
    assert reads["n"] == 1


def test_index_index_sees_a_new_event_and_a_newer_revision(store):
    reads = count_full_reads(store)
    index = GroupIndex(store)
    assert "solo" not in index.groups()
    store.upsert_event(event("joins-solo", 41.0, 45.0))
    assert index.groups()["solo"]["event_ids"] == ["solo", "joins-solo"]
    assert reads["n"] == 2
    store.upsert_event(event("joins-solo", 43.0, 45.0, rev=4))   # now it starts after "solo" has left
    assert "solo" not in index.groups()
    assert reads["n"] == 3


def test_index_a_review_or_a_duplicate_does_not_cause_another_read(store):
    reads = count_full_reads(store)
    index = GroupIndex(store)
    before = index.groups()
    store.set_review("g-2", "in_review", "reviewer", "", None)
    store.upsert_event(event("g-1", 10.0, 16.0))   # duplicate
    assert index.groups() == before
    assert reads["n"] == 1


def test_index_each_gap_value_is_computed_and_kept_separately(store):
    store.upsert_event(event("near-1", 60.0, 65.0))
    store.upsert_event(event("near-2", 66.5, 70.0))   # starts 1.5 s after near-1 ends
    reads = count_full_reads(store)
    index = GroupIndex(store)
    assert "near-1" not in index.groups()
    assert index.groups(gap_s=2.0)["near-1"]["event_ids"] == ["near-1", "near-2"]
    assert "near-1" not in index.groups(0.0) and "near-1" in index.groups(2.0)
    assert reads["n"] == 2   # one per gap value, then both answered without reading again
    with pytest.raises(ValueError):
        index.groups(gap_s=-1.0)


def test_index_requests_arriving_together_share_one_computation(store):
    reads = count_full_reads(store, delay_s=0.05)   # slow read: without a lock every thread would start its own
    index = GroupIndex(store)
    results = []
    threads = [threading.Thread(target=lambda: results.append(index.groups())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8 and all(r == results[0] for r in results)
    assert reads["n"] == 1
