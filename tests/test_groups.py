"""Events inside the same zone at the same time are shown together (server/groups.py)."""

from sceneproof.server.groups import concurrent_groups


def ev(eid, m0, m1, w0=None, w1=None, cam="c", zone="z"):
    w0 = 1000.0 + m0 if w0 is None else w0
    w1 = (None if m1 is None else 1000.0 + m1) if w1 is None else w1
    return {"event_id": eid, "camera_id": cam, "zone_id": zone, "media_entry_s": m0, "media_exit_s": m1,
            "entry_ts": w0, "exit_ts": w1}


def test_overlapping_events_in_one_zone_form_a_group():
    g = concurrent_groups([ev("a", 5.7, 9.4), ev("b", 8.8, 13.8), ev("c", 18.2, 22.5)], now=2000)
    assert g["a"] == g["b"] == {"group_id": "a", "size": 2, "event_ids": ["a", "b"]}
    assert "c" not in g


def test_groups_chain_and_keep_media_order():
    # 54.7-59.8, 55.2-58.5, 56.1-60.2 (three people in the shop-assistant video) and one that only touches the last
    g = concurrent_groups([ev("p3", 56.1, 60.2), ev("p1", 54.7, 59.8), ev("p2", 55.2, 58.5), ev("p4", 60.0, 61.0)], now=2000)
    assert g["p1"]["event_ids"] == ["p1", "p2", "p3", "p4"] and g["p4"]["group_id"] == "p1"


def test_touching_other_zone_or_other_camera_is_not_concurrent():
    g = concurrent_groups([ev("a", 0, 5), ev("b", 5, 9),                    # b starts the moment a ends
                           ev("c", 1, 4, zone="other"), ev("d", 1, 4, cam="other")], now=2000)
    assert g == {}


def test_two_analyses_of_the_same_file_are_not_grouped():
    # same media times, but the second run happened an hour later (wall clock does not overlap)
    g = concurrent_groups([ev("run1", 10, 20, 1000, 1003), ev("run2", 10, 20, 4600, 4603)], now=9000)
    assert g == {}


def test_open_event_counts_as_still_inside():
    g = concurrent_groups([ev("open", 10, None, 1010, None), ev("later", 30, 35, 1030, 1035)], now=1100)
    assert g["later"]["event_ids"] == ["open", "later"]
