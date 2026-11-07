from zonewatch.eval.gt import GroundTruth, GTEvent
from zonewatch.eval.metrics import PredEvent, match_events, summarize


def gt(events):
    return GroundTruth("v.mp4", 10, [], {}, events, [], {}, 600.0)


def test_matching_tp_fp_miss_dup_boundary():
    g = gt([GTEvent("g1", 10, 20, "intrusion", True), GTEvent("g2", 40, 45, "intrusion", False),
            GTEvent("b1", 60, 62, "boundary_pass")])
    preds = [PredEvent("p1", 11, 19, True), PredEvent("p2", 12, 15, False),  # dup of g1
             PredEvent("p3", 100, 103, False),  # fp
             PredEvent("p4", 60.5, 61.5, False)]  # boundary hit (not fp)
    m = match_events(preds, g)
    assert m.tp == [("p1", "g1")] and m.fp == ["p3"] and m.missed == ["g2"]
    assert m.duplicates == [("p2", "g1")] and m.boundary_hits == [("p4", "b1")]
    s = summarize(m, g, 600.0)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["false_alarms_per_hour"] == 6.0
    assert s["dwell"] == {"tp_dwell_correct": 1, "tp_dwell_wrong": 0}


def test_short_event_matches_by_start_tolerance():
    g = gt([GTEvent("g1", 10.0, 10.3, "intrusion")])
    m = match_events([PredEvent("p1", 10.8, 11.5, False)], g)
    assert m.tp == [("p1", "g1")]
