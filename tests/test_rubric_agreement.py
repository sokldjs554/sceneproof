"""scripts/rubric_agreement.py: agreement between two people scoring the Claude explanations."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("rubric_agreement", Path(__file__).parents[1] / "scripts" / "rubric_agreement.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)


def test_kappa_known_values():
    assert ra.cohen_kappa(["1", "0", "1", "0"], ["1", "0", "1", "0"]) == 1.0
    assert ra.cohen_kappa(["1", "1", "0", "0"], ["1", "0", "1", "0"]) == 0.0
    # both people always said the same single value: chance agreement is 1, kappa undefined
    assert ra.cohen_kappa(["1", "1", "1"], ["1", "1", "1"]) is None
    assert ra.cohen_kappa([], []) is None


def _row(eid, obs="1", hal="0", unc="1", read="3", note=""):
    return {"event_id": eid, "camera_id": "cam", "grader_observations_match_frames_0_1": obs, "grader_hallucination_0_1": hal,
            "grader_uncertainties_sensible_0_1": unc, "grader_readability_1_3": read, "grader_hallucination_note": note}


def _ans(obs="1 맞음", hal="0 없음", unc="1 타당", read="3 쉬움", note=None):
    a = {"observations_match_frames": obs, "hallucination": hal, "uncertainties_sensible": unc, "readability": read}
    if note:
        a["note"] = note
    return a


def test_compare_counts_and_lists_disagreements():
    rows = [_row("a"), _row("b", hal="1", note="first"), _row("c", read="2"), _row("d", obs="")]  # d: not scored by grader 1
    answers = {"a": _ans(), "b": _ans(hal="0 없음"), "c": _ans(read="1 어려움", note="hard"), "d": _ans()}
    r = ra.compare(rows, answers)
    assert r["events_scored_by_both"] == 3
    assert r["questions"]["hallucination"]["same"] == 2 and r["questions"]["hallucination"]["grader1_yes"] == 1
    # grader 2 answered "0" everywhere: kappa would be 0 regardless, so it is not reported
    assert r["questions"]["hallucination"]["kappa"] is None
    assert r["questions"]["readability"]["same"] == 2 and r["questions"]["readability"]["within_one"] == 3
    got = {(d["event_id"], d["question"]) for d in r["disagreements"]}
    assert got == {("b", "hallucination"), ("c", "readability")}
    b = next(d for d in r["disagreements"] if d["event_id"] == "b")
    assert b["note_grader1"] == "first" and b["grader1"] == "1" and b["grader2"] == "0"
