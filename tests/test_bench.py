"""Unit tests for the research benchmark (no dataset download needed: synthetic boxes and episodes)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sceneproof.edge.zones import ZoneSpec
from sceneproof.eval.bench.agreement import cohen_kappa, compare
from sceneproof.eval.bench.data import GTBox, Sequence
from sceneproof.eval.bench.detection import APAccumulator
from sceneproof.eval.bench.episodes import Ep, gt_episodes
from sceneproof.eval.bench.match import match, same_person

ZONE = ZoneSpec("z", "test", [(0.4, 0.4), (0.6, 0.4), (0.6, 0.8), (0.4, 0.8)], dwell_seconds=5.0,
                lost_grace_seconds=1.5, entry_confirm_seconds=0.3, exit_confirm_seconds=0.8)


def _seq(frames: int, fps: float = 10.0) -> Sequence:
    return Sequence("syn", "caviar", "syn", "syn", fps, None, None, frames, 100, 100)  # type: ignore[arg-type]


def _walk(tid: int, f0: int, f1: int, x0: float, x1: float, y_feet: float = 60.0) -> dict[int, list[GTBox]]:
    out = {}
    for f in range(f0, f1):
        x = x0 + (x1 - x0) * (f - f0) / max(1, f1 - f0 - 1)
        out[f] = [GTBox(tid, (x - 3, y_feet - 20, x + 3, y_feet))]
    return out


def _merge(*ds):
    out: dict[int, list[GTBox]] = {}
    for d in ds:
        for f, bl in d.items():
            out.setdefault(f, []).extend(bl)
    return out


def test_gt_crossing_makes_one_intrusion():
    # walks left->right across the zone (x 40..60) at feet y=60 over 100 frames (10 s)
    eps = gt_episodes(_seq(120), _walk(1, 0, 100, 10, 90), [ZONE])
    intr = [e for e in eps if e.label == "intrusion"]
    assert len(intr) == 1
    e = intr[0]
    # feet cross x=40 at ~3.8 s; the episode starts when 0.3 s inside is confirmed (same rule for GT and predictions)
    assert 3.9 <= e.start_s <= 4.4 and 6.0 < e.end_s < 7.3


def test_gt_brush_past_is_boundary_not_intrusion():
    # feet at y=38: 2 px above the zone's top edge (y=40), never inside
    eps = gt_episodes(_seq(80), _walk(2, 0, 60, 10, 90, y_feet=38.0), [ZONE])
    assert not [e for e in eps if e.label == "intrusion"]
    assert [e for e in eps if e.label == "boundary_pass"]


def test_gt_ignore_label_kept_separate():
    boxes = _walk(3, 0, 100, 10, 90)
    for bl in boxes.values():
        bl[0].label = "ignore"
    eps = gt_episodes(_seq(120), boxes, [ZONE])
    assert [e.label for e in eps] == ["ignore"]


def _ep(eid, start, end, label="pred", track=1, foot_x=50.0, frames=None):
    frames = frames or range(int(start * 10), int(end * 10) + 1, 3)
    return Ep(eid, "z", track, start, end, False, label, "", {f: (foot_x - 3, 40, foot_x + 3, 60) for f in frames})


def test_match_requires_same_person():
    gt = [_ep("g1", 2.0, 6.0, "intrusion", foot_x=45), _ep("g2", 2.0, 6.0, "intrusion", track=2, foot_x=55)]
    # one prediction at person 2's position: must match g2, not g1 (time alone cannot tell them apart)
    m = match([_ep("p1", 2.1, 5.9, foot_x=55.5)], gt)
    assert m.tp == [("p1", "g2")] and m.missed == ["g1"] and not m.fp
    # a prediction far from both people at the same time is a false alarm
    m2 = match([_ep("p2", 2.1, 5.9, foot_x=90)], gt)
    assert m2.fp == ["p2"] and len(m2.missed) == 2


def test_match_duplicates_and_neutral():
    gt = [_ep("g1", 2.0, 8.0, "intrusion"), _ep("b1", 12.0, 13.0, "boundary_pass", track=5)]
    preds = [_ep("p1", 2.0, 5.0), _ep("p2", 5.5, 8.0), _ep("p3", 12.1, 12.9, track=9)]
    m = match(preds, gt)
    assert len(m.tp) == 1 and len(m.duplicates) == 1 and m.neutral == [("p3", "b1")] and not m.fp and not m.missed


def test_same_person_none_without_common_frames():
    assert same_person(_ep("p", 1.0, 2.0), _ep("g", 5.0, 6.0, "intrusion")) is None


def test_ap_perfect_and_ignore():
    acc = APAccumulator()
    gt = [GTBox(1, (10, 10, 30, 60)), GTBox(2, (50, 10, 70, 60), label="ignore")]
    # exact hit on the person, a detection on the ignore region (dropped), nothing else
    acc.add(np.array([[10, 10, 30, 60], [50, 10, 70, 60]], np.float32), np.array([0.9, 0.8], np.float32), gt)
    r = acc.result()
    assert r["gt_boxes"] == 1 and r["ap50"] == pytest.approx(1.0) and r["ap50_95"] == pytest.approx(1.0)


def test_ap_counts_false_positive():
    acc = APAccumulator()
    gt = [GTBox(1, (10, 10, 30, 60))]
    acc.add(np.array([[80, 80, 95, 99], [10, 10, 30, 60]], np.float32), np.array([0.9, 0.5], np.float32), gt)
    r = acc.result()
    assert 0.4 < r["ap50"] < 0.6  # the higher-scored box is wrong


def test_agreement_identical_and_disjoint():
    a = {"annotator": "A", "duration_s": 20, "events": [{"start_s": 2, "end_s": 6}, {"start_s": 10, "end_s": 12}]}
    same = compare(a, json.loads(json.dumps(a)))
    assert same["event_f1"] == 1.0 and same["frame_kappa"] == 1.0
    b = {"annotator": "B", "duration_s": 20, "events": [{"start_s": 15, "end_s": 18}]}
    r = compare(a, b)
    assert r["matched"] == 0 and r["event_f1"] == 0.0 and r["frame_kappa"] < 0.1
    assert cohen_kappa(np.array([], bool), np.array([], bool)) is None


def test_frozen_split_hash_and_no_group_leak():
    from sceneproof.eval.bench.data import load_sources
    from sceneproof.eval.bench.split import load_split, load_split2

    s, s2 = load_split(), load_split2()  # raise if a frozen file was edited
    assert s2["after_split_sha256_16"] == s["sha256_16"]
    where = {sid: part for part, ids in s["sequences"].items() for sid in ids}
    second = {sid for ids in s2["sequences"].values() for sid in ids}
    assert second.isdisjoint(where), "a second held-out sequence was already in the first split"
    where.update({sid: "test2" for sid in second})
    groups: dict[str, set] = {}
    for seq in load_sources()["sequences"]:
        groups.setdefault(seq["group"], set()).add(where[seq["id"]])  # every source sequence is in exactly one split
    assert all(len(parts) == 1 for parts in groups.values()), "a group is split across train/dev/test"
    seen_sites = {seq["site"] for seq in load_sources()["sequences"] if where[seq["id"]] in ("train", "dev", "test")}
    assert set(s["unseen_sites"]).isdisjoint({seq["site"] for seq in load_sources()["sequences"]
                                              if where[seq["id"]] in ("train", "dev")})
    assert set(s2["unseen_sites"]).isdisjoint(seen_sites)


def test_ap_ignore_region_can_differ_between_iou_thresholds():
    """Regression (final run crashed): a box matched at IoU 0.5 but not at 0.75 that also overlaps an ignore
    region is kept at 0.5 and dropped at 0.75 — per-threshold score lists must stay aligned."""
    acc = APAccumulator()
    person = GTBox(1, (10, 10, 30, 60))
    ignore = GTBox(2, (8, 8, 34, 66), label="ignore")  # e.g. a distractor right behind the person
    det = np.array([[10, 16, 30, 66], [200, 200, 220, 250]], np.float32)  # IoU ~0.68 with the person; one far FP
    for _ in range(3):
        acc.add(det, np.array([0.9, 0.3], np.float32), [person, ignore])
    r = acc.result()
    assert r["gt_boxes"] == 3 and 0.0 < r["ap50_95"] < r["ap50"] <= 1.0
    assert all(len(acc.scores[t]) == len(acc.tp[t]) for t in acc.THRESHOLDS)


def test_test_split_requires_final_flag(monkeypatch, capsys):
    """The held-out split cannot be measured ad hoc: --split test without --final is refused before any work."""
    import sys

    from sceneproof.eval.bench import run

    monkeypatch.setattr(sys, "argv", ["run", "--split", "test", "--candidate", "x:models/yolox_tiny.onnx"])
    assert run.main() == 2
    assert "only with --final" in capsys.readouterr().err


def test_second_final_run_needs_a_reason(monkeypatch, tmp_path, capsys):
    """With an existing lock, --final without --rerun-reason is refused and the lock file is left untouched."""
    import sys

    from sceneproof.eval.bench import run

    lock = tmp_path / "test_lock.json"
    lock.write_text(json.dumps({"policy": "x", "runs": [{"opened": "2026-01-01T00:00:00"}]}))
    before = lock.read_text()
    monkeypatch.setattr(run, "LOCK", lock)
    monkeypatch.setattr(sys, "argv", ["run", "--split", "test", "--final"])
    assert run.main() == 3
    assert lock.read_text() == before
    assert "--rerun-reason" in capsys.readouterr().err


def test_final_run_refused_when_data_is_incomplete(monkeypatch, tmp_path, capsys):
    """A final run that could not finish (missing video, frame-count mismatch) is refused before the lock is written,
    so it cannot use up the one measurement."""
    import sys

    from sceneproof.eval.bench import run

    seqs = {"a": Sequence("a", "caviar", "s", "g", 25.0, tmp_path / "a.mp4", tmp_path / "a.xml", 10, 100, 100,
                          {"manifest": {"frames_match_gt": False}}),
            "b": Sequence("b", "caviar", "s", "g", 25.0, tmp_path / "b.mp4", tmp_path / "b.xml")}
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "a.xml").write_text("<x/>")
    monkeypatch.setattr(run, "load_sequences", lambda require_data=False: seqs)
    problems = run.preflight(["a", "b", "c"])
    assert len(problems) == 3 and "frame count differs" in problems[0] and "missing" in problems[1]

    lock = tmp_path / "test_lock.json"
    monkeypatch.setattr(run, "LOCK", lock)
    monkeypatch.setattr(run, "preflight", lambda ids: ["x: video or ground truth missing"])
    monkeypatch.setattr(run, "_dirty", lambda: [])
    monkeypatch.setattr(sys, "argv", ["run", "--split", "test", "--final"])
    assert run.main() == 4
    assert not lock.exists()
    assert "refused before touching the lock" in capsys.readouterr().err


def test_augment_clips_targets_to_the_image_not_the_padding():
    """MOT boxes extend past the frame for people cut off at the edge; training targets must stop at the image edge
    (the grey padding around a shrunken image is not part of the person)."""
    import random

    pytest.importorskip("torch")
    from sceneproof.train.finetune import augment

    img = np.zeros((100, 200, 3), np.uint8)  # black stays black under the HSV jitter; padding is 114
    boxes = np.array([[10, 50, 40, 130], [150, -20, 180, 60]], np.float32)  # past the bottom / top edge
    for seed in range(30):
        canvas, b = augment(img, boxes, 160, random.Random(seed))
        ys, xs = np.where(canvas[..., 0] == 0)
        for x0, y0, x1, y1 in b:
            assert xs.min() - 1 <= x0 and x1 <= xs.max() + 1 and ys.min() - 1 <= y0 and y1 <= ys.max() + 1


def test_candidate_options_and_prediction_only_corrections():
    from sceneproof.eval.bench.episodes import zone_specs
    from sceneproof.eval.bench.run import parse_candidate

    c = parse_candidate("t:models/yolox_tiny.onnx+track_conf=0.1+rejoin+exclude=0")
    assert c == {"name": "t", "model": "models/yolox_tiny.onnx", "variant": "yolox_tiny", "track_conf": 0.1,
                 "rejoin": True, "exclude": False}
    with pytest.raises(ValueError):
        parse_candidate("t:m.onnx+typo")
    cfg = {"rules": {"dwell_seconds": 5, "lost_grace_seconds": 1.5, "entry_confirm_seconds": 0.3, "exit_confirm_seconds": 0.8},
           "sites": {"s": {"zones": [{"zone_id": "z", "name": "z", "polygon": [[0, 0], [1, 0], [1, 1]],
                                      "exclude": [[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]]}]}}}
    gt, pred = zone_specs("s", cfg)[0], zone_specs("s", cfg, exclude=True, rejoin=True)[0]
    assert gt.exclude == [] and not gt.rejoin  # ground truth always uses the plain zone
    assert len(pred.exclude) == 1 and pred.rejoin


def test_match_assignment_is_optimal_not_greedy():
    """p1 overlaps g1 longest but also qualifies for g2; p2 qualifies only for g1. Greedy would give p1-g1 and leave
    g2 missed + p2 duplicate; the optimal assignment matches both people."""
    gt = [_ep("g1", 2.0, 8.0, "intrusion", foot_x=50), _ep("g2", 2.0, 6.0, "intrusion", track=2, foot_x=56)]
    preds = [_ep("p1", 2.0, 8.0, foot_x=53), _ep("p2", 2.0, 4.0, foot_x=45)]
    m = match(preds, gt)
    assert sorted(m.tp) == [("p1", "g2"), ("p2", "g1")] and not m.missed and not m.duplicates


def test_ap_clips_ground_truth_to_the_frame():
    """A person cut off at the bottom edge: MOT's box runs past the frame, the detector's box stops at the edge.
    Unclipped, IoU is 0.4 (a miss at IoU 0.5); clipped to the frame it is a hit."""
    gt = [GTBox(1, (10, 60, 30, 160))]  # frame is 100 px high: 60% of the box is outside
    det = np.array([[10, 60, 30, 100]], np.float32)
    plain, clipped = APAccumulator(), APAccumulator()
    plain.add(det, np.array([0.9], np.float32), gt)
    clipped.add(det, np.array([0.9], np.float32), gt, frame_wh=(100, 100))
    assert plain.result()["ap50"] == pytest.approx(0.0) and clipped.result()["ap50"] == pytest.approx(1.0)
