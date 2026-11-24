"""The measuring tool for 'where should a small VLM look' (src/sceneproof/eval/vlm_crop.py).

The model is replaced by stand-ins here: these tests check the queries, the labels and the bookkeeping,
not what SmolVLM answers. Nothing here needs torch, transformers or the crop function to be written.
"""

import numpy as np
import pytest

from sceneproof.edge.zones import point_in_polygon
from sceneproof.eval import vlm_crop as vc


@pytest.fixture(scope="module")
def queries():
    return vc.load_queries()


def test_the_demo_evidence_gives_a_balanced_set(queries):
    inside = [q for q in queries if q.label == "inside"]
    outside = [q for q in queries if q.label == "outside"]
    assert len(inside) == len(outside) >= 30
    assert {q.role for q in queries} == {"entry", "dwell", "exit"}
    assert {q.variant for q in inside} == {"zone_true"} and {q.variant for q in outside} == {"zone_shifted"}


def test_labels_follow_the_feet_and_the_drawn_zone(queries):
    for q in queries:
        fx, fy = vc.feet(q.box)
        assert point_in_polygon(fx, fy, q.zone) == (q.label == "inside"), (q.event_id, q.role, q.variant)
        if q.variant == "zone_shifted":
            h, w = q.image.shape[:2]
            assert vc._boundary_distance((fx / w, fy / h), [(x / w, y / h) for x, y in q.zone]) >= 0.06 - 1e-9


def test_the_shifted_zone_stays_in_the_frame(queries):
    for q in queries:
        h, w = q.image.shape[:2]
        assert all(-1 <= x <= w + 1 and -1 <= y <= h + 1 for x, y in q.zone)


def test_shifted_zone_returns_none_when_no_step_works():
    whole = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert vc.shifted_zone(whole, (0.5, 0.5)) is None


def test_full_and_split_send_the_whole_frame_and_crop_sends_the_region(queries):
    q = queries[0]
    h, w = q.image.shape[:2]
    for s in ("full", "split"):
        img, rect = vc.region(q, s)
        assert img is q.image and rect == (0, 0, w, h)
    img, rect = vc.region(q, "crop", crop_fn=lambda W, H, box, zone: (10, 20, 210, 220))
    assert rect == (10, 20, 210, 220) and img.shape[:2] == (200, 200)
    assert np.array_equal(img, q.image[20:220, 10:210])


@pytest.mark.parametrize("label, text, rel, outcome", [
    ("inside", '{"zone_relation": "clearly_inside"}', "clearly_inside", "right"),
    ("inside", 'Sure. {"zone_relation": "outside"}', "outside", "wrong"),
    ("outside", '{"zone_relation": "outside"}', "outside", "right"),
    ("outside", '{"zone_relation": "near_boundary"}', "near_boundary", "abstain"),
    ("inside", '{"zone_relation": "unclear"}', "unclear", "abstain"),
    ("inside", "The person is inside.", None, "unparsed"),
    ("inside", '{"zone_relation": "inside"}', None, "unparsed"),
    ("inside", ' "near_boundary".', "near_boundary", "abstain"),
    ("outside", "outside", "outside", "right"),
    ("inside", ' "clearly_inside"', "clearly_inside", "right"),
    ("inside", "clearly_inside or outside", None, "unparsed"),
    ("outside", "Near_boundary.", "near_boundary", "abstain"),
    ("outside", "Outside.", "outside", "right"),
    ("inside", "Clearly inside", "clearly_inside", "right"),
    ("inside", "Near boundary.", "near_boundary", "abstain"),
    ("inside", "Inside.", None, "unparsed"),
])
def test_judge(label, text, rel, outcome):
    assert vc.judge(label, text) == (rel, outcome)


def test_run_and_summary_with_a_stand_in_model(queries, tmp_path):
    seen = []

    def stand_in(img, prompt, split):
        seen.append(split)
        return {"text": '{"zone_relation": "clearly_inside"}', "latency_ms": 100.0 if split else 10.0,
                "input_tokens": 400 if split else 120, "visual_tokens": 320 if split else 64, "output_tokens": 8}

    qs = [q for q in queries if q.label == "inside"][:4] + [q for q in queries if q.label == "outside"][:4]
    rows = vc.run(qs, ["full", "split", "crop"], stand_in, crop_fn=lambda W, H, box, zone: (0, 0, W // 2, H // 2),
                  save_images=tmp_path, log=lambda s: None)
    assert len(rows) == 24 and seen.count(True) == 8
    assert len(list(tmp_path.glob("*.jpg"))) == 24
    s = vc.summarize(rows)
    assert s["full"]["right"] == 4 and s["full"]["wrong"] == 4 and s["full"]["accuracy"] == 0.5
    assert s["full"]["by_label"]["outside"]["wrong"] == 4
    assert s["split"]["visual_tokens_mean"] == 320 and s["split"]["latency_ms_median"] == 100.0
    assert s["crop"]["region_area_share_mean"] == 0.25 and s["full"]["region_area_share_mean"] == 1.0


def test_dry_run_needs_no_model(tmp_path, capsys):
    assert vc.main(["--dry-run", "--strategies", "full", "--limit", "3", "--save-images", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("*.jpg"))) == 3
    assert "3 queries" in capsys.readouterr().out


def test_unknown_strategy_is_refused():
    with pytest.raises(SystemExit):
        vc.main(["--dry-run", "--strategies", "full,zoom"])
