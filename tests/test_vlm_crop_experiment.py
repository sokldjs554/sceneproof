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
    assert len(inside) == len(outside) == 23          # 15 entry + 8 dwell frames
    assert {q.role for q in queries} == {"entry", "dwell"}
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


def test_exit_frames_are_left_out_by_default_but_can_be_asked_for():
    # the exit key frame's stored box is the last position inside the zone, up to about a second before the image
    assert "exit" not in vc.ROLES
    assert {q.role for q in vc.load_queries(roles=vc.ALL_ROLES)} == {"entry", "dwell", "exit"}


def test_unknown_role_is_refused():
    with pytest.raises(SystemExit):
        vc.main(["--dry-run", "--strategies", "full", "--roles", "entry,leave"])


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


# --- second question: do clearer marks help? -------------------------------------------------------------------

def test_pairs_count_scenes_where_both_zones_were_answered_right():
    def row(eid, role, variant, outcome):
        return {"event_id": eid, "role": role, "variant": variant, "outcome": outcome}
    rows = [row("a", "entry", "zone_true", "right"), row("a", "entry", "zone_shifted", "right"),
            row("b", "entry", "zone_true", "right"), row("b", "entry", "zone_shifted", "wrong"),
            row("c", "dwell", "zone_true", "abstain"), row("c", "dwell", "zone_shifted", "right"),
            row("d", "entry", "zone_true", "right")]                      # its pair is missing: not counted
    assert vc.pairs(rows) == {"pairs": 3, "pairs_both_right": 1}


def test_a_model_that_always_says_outside_gets_half_right_and_no_pair(queries):
    always_outside = lambda img, prompt, split: {"text": "outside", "latency_ms": 1.0, "input_tokens": 1,  # noqa: E731
                                                  "visual_tokens": 64, "output_tokens": 1}
    s = vc.summarize(vc.run(queries, ["full"], always_outside, log=lambda s: None))["full"]
    assert s["accuracy"] == 0.5 and s["pairs"] == 23 and s["pairs_both_right"] == 0


def test_marked_images_and_question_reach_the_model(queries):
    seen = []

    def stand_in(img, prompt, split):
        seen.append((img.copy(), prompt))
        return {"text": "unclear", "latency_ms": 1.0, "input_tokens": 1, "visual_tokens": 64, "output_tokens": 1}

    def paint_feet(image, box, zone):
        out = image.copy()
        fx, fy = int((box[0] + box[2]) / 2), min(int(box[3]), image.shape[0] - 1)
        out[max(0, fy - 3):fy + 1, max(0, fx - 3):fx + 4] = (0, 255, 0)
        return out

    q = queries[:2]
    rows = vc.run(q, ["full", "crop"], stand_in, crop_fn=lambda W, H, box, zone: (0, 0, W, H), log=lambda s: None,
                  markings=("product", "marked"), mark_fn=paint_feet, mark_prompt=" The green dot marks the feet. ")
    assert len(rows) == 8 and {r["marking"] for r in rows} == {"product", "marked"}
    assert sorted(vc.summarize(rows)) == ["crop", "crop+marked", "full", "full+marked"]
    product = [p for _, p in seen[0:2]]
    marked = [p for _, p in seen[2:4]]
    assert product == [vc.SHORT_PROMPT] * 2
    assert marked == [vc.SHORT_PROMPT + " The green dot marks the feet."] * 2
    assert not np.array_equal(seen[0][0], seen[2][0]), "the marked image is the same as the product image"
    assert np.array_equal(seen[0][0], queries[0].image), "the product image must be the analyser's drawing unchanged"


def test_earlier_results_without_a_marking_still_summarise():
    rows = [{"event_id": "a", "role": "entry", "variant": v, "label": lab, "strategy": "full", "outcome": "right",
             "latency_ms": 1.0, "visual_tokens": 64, "input_tokens": 1, "region_area_share": 1.0}
            for v, lab in (("zone_true", "inside"), ("zone_shifted", "outside"))]
    assert vc.summarize(rows)["full"]["pairs_both_right"] == 1


def test_unknown_marking_is_refused():
    with pytest.raises(SystemExit):
        vc.main(["--dry-run", "--strategies", "full", "--markings", "product,glow"])


def test_marked_needs_the_marks_written_first(monkeypatch, capsys):
    from sceneproof.server.vlm import marks

    monkeypatch.setattr(marks, "MARK_PROMPT", "")
    assert vc.main(["--dry-run", "--strategies", "full", "--markings", "marked"]) == 2
    assert "MARK_PROMPT" in capsys.readouterr().out
    monkeypatch.setattr(marks, "MARK_PROMPT", "The green dot marks the feet.")
    monkeypatch.setattr(marks, "mark_for_vlm", lambda *a: (_ for _ in ()).throw(NotImplementedError()))
    assert vc.main(["--dry-run", "--strategies", "full", "--markings", "marked"]) == 2
    assert "mark_for_vlm" in capsys.readouterr().out
