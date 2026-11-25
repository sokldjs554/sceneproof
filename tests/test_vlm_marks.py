"""What the marks drawn for a small VLM must do (src/sceneproof/server/vlm/marks.py).

Two things must become visible: where the person's feet are, and which side of the zone's line is inside.
How (colours, shapes, sizes, filling the inside or dimming the outside) is left to the implementation.
"""

import copy

import cv2
import numpy as np
import pytest

from sceneproof.server.vlm import marks

H, W = 432, 768
ZONE = [(300.0, 250.0), (580.0, 250.0), (620.0, 432.0), (270.0, 432.0)]
BOX_IN = (380.0, 60.0, 470.0, 330.0)        # feet (425, 330) inside the zone
BOX_OUT = (80.0, 80.0, 160.0, 300.0)         # feet (120, 300) outside


def grey():
    return np.full((H, W, 3), 128, np.uint8)


def mask_of(poly, shape=(H, W)):
    m = np.zeros(shape, np.uint8)
    if poly:
        cv2.fillPoly(m, [np.array([[round(x), round(y)] for x, y in poly], np.int32)], 1)
    return m.astype(bool)


def away_from(points_mask, box, poly, gap=16):
    """Pixels at least ``gap`` away from the zone's line, the person's box and the feet."""
    keep = np.ones((H, W), np.uint8)
    if poly:
        cv2.polylines(keep, [np.array([[round(x), round(y)] for x, y in poly], np.int32)], True, 0, 2 * gap)
    x1, y1, x2, y2 = (round(v) for v in box)
    cv2.rectangle(keep, (x1 - gap, y1 - gap), (x2 + gap, y2 + gap), 0, -1)
    return points_mask & keep.astype(bool)


def mean_colour(img, m):
    return img[m].astype(float).mean(axis=0)


def test_mark_prompt_tells_the_model_what_the_marks_mean():
    assert isinstance(marks.MARK_PROMPT, str) and 10 <= len(marks.MARK_PROMPT.strip()) <= 400


@pytest.mark.parametrize("box", [BOX_IN, BOX_OUT])
def test_returns_a_marked_copy_and_leaves_the_input_alone(box):
    img = grey()
    before = img.copy()
    out = marks.mark_for_vlm(img, box, copy.deepcopy(ZONE))
    assert isinstance(out, np.ndarray) and out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(img, before), "the input image was changed"
    assert out is not img


@pytest.mark.parametrize("box", [BOX_IN, BOX_OUT])
def test_the_feet_are_marked(box):
    out = marks.mark_for_vlm(grey(), box, ZONE).astype(int)
    fx, fy = round((box[0] + box[2]) / 2), round(box[3])
    yy, xx = np.ogrid[:H, :W]
    near = (xx - fx) ** 2 + (yy - fy) ** 2 <= 6 ** 2
    changed = (np.abs(out - 128).max(axis=2) > 60) & near
    assert changed.sum() >= 20, f"only {changed.sum()} pixels within 6 px of the feet {fx, fy} were clearly marked"


@pytest.mark.parametrize("box", [BOX_IN, BOX_OUT])
def test_inside_and_outside_look_clearly_different(box):
    out = marks.mark_for_vlm(grey(), box, ZONE)
    inside = away_from(mask_of(ZONE), box, ZONE)
    outside = away_from(~mask_of(ZONE), box, ZONE)
    d = np.linalg.norm(mean_colour(out, inside) - mean_colour(out, outside))
    assert d >= 40, f"inside and outside differ by only {d:.0f} in colour"


def test_the_marks_make_a_real_evidence_frame_clearer():
    from sceneproof.eval.vlm_crop import load_queries

    q = next(q for q in load_queries() if q.variant == "zone_true" and q.camera_id == "cam-worker")
    h, w = q.image.shape[:2]
    out = marks.mark_for_vlm(q.image, q.box, q.zone)
    zone_mask = mask_of(q.zone, (h, w))
    keep = np.ones((h, w), np.uint8)
    cv2.polylines(keep, [np.array([[round(x), round(y)] for x, y in q.zone], np.int32)], True, 0, 32)
    x1, y1, x2, y2 = (round(v) for v in q.box)
    cv2.rectangle(keep, (x1 - 16, y1 - 16), (x2 + 16, y2 + 16), 0, -1)
    inside, outside = zone_mask & keep.astype(bool), ~zone_mask & keep.astype(bool)
    before = np.linalg.norm(mean_colour(q.image, inside) - mean_colour(q.image, outside))
    after = np.linalg.norm(mean_colour(out, inside) - mean_colour(out, outside))
    assert after >= before + 20, f"inside/outside contrast went from {before:.0f} to {after:.0f}"


def test_a_zone_reaching_outside_the_frame():
    zone = [(-50.0, 300.0), (500.0, 300.0), (500.0, 500.0), (-50.0, 500.0)]
    out = marks.mark_for_vlm(grey(), (200.0, 100.0, 260.0, 350.0), zone)
    assert out.shape == (H, W, 3)


def test_a_person_reaching_outside_the_frame():
    out = marks.mark_for_vlm(grey(), (700.0, -20.0, 800.0, 450.0), ZONE)
    assert out.shape == (H, W, 3)


def test_without_a_zone_the_feet_are_still_marked():
    out = marks.mark_for_vlm(grey(), BOX_OUT, []).astype(int)
    fx, fy = 120, 300
    assert (np.abs(out[fy - 6:fy + 7, fx - 6:fx + 7] - 128).max(axis=2) > 60).sum() >= 20


def test_the_same_input_gives_the_same_marks():
    a = marks.mark_for_vlm(grey(), BOX_IN, ZONE)
    b = marks.mark_for_vlm(grey(), BOX_IN, ZONE)
    assert np.array_equal(a, b)
