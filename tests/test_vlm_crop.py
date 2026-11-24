"""What the region sent to a small VLM must satisfy (src/sceneproof/server/vlm/crop.py).

The VLM is asked where the tracked person's feet are relative to the restricted zone. It can only answer
if the region shows the person and the stretch of the zone boundary nearest to the feet; and the point of
cropping is that the region is smaller than the frame, so the same 64 visual tokens carry more detail.
How much margin to leave, and how to shape the rectangle, is left to the implementation.
"""

import copy
import math

import pytest

from sceneproof.server.vlm.crop import crop_for_vlm

W, H = 1920, 1080
ZONE = [(800.0, 700.0), (1300.0, 700.0), (1400.0, 1080.0), (700.0, 1080.0)]   # a floor zone at the bottom


def nearest_on_boundary(p, poly):
    """Closest point to p on the polygon's edges."""
    best, best_d = None, math.inf
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1], strict=True):
        dx, dy = bx - ax, by - ay
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
        q = (ax + t * dx, ay + t * dy)
        d = math.dist(p, q)
        if d < best_d:
            best, best_d = q, d
    return best


def feet(box):
    return ((box[0] + box[2]) / 2, box[3])


def check(rect, width, height):
    x1, y1, x2, y2 = rect
    assert all(isinstance(v, int) for v in rect), f"the rectangle is whole pixels: {rect}"
    assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, f"{rect} is not inside the {width}x{height} frame"


def contains(rect, x, y, tol=1.0):
    x1, y1, x2, y2 = rect
    return x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol


CASES = {
    "person_inside_the_zone": (980.0, 560.0, 1060.0, 760.0),
    "person_outside_left": (420.0, 640.0, 480.0, 820.0),
    "person_outside_far_above": (1000.0, 120.0, 1040.0, 220.0),
    "person_on_the_boundary": (1250.0, 520.0, 1330.0, 705.0),
    "small_person_far_away": (1150.0, 610.0, 1162.0, 640.0),
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_region_is_inside_the_frame(name):
    check(crop_for_vlm(W, H, CASES[name], ZONE), W, H)


@pytest.mark.parametrize("name", list(CASES))
def test_the_region_shows_the_whole_person(name):
    box = CASES[name]
    rect = crop_for_vlm(W, H, box, ZONE)
    assert contains(rect, box[0], box[1]) and contains(rect, box[2], box[3]), f"{rect} cuts off the person {box}"


@pytest.mark.parametrize("name", list(CASES))
def test_the_region_shows_the_zone_boundary_nearest_to_the_feet(name):
    box = CASES[name]
    q = nearest_on_boundary(feet(box), ZONE)
    rect = crop_for_vlm(W, H, box, ZONE)
    assert contains(rect, *q), f"{rect} does not show the zone boundary at {q}, the nearest to the feet {feet(box)}"


@pytest.mark.parametrize("name", list(CASES))
def test_the_region_is_not_too_small_or_too_thin(name):
    x1, y1, x2, y2 = crop_for_vlm(W, H, CASES[name], ZONE, min_side=224, max_aspect=2.0)
    w, h = x2 - x1, y2 - y1
    assert w >= 224 and h >= 224, f"{w}x{h} is below min_side"
    assert max(w, h) / min(w, h) <= 2.0 + 1e-6, f"{w}x{h} is thinner than max_aspect allows"


def test_min_side_and_max_aspect_are_parameters():
    x1, y1, x2, y2 = crop_for_vlm(W, H, CASES["small_person_far_away"], ZONE, min_side=400, max_aspect=1.2)
    w, h = x2 - x1, y2 - y1
    assert w >= 400 and h >= 400 and max(w, h) / min(w, h) <= 1.2 + 1e-6, (w, h)


@pytest.mark.parametrize("name", ["person_inside_the_zone", "small_person_far_away", "person_on_the_boundary"])
def test_the_region_is_much_smaller_than_the_frame_when_the_scene_allows(name):
    # a person and the nearby boundary that together fit in a quarter of the frame: cropping must pay off
    x1, y1, x2, y2 = crop_for_vlm(W, H, CASES[name], ZONE)
    assert (x2 - x1) * (y2 - y1) <= 0.25 * W * H, f"{(x2 - x1)}x{(y2 - y1)} is more than a quarter of the frame"


def test_a_person_reaching_outside_the_frame():
    box = (1800.0, -30.0, 1990.0, 1100.0)          # boxes from the tracker can poke out of the frame
    rect = crop_for_vlm(W, H, box, ZONE)
    check(rect, W, H)
    assert contains(rect, 1800, 0) and contains(rect, W, H), rect


def test_a_person_filling_the_frame_gives_the_whole_frame():
    assert crop_for_vlm(W, H, (0.0, 0.0, float(W), float(H)), ZONE) == (0, 0, W, H)


def test_without_a_zone_only_the_person_matters():
    box = CASES["person_outside_left"]
    rect = crop_for_vlm(W, H, box, [])
    check(rect, W, H)
    assert contains(rect, box[0], box[1]) and contains(rect, box[2], box[3])


def test_a_frame_smaller_than_min_side():
    rect = crop_for_vlm(200, 150, (50.0, 20.0, 90.0, 140.0), [(0.0, 100.0), (200.0, 100.0), (200.0, 150.0), (0.0, 150.0)])
    assert rect == (0, 0, 200, 150), rect


def test_the_frames_of_this_project():
    # the evidence frames in demo/bundle are 768x432; a person at the bench and the bench zone
    zone = [(307.2, 250.56), (583.68, 250.56), (614.4, 432.0), (276.48, 432.0)]
    box = (347.9, 0.8, 531.6, 392.5)
    rect = crop_for_vlm(768, 432, box, zone)
    check(rect, 768, 432)
    assert contains(rect, box[0], box[1]) and contains(rect, box[2], box[3])


def test_inputs_are_not_changed_and_the_answer_is_repeatable():
    box, zone = CASES["person_inside_the_zone"], copy.deepcopy(ZONE)
    first = crop_for_vlm(W, H, box, zone)
    assert zone == ZONE and box == CASES["person_inside_the_zone"]
    assert crop_for_vlm(W, H, box, zone) == first
