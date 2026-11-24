"""Which part of an evidence frame to send to a small VLM.

A small local VLM (SmolVLM) looks at each image through a fixed budget of visual tokens: one 512-pixel
tile is 64 tokens. Sending the whole frame as one tile is cheap but shrinks a person far from the
camera to a few pixels; splitting the frame into many tiles keeps the detail but multiplies the tokens
and the latency on a CPU (more than nine minutes for one three-image request on 4 vCPU). Choosing
where to look, and sending only that region as one tile, is the third option. This module chooses the
region; ``sceneproof.eval.vlm_crop`` measures what each option costs and how often the answer is right.

The rules are fixed by tests/test_vlm_crop.py. The body is to be written.
"""

from __future__ import annotations

import math

Box = tuple[float, float, float, float]          # x1, y1, x2, y2 in pixels; may reach outside the frame
Point = tuple[float, float]


def crop_for_vlm(width: int, height: int, box: Box, zone: list[Point], *,
                 min_side: int = 224, max_aspect: float = 2.0) -> tuple[int, int, int, int]:
    """The pixel rectangle (x1, y1, x2, y2) of a ``width`` x ``height`` frame to send instead of the frame.

    ``box`` is the tracked person; ``zone`` is the restricted zone's polygon in pixels (may be empty).
    The VLM has to judge where the person's feet are relative to the zone, so the rectangle must show the
    person and the part of the zone boundary nearest to the feet. Beyond that, smaller is better: the same
    64 tokens then carry more detail.
    """
    if width <= 0 or height <= 0 or min_side <= 0 or max_aspect < 1:
        raise ValueError("invalid frame size or crop constraints")
    if min(width, height) < min_side:
        return (0, 0, width, height)

    def clamp(value: float, limit: int) -> float:
        return min(max(value, 0.0), float(limit))

    # Keep the portion of the track that is visible in the frame.
    xs = [clamp(box[0], width), clamp(box[2], width)]
    ys = [clamp(box[1], height), clamp(box[3], height)]
    foot_x, foot_y = (box[0] + box[2]) / 2, box[3]
    if zone:
        closest = None
        best = math.inf
        for (ax, ay), (bx, by) in zip(zone, zone[1:] + zone[:1], strict=True):
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = max(0.0, min(1.0, ((foot_x - ax) * dx + (foot_y - ay) * dy) / length2)) if length2 else 0.0
            px, py = ax + t * dx, ay + t * dy
            distance = (px - foot_x) ** 2 + (py - foot_y) ** 2
            if distance < best:
                best, closest = distance, (px, py)
        if closest is not None:
            xs.append(clamp(closest[0], width))
            ys.append(clamp(closest[1], height))

    left, right = math.floor(min(xs)), math.ceil(max(xs))
    top, bottom = math.floor(min(ys)), math.ceil(max(ys))
    # A small margin helps retain visual context around the person and boundary.
    margin = max(8, math.ceil(0.05 * max(right - left, bottom - top)))
    left, right = max(0, left - margin), min(width, right + margin)
    top, bottom = max(0, top - margin), min(height, bottom + margin)
    crop_w = max(min_side, right - left)
    crop_h = max(min_side, bottom - top)
    crop_w = max(crop_w, math.ceil(crop_h / max_aspect))
    crop_h = max(crop_h, math.ceil(crop_w / max_aspect))
    if crop_w > width or crop_h > height:
        return (0, 0, width, height)

    x1 = max(0, min(width - crop_w, (left + right - crop_w) // 2))
    y1 = max(0, min(height - crop_h, (top + bottom - crop_h) // 2))
    return (x1, y1, x1 + crop_w, y1 + crop_h)
