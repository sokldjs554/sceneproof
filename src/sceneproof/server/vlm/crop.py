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
    raise NotImplementedError("crop_for_vlm is not implemented yet")
