"""Marks drawn on an evidence frame so a small VLM can see where the feet are and which side is the zone.

The evidence frames carry the analyser's own drawing: the zone's outline with a faint (15 %) red fill and a
yellow box round the tracked person. In the experiment of docs/experiments/vlm-crop, SmolVLM-500M could not
tell, for any of 23 scenes, both the real zone (feet inside) and the same zone moved aside (feet outside).
One reason may be that the drawing does not say clearly enough where the feet are or which side of the line
is "inside". This module draws marks that do; ``sceneproof.eval.vlm_crop --markings product,marked`` measures
whether they help.

The rules are fixed by tests/test_vlm_marks.py. The body and MARK_PROMPT are to be written.
"""

from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]          # x1, y1, x2, y2 in pixels; may reach outside the frame
Point = tuple[float, float]

# One or two English sentences, added to the question, that tell the model what the marks mean
# (for example which colour shows the feet and which shows the inside of the zone).
MARK_PROMPT = ""


def mark_for_vlm(image: np.ndarray, box: Box, zone: list[Point]) -> np.ndarray:
    """A copy of the BGR ``image`` with marks that show the person's feet and the inside of the zone.

    ``box`` is the tracked person; the feet are the middle of its bottom edge. ``zone`` is the restricted
    zone's polygon in pixels (may be empty or reach outside the frame). The input image is not changed.
    """
    raise NotImplementedError("mark_for_vlm is not implemented yet")
