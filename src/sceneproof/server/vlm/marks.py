"""Marks drawn on an evidence frame so a small VLM can see where the feet are and which side is the zone.

The evidence frames carry the analyser's own drawing: the zone's outline with a faint (15 %) red fill and a
yellow box round the tracked person. In the experiment of docs/experiments/vlm-crop, SmolVLM-500M could not
tell, for any of 23 scenes, both the real zone (feet inside) and the same zone moved aside (feet outside).
One reason may be that the drawing does not say clearly enough where the feet are or which side of the line
is "inside". This module draws marks that do; ``sceneproof.eval.vlm_crop --markings product,marked`` measures
whether they help.

The foot marker uses the same bottom-centre coordinate as the evaluator.
"""

from __future__ import annotations

import cv2
import numpy as np

Box = tuple[float, float, float, float]          # x1, y1, x2, y2 in pixels; may reach outside the frame
Point = tuple[float, float]

MARK_PROMPT = (
    "The center of the cyan dot marks the person's feet. "
    "The red shaded area is the inside of the restricted zone."
)


def mark_for_vlm(image: np.ndarray, box: Box, zone: list[Point]) -> np.ndarray:
    """A copy of the BGR ``image`` with marks that show the person's feet and the inside of the zone.

    ``box`` is the tracked person; the feet are the middle of its bottom edge. ``zone`` is the restricted
    zone's polygon in pixels (may be empty or reach outside the frame). The input image is not changed.
    """
    marked = image.copy()
    if zone:
        # Blend only the polygon so the surrounding floor keeps its original colour.
        overlay = image.copy()
        vertices = np.rint(zone).astype(np.int32)
        cv2.fillPoly(overlay, [vertices], (0, 0, 255))
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, dst=marked)

    foot = (round((box[0] + box[2]) / 2), round(box[3]))
    radius = max(5, round(min(image.shape[:2]) / 48))
    cv2.circle(marked, foot, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(marked, foot, radius, (255, 255, 0), -1, cv2.LINE_AA)
    return marked
