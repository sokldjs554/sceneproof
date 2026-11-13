"""Validation and recorded settings for the minimum detection height."""

from dataclasses import asdict

import pytest
from pydantic import ValidationError

from sceneproof.edge.config import DetectorConfig
from sceneproof.edge.detector import PersonDetector


def test_negative_minimum_box_height_is_rejected():
    with pytest.raises(ValidationError, match="min_box_height_px"):
        DetectorConfig(min_box_height_px=-10)


@pytest.mark.parametrize("minimum", [0, 40])
def test_detector_info_records_minimum_box_height(minimum):
    class StubBackend:
        name = "test"
        version = "1"

    detector = PersonDetector(StubBackend(), "test", (32, 32), min_box_height_px=minimum)
    assert asdict(detector.info)["min_box_height_px"] == minimum
    assert detector.info.__dict__["min_box_height_px"] == minimum
