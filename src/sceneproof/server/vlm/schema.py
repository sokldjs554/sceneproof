"""Structured explanation contract (schema v1) and tolerant parsing.

Every field is something a reviewer can check against the frames. ``assessment`` is a
categorical judgement, never a probability: models emit confident-sounding numbers that are
not calibrated, so the contract does not give them a place to put one.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

SCHEMA_VERSION = "1"

# The one definition of ``assessment``. It is written into the field description (which is part of the schema the
# model receives) and into the English system prompt; the Korean system prompt states the same three cases
# (tests/test_vlm_contract.py). Added to the schema in prompt v1.4 after v1.2 showed that listing the values
# without saying when to use them made the model answer "does this look dangerous?" (docs/eval/vlm-findings.md §3).
ASSESSMENT_RULE = (
    "needs_attention when a person is confirmed inside the restricted zone (red polygon) in the frames, regardless of "
    "whether the activity looks dangerous or ordinary; unclear when near the boundary or inside/outside cannot be "
    "settled; likely_benign only when the person is outside the zone or a tracking error (non-person, misplaced box) "
    "is evident. Opinions about risk go in reasons only."
)


class Observation(BaseModel):
    frame: str = Field(description="which key frame: context_before|entry|dwell|exit|context_after|all")
    text: str = Field(min_length=1, max_length=400)


class PersonInfo(BaseModel):
    count_in_zone: int | None = Field(default=None, ge=0, le=50, description="people visibly inside the zone; null if unclear")
    appearance: str | None = Field(default=None, max_length=200)
    ppe_visible: list[str] | None = Field(default=None, description="e.g. helmet, vest; null if cannot tell")


class ExplanationV1(BaseModel):
    schema_version: Literal["1"] = "1"
    scene_summary: str = Field(min_length=1, max_length=400)
    observations: list[Observation] = Field(min_length=1, max_length=12)
    person: PersonInfo = PersonInfo()
    activity: str = Field(min_length=1, max_length=120, description="short label of what the person is doing")
    zone_relation: Literal["clearly_inside", "near_boundary", "outside", "unclear"]
    assessment: Literal["needs_attention", "likely_benign", "unclear"] = Field(description=ASSESSMENT_RULE)
    reasons: list[str] = Field(default_factory=list, max_length=8)
    uncertainties: list[str] = Field(default_factory=list, max_length=8, description="what cannot be determined from the frames")
    operator_question: str | None = Field(default=None, max_length=300)


def json_schema() -> dict:
    return ExplanationV1.model_json_schema()


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def extract_json(text: str) -> dict | None:
    """Accepts a bare JSON object, a fenced block, or JSON embedded in prose."""
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = _FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def validate(obj: dict | None) -> tuple[ExplanationV1 | None, str]:
    if obj is None:
        return None, "no JSON object found in response"
    try:
        return ExplanationV1.model_validate(obj), ""
    except ValidationError as e:
        return None, f"schema violation: {e.errors()[:3]}"
