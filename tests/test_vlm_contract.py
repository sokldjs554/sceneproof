"""The assessment rule is defined once and reaches the model through the system prompt and the schema."""

import typing

from sceneproof.server.vlm.anthropic_provider import STRICT_SCHEMA
from sceneproof.server.vlm.prompt import SYSTEM_EN, SYSTEM_KO, EventContext, build_user_text
from sceneproof.server.vlm.schema import ASSESSMENT_RULE, ExplanationV1, json_schema


def _ctx():
    return EventContext("cam", "zone", "EVENT_KIND_ZONE_ENTRY", 2.0, 1.0, 3.0, "CLOSE_REASON_EXITED", ["entry", "exit"])


def test_rule_is_in_schema_description_and_english_prompt():
    assert json_schema()["properties"]["assessment"]["description"] == ASSESSMENT_RULE
    assert ASSESSMENT_RULE in SYSTEM_EN
    # the schema dump is part of the user message, so the model sees the definition next to the field
    assert ASSESSMENT_RULE in build_user_text(_ctx())


def test_korean_prompt_states_the_same_three_cases():
    rule = SYSTEM_KO.split("assessment 규칙:")[1].split("4.")[0]
    for value in ("needs_attention", "unclear", "likely_benign"):
        assert value in rule
    for idea in ("제한구역", "경계", "추적 오류", "reasons"):
        assert idea in rule


def test_api_schema_enums_match_the_contract():
    fields = ExplanationV1.model_fields
    for name in ("assessment", "zone_relation"):
        assert STRICT_SCHEMA["properties"][name]["enum"] == list(typing.get_args(fields[name].annotation))
    assert set(STRICT_SCHEMA["required"]) == set(json_schema()["properties"])
