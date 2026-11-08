"""Prompt builder. ``PROMPT_VERSION`` changes whenever wording changes, and is stored with every
explanation so results from different prompt versions are never compared silently."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .schema import json_schema

PROMPT_VERSION = "v1.3-ko"  # v1.2 → v1.3: assessment 규칙 명시 (docs/eval/vlm-findings.md §4)

SYSTEM_KO = """당신은 산업현장 CCTV 사건 검토를 돕는 보조자입니다. 입력으로 한 사건의 전후 프레임(주석 버전에는
붉은 다각형 = 제한구역, 노란 상자 = 추적 중인 사람)이 주어집니다. 규칙:
1. 프레임에서 직접 보이는 것만 서술합니다. 보이지 않는 사실(신원, 의도, 화면 밖 상황, 소리)은 단정하지 않습니다.
2. 판단이 어려우면 "unclear"를 사용하고, 무엇이 불확실한지 uncertainties에 적습니다.
3. 확률·신뢰도 숫자를 쓰지 않습니다. assessment 규칙: 프레임에서 사람이 제한구역(붉은 다각형) 안에 있는 것이 확인되면
   needs_attention — 위험해 보이는지, 평범한 행동인지와 무관합니다. 경계 부근이거나 구역 안인지 확정할 수 없으면 unclear.
   사람이 구역 밖에 있거나 추적 오류(사람이 아닌 대상, 상자 위치 오류)가 분명할 때만 likely_benign.
   위험도나 행동에 대한 의견은 reasons에만 적습니다.
4. 사람을 특정하는 표현(이름, 신원 추정)은 쓰지 않습니다. 외형은 복장·장비 수준에서만 기술합니다.
5. 출력은 반드시 아래 JSON 스키마를 따르는 JSON 객체 하나만 반환합니다. 설명 문장은 한국어로 씁니다."""

SYSTEM_EN = """You assist with reviewing industrial CCTV events. You receive key frames around one event
(annotated frames: red polygon = restricted zone, yellow box = the tracked person). Rules:
1. Describe only what is directly visible. Never assert identity, intent, off-screen context or sound.
2. When unsure say "unclear" and list what is uncertain under uncertainties.
3. No probabilities or confidence numbers. assessment rule: needs_attention when a person is confirmed inside the
   restricted zone (red polygon) in the frames — regardless of whether the activity looks dangerous or ordinary;
   unclear when near the boundary or inside/outside cannot be settled; likely_benign only when the person is
   outside the zone or a tracking error (non-person, misplaced box) is evident. Opinions about risk go in reasons only.
4. Do not identify people; describe clothing/equipment only.
5. Return exactly one JSON object following the schema below."""


@dataclass
class EventContext:
    camera_name: str
    zone_name: str
    kind: str  # ZONE_ENTRY | ZONE_DWELL
    dwell_seconds: float
    entry_media_s: float
    exit_media_s: float | None
    close_reason: str
    frame_roles: list[str]


def build_user_text(ctx: EventContext, lang: str = "ko") -> str:
    schema = json.dumps(json_schema(), ensure_ascii=False)
    if lang == "ko":
        kind = "체류(설정 시간 초과)" if ctx.kind.endswith("DWELL") else "진입"
        close = {"CLOSE_REASON_EXITED": "구역을 벗어남", "CLOSE_REASON_TRACK_LOST": "추적이 끊김(가림/화면 밖 가능)",
                 "CLOSE_REASON_STREAM_ENDED": "영상 종료"}.get(ctx.close_reason, ctx.close_reason or "미종료")
        return (f"카메라: {ctx.camera_name}\n구역: {ctx.zone_name}\n사건 종류: {kind}\n"
                f"CV 추정 체류 시간: {ctx.dwell_seconds:.1f}초 (영상 {ctx.entry_media_s:.1f}s ~ "
                f"{ctx.exit_media_s if ctx.exit_media_s is not None else '진행중'}s)\n종료 사유: {close}\n"
                f"프레임 순서: {', '.join(ctx.frame_roles)}\n\n"
                f"위 프레임들을 보고 JSON 스키마에 맞춰 답하세요.\n스키마: {schema}")
    kind = "dwell (exceeded configured time)" if ctx.kind.endswith("DWELL") else "entry"
    return (f"Camera: {ctx.camera_name}\nZone: {ctx.zone_name}\nEvent: {kind}\n"
            f"CV dwell estimate: {ctx.dwell_seconds:.1f}s (media {ctx.entry_media_s:.1f}s - "
            f"{ctx.exit_media_s if ctx.exit_media_s is not None else 'open'}s)\nClose reason: {ctx.close_reason}\n"
            f"Frame order: {', '.join(ctx.frame_roles)}\n\nAnswer with one JSON object following this schema: {schema}")


# The example is deliberately from a different scene (a loading dock, no person) so that a model that merely
# echoes it is detectable: the worker rejects results whose scene_summary equals the example's.
COMPACT_EXAMPLE = {
    "schema_version": "1",
    "scene_summary": "Empty loading dock with a parked forklift; nobody is inside the marked area.",
    "observations": [{"frame": "entry", "text": "A forklift is parked left of the red polygon; no person is visible."}],
    "person": {"count_in_zone": 0, "appearance": None, "ppe_visible": None},
    "activity": "none",
    "zone_relation": "outside",
    "assessment": "likely_benign",
    "reasons": ["no person inside the red polygon in any frame"],
    "uncertainties": ["the far corner of the dock is dark"],
    "operator_question": None,
}


def build_compact_text(ctx: EventContext) -> str:
    """Short English prompt with one example object instead of the JSON schema dump — for small local
    models (≤1B) that cannot follow a long schema. Field names and enums are the same as schema v1."""
    kind = "dwell" if ctx.kind.endswith("DWELL") else "entry"
    return (f"Camera: {ctx.camera_name}. Zone: {ctx.zone_name}. Event: {kind}, {ctx.dwell_seconds:.0f} s inside. "
            f"Frames: {', '.join(ctx.frame_roles)}.\n"
            "Describe only what is visible in THESE frames (people, clothing, position relative to the red polygon, "
            "the yellow box marks the tracked person). Answer with ONE single-line JSON object using exactly these keys "
            "(zone_relation: clearly_inside|near_boundary|outside|unclear; assessment: needs_attention|likely_benign|unclear). "
            "Format example from a DIFFERENT scene, do not copy its text:\n" + json.dumps(COMPACT_EXAMPLE))


def system_prompt(lang: str = "ko") -> str:
    return SYSTEM_KO if lang == "ko" else SYSTEM_EN


REPAIR_INSTRUCTION = {
    "ko": "이전 응답이 JSON 스키마를 만족하지 않았습니다 ({error}). 설명 없이 스키마에 맞는 JSON 객체 하나만 다시 출력하세요.",
    "en": "The previous answer did not satisfy the JSON schema ({error}). Output exactly one JSON object matching the schema, nothing else.",
}
