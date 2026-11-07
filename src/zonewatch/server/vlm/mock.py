"""Deterministic mock provider for tests, CI and the console's offline mode.

It is loud about being a mock: scene_summary starts with "[MOCK]" and the stored provider name is
"mock". Results from this provider are never reported as VLM quality numbers.
"""

from __future__ import annotations

import json
import time

from .base import ExplainRequest, ExplainResult


class MockProvider:
    name = "mock"
    model = "mock-v1"

    def __init__(self, delay_s: float = 0.2, fail_every: int = 0, invalid_every: int = 0) -> None:
        self.delay_s = delay_s
        self.fail_every = fail_every
        self.invalid_every = invalid_every
        self.calls = 0

    def explain(self, req: ExplainRequest, repair_error: str | None = None, previous_text: str | None = None) -> ExplainResult:
        self.calls += 1
        t0 = time.perf_counter()
        time.sleep(self.delay_s)
        if self.fail_every and self.calls % self.fail_every == 0:
            raise RuntimeError("mock provider failure (injected)")
        if self.invalid_every and self.calls % self.invalid_every == 0 and repair_error is None:
            return ExplainResult("this is not json", self.model, self.name, (time.perf_counter() - t0) * 1e3)
        roles = [f.role for f in req.frames]
        ctx = req.ctx
        obj = {
            "schema_version": "1",
            "scene_summary": f"[MOCK] {ctx.zone_name} 구역에서 {ctx.kind.split('_')[-1].lower()} 사건 ({len(roles)}개 프레임 입력)",
            "observations": [{"frame": r, "text": f"[MOCK] {r} 프레임을 받았습니다."} for r in roles] or
                            [{"frame": "all", "text": "[MOCK] 프레임 없음"}],
            "person": {"count_in_zone": None, "appearance": None, "ppe_visible": None},
            "activity": "unclear",
            "zone_relation": "unclear",
            "assessment": "unclear",
            "reasons": ["[MOCK] 실제 VLM이 아닙니다. 출력 형식 검증용 자리표시자입니다."],
            "uncertainties": ["[MOCK] 장면을 해석하지 않았습니다."],
            "operator_question": None,
        }
        return ExplainResult(json.dumps(obj, ensure_ascii=False), self.model, self.name, (time.perf_counter() - t0) * 1e3,
                             usage={"input_tokens": 0, "output_tokens": 0})
