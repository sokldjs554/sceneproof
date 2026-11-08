"""Anthropic Claude vision provider (Messages API, structured JSON output).

Why Claude here: the review server needs a VLM that follows a strict JSON contract, reads
several frames at once and answers in Korean; on a CPU-only box no open-weights model does all
three. The provider is still just one implementation of ``VLMProvider``: the OpenAI-compatible
provider swaps in a self-hosted vLLM model without touching the worker.

Cost control: every call records ``usage`` and an estimated USD cost; the worker stops creating
paid calls once ``vlm_budget_usd`` is exhausted.
"""

from __future__ import annotations

import base64
import time

from .base import ExplainRequest, ExplainResult
from .prompt import REPAIR_INSTRUCTION, build_user_text, system_prompt

# USD per 1M tokens (input, output, cache read) — Claude API pricing table, 2026-10.
# Cache writes (5-minute ephemeral) are billed at 1.25x the input price.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
CACHE_WRITE_MULTIPLIER = 1.25

# On Claude Opus 5.5 thinking is always on (adaptive) and its tokens count toward ``max_tokens``.
# With max_tokens=1024 the first CI smoke call ended with ``stop_reason=max_tokens`` before any JSON
# was emitted (vlm-eval run #3), so the request floor is raised to leave room for thinking + the JSON.
MIN_MAX_TOKENS = 4096

# Strict schema for output_config.format (every property required, no additional properties).
STRICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "scene_summary", "observations", "person", "activity", "zone_relation",
                 "assessment", "reasons", "uncertainties", "operator_question"],
    "properties": {
        "schema_version": {"type": "string", "enum": ["1"]},
        "scene_summary": {"type": "string"},
        "observations": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False, "required": ["frame", "text"],
                      "properties": {"frame": {"type": "string"}, "text": {"type": "string"}}},
        },
        "person": {
            "type": "object", "additionalProperties": False,
            "required": ["count_in_zone", "appearance", "ppe_visible"],
            "properties": {
                "count_in_zone": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "appearance": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "ppe_visible": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
            },
        },
        "activity": {"type": "string"},
        "zone_relation": {"type": "string", "enum": ["clearly_inside", "near_boundary", "outside", "unclear"]},
        "assessment": {"type": "string", "enum": ["needs_attention", "likely_benign", "unclear"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "operator_question": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", timeout_s: float = 120.0,
                 use_fallbacks: bool = False) -> None:
        import anthropic

        self.model = model
        self.effort = effort
        self.use_fallbacks = use_fallbacks
        self.client = anthropic.Anthropic(timeout=timeout_s, max_retries=2)
        self._anthropic = anthropic

    def _messages(self, req: ExplainRequest, repair_error: str | None, previous_text: str | None) -> list[dict]:
        content: list[dict] = []
        for f in req.frames:
            content.append({"type": "text", "text": f"[frame: {f.role} @ {f.media_s:.1f}s]"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                       "data": base64.standard_b64encode(f.jpeg).decode()}})
        content.append({"type": "text", "text": build_user_text(req.ctx, req.lang)})
        messages = [{"role": "user", "content": content}]
        if repair_error is not None and previous_text is not None:
            messages.append({"role": "assistant", "content": previous_text})
            messages.append({"role": "user", "content": REPAIR_INSTRUCTION[req.lang].format(error=repair_error)})
        return messages

    def explain(self, req: ExplainRequest, repair_error: str | None = None, previous_text: str | None = None) -> ExplainResult:
        anthropic = self._anthropic
        kwargs = dict(
            model=self.model, max_tokens=max(req.max_tokens, MIN_MAX_TOKENS),
            system=[{"type": "text", "text": system_prompt(req.lang), "cache_control": {"type": "ephemeral"}}],
            messages=self._messages(req, repair_error, previous_text),
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": STRICT_SCHEMA}},
        )
        t0 = time.perf_counter()
        try:
            if self.use_fallbacks:
                response = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                            fallbacks="default", **kwargs)
            else:
                response = self.client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise RuntimeError(f"rate limited: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"api error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"connection error: {e}") from e
        ms = (time.perf_counter() - t0) * 1e3
        usage = {
            "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        }
        pin, pout, pcache = PRICES.get(self.model, (4.0, 20.0, 0.20))
        cost = (usage["input_tokens"] * pin + usage["output_tokens"] * pout
                + usage["cache_creation_input_tokens"] * pin * CACHE_WRITE_MULTIPLIER
                + usage["cache_read_input_tokens"] * pcache) / 1e6
        if response.stop_reason == "refusal":
            cat = response.stop_details.category if response.stop_details else None
            return ExplainResult("", response.model, self.name, ms, usage=usage, cost_usd=cost,
                                 error=f"model refused (category={cat})")
        text = next((b.text for b in response.content if b.type == "text"), "")
        err = "" if response.stop_reason in ("end_turn", "stop_sequence") else f"stop_reason={response.stop_reason}"
        return ExplainResult(text, response.model, self.name, ms, usage=usage, cost_usd=cost, error=err,
                             repair_attempted=repair_error is not None)
