from __future__ import annotations

from ..config import ServerSettings
from .base import VLMProvider


def build_provider(s: ServerSettings) -> VLMProvider | None:
    p = s.vlm_provider
    if p in ("none", "", "off"):
        return None
    if p == "mock":
        from .mock import MockProvider

        return MockProvider(s.vlm_mock_delay_s, s.vlm_mock_fail_every, s.vlm_mock_invalid_every)
    if p == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=s.vlm_model or "claude-opus-5-5", timeout_s=s.vlm_timeout_s)
    if p == "openai_compat":
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(s.vlm_openai_base_url, s.vlm_model or "HuggingFaceTB/SmolVLM-500M-Instruct",
                                    s.vlm_openai_api_key, s.vlm_timeout_s)
    if p == "transformers":
        from .transformers_local import TransformersProvider

        return TransformersProvider(s.vlm_model or s.vlm_transformers_model)
    raise ValueError(f"unknown vlm provider {p!r}")
