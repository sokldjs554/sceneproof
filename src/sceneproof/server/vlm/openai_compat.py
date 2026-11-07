"""OpenAI-compatible chat-completions provider (vLLM, llama.cpp server, Ollama, LM Studio).

This is the path for a self-hosted open-weights VLM. vLLM serves Qwen2-VL / SmolVLM / etc. with
``vllm serve <model>`` and exposes /v1/chat/completions with image_url content parts (base64 data
URLs). Only httpx is used so the server has no extra dependency.
"""

from __future__ import annotations

import base64
import time

import httpx

from .base import ExplainRequest, ExplainResult
from .prompt import REPAIR_INSTRUCTION, build_user_text, system_prompt


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(self, base_url: str, model: str, api_key: str = "none", timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def explain(self, req: ExplainRequest, repair_error: str | None = None, previous_text: str | None = None) -> ExplainResult:
        content: list[dict] = []
        for f in req.frames:
            b64 = base64.b64encode(f.jpeg).decode()
            content.append({"type": "text", "text": f"[frame: {f.role} @ {f.media_s:.1f}s]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": build_user_text(req.ctx, req.lang)})
        messages = [{"role": "system", "content": system_prompt(req.lang)}, {"role": "user", "content": content}]
        if repair_error is not None and previous_text is not None:
            messages.append({"role": "assistant", "content": previous_text})
            messages.append({"role": "user", "content": REPAIR_INSTRUCTION[req.lang].format(error=repair_error)})
        body = {"model": self.model, "messages": messages, "max_tokens": req.max_tokens, "temperature": 0.0}
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(f"{self.base_url}/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        return ExplainResult(text, data.get("model", self.model), self.name, (time.perf_counter() - t0) * 1e3,
                             usage=usage, repair_attempted=repair_error is not None)
