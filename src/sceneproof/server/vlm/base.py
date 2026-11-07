from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .prompt import EventContext


@dataclass
class FrameInput:
    role: str
    jpeg: bytes
    media_s: float
    width: int = 0
    height: int = 0


@dataclass
class ExplainRequest:
    event_id: str
    ctx: EventContext
    frames: list[FrameInput]
    lang: str = "ko"
    max_tokens: int = 700


@dataclass
class ExplainResult:
    text: str
    model: str
    provider: str
    latency_ms: float
    usage: dict = field(default_factory=dict)
    error: str = ""
    cost_usd: float = 0.0
    repair_attempted: bool = False


class VLMProvider(Protocol):
    name: str
    model: str

    def explain(self, req: ExplainRequest, repair_error: str | None = None, previous_text: str | None = None) -> ExplainResult:
        ...
