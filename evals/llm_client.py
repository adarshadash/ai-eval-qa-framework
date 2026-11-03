"""Thin provider wrapper. Swap in another provider by implementing `complete()`."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = ""


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    raw_content: Any = None  # provider content blocks, appended back in tool loops


class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 1024, temperature: float | None = None,
                 timeout: float = 60, max_retries: int = 3):
        import anthropic  # imported lazily so mock mode needs no SDK / key

        self.client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, messages: list[dict], system: str | None = None, tools: list[dict] | None = None,
                 max_tokens: int | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens or self.max_tokens,
                                  "messages": messages}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        start = time.perf_counter()
        resp = self.client.messages.create(**kwargs)
        latency = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text="".join(b.text for b in resp.content if b.type == "text"),
            tool_calls=[ToolCall(b.name, dict(b.input), b.id) for b in resp.content if b.type == "tool_use"],
            stop_reason=resp.stop_reason or "",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=round(latency, 2),
            model=resp.model,
            raw_content=resp.content,
        )


def build_client(role: str, cfg: dict) -> AnthropicClient:
    """role: 'sut' or 'judge' - each can use a different model."""
    m = cfg["models"]
    return AnthropicClient(model=m[role], max_tokens=m["max_tokens"], temperature=m.get("temperature"),
                           timeout=m.get("timeout_seconds", 60), max_retries=m.get("max_retries", 3))
