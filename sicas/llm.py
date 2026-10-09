"""LLM clients + cost metering. Core uses only the stdlib; `anthropic` is optional."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass


class BudgetExceeded(Exception):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class CostMeter:
    """Tracks tokens/$ and enforces a compute budget (equal-budget comparisons, RQ1/RQ4)."""

    def __init__(self, price_in: float = 1.0, price_out: float = 5.0,
                 budget_tokens: int | None = None, budget_usd: float | None = None):
        self.price_in, self.price_out = price_in, price_out  # USD per 1M tokens
        self.budget_tokens, self.budget_usd = budget_tokens, budget_usd
        self.tokens_in = self.tokens_out = 0

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def usd(self) -> float:
        return (self.tokens_in * self.price_in + self.tokens_out * self.price_out) / 1e6

    def add(self, r: LLMResponse) -> None:
        self.tokens_in += r.input_tokens
        self.tokens_out += r.output_tokens

    def check(self) -> None:
        if self.budget_tokens is not None and self.tokens >= self.budget_tokens:
            raise BudgetExceeded(f"token budget {self.budget_tokens} reached")
        if self.budget_usd is not None and self.usd >= self.budget_usd:
            raise BudgetExceeded(f"USD budget {self.budget_usd} reached")


class LLM:
    name = "base"

    def chat(self, system: str, messages: list[dict], seed: int = 0) -> LLMResponse:
        raise NotImplementedError


class AnthropicLLM(LLM):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 2048,
                 temperature: float = 0.0):
        import anthropic  # lazy
        self.client = anthropic.Anthropic()
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.name = f"anthropic:{model}"

    def chat(self, system, messages, seed=0):
        r = self.client.messages.create(model=self.model, system=system, messages=messages,
                                        max_tokens=self.max_tokens,
                                        temperature=self.temperature)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, r.usage.input_tokens, r.usage.output_tokens)


class OpenAICompatLLM(LLM):
    """Ollama / vLLM / any OpenAI-compatible server (default: local Ollama)."""

    def __init__(self, model: str, base_url: str | None = None, temperature: float | None = None):
        self.model = model
        if temperature is None:  # SICAS_TEMPERATURE lets experiments sample (seeds need variance)
            temperature = float(os.environ.get("SICAS_TEMPERATURE", "0.0"))
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "http://localhost:11434/v1").rstrip("/")
        self.temperature = temperature
        self.name = f"openai-compat:{model}"

    def chat(self, system, messages, seed=0):
        body = {"model": self.model, "temperature": self.temperature, "seed": seed, "max_tokens": int(os.environ.get("SICAS_MAX_TOKENS", "1024")),
                "messages": [{"role": "system", "content": system}, *messages]}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json",
             "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'none')}"})
        for attempt in range(4):  # local servers occasionally stall or drop a connection
            try:
                with urllib.request.urlopen(req, timeout=int(os.environ.get("SICAS_TIMEOUT", "240"))) as f:
                    d = json.load(f)
                break
            except (OSError, ValueError):
                if attempt == 3:  # give up on this call; the run simply fails instead of crashing
                    return LLMResponse("", 0, 0)
                time.sleep(5 * (attempt + 1))
        u = d.get("usage", {})
        return LLMResponse(d["choices"][0]["message"]["content"],
                           u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


def make_llm(spec: str, tasks=None) -> LLM:
    """spec: 'mock' | 'anthropic:<model>' | 'ollama:<model>' | 'openai:<model>@<base_url>'."""
    kind, _, rest = spec.partition(":")
    if kind == "mock":
        from .mock import MockLLM
        return MockLLM(tasks or [])
    if kind == "anthropic":
        return AnthropicLLM(rest or "claude-haiku-4-5-20251001")
    if kind in ("ollama", "openai"):
        model, _, url = rest.partition("@")
        return OpenAICompatLLM(model, url or None)
    raise ValueError(f"unknown llm spec {spec!r}")
