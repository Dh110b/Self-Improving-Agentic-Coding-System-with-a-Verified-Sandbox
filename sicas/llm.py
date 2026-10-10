"""LLM clients + cost metering. Core uses only the stdlib; `anthropic` is optional."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


def load_env(path: str | None = None) -> None:
    """Load KEY=VALUE lines from a git-ignored .env into os.environ (real env vars win)."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.isfile(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k.strip(), v)


load_env()


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
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")  # needed for keys not scoped to a workspace
        self.client = anthropic.Anthropic(
            default_headers={"anthropic-workspace-id": ws} if ws else None)
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.name = f"anthropic:{model}"

    def chat(self, system, messages, seed=0):
        kw = dict(model=self.model, system=system, messages=messages, max_tokens=self.max_tokens)
        if self.temperature:  # some SDK/model versions reject `temperature`; omit it when unused
            kw["temperature"] = self.temperature
        try:
            r = self.client.messages.create(**kw)
        except (TypeError, getattr(__import__("anthropic"), "BadRequestError", TypeError)):
            if "temperature" not in kw:
                raise
            del kw["temperature"]
            self.temperature = 0.0
            r = self.client.messages.create(**kw)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, r.usage.input_tokens, r.usage.output_tokens)


class OpenAICompatLLM(LLM):
    """Ollama / vLLM / Gemini / any OpenAI-compatible server (default: local Ollama)."""

    def __init__(self, model: str, base_url: str | None = None, temperature: float | None = None,
                 api_key: str | None = None, send_seed: bool = True):
        self.model = model
        if temperature is None:  # SICAS_TEMPERATURE lets experiments sample (seeds need variance)
            temperature = float(os.environ.get("SICAS_TEMPERATURE", "0.0"))
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "http://localhost:11434/v1").rstrip("/")
        self.temperature = temperature
        self.api_key, self.send_seed = api_key, send_seed
        self.hosted = api_key is not None  # hosted APIs: fail loudly, never score an empty reply
        self.name = f"openai-compat:{model}"

    def chat(self, system, messages, seed=0):
        body = {"model": self.model, "temperature": self.temperature,
                "max_tokens": int(os.environ.get("SICAS_MAX_TOKENS", "1024")),
                "messages": [{"role": "system", "content": system}, *messages]}
        if self.send_seed:
            body["seed"] = seed
        key = self.api_key or os.environ.get("OPENAI_API_KEY", "none")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        tries = 8 if self.hosted else 4
        for attempt in range(tries):  # servers stall, drop connections, or rate-limit (429)
            try:
                with urllib.request.urlopen(req, timeout=int(os.environ.get("SICAS_TIMEOUT", "240"))) as f:
                    d = json.load(f)
                break
            except urllib.error.HTTPError as e:
                detail = e.read()[:2000].decode("utf-8", "replace")
                if self.hosted and e.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"LLM HTTP {e.code}: {detail[:300]}") from e
                if self.hosted and e.code == 429 and re.search(r"retry in (\d+h|\d{2,}m)", detail):
                    # quota exhausted for hours (e.g. free-tier daily cap): waiting seconds won't help
                    raise RuntimeError("LLM quota exhausted: " + detail[:300]) from e
                if attempt == tries - 1:
                    if self.hosted:
                        raise RuntimeError(f"LLM HTTP {e.code} after {tries} tries") from e
                    return LLMResponse("", 0, 0)
                time.sleep(min(60, 5 * 2 ** attempt) if self.hosted else 5 * (attempt + 1))
            except (OSError, ValueError):
                if attempt == tries - 1:  # local: the run simply fails instead of crashing
                    if self.hosted:
                        raise
                    return LLMResponse("", 0, 0)
                time.sleep(5 * (attempt + 1))
        u = d.get("usage", {})
        return LLMResponse(d["choices"][0]["message"]["content"] or "",
                           u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def make_llm(spec: str, tasks=None) -> LLM:
    """spec: 'mock' | 'anthropic:<model>' | 'ollama:<model>' | 'openai:<model>@<base_url>' | 'gemini:<model>'."""
    kind, _, rest = spec.partition(":")
    if kind == "mock":
        from .mock import MockLLM
        return MockLLM(tasks or [])
    if kind == "anthropic":
        return AnthropicLLM(rest or "claude-haiku-4-5-20251001")
    if kind == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set (put it in .env)")
        return OpenAICompatLLM(rest or "gemini-2.5-flash", GEMINI_URL, api_key=key,
                               send_seed=os.environ.get("SICAS_SEND_SEED") == "1")
    if kind in ("ollama", "openai"):
        model, _, url = rest.partition("@")
        return OpenAICompatLLM(model, url or None)
    raise ValueError(f"unknown llm spec {spec!r}")
