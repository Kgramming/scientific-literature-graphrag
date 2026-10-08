"""LLM provider abstraction (Phase 5B).

Providers are selected via SGRAG_LLM_PROVIDER ("mock" | "groq"). API keys
come from SGRAG_LLM_API_KEY only — never printed, never logged, never
committed. The mock provider is deterministic and used for tests and for
running the pipeline without credentials.
"""

from __future__ import annotations

import json
from typing import Protocol

import requests


class LLMError(RuntimeError):
    """Raised when the LLM call fails (network, auth, bad response)."""


class LLMProvider(Protocol):
    name: str

    def generate(self, messages: list[dict], max_tokens: int,
                 temperature: float) -> str:
        """Non-streaming chat completion. Returns the assistant text."""
        ...


class MockProvider:
    """Deterministic provider: builds a grounded-looking answer from the
    evidence block in the prompt. For tests and credential-free runs."""

    name = "mock"

    def generate(self, messages: list[dict], max_tokens: int,
                 temperature: float) -> str:
        import re

        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        cites = re.findall(r"\[E\d+\] PMID:(\d+) \| CHUNK:([A-Za-z0-9_\-]+)", user)
        titles = re.findall(r"Title: ([^\n]+)", user)
        if not cites:
            return ("Insufficient evidence was retrieved from the indexed "
                    "scientific literature to answer this question reliably.")
        lines = ["Based on the retrieved scientific literature:"]
        for (pmid, chunk), title in zip(cites[:3], titles[:3]):
            lines.append(
                f"- Evidence from PMID:{pmid} ({title.strip()}) "
                f"[PMID:{pmid} | CHUNK:{chunk}] reports findings relevant "
                f"to the question.")
        lines.append(
            "\nNote: this answer was produced by the deterministic mock LLM "
            "provider (SGRAG_LLM_PROVIDER=mock). Set SGRAG_LLM_PROVIDER=groq "
            "with SGRAG_LLM_API_KEY for live generation.")
        return "\n".join(lines)


class GroqProvider:
    """Groq via its OpenAI-compatible chat completions endpoint."""

    name = "groq"

    def __init__(self, api_key: str, model: str,
                 api_base: str = "https://api.groq.com/openai/v1",
                 timeout_s: int = 60) -> None:
        if not api_key:
            raise LLMError("SGRAG_LLM_API_KEY is not set (required for provider='groq')")
        self._api_key = api_key  # never printed or logged
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, messages: list[dict], max_tokens: int,
                 temperature: float) -> str:
        url = f"{self.api_base}/chat/completions"
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {self._api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.model, "messages": messages,
                      "temperature": temperature, "max_tokens": max_tokens},
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if resp.status_code == 401:
            raise LLMError("LLM authentication failed (HTTP 401): invalid API key")
        if resp.status_code == 429:
            raise LLMError("LLM rate limited (HTTP 429): try again later")
        if not resp.ok:
            raise LLMError(f"LLM request failed (HTTP {resp.status_code})")
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM returned an unreadable response: {exc}") from exc


def make_provider(provider: str, model: str, api_key: str = "",
                  api_base: str = "", timeout_s: int = 60) -> LLMProvider:
    if provider == "mock":
        return MockProvider()
    if provider == "groq":
        return GroqProvider(api_key=api_key, model=model,
                            api_base=api_base or "https://api.groq.com/openai/v1",
                            timeout_s=timeout_s)
    raise LLMError(f"unsupported LLM provider: {provider!r}")


def _redacted(text: str, secret: str) -> str:
    """Defensive helper: ensure a secret never appears in a message."""
    if secret and secret in text:
        return text.replace(secret, "[REDACTED]")
    return text
