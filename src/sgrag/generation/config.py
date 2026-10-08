"""Generation configuration (Phase 5B).

LLM credentials come from SGRAG_LLM_* environment variables ONLY — never
from another project's keys, never printed, never committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = "mock"          # "mock" | "groq"
    model: str = "mock-1"
    api_key: str = ""               # SGRAG_LLM_API_KEY only
    api_base: str = "https://api.groq.com/openai/v1"
    evidence_top_k: int = 8         # bounded evidence context (5-10)
    max_tokens: int = 1024
    temperature: float = 0.2
    request_timeout_s: int = 60
    max_citation_retries: int = 1   # one stricter regeneration on bad cites
    extra: dict = field(default_factory=dict)


def load_generation_config() -> GenerationConfig:
    provider = os.environ.get("SGRAG_LLM_PROVIDER", "mock").strip().lower()
    if provider not in ("mock", "groq"):
        raise ValueError(f"unsupported SGRAG_LLM_PROVIDER: {provider!r}")
    model_default = "llama-3.3-70b-versatile" if provider == "groq" else "mock-1"
    return GenerationConfig(
        provider=provider,
        model=os.environ.get("SGRAG_LLM_MODEL", model_default),
        api_key=os.environ.get("SGRAG_LLM_API_KEY", ""),
        api_base=os.environ.get(
            "SGRAG_LLM_API_BASE", "https://api.groq.com/openai/v1"),
        evidence_top_k=int(os.environ.get("SGRAG_EVIDENCE_TOP_K", "8")),
        max_tokens=int(os.environ.get("SGRAG_MAX_TOKENS", "1024")),
        temperature=float(os.environ.get("SGRAG_TEMPERATURE", "0.2")),
    )
