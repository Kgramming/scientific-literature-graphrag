"""Unit tests: LLM provider abstraction (no live API calls)."""

import pytest

from sgrag.generation.llm import (
    GroqProvider,
    LLMError,
    MockProvider,
    make_provider,
)


def test_mock_deterministic_and_grounded():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": (
            "EVIDENCE:\n[E1] PMID:111 | CHUNK:c1\nTitle: T1\nText: x\n\n"
            "QUESTION: Q?")},
    ]
    p = MockProvider()
    a1 = p.generate(msgs, max_tokens=100, temperature=0.0)
    a2 = p.generate(msgs, max_tokens=100, temperature=0.0)
    assert a1 == a2
    assert "[PMID:111 | CHUNK:c1]" in a1


def test_mock_empty_evidence_declines():
    p = MockProvider()
    a = p.generate([{"role": "user", "content": "EVIDENCE:\n\nQUESTION: Q?"}],
                   100, 0.0)
    assert "Insufficient evidence" in a


def test_groq_requires_key():
    with pytest.raises(LLMError, match="SGRAG_LLM_API_KEY"):
        GroqProvider(api_key="", model="m")


def test_groq_builds_request_without_leaking_key(monkeypatch):
    seen = {}

    class FakeResp:
        status_code = 200
        ok = True

        def json(self):
            return {"choices": [{"message": {"content": "hello"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update({"url": url, "headers": headers, "json": json})
        return FakeResp()

    monkeypatch.setattr("sgrag.generation.llm.requests.post", fake_post)
    p = GroqProvider(api_key="sk-secret", model="llama-x")
    assert p.generate([{"role": "user", "content": "hi"}], 10, 0.0) == "hello"
    assert seen["url"].endswith("/chat/completions")
    assert seen["json"]["model"] == "llama-x"
    # key only in the Authorization header, never in URL/body/logs
    assert "sk-secret" not in seen["url"]
    assert "sk-secret" not in str(seen["json"])
    assert seen["headers"]["Authorization"] == "Bearer sk-secret"


def test_groq_auth_and_rate_limit_errors(monkeypatch):
    class FakeResp:
        def __init__(self, code):
            self.status_code = code
            self.ok = False

    for code, frag in ((401, "401"), (429, "429")):
        monkeypatch.setattr(
            "sgrag.generation.llm.requests.post", lambda *a, **k: FakeResp(code))
        with pytest.raises(LLMError, match=frag):
            GroqProvider(api_key="k", model="m").generate([], 10, 0.0)


def test_make_provider_routing():
    assert make_provider("mock", "m").name == "mock"
    assert make_provider("groq", "m", api_key="k").name == "groq"
    with pytest.raises(LLMError):
        make_provider("nope", "m")
