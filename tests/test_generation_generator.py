"""Unit tests: GroundedGenerator orchestration (mock retriever + mock LLM)."""

import pytest

from sgrag.generation.config import GenerationConfig
from sgrag.generation.generator import GroundedGenerator
from sgrag.generation.llm import MockProvider
from sgrag.retrieval.models import RetrievalResult


def _res(cid, pmid="111", fused=0.9):
    return RetrievalResult(chunk_id=cid, pmid=pmid, title="T", text="txt",
                           fused_score=fused, final_score=fused, rank=1,
                           retrieval_sources=["vector"])


class FakeRetriever:
    def __init__(self, results):
        self._results = results
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        diag = {"query": query, "n_vector": len(self._results),
                "n_entity": 0, "n_semantic": 0, "n_fused": len(self._results),
                "query_entities": [], "total_ms": 1.0}
        return list(self._results), diag

    def close(self):
        pass


class CountingLLM(MockProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, messages, max_tokens, temperature):
        self.calls += 1
        return super().generate(messages, max_tokens, temperature)


def _gen(results, llm=None, **kw):
    cfg = GenerationConfig(provider="mock", model="mock-1", **kw)
    return GroundedGenerator(gen_config=cfg,
                             retriever=FakeRetriever(results),
                             provider=llm or MockProvider())


def test_insufficient_evidence_no_llm_call():
    llm = CountingLLM()
    g = _gen([], llm=llm)
    r = g.answer("What treatments help diabetes patients?")
    assert "Insufficient evidence" in r.answer
    assert llm.calls == 0
    assert any("LLM not called" in w for w in r.warnings)


def test_unsupported_question_no_retrieval():
    g = _gen([_res("c1")])
    r = g.answer("Tell me a joke")
    assert "does not appear to be a scientific" in r.answer
    assert g._retriever.calls == 0


def test_happy_path_citations_valid():
    g = _gen([_res("c1", pmid="111"), _res("c2", pmid="222")])
    r = g.answer("What treatments help diabetes patients?")
    assert r.citation_validation.valid
    assert {c.pmid for c in r.citations} <= {"111", "222"}
    assert r.evidence and r.provider == "mock"
    d = r.to_dict()
    assert d["query"] and d["answer"]
    r2 = type(r).from_dict(d)
    assert r2.answer == r.answer and len(r2.evidence) == 2


def test_fabricated_citation_repaired_or_stripped():
    class Liar(MockProvider):
        def generate(self, messages, max_tokens, temperature):
            return "Miracle cure found [PMID:999999]."

    g = _gen([_res("c1", pmid="111")], llm=Liar(),
             max_citation_retries=1)
    r = g.answer("What treatments help diabetes patients?")
    # stricter retry also lies -> invalid citations stripped, warning added
    assert "[PMID:999999]" not in r.answer
    assert r.citation_validation.regenerated is True
    assert any("unsupported citation" in w for w in r.warnings)


def test_multipart_fuses_subquery_evidence():
    g = _gen([_res("c1", pmid="111")])
    r = g.answer("What treatments help diabetes patients? How is it diagnosed?")
    assert g._retriever.calls == 2  # one retrieval per subquery
    assert r.retrieval_metadata["query_class"] == "multipart"


def test_llm_error_graceful():
    from sgrag.generation.llm import LLMError

    class Down(MockProvider):
        def generate(self, *a, **k):
            raise LLMError("boom")

    g = _gen([_res("c1")], llm=Down())
    r = g.answer("What treatments help diabetes patients?")
    assert "unavailable" in r.answer
    assert any("LLM error" in w for w in r.warnings)
