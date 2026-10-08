"""Unit tests: CrossEncoder reranker (fake model for speed; real-model smoke separate)."""

import pytest

from sgrag.retrieval.models import RetrievalResult
from sgrag.retrieval.rerank import (
    CrossEncoderReranker,
    NoOpReranker,
    RerankerError,
    resolve_model_snapshot,
)


def _results(n):
    out = []
    for i in range(n):
        out.append(RetrievalResult(
            chunk_id=f"c{i:02d}", pmid=str(100 + i), title="t", text=f"text {i}",
            fused_score=1.0 - i * 0.01, rank=i + 1,
            retrieval_sources=["vector"]))
    return out


class _FakeModel:
    """Deterministic fake: scores = -index so rerank reverses the input."""

    def __init__(self, n):
        self.n = n

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        assert len(pairs) == self.n
        return [-i for i in range(self.n)]


def _reranker_with_fake(n):
    rr = CrossEncoderReranker(candidate_k=50)
    rr._get_model = lambda: _FakeModel(n)  # noqa: SLF001 - test seam
    return rr


def test_truncates_to_candidate_k():
    rr = _reranker_with_fake(10)
    rr.candidate_k = 10
    out = rr.rerank("q", _results(30))
    assert len(out) == 30
    # only the first 10 were reranked
    assert all(r.reranker_score is not None for r in out[:10])
    assert all(r.reranker_score is None for r in out[10:])


def test_query_candidate_pair_construction():
    seen = {}

    class M:
        def predict(self, pairs, batch_size=32, show_progress_bar=False):
            seen["pairs"] = pairs
            return [0.5] * len(pairs)

    rr = CrossEncoderReranker(candidate_k=50)
    rr._get_model = lambda: M()  # noqa: SLF001
    res = _results(3)
    rr.rerank("my query", res)
    assert seen["pairs"] == [("my query", f"text {i}") for i in range(3)]


def test_score_assignment_and_deterministic_order():
    rr = _reranker_with_fake(5)
    res = _results(5)
    out = rr.rerank("q", res)
    # fake scores: c00 -> 0, c01 -> -1, ... so c00 stays first
    assert [r.chunk_id for r in out] == [f"c{i:02d}" for i in range(5)]
    assert [r.reranker_rank for r in out] == [1, 2, 3, 4, 5]
    assert [r.final_rank for r in out] == [1, 2, 3, 4, 5]
    assert out[0].reranker_score == 0.0
    assert out[0].final_score == out[0].reranker_score
    # Phase 4 scores preserved
    assert out[2].fused_score == pytest.approx(0.98)
    assert out[2].retrieval_sources == ["vector"]


def test_empty_input():
    rr = _reranker_with_fake(0)
    assert rr.rerank("q", []) == []


def test_inference_failure_raises_reranker_error():
    class Boom:
        def predict(self, *a, **k):
            raise RuntimeError("gpu gone")

    rr = CrossEncoderReranker()
    rr._get_model = lambda: Boom()  # noqa: SLF001
    with pytest.raises(RerankerError, match="inference failed"):
        rr.rerank("q", _results(2))


def test_load_failure_raises_reranker_error():
    rr = CrossEncoderReranker(model="nonexistent/model-xyz")
    with pytest.raises(RerankerError):
        rr._get_model()


def test_noop_passthrough():
    res = _results(3)
    out = NoOpReranker().rerank("q", res)
    assert out is res
    assert all(r.reranker_score is None for r in out)


def test_serialization_keeps_new_fields():
    rr = _reranker_with_fake(2)
    out = rr.rerank("q", _results(2))
    d = out[0].to_dict()
    assert {"reranker_score", "reranker_rank", "final_score", "final_rank"} <= set(d)
    back = RetrievalResult.from_dict(d)
    assert back.reranker_score == out[0].reranker_score
    assert back.final_rank == 1


def test_resolve_model_snapshot_prefers_local():
    # all-MiniLM-L6-v2 is cached locally from earlier phases
    p = resolve_model_snapshot("sentence-transformers/all-MiniLM-L6-v2")
    assert p != "sentence-transformers/all-MiniLM-L-6-v2"
    assert "snapshots" in p
    # unknown model falls back to the hub id
    assert resolve_model_snapshot("no/such-model-xyz") == "no/such-model-xyz"


def test_disabled_reranker_path_in_retriever():
    from sgrag.retrieval import HybridRetriever
    from sgrag.retrieval.config import load_retrieval_config

    cfg = load_retrieval_config("configs/retrieval.yaml")
    assert cfg.reranker.enabled is False  # backward-compatible default
    # construction must not attempt any model load
    ret = HybridRetriever(cfg)
    assert ret.reranker.name == "none"
    ret.close()


@pytest.mark.real_model
def test_real_cross_encoder_smoke():
    """Real-model smoke: scores 3 candidates, deterministic, no fabrication."""
    rr = CrossEncoderReranker(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", candidate_k=10)
    res = _results(3)
    out = rr.rerank("What causes diabetes?", res)
    assert len(out) == 3
    scores = [r.reranker_score for r in out]
    assert all(isinstance(s, float) for s in scores)
    # deterministic: re-running gives identical scores
    res2 = _results(3)
    out2 = rr.rerank("What causes diabetes?", res2)
    assert [r.reranker_score for r in out2] == scores
    # ordering is by (-score, chunk_id)
    assert [r.chunk_id for r in out] == [
        r.chunk_id for r in sorted(out, key=lambda r: (-r.reranker_score, r.chunk_id))]
