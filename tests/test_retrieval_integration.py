"""Integration tests against the live Phase 3B Neo4j graph.

Skipped when Neo4j is unavailable. Uses small top-k values; never pulls
the full 53k-chunk corpus into Python.
"""

import os

import pytest

from sgrag.retrieval import HybridRetriever, RetrievalError
from sgrag.retrieval.config import RetrievalConfig, load_retrieval_config

CONFIG = os.environ.get("SGRAG_RETRIEVAL_CONFIG",
                        "configs/retrieval.yaml")

pytestmark = pytest.mark.integration


def _needs_neo4j():
    try:
        with HybridRetriever(CONFIG) as ret:
            idx = ret.check_index()
        return idx
    except RetrievalError as exc:
        pytest.skip(f"Neo4j unavailable: {exc}")


def test_vector_index_online():
    idx = _needs_neo4j()
    assert idx["state"] == "ONLINE"
    assert idx["type"] == "VECTOR"


def test_retrieve_alzheimer_query():
    _needs_neo4j()
    cfg = load_retrieval_config(CONFIG)
    cfg = cfg.__class__(**{**cfg.__dict__, "vector_top_k": 5, "max_candidates": 20})
    with HybridRetriever(cfg) as ret:
        results, diag = ret.retrieve(
            "What are the risk factors associated with Alzheimer's disease?")
    assert diag["n_vector"] == 5
    assert diag["n_fused"] <= 20
    assert results, "expected non-empty retrieval"
    assert "DISEASE:alzheimer disease" in diag["query_entities"]
    assert any("entity" in r.retrieval_sources for r in results), \
        "expected at least one entity-sourced candidate"
    r0 = results[0]
    assert r0.rank == 1 and r0.chunk_id and r0.pmid and r0.text
    assert r0.title, "Article title should be backfilled"
    assert 0.0 <= r0.fused_score <= 1.0
    # determinism: results sorted by (-fused_score, chunk_id)
    scores = [(r.fused_score, r.chunk_id) for r in results]
    assert scores == sorted(scores, key=lambda s: (-s[0], s[1]))


def test_no_entity_fallback_is_vector_only():
    _needs_neo4j()
    cfg = load_retrieval_config(CONFIG)
    cfg = cfg.__class__(**{**cfg.__dict__, "vector_top_k": 5, "max_candidates": 20})
    with HybridRetriever(cfg) as ret:
        results, diag = ret.retrieve("recent advances in methodology")
    # graceful regardless of whether an entity was found
    assert results, "vector fallback must still return results"
    if not diag["query_entities"]:
        assert diag["n_entity"] == 0
        assert all(r.retrieval_sources == ["vector"] or
                   "semantic_graph" in r.retrieval_sources for r in results)


def test_result_serializable_round_trip():
    _needs_neo4j()
    cfg = load_retrieval_config(CONFIG)
    cfg = cfg.__class__(**{**cfg.__dict__, "vector_top_k": 3, "max_candidates": 10})
    with HybridRetriever(cfg) as ret:
        results, _ = ret.retrieve("breast cancer treatment drugs")
    d = results[0].to_dict()
    assert set(d) >= {"chunk_id", "pmid", "title", "text", "fused_score",
                      "retrieval_sources", "matched_entities", "rank"}


def test_neo4j_unavailable_raises_clear_error():
    cfg = load_retrieval_config(CONFIG)
    bad = cfg.__class__(**{**cfg.__dict__, "neo4j_uri": "bolt://localhost:1"})
    with pytest.raises(RetrievalError, match="Neo4j unavailable"):
        with HybridRetriever(bad) as ret:
            ret.retrieve("test query")
