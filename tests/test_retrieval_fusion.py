"""Unit tests: fusion math, deduplication, deterministic ranking."""

import math

from sgrag.retrieval.config import FusionConfig
from sgrag.retrieval.fusion import (
    fuse_candidates,
    merge_candidates,
    min_max_normalize,
)


def _vec(cid, score, rank):
    return {"chunk_id": cid, "pmid": "1", "vector_score": score,
            "vector_rank": rank, "sources": {"vector"}, "matched_entities": []}


def _ent(cid, score, key):
    return {"chunk_id": cid, "pmid": "1", "graph_score": score,
            "sources": {"entity"}, "matched_entities": [key]}


def _sem(cid, score):
    return {"chunk_id": cid, "pmid": "1", "semantic_score": score,
            "sources": {"semantic_graph"}, "matched_entities": []}


def test_min_max_normalize():
    assert min_max_normalize([]) == []
    assert min_max_normalize([5.0, 5.0]) == [1.0, 1.0]
    out = min_max_normalize([0.0, 0.5, 1.0])
    assert out == [0.0, 0.5, 1.0]


def test_merge_deduplicates_and_merges_provenance():
    merged = merge_candidates(
        [_vec("c1", 0.9, 1)],
        [_ent("c1", 0.4, "DISEASE:x"), _ent("c2", 0.2, "DISEASE:y")],
        [_sem("c1", 0.8)],
    )
    assert set(merged) == {"c1", "c2"}
    c1 = merged["c1"]
    assert c1["sources"] == {"vector", "entity", "semantic_graph"}
    assert c1["matched_entities"] == ["DISEASE:x"]
    assert c1["vector_score"] == 0.9 and c1["graph_score"] == 0.4
    assert c1["semantic_score"] == 0.8


def test_weighted_fusion_explainable_and_bounded():
    fusion = FusionConfig()
    res = fuse_candidates([_vec("c1", 0.9, 1), _vec("c2", 0.1, 2)],
                          [_ent("c2", 0.9, "DISEASE:y")], [], fusion, 10)
    assert len(res) == 2
    r1, r2 = res
    assert r1.chunk_id == "c1"  # strong vector signal wins with default weights
    assert r1.retrieval_sources == ["vector"]
    assert r2.retrieval_sources == ["entity", "vector"]
    assert all(0.0 <= r.fused_score <= 1.0 for r in res)
    # explainability: component scores retained
    assert r2.vector_score == 0.1 and r2.graph_score == 0.9
    assert r2.matched_entities == ["DISEASE:y"]


def test_rrf_fusion():
    fusion = FusionConfig(method="rrf", rrf_k=60)
    res = fuse_candidates([_vec("c1", 0.9, 1), _vec("c2", 0.8, 2)],
                          [_ent("c2", 0.5, "DISEASE:y")], [], fusion, 10)
    # c2 appears in 2 rankings -> higher RRF than c1 in 1 ranking
    assert res[0].chunk_id == "c2"
    assert math.isclose(res[0].fused_score, 1 / 61 + 1 / 62, rel_tol=1e-9)


def test_deterministic_tie_break():
    fusion = FusionConfig()
    res = fuse_candidates([_vec("c_b", 0.5, 1), _vec("c_a", 0.5, 2)],
                          [], [], fusion, 10)
    # identical scores -> chunk_id ascending
    assert [r.chunk_id for r in res] == ["c_a", "c_b"]
    assert [r.rank for r in res] == [1, 2]


def test_empty_fusion():
    assert fuse_candidates([], [], [], FusionConfig(), 10) == []


def test_max_candidates_cap():
    fusion = FusionConfig()
    cands = [_vec(f"c{i:03d}", 1.0 - i * 0.01, i + 1) for i in range(50)]
    res = fuse_candidates(cands, [], [], fusion, 10)
    assert len(res) == 10


def test_result_serializable():
    fusion = FusionConfig()
    res = fuse_candidates([_vec("c1", 0.9, 1)], [_ent("c1", 0.3, "DISEASE:x")],
                          [], fusion, 10)
    d = res[0].to_dict()
    assert d["chunk_id"] == "c1" and d["rank"] == 1
    assert isinstance(d["retrieval_sources"], list)
