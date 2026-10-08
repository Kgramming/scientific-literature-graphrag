"""Unit tests for bounded similarity-edge construction (no live Neo4j)."""

import numpy as np
import pytest

from sgrag.graph.similarity import (
    l2_normalize,
    similarity_score_distribution,
    top_k_neighbors,
)


def _embs():
    # 4 points: a~b close, c far from both, d == a direction (tie scenario avoided)
    return np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.7, 0.7],
    ])


def test_top_k_excludes_self_and_ranks():
    ids = ["a", "b", "c", "d"]
    edges = top_k_neighbors(ids, _embs(), top_k=2, model="m")
    by_src = {}
    for e in edges:
        by_src.setdefault(e.src_chunk_id, []).append(e)
    assert set(by_src) == set(ids)
    for src, es in by_src.items():
        assert len(es) == 2
        assert [e.rank for e in es] == [1, 2]
        assert all(e.dst_chunk_id != src for e in es)
        assert es[0].score >= es[1].score
        assert all(e.model == "m" for e in es)
    # a's nearest neighbor must be b (cosine ~0.994)
    assert by_src["a"][0].dst_chunk_id == "b"


def test_exact_edge_count_bounded():
    ids = [f"c{i}" for i in range(10)]
    rng = np.random.default_rng(42)
    edges = top_k_neighbors(ids, rng.normal(size=(10, 8)), top_k=3)
    assert len(edges) == 30  # exactly n * top_k, never O(n^2)


def test_top_k_capped_at_n_minus_1():
    ids = ["a", "b"]
    edges = top_k_neighbors(ids, _embs()[:2], top_k=5)
    assert len(edges) == 2  # each chunk links only to the other


def test_deterministic_ties():
    ids = ["x", "y", "z"]
    embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])  # y,z equidistant from x
    e1 = top_k_neighbors(ids, embs, top_k=2)
    e2 = top_k_neighbors(ids, embs, top_k=2)
    assert [(e.src_chunk_id, e.dst_chunk_id) for e in e1] == \
           [(e.src_chunk_id, e.dst_chunk_id) for e in e2]


def test_empty_and_mismatched():
    assert top_k_neighbors([], np.zeros((0, 4)), top_k=2) == []
    with pytest.raises(ValueError):
        top_k_neighbors(["a"], np.zeros((2, 4)), top_k=2)


def test_l2_normalize_zero_row():
    out = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.allclose(np.linalg.norm(out[1]), 1.0)
    assert np.allclose(out[0], [0.0, 0.0])


def test_score_distribution():
    from sgrag.graph.similarity import SimilarityEdge
    edges = [SimilarityEdge("a", "b", 0.9, 1, "m"), SimilarityEdge("a", "c", 0.5, 2, "m")]
    d = similarity_score_distribution(edges, bins=2)
    assert d["count"] == 2 and d["min"] == 0.5 and d["max"] == 0.9
    assert sum(d["histogram"]) == 2
    assert similarity_score_distribution([]) == {"count": 0}
