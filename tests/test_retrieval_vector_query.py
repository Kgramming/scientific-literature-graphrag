"""Unit tests: vector query construction/parsing; query-side helpers."""

import numpy as np
import pytest

from sgrag.retrieval.models import QueryEntities
from sgrag.retrieval.query import _is_informative, embed_query
from sgrag.retrieval.vector import build_vector_query, parse_vector_rows, vector_search


def test_build_vector_query_safe_name():
    q = build_vector_query("chunk_embedding_idx")
    assert "CALL db.index.vector.queryNodes('chunk_embedding_idx'" in q
    assert "ORDER BY vector_score DESC, chunk_id ASC" in q
    with pytest.raises(ValueError):
        build_vector_query("x'); DROP INDEX x;--")


def test_parse_vector_rows_ranking():
    rows = [
        {"chunk_id": "c1", "pmid": 111, "strategy": "semantic_cluster",
         "text": "a", "token_count": 5, "vector_score": 0.9},
        {"chunk_id": "c2", "pmid": 111, "strategy": "semantic_cluster",
         "text": "b", "token_count": 6, "vector_score": 0.8},
    ]
    out = parse_vector_rows(rows)
    assert [c["chunk_id"] for c in out] == ["c1", "c2"]
    assert [c["vector_rank"] for c in out] == [1, 2]
    assert out[0]["sources"] == {"vector"}
    assert out[0]["pmid"] == "111"


def test_vector_search_wires_callback():
    seen: dict = {}

    def run(cypher, **params):
        seen.update(params)
        assert "db.index.vector.queryNodes" in cypher
        return []

    out = vector_search(run, np.zeros(384, dtype=np.float32),
                        "chunk_embedding_idx", top_k=7, min_score=0.1)
    assert out == []
    assert seen["top_k"] == 7 and seen["min_score"] == 0.1
    assert len(seen["embedding"]) == 384


def test_is_informative_filters_noise():
    assert _is_informative("alzheimer disease")
    assert _is_informative("aspirin")
    assert not _is_informative("0 035")      # from '±0.035' style spans
    assert not _is_informative("ab")          # too short
    assert not _is_informative("12")


def test_query_entities_container():
    qe = QueryEntities(items=[{"normalized_key": "DISEASE:x"}])
    assert qe.keys == ["DISEASE:x"]
    assert len(qe) == 1
    assert len(QueryEntities(items=[])) == 0


def test_embed_query_shape():
    vec = embed_query("What are risk factors for Alzheimer's disease?")
    assert vec.shape == (384,)
    assert vec.dtype == np.float32
