"""Unit tests: bounded graph expansion (hub cap, dedup, provenance)."""

import math

from sgrag.retrieval.graph import (
    expand_entities,
    expand_semantic,
    entity_graph_score,
)


def test_entity_graph_score_downweights_hubs():
    assert entity_graph_score(0) == 1.0
    rare = entity_graph_score(1)
    hub = entity_graph_score(635)  # 'pain' in the Phase 3B graph
    assert 0.0 < hub < rare < 1.0
    assert math.isclose(hub, 1.0 / (1.0 + math.log1p(635)), rel_tol=1e-9)


def _fake_expand(calls):
    def run(cypher, **params):
        calls.append(params)
        # hub entities return no rows (WHERE degree <= hub_cap filters them)
        if params.get("key") == "DISEASE:hub":
            return []
        return [
            {"chunk_id": "c1", "pmid": "111", "strategy": "semantic_cluster",
             "text": "t1", "token_count": 10, "surface_form": "Alzheimer",
             "entity_degree": 5, "entity_key": params["key"]},
            {"chunk_id": "c2", "pmid": "111", "strategy": "semantic_cluster",
             "text": "t2", "token_count": 12, "surface_form": "Alzheimer",
             "entity_degree": 5, "entity_key": params["key"]},
        ]
    return run


def test_expand_entities_bounded_and_provenance():
    calls: list = []
    out = expand_entities(_fake_expand(calls), ["DISEASE:alzheimer disease", "DISEASE:hub"],
                          entity_top_k=15, hub_degree_cap=1000)
    assert len(out) == 2  # hub entity contributes nothing
    assert calls[0]["limit"] == 15 and calls[0]["hub_cap"] == 1000
    assert out[0]["sources"] == {"entity"}
    assert out[0]["matched_entities"] == ["DISEASE:alzheimer disease"]
    assert out[0]["graph_score"] == entity_graph_score(5)


def test_expand_entities_empty_keys():
    assert expand_entities(lambda *a, **k: [], [], 15, 1000) == []


def test_expand_semantic_empty_sources():
    assert expand_semantic(lambda *a, **k: [], []) == []


def test_expand_semantic_passes_per_chunk_and_provenance():
    calls: list = []

    def run(cypher, **params):
        calls.append(params)
        return [
            {"chunk_id": "n1", "pmid": "222", "strategy": "semantic_cluster",
             "text": "s", "token_count": 8, "semantic_score": 0.77,
             "src_chunk_ids": ["v1", "v2"]},
        ]

    out = expand_semantic(run, ["v1", "v2"], per_chunk=3)
    assert calls[0]["per_chunk"] == 3
    assert calls[0]["chunk_ids"] == ["v1", "v2"]
    assert out[0]["semantic_score"] == 0.77
    assert out[0]["sources"] == {"semantic_graph"}
    assert out[0]["src_chunk_ids"] == ["v1", "v2"]
