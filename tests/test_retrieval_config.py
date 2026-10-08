"""Unit tests: retrieval configuration."""

import pytest

from sgrag.retrieval.config import (
    FusionConfig,
    RetrievalConfig,
    load_retrieval_config,
    validate_retrieval_config,
)


def test_defaults_are_bounded():
    cfg = RetrievalConfig()
    assert cfg.vector_top_k == 20
    assert cfg.hub_degree_cap <= 1000
    assert cfg.graph_hop_limit <= 1


def test_yaml_round_trip(tmp_path):
    p = tmp_path / "retrieval.yaml"
    p.write_text(
        "vector_top_k: 10\nentity_top_k: 5\nhub_degree_cap: 500\n"
        "fusion:\n  method: rrf\n  rrf_k: 30\n"
        "neo4j:\n  uri: bolt://localhost:7687\n"
    )
    cfg = load_retrieval_config(p)
    assert cfg.vector_top_k == 10
    assert cfg.entity_top_k == 5
    assert cfg.hub_degree_cap == 500
    assert cfg.fusion.method == "rrf"
    assert cfg.fusion.rrf_k == 30
    assert cfg.neo4j_uri == "bolt://localhost:7687"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__class__(**{**c.__dict__, "hub_degree_cap": 1001}),
        lambda c: c.__class__(**{**c.__dict__, "vector_top_k": 0}),
        lambda c: c.__class__(**{**c.__dict__, "graph_hop_limit": 2}),
        lambda c: c.__class__(
            **{**c.__dict__, "fusion": FusionConfig(weight_vector=0.5,
                                                   weight_entity=0.5,
                                                   weight_semantic=0.5)}
        ),
        lambda c: c.__class__(
            **{**c.__dict__, "fusion": FusionConfig(method="nope")}
        ),
    ],
)
def test_validation_rejects_bad_config(mutate):
    with pytest.raises(ValueError):
        validate_retrieval_config(mutate(RetrievalConfig()))
