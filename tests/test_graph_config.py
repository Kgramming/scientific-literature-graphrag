"""Unit tests for graph config loading and validation (no live Neo4j)."""

import os

import pytest
import yaml

from sgrag.graph.config import GraphConfig, load_graph_config, validate_graph_config


def _write(tmp_path, data):
    p = tmp_path / "graph_build.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_load_defaults(tmp_path):
    cfg = load_graph_config(_write(tmp_path, {}))
    assert cfg.ingestion.chunk_strategy == "semantic_cluster"
    assert cfg.similarity.top_k == 5
    assert cfg.vector_index.dimensions == 384
    assert cfg.neo4j.password == os.environ.get("SGRAG_NEO4J_PASSWORD", "")


def test_env_overrides_uri_user(tmp_path, monkeypatch):
    monkeypatch.setenv("SGRAG_NEO4J_URI", "bolt://db:7687")
    monkeypatch.setenv("SGRAG_NEO4J_USER", "sgrag")
    cfg = load_graph_config(_write(tmp_path, {}))
    assert cfg.neo4j.uri == "bolt://db:7687"
    assert cfg.neo4j.user == "sgrag"


def test_password_never_in_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("SGRAG_NEO4J_PASSWORD", "s3cret")
    raw = yaml.safe_load((tmp_path / "graph_build.yaml").read_text()) if False else {}
    cfg = load_graph_config(_write(tmp_path, {"neo4j": {"password": "ignored"}}))
    assert cfg.neo4j.password == "s3cret"  # env wins; yaml password ignored


def test_invalid_chunk_strategy(tmp_path):
    with pytest.raises(ValueError, match="chunk_strategy"):
        load_graph_config(_write(tmp_path, {"ingestion": {"chunk_strategy": "nope"}}))


def test_invalid_top_k(tmp_path):
    with pytest.raises(ValueError, match="top_k"):
        load_graph_config(_write(tmp_path, {"similarity": {"top_k": 0}}))


def test_minilm_dimension_guard(tmp_path):
    with pytest.raises(ValueError, match="dimensions"):
        load_graph_config(_write(
            tmp_path,
            {"vector_index": {"embedding_model": "all-MiniLM-L6-v2", "dimensions": 768}},
        ))


def test_validate_rejects_bad_metric():
    cfg = GraphConfig()
    bad = GraphConfig(
        similarity=type(cfg.similarity)(top_k=5, metric="manhattan",
                                        embedding_model=cfg.similarity.embedding_model)
    )
    with pytest.raises(ValueError, match="metric"):
        validate_graph_config(bad)
