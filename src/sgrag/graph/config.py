"""Configuration for the Neo4j knowledge-graph layer.

Secrets (Neo4j password) come from environment variables only and are never
written to config files or logs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Neo4jConnection:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""  # filled from SGRAG_NEO4J_PASSWORD at load time
    database: str = "neo4j"


@dataclass(frozen=True)
class IngestionConfig:
    chunk_strategy: str = "semantic_cluster"
    batch_size: int = 1000


@dataclass(frozen=True)
class EntityExtractionConfig:
    model: str = "en_ner_bc5cdr_md"
    batch_size: int = 256
    # entity types emitted by the extractor; anything else is dropped
    allowed_types: tuple[str, ...] = ("DISEASE", "CHEMICAL")


@dataclass(frozen=True)
class SimilarityConfig:
    top_k: int = 5
    metric: str = "cosine"
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class VectorIndexConfig:
    name: str = "chunk_embedding_idx"
    dimensions: int = 384
    similarity_function: str = "cosine"  # Neo4j vector index option
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class GraphConfig:
    neo4j: Neo4jConnection = field(default_factory=Neo4jConnection)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    entities: EntityExtractionConfig = field(default_factory=EntityExtractionConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)


def _from_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def load_graph_config(path: str | Path) -> GraphConfig:
    """Load graph config from YAML; secrets are resolved from the environment."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    neo4j_raw = raw.get("neo4j", {})
    ingestion_raw = raw.get("ingestion", {})
    entities_raw = raw.get("entities", {})
    similarity_raw = raw.get("similarity", {})
    vector_raw = raw.get("vector_index", {})

    password = os.environ.get("SGRAG_NEO4J_PASSWORD", "")
    cfg = GraphConfig(
        neo4j=Neo4jConnection(
            uri=_from_env("SGRAG_NEO4J_URI", neo4j_raw.get("uri", "bolt://localhost:7687")),
            user=_from_env("SGRAG_NEO4J_USER", neo4j_raw.get("user", "neo4j")),
            password=password,
            database=neo4j_raw.get("database", "neo4j"),
        ),
        ingestion=IngestionConfig(
            chunk_strategy=ingestion_raw.get("chunk_strategy", "semantic_cluster"),
            batch_size=int(ingestion_raw.get("batch_size", 1000)),
        ),
        entities=EntityExtractionConfig(
            model=entities_raw.get("model", "en_ner_bc5cdr_md"),
            batch_size=int(entities_raw.get("batch_size", 256)),
            allowed_types=tuple(entities_raw.get("allowed_types", ["DISEASE", "CHEMICAL"])),
        ),
        similarity=SimilarityConfig(
            top_k=int(similarity_raw.get("top_k", 5)),
            metric=similarity_raw.get("metric", "cosine"),
            embedding_model=similarity_raw.get("embedding_model", "all-MiniLM-L6-v2"),
        ),
        vector_index=VectorIndexConfig(
            name=vector_raw.get("name", "chunk_embedding_idx"),
            dimensions=int(vector_raw.get("dimensions", 384)),
            similarity_function=vector_raw.get("similarity_function", "cosine"),
            embedding_model=vector_raw.get("embedding_model", "all-MiniLM-L6-v2"),
        ),
    )
    validate_graph_config(cfg)
    return cfg


def validate_graph_config(cfg: GraphConfig) -> None:
    """Raise ValueError on any invalid graph configuration."""
    if not cfg.neo4j.uri:
        raise ValueError("neo4j.uri must be non-empty")
    if not cfg.neo4j.user:
        raise ValueError("neo4j.user must be non-empty")
    if cfg.ingestion.chunk_strategy not in {"semantic_cluster", "sentence_boundary", "fixed_token"}:
        raise ValueError(f"unknown chunk_strategy: {cfg.ingestion.chunk_strategy}")
    if cfg.ingestion.batch_size < 1:
        raise ValueError("ingestion.batch_size must be >= 1")
    if cfg.entities.batch_size < 1:
        raise ValueError("entities.batch_size must be >= 1")
    if not cfg.entities.allowed_types:
        raise ValueError("entities.allowed_types must be non-empty")
    if cfg.similarity.top_k < 1:
        raise ValueError("similarity.top_k must be >= 1")
    if cfg.similarity.metric not in {"cosine", "euclidean"}:
        raise ValueError(f"unsupported similarity.metric: {cfg.similarity.metric}")
    if cfg.vector_index.dimensions != 384 and cfg.vector_index.embedding_model == "all-MiniLM-L6-v2":
        raise ValueError("all-MiniLM-L6-v2 must use dimensions=384")
    if cfg.vector_index.similarity_function not in {"cosine", "euclidean"}:
        raise ValueError(f"unsupported vector similarity_function: {cfg.vector_index.similarity_function}")
