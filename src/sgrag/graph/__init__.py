"""Knowledge-graph layer: schema, entity extraction/normalization, ingestion, similarity edges."""

from sgrag.graph.config import GraphConfig, load_graph_config
from sgrag.graph.entities import extract_entities, normalize_entity
from sgrag.graph.schema import (
    CONSTRAINT_STATEMENTS,
    INDEX_STATEMENTS,
    VECTOR_INDEX_STATEMENT,
    build_article_params,
    build_chunk_params,
    build_mentions_params,
    build_similarity_params,
)

__all__ = [
    "CONSTRAINT_STATEMENTS",
    "INDEX_STATEMENTS",
    "VECTOR_INDEX_STATEMENT",
    "GraphConfig",
    "build_article_params",
    "build_chunk_params",
    "build_mentions_params",
    "build_similarity_params",
    "extract_entities",
    "load_graph_config",
    "normalize_entity",
]
