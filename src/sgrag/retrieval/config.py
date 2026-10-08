"""Retrieval configuration (Phase 4).

All expansion is bounded: hub_degree_cap <= 1000 is enforced, and every
limit below keeps a single query from exploding on the 53,091-chunk graph.
Secrets (Neo4j password) come from SGRAG_NEO4J_PASSWORD only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False          # backward-compatible default: Phase 4 only
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_k: int = 50          # bounded pool reranked (never the 53k corpus)
    batch_size: int = 32
    fallback_to_fusion: bool = True  # on reranker failure keep fused ranking


@dataclass(frozen=True)
class FusionConfig:
    method: str = "weighted"  # "weighted" | "rrf"
    weight_vector: float = 0.6
    weight_entity: float = 0.25
    weight_semantic: float = 0.15
    rrf_k: int = 60


@dataclass(frozen=True)
class RetrievalConfig:
    # candidate generation
    vector_top_k: int = 20
    entity_top_k: int = 15          # max chunks per query entity (MENTIONS)
    semantic_similar_top_k: int = 3  # SEMANTIC_SIMILAR edges per vector hit
    graph_hop_limit: int = 1        # expansion hops from vector/entity hits
    hub_degree_cap: int = 1000      # entities above this degree are skipped
    max_candidates: int = 100       # hard cap after fusion/dedup
    min_vector_score: float = 0.0   # cosine floor for vector candidates
    # models
    embedding_model: str = "all-MiniLM-L6-v2"
    entity_model: str = "en_ner_bc5cdr_md"
    vector_index_name: str = "chunk_embedding_idx"
    # neo4j (password from env only)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    fusion: FusionConfig = field(default_factory=FusionConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)


def load_retrieval_config(path: str | Path) -> RetrievalConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    fusion_raw = raw.get("fusion", {})
    rerank_raw = raw.get("reranker", {})
    neo4j_raw = raw.get("neo4j", {})
    cfg = RetrievalConfig(
        vector_top_k=int(raw.get("vector_top_k", 20)),
        entity_top_k=int(raw.get("entity_top_k", 15)),
        semantic_similar_top_k=int(raw.get("semantic_similar_top_k", 3)),
        graph_hop_limit=int(raw.get("graph_hop_limit", 1)),
        hub_degree_cap=int(raw.get("hub_degree_cap", 1000)),
        max_candidates=int(raw.get("max_candidates", 100)),
        min_vector_score=float(raw.get("min_vector_score", 0.0)),
        embedding_model=raw.get("embedding_model", "all-MiniLM-L6-v2"),
        entity_model=raw.get("entity_model", "en_ner_bc5cdr_md"),
        vector_index_name=raw.get("vector_index_name", "chunk_embedding_idx"),
        neo4j_uri=os.environ.get("SGRAG_NEO4J_URI", neo4j_raw.get("uri", raw.get("neo4j_uri", "bolt://localhost:7687"))),
        neo4j_user=os.environ.get("SGRAG_NEO4J_USER", neo4j_raw.get("user", raw.get("neo4j_user", "neo4j"))),
        neo4j_password=os.environ.get("SGRAG_NEO4J_PASSWORD", ""),
        neo4j_database=neo4j_raw.get("database", raw.get("neo4j_database", "neo4j")),
        fusion=FusionConfig(
            method=fusion_raw.get("method", "weighted"),
            weight_vector=float(fusion_raw.get("weight_vector", 0.6)),
            weight_entity=float(fusion_raw.get("weight_entity", 0.25)),
            weight_semantic=float(fusion_raw.get("weight_semantic", 0.15)),
            rrf_k=int(fusion_raw.get("rrf_k", 60)),
        ),
        reranker=RerankerConfig(
            enabled=bool(rerank_raw.get("enabled", False)),
            model=rerank_raw.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            candidate_k=int(rerank_raw.get("candidate_k", 50)),
            batch_size=int(rerank_raw.get("batch_size", 32)),
            fallback_to_fusion=bool(rerank_raw.get("fallback_to_fusion", True)),
        ),
    )
    validate_retrieval_config(cfg)
    return cfg


def validate_retrieval_config(cfg: RetrievalConfig) -> None:
    if cfg.vector_top_k < 1:
        raise ValueError("vector_top_k must be >= 1")
    if cfg.entity_top_k < 1:
        raise ValueError("entity_top_k must be >= 1")
    if cfg.semantic_similar_top_k < 0:
        raise ValueError("semantic_similar_top_k must be >= 0")
    if cfg.graph_hop_limit not in (0, 1):
        raise ValueError("graph_hop_limit must be 0 or 1 in Phase 4")
    if not 1 <= cfg.hub_degree_cap <= 1000:
        raise ValueError("hub_degree_cap must be within [1, 1000]")
    if cfg.max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if cfg.fusion.method not in ("weighted", "rrf"):
        raise ValueError(f"unknown fusion.method: {cfg.fusion.method}")
    w = cfg.fusion.weight_vector + cfg.fusion.weight_entity + cfg.fusion.weight_semantic
    if cfg.fusion.method == "weighted" and abs(w - 1.0) > 1e-6:
        raise ValueError(f"fusion weights must sum to 1.0, got {w}")
    if cfg.fusion.rrf_k < 1:
        raise ValueError("fusion.rrf_k must be >= 1")
    if cfg.reranker.candidate_k < 1:
        raise ValueError("reranker.candidate_k must be >= 1")
    if cfg.reranker.batch_size < 1:
        raise ValueError("reranker.batch_size must be >= 1")
