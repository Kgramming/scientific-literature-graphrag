"""Typed retrieval result model (serializable for later RAG/UI phases)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RetrievalResult:
    chunk_id: str
    pmid: str
    title: str
    text: str
    strategy: str = "semantic_cluster"
    vector_score: float | None = None
    vector_rank: int | None = None
    graph_score: float | None = None   # entity-match score in [0, 1]
    graph_rank: int | None = None
    semantic_score: float | None = None  # max SEMANTIC_SIMILAR edge score
    semantic_rank: int | None = None
    fused_score: float = 0.0
    rank: int = 0                      # fused rank (1-based)
    retrieval_sources: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    # Phase 5A reranking (None when the reranker is disabled or not applied)
    reranker_score: float | None = None
    reranker_rank: int | None = None
    final_score: float | None = None   # = reranker_score when reranked
    final_rank: int | None = None      # final 1-based rank after reranking

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RetrievalResult:
        return cls(**data)


@dataclass
class QueryEntities:
    """Entities extracted from a user query, with normalized keys."""

    items: list[dict]  # each: {surface_form, entity_type, normalized_key}

    @property
    def keys(self) -> list[str]:
        return [i["normalized_key"] for i in self.items]

    def __len__(self) -> int:
        return len(self.items)
