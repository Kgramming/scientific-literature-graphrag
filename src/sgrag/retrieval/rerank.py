"""Optional reranker interface (Phase 5 will plug in cross-encoder/ms-marco-MiniLM-L-6-v2).

Phase 4 retrieval is complete and independently testable without any
reranker. The HybridRetriever accepts an optional Reranker; the default is
a pass-through. Phase 5 implements CrossEncoderReranker against this
interface without rewriting retrieval.
"""

from __future__ import annotations

from typing import Protocol

from sgrag.retrieval.models import RetrievalResult


class Reranker(Protocol):
    """Rerank fused retrieval results for a query."""

    name: str

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        ...


class NoOpReranker:
    """Default pass-through: keeps fused ranking unchanged."""

    name = "none"

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return results
