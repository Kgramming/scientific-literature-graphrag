"""Bounded evidence packing: ranked RetrievalResults -> EvidenceItems.

Only a configurable top-k (default 8) is ever sent to the LLM. The full
53k-chunk corpus is never packed.
"""

from __future__ import annotations

from sgrag.generation.models import EvidenceItem


def pack_evidence(results, top_k: int = 8) -> list[EvidenceItem]:
    """Pack the top-k ranked results into evidence items.

    Expects results ordered best-first (final reranked order). Preserves
    chunk_id, PMID, title, text, reranker/fused scores, sources and
    matched entities for grounding and citation validation.
    """
    items: list[EvidenceItem] = []
    for r in results[: max(int(top_k), 0)]:
        items.append(
            EvidenceItem(
                chunk_id=r.chunk_id,
                pmid=str(r.pmid),
                title=r.title or "",
                text=r.text or "",
                reranker_score=r.reranker_score,
                fused_score=float(r.fused_score or 0.0),
                retrieval_sources=list(r.retrieval_sources or []),
                matched_entities=list(r.matched_entities or []),
            )
        )
    return items
