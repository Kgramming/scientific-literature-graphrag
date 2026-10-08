"""Candidate fusion: weighted scores (default) or Reciprocal Rank Fusion.

Every candidate keeps its component scores, per-strategy ranks, and sources,
so the final ranking is explainable: why a chunk was retrieved is visible in
the RetrievalResult.
"""

from __future__ import annotations

from sgrag.retrieval.config import FusionConfig
from sgrag.retrieval.models import RetrievalResult


def min_max_normalize(values: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]; constant input maps to 1.0."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def merge_candidates(
    vector_cands: list[dict],
    entity_cands: list[dict],
    semantic_cands: list[dict],
) -> dict[str, dict]:
    """Deduplicate by chunk_id, merging sources/scores/mathed entities."""
    merged: dict[str, dict] = {}

    def _add(cand: dict) -> None:
        cid = cand["chunk_id"]
        cur = merged.get(cid)
        if cur is None:
            merged[cid] = {
                "chunk_id": cid,
                "pmid": cand["pmid"],
                "strategy": cand.get("strategy") or "semantic_cluster",
                "text": cand.get("text") or "",
                "token_count": cand.get("token_count"),
                "vector_score": cand.get("vector_score"),
                "vector_rank": cand.get("vector_rank"),
                "graph_score": cand.get("graph_score"),
                "semantic_score": cand.get("semantic_score"),
                "sources": set(cand.get("sources") or set()),
                "matched_entities": list(cand.get("matched_entities") or []),
            }
            return
        for key in ("vector_score", "vector_rank", "graph_score", "semantic_score"):
            if cur.get(key) is None and cand.get(key) is not None:
                cur[key] = cand[key]
        # entity matches from several query entities: keep the max score
        if cand.get("graph_score") is not None:
            cur["graph_score"] = max(cur["graph_score"] or 0.0, cand["graph_score"])
        cur["sources"] |= set(cand.get("sources") or set())
        for ent in cand.get("matched_entities") or []:
            if ent not in cur["matched_entities"]:
                cur["matched_entities"].append(ent)

    for c in vector_cands:
        _add(c)
    for c in entity_cands:
        _add(c)
    for c in semantic_cands:
        _add(c)
    return merged


def _assign_ranks(merged: dict[str, dict], score_key: str, rank_key: str) -> None:
    order = sorted(
        (c for c in merged.values() if c.get(score_key) is not None),
        key=lambda c: (-c[score_key], c["chunk_id"]),
    )
    for rank, cand in enumerate(order, start=1):
        cand[rank_key] = rank


def fuse_weighted(merged: dict[str, dict], fusion: FusionConfig) -> list[RetrievalResult]:
    """Weighted fusion of min-max-normalized component scores."""
    cands = list(merged.values())
    _assign_ranks(merged, "vector_score", "vector_rank")
    _assign_ranks(merged, "graph_score", "graph_rank")
    _assign_ranks(merged, "semantic_score", "semantic_rank")

    vec_norm = min_max_normalize([c["vector_score"] or 0.0 for c in cands])
    ent_norm = min_max_normalize([c["graph_score"] or 0.0 for c in cands])
    sem_norm = min_max_normalize([c["semantic_score"] or 0.0 for c in cands])

    results: list[RetrievalResult] = []
    for cand, vn, en, sn in zip(cands, vec_norm, ent_norm, sem_norm):
        fused = (
            fusion.weight_vector * vn
            + fusion.weight_entity * en
            + fusion.weight_semantic * sn
        )
        results.append(_to_result(cand, fused))
    results.sort(key=lambda r: (-r.fused_score, r.chunk_id))
    for rank, r in enumerate(results, start=1):
        r.rank = rank
    return results


def fuse_rrf(merged: dict[str, dict], fusion: FusionConfig) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion over the per-strategy rankings."""
    _assign_ranks(merged, "vector_score", "vector_rank")
    _assign_ranks(merged, "graph_score", "graph_rank")
    _assign_ranks(merged, "semantic_score", "semantic_rank")
    k = fusion.rrf_k
    results: list[RetrievalResult] = []
    for cand in merged.values():
        score = 0.0
        for rank_key in ("vector_rank", "graph_rank", "semantic_rank"):
            rank = cand.get(rank_key)
            if rank:
                score += 1.0 / (k + rank)
        results.append(_to_result(cand, score))
    results.sort(key=lambda r: (-r.fused_score, r.chunk_id))
    for rank, r in enumerate(results, start=1):
        r.rank = rank
    return results


def _to_result(cand: dict, fused: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=cand["chunk_id"],
        pmid=cand["pmid"],
        title="",  # filled by the retriever from the Article node
        text=cand.get("text") or "",
        strategy=cand.get("strategy") or "semantic_cluster",
        vector_score=cand.get("vector_score"),
        vector_rank=cand.get("vector_rank"),
        graph_score=cand.get("graph_score"),
        graph_rank=cand.get("graph_rank"),
        semantic_score=cand.get("semantic_score"),
        semantic_rank=cand.get("semantic_rank"),
        fused_score=float(fused),
        retrieval_sources=sorted(cand.get("sources") or set()),
        matched_entities=list(cand.get("matched_entities") or []),
    )


def fuse_candidates(
    vector_cands: list[dict],
    entity_cands: list[dict],
    semantic_cands: list[dict],
    fusion: FusionConfig,
    max_candidates: int,
) -> list[RetrievalResult]:
    """Merge, fuse, rank, and cap candidates (deterministic)."""
    merged = merge_candidates(vector_cands, entity_cands, semantic_cands)
    if not merged:
        return []
    if fusion.method == "rrf":
        results = fuse_rrf(merged, fusion)
    else:
        results = fuse_weighted(merged, fusion)
    return results[:max_candidates]
