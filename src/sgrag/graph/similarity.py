"""Bounded semantic-similarity edge construction.

Computes top-k nearest neighbors per chunk from L2-normalized embeddings with
cosine similarity. The graph stays bounded: exactly ``top_k`` outgoing
SEMANTIC_SIMILAR edges per chunk (self matches excluded), never an O(n^2)
all-pairs graph.

Pure NumPy/sklearn implementation: no Neo4j connection required, so the edge
construction logic is fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityEdge:
    src_chunk_id: str
    dst_chunk_id: str
    score: float  # cosine similarity in [-1, 1]
    rank: int     # 1-based rank among the source chunk's neighbors
    model: str


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def top_k_neighbors(
    chunk_ids: list[str],
    embeddings: np.ndarray,
    top_k: int = 5,
    model: str = "all-MiniLM-L6-v2",
) -> list[SimilarityEdge]:
    """Return top-k cosine-similar neighbors per chunk (excluding self).

    Uses a blocked brute-force cosine computation on L2-normalized embeddings.
    Deterministic: ties broken by chunk_id ordering for reproducibility.
    """
    if len(chunk_ids) != embeddings.shape[0]:
        raise ValueError("chunk_ids and embeddings row count must match")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    n = len(chunk_ids)
    if n == 0:
        return []
    k = min(top_k, n - 1)
    if k < 1:
        return []

    order = np.argsort(chunk_ids, kind="stable")  # deterministic tie-break basis
    emb = l2_normalize(np.asarray(embeddings, dtype=np.float64))

    edges: list[SimilarityEdge] = []
    block = 1024  # bound peak RAM: one block's sims matrix is ~1024 x n float64
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = emb[start:stop] @ emb.T  # cosine on normalized vectors
        for i in range(start, stop):
            sims[i - start, i] = -np.inf  # exclude self
            # argpartition for top-k, then full sort of the k for rank order
            cand = np.argpartition(sims[i - start], -k)[-k:]
            # deterministic: sort by (-score, chunk_id)
            cand = sorted(cand, key=lambda j: (-sims[i - start, j], chunk_ids[j]))
            for rank, j in enumerate(cand, start=1):
                edges.append(
                    SimilarityEdge(
                        src_chunk_id=chunk_ids[i],
                        dst_chunk_id=chunk_ids[j],
                        score=float(sims[i - start, j]),
                        rank=rank,
                        model=model,
                    )
                )
    return edges


def similarity_score_distribution(edges: list[SimilarityEdge], bins: int = 10) -> dict:
    """Histogram + summary stats of edge scores for reporting."""
    scores = np.array([e.score for e in edges], dtype=np.float64)
    if scores.size == 0:
        return {"count": 0}
    hist, bin_edges = np.histogram(scores, bins=bins)
    return {
        "count": int(scores.size),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "histogram": [int(h) for h in hist],
        "bin_edges": [float(b) for b in bin_edges],
    }
