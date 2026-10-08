"""Reranker implementations (Phase 5A).

Phase 4 defined the Reranker protocol with a NoOpReranker default. Phase 5A
adds CrossEncoderReranker (cross-encoder/ms-marco-MiniLM-L-6-v2), which scores
(query, chunk_text) pairs and reorders the bounded Phase 4 candidate pool.

The 53k corpus is NEVER reranked — only the top candidate_k fused candidates.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Protocol

from sgrag.retrieval.models import RetrievalResult

log = logging.getLogger(__name__)


class Reranker(Protocol):
    """Rerank fused retrieval results for a query."""

    name: str

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        ...


class RerankerError(RuntimeError):
    """Raised when the reranker cannot load or score (never silently faked)."""


class NoOpReranker:
    """Default pass-through: keeps fused ranking unchanged."""

    name = "none"

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return results


def resolve_model_snapshot(model_name: str) -> str:
    """Prefer a local HF-hub snapshot; fall back to the hub id (download)."""
    from pathlib import Path

    cache = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_name.replace("/", "--")
    snap_dir = cache / slug / "snapshots"
    if snap_dir.is_dir():
        snaps = sorted(p for p in snap_dir.iterdir() if p.is_dir())
        if snaps:
            return str(snaps[0])
    return model_name


@lru_cache(maxsize=4)
def _load_cross_encoder(model_path: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_path, max_length=512)


class CrossEncoderReranker:
    """Cross-encoder reranker over the bounded candidate pool.

    Scores (query, chunk_text) pairs; deterministic ordering by
    (-reranker_score, chunk_id). Preserves all Phase 4 component scores and
    provenance; adds reranker_score/reranker_rank/final_score/final_rank.
    """

    name = "cross-encoder"

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_k: int = 50,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model
        self.candidate_k = int(candidate_k)
        self.batch_size = int(batch_size)

    def _get_model(self):
        # Offline-first on this VM: the egress proxy breaks hub HTTP, so a
        # missing local snapshot is a hard error, not a silent hang.
        path = resolve_model_snapshot(self.model_name)
        if path == self.model_name and os.environ.get("HF_HUB_OFFLINE") == "1":
            raise RerankerError(
                f"cross-encoder model '{self.model_name}' not cached locally and "
                "HF_HUB_OFFLINE=1 (download it first)"
            )
        try:
            return _load_cross_encoder(path)
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(
                f"failed to load cross-encoder '{self.model_name}': {exc}"
            ) from exc

    def rerank(
        self, query: str, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        if not results:
            return []
        model = self._get_model()
        pool = results[: self.candidate_k]
        rest = results[self.candidate_k :]
        pairs = [(query, r.text) for r in pool]
        try:
            scores = model.predict(pairs, batch_size=self.batch_size,
                                   show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(f"cross-encoder inference failed: {exc}") from exc
        for r, s in zip(pool, scores):
            r.reranker_score = float(s)
        pool.sort(key=lambda r: (-r.reranker_score, r.chunk_id))
        for i, r in enumerate(pool, start=1):
            r.reranker_rank = i
            r.final_score = r.reranker_score
            r.final_rank = i
        # unreranked tail keeps fused order after the reranked head
        for j, r in enumerate(rest, start=len(pool) + 1):
            r.final_score = r.fused_score
            r.final_rank = j
        return pool + rest
