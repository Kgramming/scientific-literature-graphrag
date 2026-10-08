"""Query-side processing: embedding + entity extraction.

Reuses the Phase 3B entity extraction/normalization logic verbatim so query
entities and graph entities share the same key space.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from sgrag.graph.entities import normalize_entity, normalize_entity_name
from sgrag.retrieval.models import QueryEntities

_MIN_NORM_LEN = 3


def resolve_embedding_model(model_name: str) -> str:
    """Prefer the local HF-hub snapshot (egress proxy breaks hub HTTP)."""
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_name.replace("/", "--")
    snap_dir = cache / slug / "snapshots"
    if snap_dir.is_dir():
        snaps = sorted(p for p in snap_dir.iterdir() if p.is_dir())
        if snaps:
            return str(snaps[0])
    return model_name


@lru_cache(maxsize=2)
def _load_text_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(resolve_embedding_model(model_name))


def embed_query(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a user query -> 384-dim float32 vector.

    The Phase 3B vector index uses cosine similarity, which is
    scale-invariant, so the raw (unnormalized) embedding is correct here.
    """
    model = _load_text_encoder(model_name)
    vec = model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
    return np.asarray(vec, dtype=np.float32)


def _is_informative(normalized_name: str) -> bool:
    """Drop noisy spans (e.g. '0 035' from '±0.035'): not every CHEMICAL span
    is equally important. Keeps the entity->chunk expansion precise."""
    return len(normalized_name) >= _MIN_NORM_LEN and bool(re.search(r"[a-z]", normalized_name))


@lru_cache(maxsize=2)
def _load_entity_extractor(model_name: str):
    from sgrag.graph.entities import BiomedicalEntityExtractor

    # SGRAG_SCISPACY_MODEL may point at a manually extracted model directory
    # (the legacy bc5cdr wheel does not build on Python 3.12).
    path = os.environ.get("SGRAG_SCISPACY_MODEL", model_name)
    return BiomedicalEntityExtractor(model=path)


def extract_query_entities(
    query: str,
    model_name: str = "en_ner_bc5cdr_md",
    allowed_types: tuple[str, ...] = ("DISEASE", "CHEMICAL"),
) -> QueryEntities:
    """Extract + normalize entities from a user query.

    Returns an empty QueryEntities when nothing informative is detected;
    callers fall back to vector-only retrieval in that case.
    """
    extractor = _load_entity_extractor(model_name)
    # temporarily narrow allowed types without rebuilding the extractor
    allowed = tuple(t.upper() for t in allowed_types)
    items: list[dict] = []
    seen: set[str] = set()
    for per_text in extractor.extract_batch([query], batch_size=8):
        for ent in per_text:
            if ent.entity_type not in allowed:
                continue
            norm = normalize_entity(ent.entity_type, ent.surface_form)
            if not _is_informative(normalize_entity_name(ent.surface_form)):
                continue
            if norm["normalized_key"] in seen:
                continue
            seen.add(norm["normalized_key"])
            items.append(
                {
                    "surface_form": ent.surface_form,
                    "entity_type": norm["entity_type"],
                    "normalized_key": norm["normalized_key"],
                }
            )
    return QueryEntities(items=items)
