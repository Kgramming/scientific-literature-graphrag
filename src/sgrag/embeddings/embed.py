"""Embedding generation with deterministic disk caching.

encode() returns the identical matrix for identical (config, texts), loading
from data/processed/embeddings_cache/ when the cache entry's provenance
matches. Every cache entry carries a meta sidecar recording model, dimension,
normalization, device, runtime, text count/fingerprint, and timestamp, so a
later run can be audited without recomputation.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=4)
def load_model(model_name, device="cpu"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def _meta_path(cache_dir, slug, key):
    return Path(cache_dir) / f"{slug}.{key}.meta.json"


def _emb_path(cache_dir, slug, key):
    return Path(cache_dir) / f"{slug}.{key}.npy"


def encode(texts, config, use_cache=True, _encoder=None):
    """Encode texts -> (embeddings, provenance).

    _encoder is an injectable (model.encode-like) for tests; production
    passes None and uses the cached SentenceTransformer.
    """
    texts = list(texts)
    key = config.cache_key(texts)
    slug = config.cache_slug()
    cache_dir = Path(config.cache_dir)
    meta_p, emb_p = _meta_path(cache_dir, slug, key), _emb_path(cache_dir, slug, key)

    if use_cache and meta_p.exists() and emb_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if (meta.get("model_name") == config.model_name
                and meta.get("normalize") == config.normalize
                and meta.get("n_texts") == len(texts)):
            return np.load(emb_p), {**meta, "cache_hit": True}

    model = _encoder if _encoder is not None else load_model(
        config.model_name, config.device)
    t0 = time.perf_counter()
    embs = np.asarray(model.encode(texts, batch_size=config.batch_size,
                                   show_progress_bar=False,
                                   convert_to_numpy=True), dtype=np.float64)
    encode_s = time.perf_counter() - t0
    if config.normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms

    provenance = {
        "model_name": config.model_name,
        "embedding_dim": int(embs.shape[1]) if embs.ndim == 2 else None,
        "normalize": config.normalize,
        "device": config.device,
        "batch_size": config.batch_size,
        "n_texts": len(texts),
        "texts_sha256": hashlib.sha256(
            "\x00".join(texts).encode()).hexdigest(),
        "encode_seconds": encode_s,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cache_hit": False,
    }
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_p, embs)
        meta_p.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        provenance["cache_path"] = str(emb_p)
    return embs, provenance
