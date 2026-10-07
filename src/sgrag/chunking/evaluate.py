"""Shared chunking-evaluation metrics.

Single implementation used by both the three-strategy pilot evaluation and
the controlled semantic-configuration comparison, so the numbers are
computed identically everywhere.
"""

import numpy as np

from .base import Chunk
from .coherence import coherence_for_article


def describe(vals):
    vals = np.asarray(vals, dtype=float)
    return {"n": int(len(vals)), "mean": float(np.mean(vals)),
            "median": float(np.median(vals)), "min": float(np.min(vals)),
            "max": float(np.max(vals)), "std": float(np.std(vals))}


def strategy_metrics(chunks, records, art_sentences, art_embs):
    """Compute the full metric set for one strategy's chunk list.

    chunks: list of chunk dicts (JSONL-decoded). records: {pmid: record}.
    art_sentences/art_embs: {pmid: ...} shared across strategies.
    """
    by_art = {}
    for ch in chunks:
        by_art.setdefault(ch["pmid"], []).append(ch)
    for v in by_art.values():
        v.sort(key=lambda c: c["ordinal"])

    n_chunks = len(chunks)
    per_art = [len(by_art.get(p, [])) for p in records]
    toks = [c["token_count"] for c in chunks]
    chars = [c["char_count"] for c in chunks]
    cuts = sum(1 for c in chunks if c["cuts_sentences"])
    invalid = sum(1 for c in chunks
                  if not c["text"].strip() or not c["pmid"] or c["token_count"] <= 0)
    empty = sum(1 for c in chunks if not c["text"].strip())

    coverage_ok, dup, dropped = 0, 0, 0
    for pmid in records:
        seen = []
        for c in by_art.get(pmid, []):
            if c["sentence_start"] is not None:
                seen.extend(range(c["sentence_start"], c["sentence_end"] + 1))
        n_sent = len(art_sentences[pmid])
        if n_sent > 0 and sorted(seen) == list(range(n_sent)):
            coverage_ok += 1
        if len(seen) != len(set(seen)):
            dup += 1
        if any(i < 0 or i >= n_sent for i in seen):
            dropped += 1

    coh = [coherence_for_article([Chunk(**c) for c in by_art.get(p, [])], art_embs[p])
           for p in records if art_embs[p].shape[0] > 0]
    intra = [c["intra_mean"] for c in coh if c["intra_mean"] == c["intra_mean"]]
    inter = [c["inter_mean"] for c in coh if c["inter_mean"] == c["inter_mean"]]
    gaps = [c["gap"] for c in coh if c["gap"] == c["gap"]]

    return {
        "n_chunks": n_chunks,
        "chunks_per_article": describe(per_art),
        "token_length": describe(toks),
        "char_length": describe(chars),
        "sentence_integrity_violations": cuts,
        "integrity_violation_rate": cuts / n_chunks if n_chunks else 0.0,
        "empty_chunks": empty,
        "invalid_chunks": invalid,
        "coverage_exact_once_articles": coverage_ok,
        "articles_with_duplicate_sentences": dup,
        "articles_with_out_of_range_sentences": dropped,
        "coherence": {
            "intra_chunk_cosine_mean": float(np.mean(intra)) if intra else None,
            "inter_chunk_cosine_mean": float(np.mean(inter)) if inter else None,
            "gap_mean": float(np.mean(gaps)) if gaps else None,
            "n_articles_with_intra": len(intra),
            "n_articles_with_inter": len(inter),
        },
    }
