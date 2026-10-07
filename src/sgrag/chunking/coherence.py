"""Semantic coherence measurement.

Uses the required all-MiniLM-L6-v2 sentence embeddings to score every
strategy on the same yardstick:
  - intra-chunk coherence: mean pairwise cosine similarity of the sentences
    inside each multi-sentence chunk (higher = tighter topic grouping);
  - inter-chunk separation: mean cosine similarity between adjacent chunk
    centroids within an article (lower = cleaner topic boundaries);
  - gap = intra - inter: the headline number. A chunker that groups
    semantically related sentences should show a positive gap.
Single-sentence chunks are excluded from the intra mean (undefined) and
counted separately.
"""

import numpy as np


def _get_model(model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def coherence_for_article(chunks, sentence_embeddings):
    """chunks: Chunk list (document order) for one article.
    sentence_embeddings: np array aligned to sentence indices.
    Returns dict with intra/inter means, gap, and counts.
    """
    intra_vals, inter_vals = [], []
    centroids = []
    singletons = 0
    for ch in chunks:
        if ch.sentence_start is None:
            continue
        idxs = list(range(ch.sentence_start, ch.sentence_end + 1))
        embs = sentence_embeddings[idxs]
        centroids.append(embs.mean(axis=0))
        if len(idxs) < 2:
            singletons += 1
            continue
        sims = [cosine_sim(embs[i], embs[j])
                for i in range(len(idxs)) for j in range(i + 1, len(idxs))]
        intra_vals.append(float(np.mean(sims)))
    for a, b in zip(centroids, centroids[1:]):
        inter_vals.append(cosine_sim(a, b))
    intra = float(np.mean(intra_vals)) if intra_vals else float("nan")
    inter = float(np.mean(inter_vals)) if inter_vals else float("nan")
    gap = intra - inter if intra == intra and inter == inter else float("nan")
    return {"intra_mean": intra, "inter_mean": inter, "gap": gap,
            "n_intra_chunks": len(intra_vals), "n_inter_pairs": len(inter_vals),
            "n_singletons": singletons}
