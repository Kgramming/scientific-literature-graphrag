"""Strategy C — semantic clustering.

Sentences are embedded with the required all-MiniLM-L6-v2 model and grouped
with HDBSCAN. Crucially, clusters are then split into maximal CONTIGUOUS
runs: a cluster whose sentences are scattered across the abstract becomes
one chunk per contiguous section, so every chunk remains a coherent section
of the original abstract in original order (never globally reordered).
Noise points (HDBSCAN label -1) become singleton chunks. Abstracts with
fewer than min_sentences sentences fall back to a single chunk.
"""

from dataclasses import asdict

import numpy as np

from .base import Chunk, SemanticClusterConfig, make_chunk_id
from .sentences import split_sentences
from .tokenizer import count_tokens, get_tokenizer


def _default_embed_fn(model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def embed(sentences):
        return np.asarray(model.encode(sentences, show_progress_bar=False,
                                       convert_to_numpy=True))
    return embed


def make_embed_fn(model_name):
    """Build a reusable sentence-embedding function (load the model once)."""
    return _default_embed_fn(model_name)


def _cluster_labels(embeddings, config):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=config.min_cluster_size,
                                metric=config.metric)
    return clusterer.fit_predict(embeddings)


def _contiguous_runs(sorted_indices):
    """Split sorted sentence indices into maximal contiguous runs."""
    runs, run = [], [sorted_indices[0]]
    for i in sorted_indices[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            runs.append(run)
            run = [i]
    runs.append(run)
    return runs


def _enforce_max_tokens(runs, sentences, tokenizer, max_tokens):
    """Split runs exceeding max_tokens greedily on sentence boundaries.

    Sentences are never cut: a single sentence longer than max_tokens is
    kept whole (documented, counted in evaluation).
    """
    out = []
    for run in runs:
        cur, cur_t = [], 0
        for i in run:
            t = count_tokens(sentences[i], tokenizer)
            if cur and cur_t + t > max_tokens:
                out.append(cur)
                cur, cur_t = [], 0
            cur.append(i)
            cur_t += t
        if cur:
            out.append(cur)
    return out


def group_sentences(sentences, embeddings, config):
    """Cluster sentence indices -> ordered list of contiguous index runs.

    Returns (runs, info) where runs is a list of index lists in document
    order and info records clustering diagnostics.
    """
    labels = _cluster_labels(embeddings, config)
    info = {"n_sentences": len(sentences),
            "labels": [int(l) for l in labels]}
    groups = {}
    noise = []
    for idx, label in enumerate(labels):
        label = int(label)
        if label == -1:
            noise.append(idx)
        else:
            groups.setdefault(label, []).append(idx)

    runs = []
    for label in sorted(groups, key=lambda l: groups[l][0]):
        runs.extend(_contiguous_runs(sorted(groups[label])))
    for idx in sorted(noise):
        runs.append([idx])
    # Order chunks by first sentence for document-order output.
    runs.sort(key=lambda r: r[0])
    info.update({
        "n_clusters": len(groups),
        "n_noise": len(noise),
        "n_chunks": len(runs),
    })
    return runs, info


def chunk_record(record, config=None, tokenizer=None, embed_fn=None):
    """Chunk one pilot-corpus record by semantic clustering.

    Returns (chunks, info); info carries embedding model, clustering
    algorithm/parameters, cluster counts and any fallback taken.
    """
    config = config or SemanticClusterConfig()
    if isinstance(config, dict):
        config = SemanticClusterConfig(**config)
    tokenizer = tokenizer or get_tokenizer(config.tokenizer_model)

    pmid = str(record["pmid"])
    text = record.get("abstract") or ""
    source = {"title": record.get("title"), "journal": record.get("journal"),
              "year": record.get("year")}
    info = {"embedding_model": config.embedding_model,
            "clustering": config.clustering,
            "min_cluster_size": config.min_cluster_size,
            "metric": config.metric,
            "normalize_embeddings": config.normalize_embeddings,
            "fallback": None}

    sentences = split_sentences(text)
    if not sentences:
        return [], info
    if len(sentences) < config.min_sentences:
        info["fallback"] = f"fewer than {config.min_sentences} sentences"
        runs = [[i for i in range(len(sentences))]]
    else:
        embed = embed_fn or _default_embed_fn(config.embedding_model)
        embeddings = np.asarray(embed(sentences), dtype=float)
        if config.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        try:
            runs, grp_info = group_sentences(sentences, embeddings, config)
        except Exception as exc:  # clustering must never kill the pipeline
            info["fallback"] = f"clustering failed: {type(exc).__name__}"
            runs, grp_info = [[i for i in range(len(sentences))]], {}
        info.update(grp_info)

    # Enforce the configured maximum chunk size (sentence-safe).
    n_runs_before = len(runs)
    runs = _enforce_max_tokens(runs, sentences, tokenizer, config.max_tokens)
    info["n_size_splits"] = len(runs) - n_runs_before
    info["max_tokens"] = config.max_tokens

    chunks = []
    for ordinal, idxs in enumerate(runs):
        chunk_text = " ".join(sentences[i] for i in idxs)
        chunks.append(Chunk(
            chunk_id=make_chunk_id(pmid, "semantic_cluster", ordinal),
            pmid=pmid, article_id=pmid, strategy="semantic_cluster",
            ordinal=ordinal, text=chunk_text,
            token_count=count_tokens(chunk_text, tokenizer),
            char_count=len(chunk_text), n_sentences=len(idxs),
            sentence_start=idxs[0], sentence_end=idxs[-1],
            cuts_sentences=False,
            config=asdict(config), source=source,
        ))
    return chunks, info
