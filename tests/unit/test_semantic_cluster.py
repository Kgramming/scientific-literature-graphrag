"""Tests for the semantic-clustering chunker (strategy C).

Clustering is tested with injected deterministic embeddings so the tests
never download the MiniLM model. The key invariant under test: chunks are
contiguous sections in original document order — clusters are never
globally reordered.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sgrag.chunking import semantic_cluster
from sgrag.chunking.base import SemanticClusterConfig


def _rec(text, pmid="21645374"):
    return {"pmid": pmid, "title": "T", "abstract": text,
            "journal": "J", "year": 2011}


def _embed_fn(vectors):
    def embed(sentences):
        assert len(sentences) == len(vectors)
        return np.asarray(vectors, dtype=float)
    return embed


TOPIC_A = [1.0, 0.0]
TOPIC_B = [0.0, 1.0]

TEXT_4 = ("Cats are great pets. Dogs are great pets. "
          "Quantum physics is hard. Relativity is hard.")


TEXT_6 = ("Cats are great pets. Dogs are great pets. Fish are quiet pets. "
          "Quantum physics is hard. Relativity is hard. Thermodynamics is hard.")


def test_semantic_groups_become_chunks_in_order():
    cfg = SemanticClusterConfig(min_cluster_size=2)
    vecs = [[1.0, 0.0], [1.0, 0.05], [1.0, 0.1],
            [0.0, 1.0], [0.05, 1.0], [0.1, 1.0]]
    chunks, info = semantic_cluster.chunk_record(
        _rec(TEXT_6), config=cfg, embed_fn=_embed_fn(vecs))
    assert len(chunks) == 2
    assert (chunks[0].sentence_start, chunks[0].sentence_end) == (0, 2)
    assert (chunks[1].sentence_start, chunks[1].sentence_end) == (3, 5)
    assert all(not c.cuts_sentences for c in chunks)
    assert all(c.strategy == "semantic_cluster" for c in chunks)


def test_noncontiguous_cluster_splits_into_runs():
    # Same topic at positions 0 and 2 -> two separate chunks, document order.
    text = "Cats are great. Quantum is hard. Dogs are great."
    vecs = [TOPIC_A, TOPIC_B, [1.0, 0.1]]
    cfg = SemanticClusterConfig(min_cluster_size=2)
    chunks, info = semantic_cluster.chunk_record(
        _rec(text), config=cfg, embed_fn=_embed_fn(vecs))
    spans = [(c.sentence_start, c.sentence_end) for c in chunks]
    # middle sentence is noise (singleton); topic-A sentences stay separate runs
    assert spans == [(0, 0), (1, 1), (2, 2)]
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_noise_becomes_singleton_and_diagnostics_recorded():
    cfg = SemanticClusterConfig(min_cluster_size=2)
    text = "Cats are great pets. Dogs are great pets. Quantum physics is hard."
    vecs = [TOPIC_A, [1.0, 0.1], TOPIC_B]
    chunks, info = semantic_cluster.chunk_record(
        _rec(text), config=cfg, embed_fn=_embed_fn(vecs))
    assert info["embedding_model"] == "all-MiniLM-L6-v2"
    assert info["clustering"] == "hdbscan"
    assert info["min_cluster_size"] == 2
    assert info["fallback"] is None
    assert info["n_noise"] >= 1


def test_fallback_for_short_abstract():
    cfg = SemanticClusterConfig(min_sentences=3)
    chunks, info = semantic_cluster.chunk_record(
        _rec("Only one sentence here."), config=cfg,
        embed_fn=_embed_fn([[1.0, 0.0]]))
    assert len(chunks) == 1
    assert info["fallback"] is not None
    assert chunks[0].n_sentences == 1


def test_deterministic_given_embeddings():
    cfg = SemanticClusterConfig(min_cluster_size=2)
    vecs = [[1.0, 0.0], [1.0, 0.05], [1.0, 0.1],
            [0.0, 1.0], [0.05, 1.0], [0.1, 1.0]]
    chunks_a, _ = semantic_cluster.chunk_record(
        _rec(TEXT_6), config=cfg, embed_fn=_embed_fn(vecs))
    chunks_b, _ = semantic_cluster.chunk_record(
        _rec(TEXT_6), config=cfg, embed_fn=_embed_fn(vecs))
    a = [c.to_dict() for c in chunks_a]
    b = [c.to_dict() for c in chunks_b]
    assert a == b


def test_empty_input_and_provenance():
    cfg = SemanticClusterConfig()
    chunks, info = semantic_cluster.chunk_record(
        _rec(""), config=cfg, embed_fn=_embed_fn([]))
    assert chunks == []
    chunks, info = semantic_cluster.chunk_record(
        _rec(TEXT_4), config=cfg,
        embed_fn=_embed_fn([TOPIC_A, [1.0, 0.1], TOPIC_B, [0.1, 1.0]]))
    assert chunks[0].chunk_id == "21645374_semantic_cluster_000"
    assert all(c.pmid == "21645374" for c in chunks)
    assert all(c.source["title"] == "T" for c in chunks)


def test_enforce_max_tokens_splits_long_runs():
    from sgrag.chunking.semantic_cluster import _enforce_max_tokens
    from sgrag.chunking.tokenizer import get_tokenizer
    tok = get_tokenizer()
    sents = [" ".join(["word"] * 40) + f" number {i}." for i in range(8)]
    runs = _enforce_max_tokens([[0, 1, 2, 3, 4, 5, 6, 7]], sents, tok, 256)
    assert len(runs) == 2
    assert [i for r in runs for i in r] == list(range(8))
    from sgrag.chunking.tokenizer import count_tokens
    for r in runs:
        assert sum(count_tokens(sents[i], tok) for i in r) <= 256


def test_enforce_max_tokens_keeps_long_single_sentence_whole():
    from sgrag.chunking.semantic_cluster import _enforce_max_tokens
    from sgrag.chunking.tokenizer import get_tokenizer
    tok = get_tokenizer()
    sents = [" ".join(["word"] * 300) + "."]
    runs = _enforce_max_tokens([[0]], sents, tok, 256)
    assert runs == [[0]]


def test_max_tokens_enforced_end_to_end(monkeypatch):
    # Stub clustering: all 8 sentences in one cluster -> run must be split.
    import numpy as np
    monkeypatch.setattr(semantic_cluster, "_cluster_labels",
                        lambda emb, cfg: np.zeros(emb.shape[0], dtype=int))
    sents = [" ".join(["word"] * 40) + f" number {i}." for i in range(8)]
    cfg = SemanticClusterConfig(min_cluster_size=2, max_tokens=256)
    chunks, info = semantic_cluster.chunk_record(
        _rec(" ".join(sents)), config=cfg,
        embed_fn=_embed_fn([[1.0, 0.0]] * 8))
    assert len(chunks) == 2
    assert all(c.token_count <= 256 for c in chunks)
    assert all(not c.cuts_sentences for c in chunks)
    covered = []
    for c in chunks:
        covered.extend(range(c.sentence_start, c.sentence_end + 1))
    assert covered == list(range(8))
    assert info["n_size_splits"] == 1


def test_single_long_sentence_kept_whole():
    # Sentence integrity wins: an over-long single sentence is never cut.
    text = " ".join(["word"] * 300) + "."
    cfg = SemanticClusterConfig(min_sentences=3, max_tokens=256)
    chunks, info = semantic_cluster.chunk_record(
        _rec(text), config=cfg, embed_fn=_embed_fn([[1.0, 0.0]]))
    assert len(chunks) == 1
    assert chunks[0].n_sentences == 1
    assert not chunks[0].cuts_sentences


def test_normalize_embeddings_flag_recorded_and_applied(monkeypatch):
    # Normalization must change the geometry HDBSCAN sees, not just metadata.
    import numpy as np
    seen = {}

    def fake_cluster(emb, cfg):
        seen["unit_norms"] = np.linalg.norm(emb, axis=1)
        return np.full(emb.shape[0], -1)  # all noise -> singletons

    monkeypatch.setattr(semantic_cluster, "_cluster_labels", fake_cluster)
    vecs = [[3.0, 4.0], [6.0, 8.0], [1.0, 1.0]]
    cfg = SemanticClusterConfig(normalize_embeddings=True)
    chunks, info = semantic_cluster.chunk_record(
        _rec("First sentence here. Second sentence here. Third sentence here."),
        config=cfg, embed_fn=_embed_fn(vecs))
    assert info["normalize_embeddings"] is True
    assert np.allclose(seen["unit_norms"], 1.0)
    assert len(chunks) == 3  # all noise -> 3 singletons

    cfg2 = SemanticClusterConfig(normalize_embeddings=False)
    chunks2, info2 = semantic_cluster.chunk_record(
        _rec("First sentence here. Second sentence here. Third sentence here."),
        config=cfg2, embed_fn=_embed_fn(vecs))
    assert info2["normalize_embeddings"] is False
    assert not np.allclose(np.linalg.norm(np.asarray(vecs), axis=1), 1.0)
