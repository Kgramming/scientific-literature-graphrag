"""Tests for the embeddings module and comparison-report schema.

Model-loading and shape tests use the cached all-MiniLM-L6-v2 (no download).
Cache-behavior tests use an injected stub encoder and a tmp cache dir.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sgrag.embeddings import EmbeddingConfig, encode, load_model


def _cfg(**kw):
    d = dict(model_name="sentence-transformers/all-MiniLM-L6-v2",
             cache_dir=str(Path(__file__).parent / "tmp_cache_test"))
    d.update(kw)
    return EmbeddingConfig(**d)


def test_config_cache_key_deterministic_and_sensitive():
    c = _cfg()
    texts = ["hello world", "second sentence"]
    assert c.cache_key(texts) == c.cache_key(texts)
    assert c.cache_key(texts) != c.cache_key(["hello world", "different"])
    assert c.cache_key(texts) != _cfg(model_name="other/model").cache_key(texts)
    assert c.cache_key(texts) != _cfg(normalize=True).cache_key(texts)
    assert "/" not in c.cache_slug()


def test_load_model_and_shape():
    model = load_model("sentence-transformers/all-MiniLM-L6-v2", "cpu")
    assert model.get_sentence_embedding_dimension() == 384


def test_encode_deterministic_shape(tmp_path):
    cfg = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2",
                          cache_dir=str(tmp_path / "cache"))
    texts = ["Programmed cell death.", "Mitochondria play a role."]
    e1, p1 = encode(texts, cfg)
    e2, p2 = encode(texts, cfg)  # second call should hit the disk cache
    assert e1.shape == (2, 384)
    assert np.allclose(e1, e2)
    assert p2["cache_hit"] is True
    assert p1["model_name"] == cfg.model_name
    assert p1["n_texts"] == 2


class _StubEncoder:
    def __init__(self, dim=8):
        self.calls = 0
        self.dim = dim

    def encode(self, texts, **kw):
        self.calls += 1
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(texts), self.dim))


@pytest.fixture
def stub_cfg(tmp_path):
    return EmbeddingConfig(model_name="test/stub", cache_dir=str(tmp_path / "cache"))


def test_cache_avoids_recompute(stub_cfg):
    stub = _StubEncoder()
    texts = ["a", "b", "c"]
    e1, p1 = encode(texts, stub_cfg, _encoder=stub)
    assert stub.calls == 1 and p1["cache_hit"] is False
    e2, p2 = encode(texts, stub_cfg, _encoder=stub)
    assert stub.calls == 1 and p2["cache_hit"] is True
    assert np.array_equal(e1, e2)
    # Different texts -> recompute.
    encode(["a", "b", "d"], stub_cfg, _encoder=stub)
    assert stub.calls == 2
    # Cache files exist with provenance sidecar.
    files = list(Path(stub_cfg.cache_dir).glob("*"))
    assert any(f.suffix == ".npy" for f in files)
    assert any(f.suffix == ".json" for f in files)
    meta = json.loads(next(f for f in files if f.suffix == ".json").read_text())
    assert meta["model_name"] == "test/stub" and meta["n_texts"] == 3


def test_normalize_flag_produces_unit_norms(stub_cfg):
    stub = _StubEncoder(dim=8)
    cfg = EmbeddingConfig(model_name="test/stub", normalize=True,
                          cache_dir=stub_cfg.cache_dir)
    embs, _ = encode(["x", "y"], cfg, _encoder=stub)
    assert np.allclose(np.linalg.norm(embs, axis=1), 1.0)


def test_build_report_schema():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import compare_embeddings

    def fake_model(mrr, gap, dim, secs):
        return {
            "config": {"model_name": "m", "normalize": False, "device": "cpu"},
            "provenance": {"embedding_dim": dim, "encode_seconds": secs,
                           "n_texts": 10, "cache_hit": False,
                           "timestamp_utc": "2026-01-01T00:00:00+00:00"},
            "title_provenance": {"encode_seconds": 0.5},
            "coherence": {"gap_mean": gap},
            "retrieval": {"mrr": mrr, "success@1": 0.5, "success@5": 0.7,
                          "success@10": 0.8},
            "storage_bytes": 1000 * dim,
        }

    report = compare_embeddings.build_report(
        {"baseline": fake_model(0.60, 0.20, 384, 10.0),
         "candidate": fake_model(0.65, 0.25, 768, 40.0)},
        {"experiment": "x", "recall_k": [1, 5, 10]})
    for name in ("baseline", "candidate"):
        m = report["models"][name]
        for key in ("config", "embedding_dim", "normalize", "device",
                    "encode_seconds_sentences", "n_texts", "coherence",
                    "title_retrieval", "storage_bytes", "timestamp_utc"):
            assert key in m, f"{name} missing {key}"
    d = report["deltas"]
    assert d["mrr"] == pytest.approx(0.05)
    assert d["coherence_gap"] == pytest.approx(0.05)
    assert d["dim_ratio"] == pytest.approx(2.0)
    assert d["encode_time_ratio"] == pytest.approx(4.0)
