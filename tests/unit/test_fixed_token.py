"""Tests for the fixed-token baseline chunker (strategy A)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sgrag.chunking import fixed_token
from sgrag.chunking.base import FixedTokenConfig
from sgrag.chunking.tokenizer import get_tokenizer

TEXT = ("Programmed cell death removes unwanted cells. "
        "The lace plant forms leaf perforations through this process. "
        "Mitochondria may play a role in remodelling. "
        "We examined leaves at several developmental stages. "
        "Results suggest mitochondria are involved early. "
        "Further work will confirm the mechanism. " * 6)


def _rec(text=TEXT, pmid="21645374"):
    return {"pmid": pmid, "title": "T", "abstract": text,
            "journal": "J", "year": 2011}


def test_windows_respect_token_limit():
    cfg = FixedTokenConfig(token_limit=64, overlap=8)
    chunks = fixed_token.chunk_record(_rec(), config=cfg)
    assert chunks
    for ch in chunks:
        assert ch.token_count <= 64
        assert ch.strategy == "fixed_token"
        assert ch.pmid == "21645374"


def test_overlap_between_windows():
    cfg = FixedTokenConfig(token_limit=64, overlap=16)
    tok = get_tokenizer(cfg.tokenizer_model)
    chunks = fixed_token.chunk_record(_rec(), config=cfg, tokenizer=tok)
    assert len(chunks) >= 2
    # overlap: tail tokens of chunk 0 == head tokens of chunk 1
    t0 = tok.encode(chunks[0].text, add_special_tokens=False)
    t1 = tok.encode(chunks[1].text, add_special_tokens=False)
    assert t0[-16:] == t1[:16]


def test_deterministic():
    cfg = FixedTokenConfig(token_limit=64, overlap=8)
    a = [c.to_dict() for c in fixed_token.chunk_record(_rec(), config=cfg)]
    b = [c.to_dict() for c in fixed_token.chunk_record(_rec(), config=cfg)]
    assert a == b


def test_cuts_sentences_flagged():
    # One very long sentence must be cut by a small window.
    long_sent = "word " * 200
    cfg = FixedTokenConfig(token_limit=32, overlap=4)
    chunks = fixed_token.chunk_record(_rec(text=long_sent), config=cfg)
    assert any(c.cuts_sentences for c in chunks)


def test_chunk_ids_and_provenance():
    cfg = FixedTokenConfig(token_limit=64, overlap=8)
    chunks = fixed_token.chunk_record(_rec(), config=cfg)
    ids = [c.chunk_id for c in chunks]
    assert ids == [f"21645374_fixed_token_{i:03d}" for i in range(len(ids))]
    assert all(c.config["token_limit"] == 64 for c in chunks)
    assert all(c.source["year"] == 2011 for c in chunks)


def test_empty_and_malformed_input():
    cfg = FixedTokenConfig()
    assert fixed_token.chunk_record(_rec(text="")) == []
    assert fixed_token.chunk_record(_rec(text="   ")) == []
    rec = _rec(); rec["abstract"] = None
    assert fixed_token.chunk_record(rec) == []
