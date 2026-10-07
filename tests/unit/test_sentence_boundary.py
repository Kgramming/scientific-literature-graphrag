"""Tests for the sentence-boundary baseline chunker (strategy B)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sgrag.chunking import sentence_boundary
from sgrag.chunking.base import SentenceBoundaryConfig
from sgrag.chunking.sentences import split_sentences
from sgrag.chunking.tokenizer import get_tokenizer

TEXT = ("Programmed cell death removes unwanted cells. "
        "The lace plant forms leaf perforations through this process. "
        "Mitochondria may play a role in remodelling. "
        "We examined leaves at several developmental stages. "
        "Results suggest mitochondria are involved early. "
        "Further work will confirm the mechanism. " * 8)


def _rec(text=TEXT, pmid="21645374"):
    return {"pmid": pmid, "title": "T", "abstract": text,
            "journal": "J", "year": 2011}


def test_never_cuts_sentences_and_respects_limit():
    cfg = SentenceBoundaryConfig(max_tokens=64)
    chunks = sentence_boundary.chunk_record(_rec(), config=cfg)
    assert chunks
    assert all(not c.cuts_sentences for c in chunks)
    for c in chunks:
        assert c.token_count <= 64
        assert c.strategy == "sentence_boundary"


def test_sentences_covered_exactly_once_in_order():
    cfg = SentenceBoundaryConfig(max_tokens=64)
    sentences = split_sentences(TEXT)
    chunks = sentence_boundary.chunk_record(_rec(), config=cfg)
    covered = []
    for c in chunks:
        covered.extend(range(c.sentence_start, c.sentence_end + 1))
    assert covered == list(range(len(sentences)))
    # chunk text is exactly the joined source sentences
    assert " ".join(sentences) == " ".join(c.text for c in chunks)


def test_deterministic():
    cfg = SentenceBoundaryConfig(max_tokens=64)
    a = [c.to_dict() for c in sentence_boundary.chunk_record(_rec(), config=cfg)]
    b = [c.to_dict() for c in sentence_boundary.chunk_record(_rec(), config=cfg)]
    assert a == b


def test_single_sentence_and_empty_input():
    cfg = SentenceBoundaryConfig(max_tokens=64)
    one = sentence_boundary.chunk_record(_rec(text="Only one sentence here."))
    assert len(one) == 1 and one[0].n_sentences == 1
    assert one[0].sentence_start == one[0].sentence_end == 0
    assert sentence_boundary.chunk_record(_rec(text="")) == []
    assert sentence_boundary.chunk_record(_rec(text="   ")) == []


def test_provenance_preserved():
    cfg = SentenceBoundaryConfig(max_tokens=64)
    chunks = sentence_boundary.chunk_record(_rec(), config=cfg)
    assert chunks[0].chunk_id == "21645374_sentence_boundary_000"
    assert all(c.pmid == "21645374" and c.article_id == "21645374" for c in chunks)
    assert all(c.config["max_tokens"] == 64 for c in chunks)
    assert all(c.source["journal"] == "J" for c in chunks)
    # token counts are consistent with the shared tokenizer
    tok = get_tokenizer(cfg.tokenizer_model)
    for c in chunks[:3]:
        assert c.token_count == len(tok.encode(c.text, add_special_tokens=False))
