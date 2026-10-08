"""Unit tests: evidence packing, prompts, citations."""

import pytest

from sgrag.generation import citations as cite_mod
from sgrag.generation.evidence import pack_evidence
from sgrag.generation.models import EvidenceItem
from sgrag.generation.prompts import build_messages


def _res(cid, pmid="111"):
    from sgrag.retrieval.models import RetrievalResult
    return RetrievalResult(chunk_id=cid, pmid=pmid, title="T", text="txt",
                           fused_score=0.9, reranker_score=0.5,
                           retrieval_sources=["vector"],
                           matched_entities=["DISEASE:x"])


def test_pack_evidence_bounded_and_fields():
    items = pack_evidence([_res(f"c{i}") for i in range(20)], top_k=5)
    assert len(items) == 5
    it = items[0]
    assert (it.chunk_id, it.pmid, it.title, it.text) == ("c0", "111", "T", "txt")
    assert it.reranker_score == 0.5 and it.fused_score == 0.9
    assert it.retrieval_sources == ["vector"]
    assert it.matched_entities == ["DISEASE:x"]


def test_pack_evidence_empty():
    assert pack_evidence([], top_k=8) == []


def test_prompt_contains_evidence_and_format():
    items = [EvidenceItem(chunk_id="c1", pmid="111", title="T", text="body")]
    msgs = build_messages("What?", items)
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "PMID:111" in user and "CHUNK:c1" in user and "body" in user
    assert "[PMID:<pmid>" in user
    assert "insufficient" in msgs[0]["content"].lower()


def test_extract_citations():
    text = ("Claim one [PMID:12345]. Claim two [PMID:67890 | CHUNK:abc_001]. "
            "Repeat [PMID:12345].")
    cites = cite_mod.extract_citations(text)
    assert [(c.pmid, c.chunk_id) for c in cites] == [
        ("12345", None), ("67890", "abc_001")]


def test_validate_citations_ok():
    v = cite_mod.validate_citations("x [PMID:111] y [PMID:222 | CHUNK:c2]",
                                    {"111", "222"}, {"c1", "c2"})
    assert v.valid and not v.invalid
    assert len(v.citations) == 2


def test_validate_citations_fabricated():
    v = cite_mod.validate_citations("x [PMID:999] y [PMID:111 | CHUNK:nope]",
                                    {"111"}, {"c1"})
    assert not v.valid
    assert len(v.invalid) == 2
    assert all(not c.valid for c in v.invalid)


def test_strip_invalid_citations():
    text = "Good [PMID:111]. Bad [PMID:999]. Also bad [PMID:111 | CHUNK:zz]."
    v = cite_mod.validate_citations(text, {"111"}, {"c1"})
    out = cite_mod.strip_invalid_citations(text, v.invalid)
    assert "[PMID:999]" not in out and "CHUNK:zz" not in out
    assert "[PMID:111]" in out  # the valid bare citation stays
