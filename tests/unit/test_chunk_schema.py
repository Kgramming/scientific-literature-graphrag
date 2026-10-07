"""Tests for the Chunk schema: provenance, validity, JSONL round-trip."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sgrag.chunking.base import Chunk, make_chunk_id


def _chunk(**kw):
    d = dict(chunk_id="1_fixed_token_000", pmid="1", article_id="1",
             strategy="fixed_token", ordinal=0, text="some text",
             token_count=2, char_count=9, n_sentences=1,
             sentence_start=0, sentence_end=0, cuts_sentences=False,
             config={"token_limit": 256}, source={"title": "T"})
    d.update(kw)
    return Chunk(**d)


def test_required_fields_present():
    d = _chunk().to_dict()
    for f in Chunk.required_fields():
        assert f in d, f"missing {f}"


def test_jsonl_round_trip():
    c = _chunk()
    c2 = Chunk(**json.loads(json.dumps(c.to_dict())))
    assert c2 == c


def test_make_chunk_id_format():
    assert make_chunk_id("21645374", "fixed_token", 3) == "21645374_fixed_token_003"


def test_is_valid_rejects_bad_chunks():
    assert _chunk().is_valid()
    assert not _chunk(text="   ").is_valid()
    assert not _chunk(text="").is_valid()
    assert not _chunk(pmid="").is_valid()
    assert not _chunk(token_count=0).is_valid()
