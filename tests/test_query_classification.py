"""Unit tests: query classification + bounded decomposition."""

import pytest

from sgrag.query import (
    MAX_QUERY_CHARS,
    MAX_SUBQUERIES,
    QueryClass,
    classify_query,
)


@pytest.mark.parametrize("q,expected", [
    ("What are the risk factors for Alzheimer's disease?", QueryClass.FACTUAL),
    ("What drugs treat breast cancer?", QueryClass.ENTITY_FOCUSED),
    ("Which genes are linked to diabetes?", QueryClass.ENTITY_FOCUSED),
    ("Aspirin vs ibuprofen for inflammation?", QueryClass.COMPARISON),
    ("Compare metformin and insulin", QueryClass.COMPARISON),
    ("Tell me a joke", QueryClass.UNSUPPORTED),
    ("", QueryClass.UNSUPPORTED),
    ("What is the weather today?", QueryClass.UNSUPPORTED),
])
def test_classification(q, expected):
    assert classify_query(q).query_class == expected


def test_multipart_bounded():
    c = classify_query(
        "What treats diabetes? How is it diagnosed? What causes it? "
        "Who discovered insulin? When was it found?")
    assert c.query_class == QueryClass.MULTIPART
    assert 2 <= len(c.subqueries) <= MAX_SUBQUERIES
    assert all(s.strip() for s in c.subqueries)


def test_multipart_no_duplicates():
    c = classify_query("What drugs treat breast cancer?")
    assert c.query_class != QueryClass.MULTIPART
    assert c.subqueries == []


def test_excessively_long_unsupported():
    c = classify_query("diabetes treatment " * 500)
    assert c.query_class == QueryClass.UNSUPPORTED
    assert "too long" in c.reason


def test_subqueries_are_valid_questions():
    c = classify_query("What causes migraines? What relieves them?")
    assert c.query_class == QueryClass.MULTIPART
    for s in c.subqueries:
        assert len(s) >= 12
