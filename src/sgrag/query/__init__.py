"""Query routing layer: classification + bounded decomposition (Phase 5C)."""

from sgrag.query.classification import (
    MAX_QUERY_CHARS,
    MAX_SUBQUERIES,
    ClassifiedQuery,
    QueryClass,
    classify_query,
)

__all__ = [
    "MAX_QUERY_CHARS",
    "MAX_SUBQUERIES",
    "ClassifiedQuery",
    "QueryClass",
    "classify_query",
]
