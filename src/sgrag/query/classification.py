"""Lightweight, deterministic query classification + bounded decomposition (5C).

No LLM, no recursion, no uncontrolled loops. Rule-based so routing is
validated and reproducible:

  FACTUAL        - normal scientific question -> direct retrieval
  ENTITY_FOCUSED - asks about drugs/genes/treatments for an entity
  COMPARISON     - "X vs Y" / "compare" questions
  MULTIPART      - several sub-questions -> bounded subqueries (<=3)
  UNSUPPORTED    - non-scientific / empty / excessively long
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

MAX_SUBQUERIES = 3
MAX_QUERY_CHARS = 2000

_NON_SCIENCE = re.compile(
    r"\b(write|compose)\s+(a\s+)?(poem|song|story|joke|essay)|"
    r"\b(weather|lottery|stock price|who won|score of|what time)\b|"
    r"\b(tell me a joke|how to cook|recipe for)\b",
    re.IGNORECASE,
)
_COMPARISON = re.compile(r"\b(vs\.?|versus|compar\w+|difference between)\b", re.IGNORECASE)
_ENTITY_Q = re.compile(
    r"\b(what|which)\s+(drugs?|genes?|proteins?|treatments?|therap\w+|"
    r"medications?|compounds?|biomarkers?)\b",
    re.IGNORECASE,
)
_BIOMED_HINT = re.compile(
    r"\b(disease|cancer|drug|treatment|gene|protein|cell|clinical|patient|"
    r"therapy|syndrome|disorder|infection|tumor|biomarker|enzyme|receptor)\b",
    re.IGNORECASE,
)


class QueryClass(str, Enum):
    FACTUAL = "factual"
    ENTITY_FOCUSED = "entity_focused"
    COMPARISON = "comparison"
    MULTIPART = "multipart"
    UNSUPPORTED = "unsupported"


@dataclass
class ClassifiedQuery:
    text: str
    query_class: QueryClass
    subqueries: list[str] = field(default_factory=list)
    reason: str = ""


def _split_parts(text: str) -> list[str]:
    """Split a multi-part question into candidate subqueries."""
    chunks = re.split(r"\?\s*|\bsubparts?\b", text)
    parts = []
    for ch in chunks:
        ch = re.sub(r"^(and|or|also)\b", "", ch.strip(), flags=re.IGNORECASE).strip(" ,;")
        if len(ch) >= 12:
            parts.append(ch if ch.endswith("?") else ch + "?")
    # also try ' and ' splits for imperative multi-parts (only if it splits)
    and_chunks = re.split(r"\s+and\s+", text)
    if len(parts) < 2 and len(and_chunks) > 1:
        norm = {p.rstrip("?").lower() for p in parts}
        for ch in and_chunks:
            ch = ch.strip(" ,;?")
            if len(ch) >= 12 and ch.lower() not in norm:
                norm.add(ch.lower())
                parts.append(ch + "?")
    return parts


def classify_query(text: str) -> ClassifiedQuery:
    t = (text or "").strip()
    if not t:
        return ClassifiedQuery(text=t, query_class=QueryClass.UNSUPPORTED,
                               reason="empty query")
    if len(t) > MAX_QUERY_CHARS:
        return ClassifiedQuery(text=t, query_class=QueryClass.UNSUPPORTED,
                               reason=f"query too long ({len(t)} chars)")
    if _NON_SCIENCE.search(t):
        return ClassifiedQuery(text=t, query_class=QueryClass.UNSUPPORTED,
                               reason="non-scientific question")
    if not _BIOMED_HINT.search(t) and len(t.split()) < 4:
        return ClassifiedQuery(text=t, query_class=QueryClass.UNSUPPORTED,
                               reason="no biomedical signal")
    if _COMPARISON.search(t):
        return ClassifiedQuery(text=t, query_class=QueryClass.COMPARISON,
                               reason="comparison phrasing")
    parts = _split_parts(t)
    if len(parts) >= 2:
        subs = parts[:MAX_SUBQUERIES]
        return ClassifiedQuery(text=t, query_class=QueryClass.MULTIPART,
                               subqueries=subs,
                               reason=f"{len(subs)} subqueries")
    if _ENTITY_Q.search(t):
        return ClassifiedQuery(text=t, query_class=QueryClass.ENTITY_FOCUSED,
                               reason="entity-seeking phrasing")
    return ClassifiedQuery(text=t, query_class=QueryClass.FACTUAL,
                           reason="default scientific question")
