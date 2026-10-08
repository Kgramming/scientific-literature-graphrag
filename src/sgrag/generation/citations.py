"""Deterministic citation extraction + validation (Phase 5B, REQUIRED).

Citation format: [PMID:<digits>] or [PMID:<digits> | CHUNK:<chunk_id>].
A citation is valid only if its PMID (and chunk ID, when given) appear in
the evidence actually supplied to the LLM. Fabricated citations are flagged,
never silently accepted.
"""

from __future__ import annotations

import re

from sgrag.generation.models import Citation, CitationValidation

_CITE_RE = re.compile(
    r"\[PMID\s*:\s*(\d+)(?:\s*\|\s*CHUNK\s*:\s*([A-Za-z0-9_\-]+))?\s*\]",
    re.IGNORECASE,
)


def extract_citations(answer: str) -> list[Citation]:
    """Extract all [PMID:...] citations from generated text (deduped)."""
    seen: set[tuple[str, str | None]] = set()
    out: list[Citation] = []
    for m in _CITE_RE.finditer(answer or ""):
        key = (m.group(1), m.group(2))
        if key not in seen:
            seen.add(key)
            out.append(Citation(pmid=m.group(1), chunk_id=m.group(2)))
    return out


def validate_citations(
    answer: str, evidence_pmids: set[str], evidence_chunk_ids: set[str]
) -> CitationValidation:
    """Validate extracted citations against the supplied evidence."""
    citations = extract_citations(answer)
    invalid: list[Citation] = []
    for c in citations:
        ok = c.pmid in evidence_pmids
        if ok and c.chunk_id is not None:
            ok = c.chunk_id in evidence_chunk_ids
        c.valid = ok
        if not ok:
            invalid.append(c)
    return CitationValidation(
        citations=citations, valid=not invalid, invalid=invalid)


def strip_invalid_citations(answer: str, invalid: list[Citation]) -> str:
    """Remove fabricated citation markers from text (no invented replacement)."""
    out = answer
    for c in invalid:
        if c.chunk_id:
            pat = re.compile(
                rf"\[PMID\s*:\s*{c.pmid}\s*\|\s*CHUNK\s*:\s*{re.escape(c.chunk_id)}\s*\]",
                re.IGNORECASE)
        else:
            pat = re.compile(rf"\[PMID\s*:\s*{c.pmid}\s*\]", re.IGNORECASE)
        out = pat.sub("", out)
    # tidy leftover double spaces left by removal
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()
