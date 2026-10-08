"""Typed generation models (serializable)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EvidenceItem:
    chunk_id: str
    pmid: str
    title: str
    text: str
    reranker_score: float | None = None
    fused_score: float = 0.0
    retrieval_sources: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)


@dataclass
class Citation:
    pmid: str
    chunk_id: str | None = None
    valid: bool = True


@dataclass
class CitationValidation:
    citations: list[Citation] = field(default_factory=list)
    valid: bool = True
    invalid: list[Citation] = field(default_factory=list)
    regenerated: bool = False


@dataclass
class GenerationResult:
    query: str
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    retrieval_metadata: dict = field(default_factory=dict)
    generation_metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    citation_validation: CitationValidation = field(
        default_factory=CitationValidation)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> GenerationResult:
        data = dict(data)
        data["citations"] = [Citation(**c) for c in data.get("citations", [])]
        data["evidence"] = [EvidenceItem(**e) for e in data.get("evidence", [])]
        cv = data.get("citation_validation", {})
        data["citation_validation"] = CitationValidation(
            citations=[Citation(**c) for c in cv.get("citations", [])],
            valid=cv.get("valid", True),
            invalid=[Citation(**c) for c in cv.get("invalid", [])],
            regenerated=cv.get("regenerated", False),
        )
        return cls(**data)
