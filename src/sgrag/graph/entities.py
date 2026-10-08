"""Biomedical entity extraction and deterministic normalization.

Extraction uses the scispaCy ``en_ner_bc5cdr_md`` model (disease/chemical NER)
behind a thin adapter so unit tests can run without the model installed.

Normalization is deliberately conservative: case, punctuation, whitespace and
possessives only. No alias dictionary and no fuzzy merging are applied, so no
synonym-resolution capability beyond the rules below is claimed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_NORMALIZE_PUNCT = re.compile(r"[^a-z0-9\s]")
_NORMALIZE_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class ExtractedEntity:
    surface_form: str
    entity_type: str
    char_start: int
    char_end: int


def normalize_entity_name(name: str) -> str:
    """Deterministic surface-form normalization.

    Rules (in order):
      1. Unicode NFKC normalization
      2. lowercase
      3. drop possessive markers (``'s`` / trailing ``'``)
      4. replace remaining punctuation with a single space
      5. collapse whitespace and strip

    Examples
    --------
    >>> normalize_entity_name("Alzheimer's Disease")
    'alzheimer disease'
    >>> normalize_entity_name("ALZHEIMER'S DISEASE")
    'alzheimer disease'
    """
    text = unicodedata.normalize("NFKC", name)
    text = text.lower()
    text = re.sub(r"'s\b", "", text)
    text = text.replace("'", "")
    text = _NORMALIZE_PUNCT.sub(" ", text)
    text = _NORMALIZE_WS.sub(" ", text).strip()
    return text


def normalized_key(entity_type: str, name: str) -> str:
    """Deterministic unique key for an Entity node: ``TYPE:normalized-name``."""
    return f"{entity_type.strip().upper()}:{normalize_entity_name(name)}"


def normalize_entity(entity_type: str, surface_form: str) -> dict:
    """Return the Entity-node identity fields for one mention."""
    return {
        "normalized_key": normalized_key(entity_type, surface_form),
        "entity_type": entity_type.strip().upper(),
        "surface_form": surface_form,
    }


class BiomedicalEntityExtractor:
    """scispaCy bc5cdr extraction adapter (lazy model load)."""

    def __init__(self, model: str = "en_ner_bc5cdr_md",
                 allowed_types: tuple[str, ...] = ("DISEASE", "CHEMICAL")) -> None:
        self.model_name = model
        self.allowed_types = tuple(t.upper() for t in allowed_types)
        self._nlp = None

    def _load(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.model_name)
        return self._nlp

    def extract_batch(self, texts: list[str], batch_size: int = 256) -> list[list[ExtractedEntity]]:
        """Extract entities for each text; never hallucinates (model spans only)."""
        nlp = self._load()
        results: list[list[ExtractedEntity]] = []
        for doc in nlp.pipe(texts, batch_size=batch_size):
            ents: list[ExtractedEntity] = []
            for ent in doc.ents:
                label = ent.label_.upper()
                if label not in self.allowed_types:
                    continue
                surface = ent.text.strip()
                if not surface or not normalize_entity_name(surface):
                    continue
                ents.append(
                    ExtractedEntity(
                        surface_form=surface,
                        entity_type=label,
                        char_start=ent.start_char,
                        char_end=ent.end_char,
                    )
                )
            results.append(ents)
        return results


def extract_entities(
    texts: list[str],
    model: str = "en_ner_bc5cdr_md",
    batch_size: int = 256,
    allowed_types: tuple[str, ...] = ("DISEASE", "CHEMICAL"),
) -> list[list[ExtractedEntity]]:
    """Convenience wrapper around :class:`BiomedicalEntityExtractor`."""
    extractor = BiomedicalEntityExtractor(model=model, allowed_types=allowed_types)
    return extractor.extract_batch(texts, batch_size=batch_size)
