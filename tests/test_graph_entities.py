"""Unit tests for entity normalization and mention aggregation (no live Neo4j)."""

from sgrag.graph.entities import (
    ExtractedEntity,
    normalize_entity,
    normalize_entity_name,
    normalized_key,
)
from sgrag.graph.ingest import aggregate_entities, build_entity_rows


def test_normalize_alzheimers_variants_agree():
    keys = {
        normalized_key("DISEASE", "Alzheimer's Disease"),
        normalized_key("DISEASE", "alzheimer disease"),
        normalized_key("DISEASE", "ALZHEIMER'S DISEASE"),
    }
    assert len(keys) == 1


def test_normalize_rules():
    assert normalize_entity_name("  Type-2   Diabetes\n") == "type 2 diabetes"
    assert normalize_entity_name("IL-6") == "il 6"
    assert normalize_entity_name("Cancers'") == "cancers"


def test_key_includes_type():
    assert normalized_key("DISEASE", "diabetes") != normalized_key("CHEMICAL", "diabetes")


def test_key_deterministic():
    assert normalized_key("DISEASE", "Asthma") == normalized_key("disease", "  asthma ")


def test_normalize_entity_fields():
    d = normalize_entity("disease", "Asthma")
    assert d["entity_type"] == "DISEASE"
    assert d["normalized_key"] == "DISEASE:asthma"


def test_aggregate_entities_canonical_and_aliases():
    mentions = [
        ("c1", ExtractedEntity("Diabetes", "DISEASE", 0, 8)),
        ("c2", ExtractedEntity("diabetes", "DISEASE", 5, 13)),
        ("c2", ExtractedEntity("Diabetes", "DISEASE", 20, 28)),
        ("c3", ExtractedEntity("Aspirin", "CHEMICAL", 0, 7)),
    ]
    entity_rows, mention_rows = aggregate_entities(mentions)
    by_key = {r["normalized_key"]: r for r in entity_rows}
    assert set(by_key) == {"DISEASE:diabetes", "CHEMICAL:aspirin"}
    d = by_key["DISEASE:diabetes"]
    assert d["canonical_name"] == "Diabetes"  # most frequent surface form
    assert d["aliases"] == ["Diabetes", "diabetes"]
    assert len(mention_rows) == 4
    assert mention_rows[0]["chunk_id"] == "c1"
    assert mention_rows[0]["surface_form"] == "Diabetes"


def test_aggregate_canonical_tiebreak_deterministic():
    mentions = [
        ("c1", ExtractedEntity("b form", "DISEASE", 0, 6)),
        ("c2", ExtractedEntity("a form", "DISEASE", 0, 6)),
    ]
    entity_rows, _ = aggregate_entities(mentions)
    assert entity_rows[0]["canonical_name"] == "a form"  # lexicographic tie-break


def test_build_entity_rows_from_counters():
    from collections import Counter
    key_surfaces = {
        "DISEASE:diabetes": Counter({"Diabetes": 2, "diabetes": 1}),
        "CHEMICAL:aspirin": Counter({"Aspirin": 1}),
    }
    key_types = {"DISEASE:diabetes": "DISEASE", "CHEMICAL:aspirin": "CHEMICAL"}
    rows = build_entity_rows(key_surfaces, key_types)
    by_key = {r["normalized_key"]: r for r in rows}
    assert by_key["DISEASE:diabetes"]["canonical_name"] == "Diabetes"
    assert by_key["DISEASE:diabetes"]["aliases"] == ["Diabetes", "diabetes"]
    assert by_key["CHEMICAL:aspirin"]["entity_type"] == "CHEMICAL"


def test_empty_mentions():
    assert aggregate_entities([]) == ([], [])
