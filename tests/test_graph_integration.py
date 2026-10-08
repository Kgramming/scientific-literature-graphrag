"""Integration tests against a live Neo4j (skipped unless SGRAG_TEST_NEO4J=1).

Test data uses a TEST- prefix and a nonsense entity name so it can never
collide with the real graph; teardown removes exactly the test nodes.

Run: SGRAG_NEO4J_PASSWORD=... SGRAG_TEST_NEO4J=1 \
     .venv/bin/python -m pytest tests/test_graph_integration.py -q
"""

import os

import pytest

from sgrag.graph.config import load_graph_config
from sgrag.graph.entities import ExtractedEntity
from sgrag.graph.ingest import GraphStore, aggregate_entities
from sgrag.graph.schema import build_article_params, build_chunk_params

pytestmark = pytest.mark.skipif(
    os.environ.get("SGRAG_TEST_NEO4J") != "1", reason="needs live Neo4j"
)

TEST_PMID = "TEST-999"
TEST_CHUNK = "TEST-999:semantic:0"


@pytest.fixture()
def store():
    cfg_path = os.environ.get("SGRAG_GRAPH_CONFIG", "configs/graph_build.yaml")
    cfg = load_graph_config(cfg_path)
    with GraphStore(cfg) as s:
        s.ensure_schema()
        yield s
        # teardown: remove exactly the test nodes (chunks first, then
        # article; test-only orphan entities last)
        s.query(
            "MATCH (c:Chunk) WHERE c.chunk_id STARTS WITH 'TEST-' DETACH DELETE c")
        s.query(
            "MATCH (a:Article) WHERE a.pmid STARTS WITH 'TEST-' DETACH DELETE a")
        s.query(
            "MATCH (e:Entity) WHERE e.normalized_key STARTS WITH 'DISEASE:xyzzytest' "
            "AND NOT ()-[:MENTIONS]->(e) DELETE e")


def test_idempotent_roundtrip(store):
    articles = [build_article_params(
        {"pmid": TEST_PMID, "title": "T", "journal": "J", "year": 2020, "abstract": "A"})]
    chunks = [build_chunk_params(
        {"chunk_id": TEST_CHUNK, "pmid": TEST_PMID, "strategy": "semantic_cluster",
         "text": "Xyzzytest disease case report.", "token_count": 4,
         "sentence_start": 0, "sentence_end": 1}, [0.1] * 384)]
    mentions = [(TEST_CHUNK, ExtractedEntity("Xyzzytest disease", "DISEASE", 0, 16))]
    entity_rows, mention_rows = aggregate_entities(mentions)
    assert entity_rows[0]["normalized_key"] == "DISEASE:xyzzytest disease"

    before = store.counts()
    for _ in range(2):  # run twice: nothing may duplicate
        store.upsert_articles(articles)
        store.upsert_chunks(chunks)
        store.upsert_entities(entity_rows)
        store.upsert_mentions(mention_rows)
    after = store.counts()

    assert after["articles"] == before["articles"] + 1
    assert after["chunks"] == before["chunks"] + 1
    assert after["entities"] == before["entities"] + 1
    assert after["has_chunk"] == before["has_chunk"] + 1
    assert after["mentions"] == before["mentions"] + 1
