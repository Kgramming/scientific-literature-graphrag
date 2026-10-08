"""Unit tests for graph schema Cypher and param builders (no live Neo4j)."""

from sgrag.graph import schema


def test_constraints_are_idempotent_unique():
    text = "\n".join(schema.CONSTRAINT_STATEMENTS)
    assert "IF NOT EXISTS" in text
    assert "article_pmid" in text and "a.pmid IS UNIQUE" in text
    assert "chunk_id" in text and "c.chunk_id IS UNIQUE" in text
    assert "entity_key" in text and "e.normalized_key IS UNIQUE" in text


def test_vector_index_statement():
    stmt = schema.vector_index_statement("chunk_embedding_idx", 384, "cosine")
    assert "CREATE VECTOR INDEX chunk_embedding_idx IF NOT EXISTS" in stmt
    assert "`vector.dimensions`: 384" in stmt
    assert "`vector.similarity_function`: 'cosine'" in stmt
    assert "ON (c.embedding)" in stmt


def test_ingest_cypher_uses_unwind_merge():
    for name in ("UPSERT_ARTICLES", "UPSERT_CHUNKS", "UPSERT_ENTITIES",
                 "UPSERT_MENTIONS", "UPSERT_SIMILAR"):
        stmt = getattr(schema, name)
        assert "UNWIND $rows AS row" in stmt, name
        assert "MERGE" in stmt, name


def test_chunk_ingest_links_article():
    assert "MATCH (a:Article {pmid: row.pmid})" in schema.UPSERT_CHUNKS
    assert "MERGE (a)-[:HAS_CHUNK]->(c)" in schema.UPSERT_CHUNKS


def test_similarity_edge_properties():
    assert "r.score = row.score" in schema.UPSERT_SIMILAR
    assert "r.rank = row.rank" in schema.UPSERT_SIMILAR
    assert "r.model = row.model" in schema.UPSERT_SIMILAR


def test_build_article_params():
    p = schema.build_article_params(
        {"pmid": 123, "title": "T", "journal": "J", "year": 2020, "abstract": "A"})
    assert p["pmid"] == "123" and p["title"] == "T" and p["year"] == 2020


def test_build_chunk_params_deterministic():
    chunk = {"chunk_id": "1:semantic:0", "pmid": "1", "strategy": "semantic_cluster",
             "text": "hi", "token_count": 2, "sentence_start": 0, "sentence_end": 1}
    p = schema.build_chunk_params(chunk, [0.1, 0.2])
    assert p["chunk_id"] == "1:semantic:0"
    assert p["embedding"] == [0.1, 0.2]
    assert p["sentence_start"] == 0 and p["sentence_end"] == 1


def test_build_similarity_params():
    p = schema.build_similarity_params("a", "b", 0.9, 1, "all-MiniLM-L6-v2")
    assert p == {"src_chunk_id": "a", "dst_chunk_id": "b", "score": 0.9,
                 "rank": 1, "model": "all-MiniLM-L6-v2"}
