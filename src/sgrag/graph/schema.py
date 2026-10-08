"""Graph schema: node/relationship definitions and idempotent DDL Cypher.

Nodes
-----
(:Article {pmid, title, journal, year, ...})
(:Chunk {chunk_id, pmid, strategy, text, token_count, sentence_start, sentence_end,
         embedding, ...})
(:Entity {normalized_key, canonical_name, entity_type, aliases})

Relationships
-------------
(:Article)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:MENTIONS {surface_form, char_start, char_end}]->(:Entity)
(:Chunk)-[:SEMANTIC_SIMILAR {score, rank, model}]->(:Chunk)

All DDL uses IF NOT EXISTS so the build is safely rerunnable.
"""

from __future__ import annotations

# --- uniqueness constraints -------------------------------------------------
CONSTRAINT_STATEMENTS: tuple[str, ...] = (
    "CREATE CONSTRAINT article_pmid IF NOT EXISTS FOR (a:Article) REQUIRE a.pmid IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.normalized_key IS UNIQUE",
)

# --- supporting indexes ------------------------------------------------------
INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX chunk_pmid IF NOT EXISTS FOR (c:Chunk) ON (c.pmid)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
)

VECTOR_INDEX_STATEMENT = (
    "CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
    "OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, "
    "`vector.similarity_function`: '{similarity_function}'}}}}"
)

# --- batched ingestion Cypher (UNWIND, MERGE => idempotent) -------------------

UPSERT_ARTICLES = """
UNWIND $rows AS row
MERGE (a:Article {pmid: row.pmid})
SET a.title = row.title,
    a.journal = row.journal,
    a.year = row.year,
    a.abstract = row.abstract
"""

UPSERT_CHUNKS = """
UNWIND $rows AS row
MATCH (a:Article {pmid: row.pmid})
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.pmid = row.pmid,
    c.strategy = row.strategy,
    c.text = row.text,
    c.token_count = row.token_count,
    c.sentence_start = row.sentence_start,
    c.sentence_end = row.sentence_end,
    c.embedding = row.embedding
MERGE (a)-[:HAS_CHUNK]->(c)
"""

UPSERT_ENTITIES = """
UNWIND $rows AS row
MERGE (e:Entity {normalized_key: row.normalized_key})
SET e.canonical_name = row.canonical_name,
    e.entity_type = row.entity_type,
    e.aliases = row.aliases
"""

UPSERT_MENTIONS = """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
MATCH (e:Entity {normalized_key: row.normalized_key})
MERGE (c)-[r:MENTIONS]->(e)
SET r.surface_form = row.surface_form,
    r.char_start = row.char_start,
    r.char_end = row.char_end
"""

UPSERT_SIMILAR = """
UNWIND $rows AS row
MATCH (a:Chunk {chunk_id: row.src_chunk_id})
MATCH (b:Chunk {chunk_id: row.dst_chunk_id})
MERGE (a)-[r:SEMANTIC_SIMILAR]->(b)
SET r.score = row.score,
    r.rank = row.rank,
    r.model = row.model
"""


def vector_index_statement(name: str, dimensions: int, similarity_function: str) -> str:
    return VECTOR_INDEX_STATEMENT.format(
        name=name, dimensions=dimensions, similarity_function=similarity_function
    )


def build_article_params(record: dict) -> dict:
    """Project a corpus record onto the Article node property map."""
    return {
        "pmid": str(record["pmid"]),
        "title": record.get("title") or "",
        "journal": record.get("journal") or "",
        "year": record.get("year"),
        "abstract": record.get("abstract") or "",
    }


def build_chunk_params(chunk: dict, embedding: list[float] | None) -> dict:
    """Project a chunk record (+ embedding) onto the Chunk node property map."""
    return {
        "chunk_id": chunk["chunk_id"],
        "pmid": str(chunk["pmid"]),
        "strategy": chunk.get("strategy", ""),
        "text": chunk.get("text", ""),
        "token_count": chunk.get("token_count"),
        "sentence_start": chunk.get("sentence_start"),
        "sentence_end": chunk.get("sentence_end"),
        "embedding": embedding,
    }


def build_mentions_params(
    chunk_id: str, normalized_key: str, surface_form: str, char_start: int, char_end: int
) -> dict:
    return {
        "chunk_id": chunk_id,
        "normalized_key": normalized_key,
        "surface_form": surface_form,
        "char_start": char_start,
        "char_end": char_end,
    }


def build_similarity_params(
    src_chunk_id: str, dst_chunk_id: str, score: float, rank: int, model: str
) -> dict:
    return {
        "src_chunk_id": src_chunk_id,
        "dst_chunk_id": dst_chunk_id,
        "score": float(score),
        "rank": int(rank),
        "model": model,
    }
