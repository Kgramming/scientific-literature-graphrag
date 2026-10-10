# 05 — Neo4j Graph and Data Model

## 1. What this component does

Stores the entire corpus as a queryable knowledge graph: articles, chunks, entities, and the relationships between them — plus a vector index for hybrid search. This is the database everything reads from at query time.

## 2. Why it exists

A vector database can answer "find text similar to X." It cannot answer "find all chunks mentioning the same disease as this chunk" or "find chunks semantically similar to my vector hits." Neo4j stores *relationships as first-class data*, which is what makes graph expansion (file 06) possible.

## 3. Where it lives in the repository

- `src/sgrag/graph/schema.py` — node/relationship definitions, constraints, indexes, idempotent DDL Cypher
- `src/sgrag/graph/ingest.py` — `GraphStore`: `ensure_schema()`, `upsert_articles/chunks/entities/mentions/similar()`, `counts()`
- `src/sgrag/graph/similarity.py` — `SimilarityEdge`: top-5 semantic edges per chunk
- `src/sgrag/graph/config.py` — connection config (reads `SGRAG_NEO4J_*` from env)
- `docker-compose.yml` — Neo4j 5.26 + APOC + GDS service definition
- `scripts/build_graph.py`, `scripts/validate_graph.py`

## 4. Important classes/functions

- **`GraphStore`** (`ingest.py`) — the write path. Context manager; `ensure_schema()` creates constraints/indexes (`IF NOT EXISTS`, safely rerunnable); `upsert_*` methods batch-write via `UNWIND` + `MERGE` (idempotent — rerunning never duplicates).
- **`CONSTRAINT_STATEMENTS`** (`schema.py`) — uniqueness on `Article.pmid`, `Chunk.chunk_id`, `Entity.normalized_key`.
- **`vector_index_statement()`** — creates `chunk_embedding_idx`: 384-dim, cosine similarity, on `Chunk.embedding`.
- **`SimilarityEdge`** (`similarity.py`) — `{source_chunk, target_chunk, score, rank, model}`; top-5 per chunk → 265,455 edges.
- **`counts()`** — live `MATCH (n:Article) RETURN count(n)` etc.; powers the UI's "live corpus stats."

## 5. Step-by-step execution (graph build)

```text
scripts/build_graph.py
  → GraphStore.ensure_schema()     [constraints + indexes + vector index]
  → upsert_articles(rows)          [MERGE (:Article {pmid})]
  → upsert_chunks(rows)            [MERGE (:Chunk {chunk_id}), SET embedding[384]]
  → upsert_entities(rows)          [MERGE (:Entity {normalized_key})]
  → upsert_mentions(rows)          [MERGE (:Chunk)-[:MENTIONS]->(:Entity)]
  → upsert_similar(edges)          [MERGE (:Chunk)-[:SEMANTIC_SIMILAR {score,rank}]->(:Chunk)]
```

## 6. Data flow

`articles + chunks + vectors + entities` → `GraphStore` → Neo4j. At query time, `HybridRetriever` issues Cypher: vector index queries, `MENTIONS` traversals, `SEMANTIC_SIMILAR` traversals.

## 7. How it connects to the rest of the system

The graph is the *single source of truth* at query time. Retrieval reads it (vector + traversals); the UI reads it (live counts via `counts()`, PyVis visualization via `build_evidence_graph()`); evaluation reads it (retrieval over the same index). Nothing at query time touches raw files.

## 8. Important design decisions

- **Why Neo4j, not just a vector DB:** vector search is one access pattern; the project needs three (vector + entity traversal + similarity traversal). Neo4j does all three, with Cypher expressing traversals declaratively.
- **Idempotent DDL and writes:** `IF NOT EXISTS` + `MERGE` mean the build is safely rerunnable — no "drop everything and pray" rebuilds.
- **Top-5 semantic edges:** full pairwise similarity over 53k chunks is ~1.4B pairs; top-5 per chunk captures the useful neighborhood at 265k edges.
- **APOC + GDS installed:** APOC for batch Cypher utilities; GDS plugin present and used in `scripts/validate_graph.py`. **Accurately:** GDS procedures are *not* called in the live retrieval path — the graph algorithms available are future scope, not current behavior.
- **Vector index inside Neo4j:** one database serves vector + graph queries, so hybrid retrieval needs no second system.

## 9. Example from the actual project

Live counts (via UI): 5,000 Articles, 53,091 Chunks, 9,945 Entities, 265,455 SEMANTIC_SIMILAR edges. A `:Chunk` node: `{chunk_id, pmid, strategy, text, token_count, embedding[384]}`. A `:MENTIONS` edge: `{surface_form: "breast cancer", char_start, char_end}`.

## 10. Common misunderstandings

- **"GDS runs the retrieval."** No — GDS is installed and used in validation; live retrieval uses the vector index + plain Cypher traversals. Say exactly this.
- **"The graph replaces vector search."** No — it *augments* it. Vector search is still the highest-weighted signal (0.6).
- **"SEMANTIC_SIMILAR means the chunks say the same thing."** It means high embedding cosine similarity — related content, not paraphrases.

## 11. Professor Viva Questions

- Why Neo4j instead of a plain vector database?
- What nodes and relationships exist? Why these?
- How is the graph populated? Is it rerunnable?
- What is the vector index? Where does it live?
- What does the graph contain that a vector DB wouldn't?
- What are APOC and GDS used for in *your* code?

## 12. Natural Spoken Answers

**"Why Neo4j?"**
> "Sir, we need three access patterns: vector similarity, entity-to-chunk traversal, and chunk-to-similar-chunk traversal. A vector database does only the first. Neo4j stores relationships as first-class data, so 'find every chunk mentioning this disease' is a single Cypher traversal, and we keep the vector index in the same database — one system for hybrid retrieval."

**"What do APOC and GDS do in your project?"**
> "Honestly, sir — APOC helps with batch Cypher operations during the build, and GDS is installed as a plugin and used in our graph validation script. The live retrieval path uses the vector index and plain traversals; GDS graph algorithms are available but not in the query path — that's future scope."

## 13. Knowledge Bites

- Nodes: `:Article` (pmid), `:Chunk` (chunk_id, 384-dim embedding), `:Entity` (normalized_key).
- Edges: `:HAS_CHUNK`, `:MENTIONS` (with surface form + offsets), `:SEMANTIC_SIMILAR` (score, rank, top-5).
- Uniqueness constraints on all three node keys; `IF NOT EXISTS` DDL.
- `chunk_embedding_idx`: cosine vector index on `Chunk.embedding`.
- All writes via `UNWIND` + `MERGE` → idempotent, rerunnable builds.
- 265,455 semantic edges (top-5 per chunk, not full pairwise).
- `GraphStore.counts()` powers the UI's live corpus statistics.
- APOC: batch utilities. GDS: installed, used in validation — not in live retrieval.
- Query time reads *only* Neo4j; raw files are never touched.
