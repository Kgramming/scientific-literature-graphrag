"""Batch ingestion pipeline into Neo4j.

Pipeline: Articles -> Chunks (+embeddings) -> entity extraction ->
normalization -> Entity nodes -> MENTIONS -> SEMANTIC_SIMILAR.

All writes use UNWIND batches with MERGE, so the build is idempotent and a
second run duplicates nothing.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Iterable

from sgrag.graph import schema
from sgrag.graph.config import GraphConfig
from sgrag.graph.entities import ExtractedEntity, normalized_key
from sgrag.graph.similarity import SimilarityEdge


def batched(iterable: Iterable[dict], size: int):
    batch: list[dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_entity_rows(
    key_surfaces: dict[str, Counter], key_types: dict[str, str]
) -> list[dict]:
    """Build Entity node rows from per-key surface-form counters.

    canonical_name = most frequent surface form (lexicographic tie-break);
    aliases = sorted distinct surface forms. Deterministic.
    """
    rows: list[dict] = []
    for key in sorted(key_surfaces):
        surfaces = key_surfaces[key]
        top = max(surfaces.values())
        canonical = sorted(s for s, c in surfaces.items() if c == top)[0]
        rows.append(
            {
                "normalized_key": key,
                "canonical_name": canonical,
                "entity_type": key_types[key],
                "aliases": sorted(surfaces),
            }
        )
    return rows


def aggregate_entities(
    mentions: list[tuple[str, ExtractedEntity]],
) -> tuple[list[dict], list[dict]]:
    """Aggregate chunk-level mentions into Entity nodes + MENTIONS rows.

    Returns (entity_rows, mention_rows). Entity identity is the deterministic
    normalized key. canonical_name = most frequent surface form (ties broken
    by lexicographic order for determinism); aliases = sorted distinct
    surface forms.
    """
    by_key: dict[str, dict] = defaultdict(lambda: {"type": "", "surfaces": Counter()})
    mention_rows: list[dict] = []
    for chunk_id, ent in mentions:
        key = normalized_key(ent.entity_type, ent.surface_form)
        slot = by_key[key]
        slot["type"] = ent.entity_type
        slot["surfaces"][ent.surface_form] += 1
        mention_rows.append(
            schema.build_mentions_params(
                chunk_id, key, ent.surface_form, ent.char_start, ent.char_end
            )
        )
    entity_rows: list[dict] = []
    for key in sorted(by_key):
        slot = by_key[key]
        surfaces = slot["surfaces"]
        top_count = max(surfaces.values())
        canonical = sorted(s for s, c in surfaces.items() if c == top_count)[0]
        entity_rows.append(
            {
                "normalized_key": key,
                "canonical_name": canonical,
                "entity_type": slot["type"],
                "aliases": sorted(surfaces),
            }
        )
    return entity_rows, mention_rows


class GraphStore:
    """Thin wrapper around the Neo4j driver with batched, idempotent writes."""

    def __init__(self, cfg: GraphConfig) -> None:
        self.cfg = cfg
        self._driver = None

    def connect(self):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("the 'neo4j' package is required for graph ingestion") from exc
        if not self.cfg.neo4j.password:
            raise RuntimeError("SGRAG_NEO4J_PASSWORD is not set; refusing to connect")
        self._driver = GraphDatabase.driver(
            self.cfg.neo4j.uri,
            auth=(self.cfg.neo4j.user, self.cfg.neo4j.password),
        )
        return self

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> GraphStore:
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    def _run(self, cypher: str, **params):
        with self._driver.session(database=self.cfg.neo4j.database) as session:
            return session.run(cypher, **params)

    def run_write(self, cypher: str, rows: list[dict], batch_size: int | None = None) -> int:
        """Execute a UNWIND write in batches; returns total rows written."""
        size = batch_size or self.cfg.ingestion.batch_size
        total = 0

        def _tx(tx, batch):
            tx.run(cypher, rows=batch)

        with self._driver.session(database=self.cfg.neo4j.database) as session:
            for batch in batched(rows, size):
                session.execute_write(_tx, batch)
                total += len(batch)
        return total

    def query(self, cypher: str, **params) -> list[dict]:
        with self._driver.session(database=self.cfg.neo4j.database) as session:
            return [dict(r) for r in session.run(cypher, **params)]

    def query_system(self, cypher: str, **params) -> list[dict]:
        """Run an admin query against the system database (e.g. CREATE/DROP DATABASE)."""
        with self._driver.session(database="system") as session:
            return [dict(r) for r in session.run(cypher, **params)]

    # -- DDL -----------------------------------------------------------------
    def ensure_schema(self) -> None:
        for stmt in schema.CONSTRAINT_STATEMENTS:
            self._run(stmt)
        for stmt in schema.INDEX_STATEMENTS:
            self._run(stmt)
        self._run(
            schema.vector_index_statement(
                self.cfg.vector_index.name,
                self.cfg.vector_index.dimensions,
                self.cfg.vector_index.similarity_function,
            )
        )

    # -- nodes / relationships ------------------------------------------------
    def upsert_articles(self, rows: list[dict]) -> int:
        return self.run_write(schema.UPSERT_ARTICLES, rows)

    def upsert_chunks(self, rows: list[dict]) -> int:
        return self.run_write(schema.UPSERT_CHUNKS, rows)

    def upsert_entities(self, rows: list[dict]) -> int:
        return self.run_write(schema.UPSERT_ENTITIES, rows)

    def upsert_mentions(self, rows: list[dict]) -> int:
        return self.run_write(schema.UPSERT_MENTIONS, rows)

    def upsert_similar(self, edges: list[SimilarityEdge]) -> int:
        rows = [
            schema.build_similarity_params(e.src_chunk_id, e.dst_chunk_id, e.score, e.rank, e.model)
            for e in edges
        ]
        return self.run_write(schema.UPSERT_SIMILAR, rows)

    def clear_graph(self) -> None:
        """Delete all nodes/relationships (schema objects are kept)."""
        self._run("MATCH (n) DETACH DELETE n")

    def counts(self) -> dict:
        q = """
        RETURN
          count { (a:Article) } AS articles,
          count { (c:Chunk) } AS chunks,
          count { (e:Entity) } AS entities,
          count { ()-[:HAS_CHUNK]->() } AS has_chunk,
          count { ()-[:MENTIONS]->() } AS mentions,
          count { ()-[:SEMANTIC_SIMILAR]->() } AS similar
        """
        rows = self.query(q)
        return dict(rows[0]) if rows else {}
