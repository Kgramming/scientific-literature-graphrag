"""Bounded graph expansion for retrieval (Phase 4).

Two expansion paths, both bounded:

1. Entity -> MENTIONS -> Chunk: for each query entity key, fetch mentioning
   chunks. Entities whose mention-degree exceeds hub_degree_cap are SKIPPED
   (never traverse a hub's full neighborhood). Per-entity results are capped
   at entity_top_k (deterministic chunk_id order).
2. Chunk -> SEMANTIC_SIMILAR -> Chunk: expand from vector hits, taking at most
   semantic_similar_top_k outgoing edges per source chunk (the Phase 3B
   bounded rule guarantees <= 5 exist).

Every candidate keeps provenance (sources, matched entities, edge scores).
"""

from __future__ import annotations

import math

ENTITY_MENTIONS_QUERY = """
MATCH (e:Entity {normalized_key: $key})
WITH e, count { (e)<-[:MENTIONS]-() } AS degree
WHERE degree <= $hub_cap
MATCH (e)<-[m:MENTIONS]-(c:Chunk)
RETURN c.chunk_id AS chunk_id,
       c.pmid AS pmid,
       c.strategy AS strategy,
       c.text AS text,
       c.token_count AS token_count,
       m.surface_form AS surface_form,
       degree AS entity_degree,
       $key AS entity_key
ORDER BY chunk_id ASC
LIMIT $limit
"""

SEMANTIC_EXPAND_QUERY = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})-[r:SEMANTIC_SIMILAR]->(n:Chunk)
WITH cid, n, r.score AS s ORDER BY s DESC, n.chunk_id ASC
WITH cid, collect({node: n, score: s})[0..$per_chunk] AS top
UNWIND top AS t
WITH t.node AS n, t.score AS s, cid
WITH n, max(s) AS sem_score, collect(DISTINCT cid) AS src_ids
RETURN n.chunk_id AS chunk_id,
       n.pmid AS pmid,
       n.strategy AS strategy,
       n.text AS text,
       n.token_count AS token_count,
       sem_score AS semantic_score,
       src_ids AS src_chunk_ids
ORDER BY semantic_score DESC, chunk_id ASC
"""


def entity_graph_score(entity_degree: int) -> float:
    """Importance weight for an entity match: rare entities count more.

    graph_score = 1 / (1 + ln(1 + degree)) in (0, 1]. A singleton entity
    scores ~1.0; the largest hub ('pain', 635 mentions) scores ~0.13.
    Not every CHEMICAL/DISEASE span is equally important.
    """
    return 1.0 / (1.0 + math.log1p(max(int(entity_degree), 0)))


def expand_entities(
    run_query_fn,
    entity_keys: list[str],
    entity_top_k: int,
    hub_degree_cap: int,
) -> list[dict]:
    """Expand query entities to chunks via MENTIONS (bounded, hub-safe)."""
    out: list[dict] = []
    for key in entity_keys:
        rows = run_query_fn(
            ENTITY_MENTIONS_QUERY,
            key=key,
            hub_cap=int(hub_degree_cap),
            limit=int(entity_top_k),
        )
        for row in rows:
            row = dict(row)
            out.append(
                {
                    "chunk_id": row["chunk_id"],
                    "pmid": str(row["pmid"]),
                    "strategy": row.get("strategy") or "semantic_cluster",
                    "text": row.get("text") or "",
                    "token_count": row.get("token_count"),
                    "graph_score": entity_graph_score(row["entity_degree"]),
                    "sources": {"entity"},
                    "matched_entities": [row["entity_key"]],
                    "surface_form": row.get("surface_form"),
                }
            )
    return out


def expand_semantic(
    run_query_fn,
    source_chunk_ids: list[str],
    per_chunk: int = 3,
) -> list[dict]:
    """Expand vector hits through SEMANTIC_SIMILAR (1 hop).

    Takes at most ``per_chunk`` outgoing edges per source chunk (Phase 3B
    materialized at most 5); targets reached from several sources keep the
    max edge score and the list of source chunk_ids as provenance.
    """
    if not source_chunk_ids:
        return []
    rows = run_query_fn(
        SEMANTIC_EXPAND_QUERY, chunk_ids=list(source_chunk_ids), per_chunk=int(per_chunk))
    out: list[dict] = []
    for row in rows:
        row = dict(row)
        out.append(
            {
                "chunk_id": row["chunk_id"],
                "pmid": str(row["pmid"]),
                "strategy": row.get("strategy") or "semantic_cluster",
                "text": row.get("text") or "",
                "token_count": row.get("token_count"),
                "semantic_score": float(row["semantic_score"]),
                "sources": {"semantic_graph"},
                "matched_entities": [],
                "src_chunk_ids": row.get("src_chunk_ids") or [],
            }
        )
    return out
