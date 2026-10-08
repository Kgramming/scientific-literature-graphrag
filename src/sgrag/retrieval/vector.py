"""Vector retrieval through the Neo4j vector index (Phase 3B: chunk_embedding_idx).

The 53,091 chunk embeddings stay in Neo4j; retrieval happens index-side via
``db.index.vector.queryNodes`` — Python never loads the full embedding matrix.
"""

from __future__ import annotations

import numpy as np

VECTOR_QUERY = """
CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
YIELD node AS c, score
WHERE score >= $min_score
RETURN c.chunk_id AS chunk_id,
       c.pmid AS pmid,
       c.strategy AS strategy,
       c.text AS text,
       c.token_count AS token_count,
       score AS vector_score
ORDER BY vector_score DESC, chunk_id ASC
"""


def build_vector_query(index_name: str) -> str:
    """Render the vector-search Cypher for an index name (no string injection
    of user data: only the configured index name is interpolated)."""
    if not index_name.replace("_", "").isalnum():
        raise ValueError(f"unsafe index name: {index_name!r}")
    return VECTOR_QUERY.replace("$index_name", f"'{index_name}'")


def parse_vector_rows(rows: list[dict]) -> list[dict]:
    """Parse raw vector-search rows into candidate dicts with 1-based rank."""
    out: list[dict] = []
    for rank, row in enumerate(rows, start=1):
        out.append(
            {
                "chunk_id": row["chunk_id"],
                "pmid": str(row["pmid"]),
                "strategy": row.get("strategy") or "semantic_cluster",
                "text": row.get("text") or "",
                "token_count": row.get("token_count"),
                "vector_score": float(row["vector_score"]),
                "vector_rank": rank,
                "sources": {"vector"},
                "matched_entities": [],
            }
        )
    return out


def vector_search(
    run_query_fn,
    embedding: np.ndarray,
    index_name: str,
    top_k: int,
    min_score: float = 0.0,
) -> list[dict]:
    """Run vector search via callback ``run_query_fn(cypher, **params)``.

    The callback indirection keeps this unit-testable without a live Neo4j.
    """
    cypher = build_vector_query(index_name)
    rows = run_query_fn(
        cypher,
        embedding=[float(x) for x in np.asarray(embedding, dtype=np.float32)],
        top_k=int(top_k),
        min_score=float(min_score),
    )
    return parse_vector_rows([dict(r) for r in rows])
