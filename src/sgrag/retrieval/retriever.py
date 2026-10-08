"""HybridRetriever: the Phase 4 retrieval orchestrator.

Pipeline:
    query
    -> embed (all-MiniLM-L6-v2, 384-dim)
    -> Neo4j vector search (chunk_embedding_idx, cosine, bounded top-k)
    -> query entity extraction (Phase 3B scispaCy + normalization)
    -> bounded graph expansion (MENTIONS, hub-capped; SEMANTIC_SIMILAR 1-hop)
    -> fusion (weighted | RRF), deterministic ranking
    -> optional reranker (Phase 5 interface; default pass-through)

No LLM, no generation, no UI. Independently testable.
"""

from __future__ import annotations

import time

from neo4j import Driver, GraphDatabase

from sgrag.retrieval import graph as graph_mod
from sgrag.retrieval import vector as vector_mod
from sgrag.retrieval.config import RetrievalConfig, load_retrieval_config
from sgrag.retrieval.fusion import fuse_candidates
from sgrag.retrieval.models import RetrievalResult
from sgrag.retrieval.query import embed_query, extract_query_entities
from sgrag.retrieval.rerank import (
    CrossEncoderReranker,
    NoOpReranker,
    Reranker,
    RerankerError,
)

TITLE_LOOKUP = """
UNWIND $pmids AS pmid
MATCH (a:Article {pmid: pmid})
RETURN a.pmid AS pmid, a.title AS title
"""


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot proceed (e.g. Neo4j unavailable)."""


class HybridRetriever:
    def __init__(
        self,
        config: RetrievalConfig | str,
        driver: Driver | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = (
            load_retrieval_config(config) if isinstance(config, str) else config
        )
        self._driver = driver
        self._owns_driver = driver is None
        # Reranker selection: explicit instance wins; otherwise the config
        # decides. Default is the Phase 4 pass-through (backward compatible).
        if reranker is not None:
            self.reranker: Reranker = reranker
        elif self.config.reranker.enabled:
            rc = self.config.reranker
            self.reranker = CrossEncoderReranker(
                model=rc.model, candidate_k=rc.candidate_k,
                batch_size=rc.batch_size)
        else:
            self.reranker = NoOpReranker()

    # -- connection -----------------------------------------------------
    def _connect(self) -> Driver:
        if self._driver is None:
            cfg = self.config
            try:
                self._driver = GraphDatabase.driver(
                    cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
                )
                self._driver.verify_connectivity()
            except Exception as exc:  # noqa: BLE001 - wrap any driver failure
                raise RetrievalError(
                    f"Neo4j unavailable at {cfg.neo4j_uri}: {exc}"
                ) from exc
            self._owns_driver = True
        return self._driver

    def _run(self, cypher: str, **params):
        driver = self._connect()
        try:
            with driver.session(database=self.config.neo4j_database) as session:
                return list(session.run(cypher, **params))
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Neo4j query failed: {exc}") from exc

    def close(self) -> None:
        if self._driver is not None and self._owns_driver:
            self._driver.close()
            self._driver = None

    # -- pipeline --------------------------------------------------------
    def retrieve(self, query: str) -> tuple[list[RetrievalResult], dict]:
        """Run the full hybrid retrieval pipeline.

        Returns (results, diagnostics) where diagnostics carries per-stage
        counts and latency for smoke tests / debugging.
        """
        t0 = time.perf_counter()
        diag: dict = {"query": query}
        cfg = self.config

        # 1. query embedding
        t = time.perf_counter()
        qvec = embed_query(query, cfg.embedding_model)
        diag["embed_ms"] = round((time.perf_counter() - t) * 1000, 1)

        # 2. vector search (index-side; never loads 53k embeddings)
        t = time.perf_counter()
        vector_cands = vector_mod.vector_search(
            self._run,
            qvec,
            cfg.vector_index_name,
            cfg.vector_top_k,
            cfg.min_vector_score,
        )
        diag["vector_ms"] = round((time.perf_counter() - t) * 1000, 1)
        diag["n_vector"] = len(vector_cands)

        # 3. query entities (Phase 3B extractor + normalization)
        t = time.perf_counter()
        qents = extract_query_entities(query, cfg.entity_model)
        diag["entity_ms"] = round((time.perf_counter() - t) * 1000, 1)
        diag["query_entities"] = [e["normalized_key"] for e in qents.items]

        # 4a. entity -> MENTIONS -> chunk (bounded, hub-capped)
        entity_cands: list[dict] = []
        if qents.keys:
            t = time.perf_counter()
            entity_cands = graph_mod.expand_entities(
                self._run, qents.keys, cfg.entity_top_k, cfg.hub_degree_cap
            )
            diag["entity_expand_ms"] = round((time.perf_counter() - t) * 1000, 1)
        diag["n_entity"] = len(entity_cands)

        # 4b. chunk -> SEMANTIC_SIMILAR -> chunk (1 hop, bounded)
        semantic_cands: list[dict] = []
        if cfg.graph_hop_limit >= 1 and vector_cands:
            t = time.perf_counter()
            semantic_cands = graph_mod.expand_semantic(
                self._run,
                [c["chunk_id"] for c in vector_cands],
                cfg.semantic_similar_top_k,
            )
            diag["semantic_expand_ms"] = round((time.perf_counter() - t) * 1000, 1)
        diag["n_semantic"] = len(semantic_cands)

        # 5. fusion + ranking (deterministic)
        t = time.perf_counter()
        results = fuse_candidates(
            vector_cands, entity_cands, semantic_cands, cfg.fusion, cfg.max_candidates
        )
        diag["fusion_ms"] = round((time.perf_counter() - t) * 1000, 1)
        diag["n_fused"] = len(results)
        diag["n_multi_source"] = sum(1 for r in results if len(r.retrieval_sources) > 1)

        # 6. titles (single batched lookup, no N+1)
        if results:
            pmids = sorted({r.pmid for r in results})
            rows = self._run(TITLE_LOOKUP, pmids=pmids)
            titles = {str(r["pmid"]): r["title"] for r in rows}
            for r in results:
                r.title = titles.get(r.pmid, "")

        # 7. optional reranker (Phase 5A; default pass-through)
        t = time.perf_counter()
        diag["reranker"] = self.reranker.name
        try:
            results = self.reranker.rerank(query, results)
        except RerankerError as exc:
            if self.config.reranker.fallback_to_fusion:
                print(f"[rerank] {exc} -- falling back to fused ranking")
                results = NoOpReranker().rerank(query, results)
                diag["reranker_fallback"] = True
            else:
                raise RetrievalError(f"reranker failed: {exc}") from exc
        diag["rerank_ms"] = round((time.perf_counter() - t) * 1000, 1)
        for rank, r in enumerate(results, start=1):
            r.rank = rank
            # final_* mirrors the reranked order; without a reranker the
            # fused ranking stands as final.
            if r.final_rank is None:
                r.final_rank = rank
                r.final_score = r.fused_score

        diag["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return results, diag

    def check_index(self) -> dict:
        """Verify the Phase 3B vector index is ONLINE."""
        rows = self._run(
            "SHOW INDEXES YIELD name, type, state "
            "WHERE name = $name RETURN name, type, state",
            name=self.config.vector_index_name,
        )
        if not rows:
            raise RetrievalError(
                f"vector index '{self.config.vector_index_name}' not found"
            )
        return dict(rows[0])

    # context-manager sugar
    def __enter__(self) -> HybridRetriever:
        self._connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
