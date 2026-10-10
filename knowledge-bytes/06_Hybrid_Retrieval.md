# 06 — Hybrid Retrieval

## 1. What this component does

Takes a query string and returns ranked chunk candidates by combining three independent signals — vector similarity, entity-graph expansion, and semantic-graph expansion — fused into one deterministic ranking.

## 2. Why it exists

No single signal is enough: vector search misses paraphrased entities, entity search misses semantically-related-but-entity-free chunks. Hybrid retrieval covers all three failure modes, and the measured result (Recall@5 0.9850 with reranking) proves the combination beats any part alone.

## 3. Where it lives in the repository

- `src/sgrag/retrieval/retriever.py` — `HybridRetriever.retrieve()`: the 7-stage orchestrator
- `src/sgrag/retrieval/vector.py` — `build_vector_query()`, `vector_search()`: Neo4j vector index Cypher
- `src/sgrag/retrieval/graph.py` — `expand_entities()`, `expand_semantic()`, `entity_graph_score()`
- `src/sgrag/retrieval/fusion.py` — `merge_candidates()`, `fuse_weighted()`, `fuse_rrf()`, `fuse_candidates()`
- `src/sgrag/retrieval/query.py` — `embed_query()`, `extract_query_entities()`
- `src/sgrag/retrieval/models.py` — `RetrievalResult`
- `src/sgrag/retrieval/config.py` — `RetrievalConfig` (all bounds live here)
- `configs/retrieval.yaml` — the actual tuned values

## 4. Important classes/functions

- **`HybridRetriever.retrieve(query)`** — returns `(list[RetrievalResult], diagnostics)`. Seven stages: embed → vector search → entity extraction → entity expansion → semantic expansion → fusion → rerank. Never loads 53k embeddings — the vector search runs index-side in Neo4j.
- **`vector_search(run, qvec, index, top_k, min_score)`** — Cypher against `chunk_embedding_idx`, cosine, bounded top-k (default 20).
- **`expand_entities(run, keys, top_k, hub_cap)`** — for each query entity key: `MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)`, top-k chunks per entity. **Hub cap:** entities with mention-degree > 1000 are skipped entirely (a disease mentioned in 5,000 chunks adds noise, not signal).
- **`expand_semantic(run, chunk_ids, top_k)`** — 1-hop `SEMANTIC_SIMILAR` from each vector hit (default 3 per hit).
- **`fuse_candidates(...)`** — merges the three candidate lists by `chunk_id`, then `fuse_weighted` (default weights 0.6/0.25/0.15) or `fuse_rrf`; deterministic tie-break by `chunk_id`.
- **`RetrievalResult`** — carries `chunk_id`, `pmid`, `text`, `title`, per-source scores, `retrieval_sources`, `fused_score`, `rank`, and later `reranker_score`/`final_score`.

## 5. Step-by-step execution

```text
retrieve("What drugs are associated with breast cancer treatment?")
  1. embed_query() → 384-dim vector                        [~ms]
  2. vector_search() → 20 chunks (cosine vs chunk_embedding_idx)
  3. extract_query_entities() → ["disease:breast cancer"]
  4a. expand_entities() → ≤15 chunks mentioning the entity (hub-capped)
  4b. expand_semantic() → ≤3 similar chunks per vector hit (1 hop)
  5. fuse_candidates() → merged, weighted 0.6/0.25/0.15, ranked
  6. TITLE_LOOKUP → batched title fetch (no N+1)
  7. reranker.rerank() → cross-encoder rescores top-50
  → (results, diagnostics with per-stage ms + counts)
```

## 6. Data flow

`str query` → `vector[384]` + `list[entity keys]` → three `list[chunk dict]` candidate sets → `list[RetrievalResult]` (fused, ranked) → reranked `list[RetrievalResult]`. The diagnostics dict (`embed_ms`, `n_vector`, `n_entity`, …) flows to the UI.

## 7. How it connects to the rest of the system

Called per subquery by `GroundedGenerator._retrieve_all()`; its `RetrievalResult` list becomes evidence via `pack_evidence()`. All bounds come from `RetrievalConfig` (YAML + env). It owns the Neo4j driver for the query path.

## 8. Important design decisions

- **Bounded everything:** top-k at every stage + hub-degree cap. Without the cap, a common entity ("cancer") would flood candidates.
- **Weights 0.6/0.25/0.15:** vector dominates (it's the most precise), entity and semantic add recall. Configurable per experiment.
- **Deterministic fusion:** ties broken by `chunk_id` — same query, same ranking, every run.
- **Index-side vector search:** the 53k×384 matrix never enters Python; Neo4j's index does the cosine math.
- **Diagnostics by design:** per-stage timings/counts aren't logging — they're the UI's "retrieval intelligence" panel.

## 9. Example from the actual project

Query: *"What are the effects of obesity on cardiovascular disease?"*
1. Embedded → vector search returns 20 chunks about obesity/cardiovascular topics.
2. Entities extracted: `disease:obesity`, `disease:cardiovascular disease`.
3. Entity expansion: chunks *mentioning* those entities — including ones whose wording differs from the query.
4. Semantic expansion: chunks similar to the vector hits (1 hop).
5. Fusion merges (deduplicating by chunk_id), weights, ranks → top candidates → cross-encoder reranks top 50 → final list.

## 10. Common misunderstandings

- **"Hybrid means vector + keyword."** Here it means vector + entity-graph + semantic-graph — three Neo4j-backed signals, no BM25.
- **"More candidates is always better."** The bounds exist because noise hurts: hub entities and far hops add irrelevant chunks that dilute fusion.
- **"Fusion is averaging."** Weighted fusion combines *normalized per-source scores*; RRF combines *ranks*. Both are deterministic and configurable.

## 11. Professor Viva Questions

- What are the three retrieval signals? Why three?
- What is the hub cap and why does it exist?
- How do you combine the signals? What are the weights?
- Why is vector search still the highest-weighted signal?
- What does "deterministic ranking" mean here?
- Where do the 53,091 embeddings live during search? (Answer: in Neo4j — never loaded into Python.)

## 12. Natural Spoken Answers

**"Explain your hybrid retrieval."**
> "Sir, for each query we run three searches in parallel. First, vector search — embed the query and take the top 20 chunks from Neo4j's vector index. Second, we extract biomedical entities from the query with scispaCy and walk MENTIONS edges to find chunks discussing those entities — skipping hub entities mentioned in over a thousand chunks, since those add noise. Third, we take one hop along SEMANTIC_SIMILAR edges from the vector hits. Then we fuse the three lists with weights 0.6, 0.25, 0.15, breaking ties deterministically, and rerank the top 50 with a cross-encoder."

## 13. Knowledge Bites

- `HybridRetriever.retrieve()` = 7 stages, returns (results, diagnostics).
- Three signals: vector (top-20) + entity expansion (top-15/entity) + semantic 1-hop (top-3/hit).
- Hub cap 1000: ultra-common entities are skipped, not expanded.
- Fusion: weighted 0.6/0.25/0.15 (default) or RRF; ties → chunk_id.
- Vector search runs index-side; 53k embeddings never enter Python.
- `RetrievalResult` carries per-source scores + provenance through the pipeline.
- Diagnostics dict powers the UI's retrieval intelligence panel.
- All bounds in `configs/retrieval.yaml` — tunable per experiment.
