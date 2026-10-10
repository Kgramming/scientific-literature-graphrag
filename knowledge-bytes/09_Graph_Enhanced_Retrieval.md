# 09 — Graph-Enhanced Retrieval (Why the Graph Matters)

## 1. What this component does

Explains *how* the Neo4j graph structure improves evidence discovery beyond what vector search alone can do — the two graph expansion paths, when each helps, and what the measured impact is.

## 2. Why it exists

"Why GraphRAG?" is the project's thesis question. This file is the answer: concrete mechanisms (not slogans) by which `MENTIONS` and `SEMANTIC_SIMILAR` edges find evidence that vector search misses.

## 3. Where it lives in the repository

- `src/sgrag/retrieval/graph.py` — `expand_entities()`, `expand_semantic()`, `entity_graph_score()`
- `src/sgrag/retrieval/fusion.py` — where graph signals merge with vector scores
- `src/sgrag/graph/schema.py` — the edges being traversed
- `configs/retrieval.yaml` — `entity_top_k`, `semantic_similar_top_k`, `hub_degree_cap`, `graph_hop_limit`

## 4. Important classes/functions

- **`expand_entities(run, keys, top_k=15, hub_cap=1000)`** — Cypher: `MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk) WHERE e.normalized_key IN $keys`, top-k chunks per entity ordered by mention relevance. Skips entities with degree > hub_cap.
- **`expand_semantic(run, chunk_ids, top_k=3)`** — Cypher: `MATCH (c:Chunk)-[:SEMANTIC_SIMILAR]->(s:Chunk) WHERE c.chunk_id IN $ids`, top-k outgoing edges per vector hit (1 hop only, `graph_hop_limit=1`).
- **`entity_graph_score(degree)`** — downweights chunks from ultra-connected entities so hub mentions don't dominate.

## 5. Step-by-step execution (the two paths)

**Entity path** — *"What drugs treat hypertension?"*
```text
query entities: ["disease:hypertension"]
  → MATCH (e:Entity {normalized_key:"disease:hypertension"})<-[:MENTIONS]-(c:Chunk)
  → top 15 chunks mentioning hypertension
  → includes chunks whose wording differs entirely from the query
     (e.g. "antihypertensive agents were administered...")
```

**Semantic path** — vector hits seed graph walking:
```text
vector hits: [chunk_A, chunk_B, ...] (top 20)
  → MATCH (c:Chunk)-[:SEMANTIC_SIMILAR]->(s:Chunk) WHERE c IN hits
  → ≤3 neighbors per hit
  → finds chunks similar to *good* chunks, even if dissimilar to the query
```

## 6. Data flow

`query entities` → `MENTIONS` traversal → `entity candidates`. `vector hit IDs` → `SEMANTIC_SIMILAR` traversal → `semantic candidates`. Both merge with vector candidates in `fuse_candidates()`.

## 7. How it connects to the rest of the system

These are stages 4a/4b of `HybridRetriever.retrieve()`. Their outputs are *candidates*, not answers — fusion weights them (0.25 entity, 0.15 semantic vs 0.6 vector), the cross-encoder rescores, and only then does evidence selection happen.

## 8. Important design decisions

- **1 hop only:** 2-hop expansion on a 265k-edge graph explodes combinatorially. One hop from 20 vector hits is bounded and sufficient.
- **Hub cap (1000):** "cancer" is mentioned in thousands of chunks; expanding it adds noise. Skipping hubs is a precision decision.
- **Entity path needs the query entity to exist:** if scispaCy finds no entities, this path contributes nothing — the system degrades gracefully to vector+semantic.
- **Why not just lower the vector threshold?** Vector search ranks by *query* similarity; the graph paths rank by *entity co-occurrence* and *chunk neighborhood* — genuinely different signals, not just "more vector results."

## 9. Example from the actual project

Query *"What drugs are associated with breast cancer treatment?"*: vector search finds chunks with similar wording; entity expansion finds *every* chunk mentioning `disease:breast cancer` — including ones saying "tamoxifen was administered to the cohort" with no lexical overlap with "drugs associated"; semantic expansion finds chunks similar to the best vector hits. Fusion + rerank then picks the winners.

## 10. Common misunderstandings

- **"The graph replaces embeddings."** No — vector similarity is still the top-weighted signal (0.6). The graph adds recall where vectors are blind.
- **"More hops = better."** No — bounded 1-hop is a deliberate precision/latency choice on a 265k-edge graph.
- **"GDS powers this."** No — these are plain Cypher traversals. GDS is installed and used in validation only (file 05).
- **"There's a graph-aware reranker."** No such module. Graph influence flows: expansion → fusion weights → cross-encoder input. Say exactly this.

## 11. Professor Viva Questions

- Vector-only vs graph-enhanced retrieval — what's the difference in *your* system?
- When does graph expansion help? When could it hurt?
- What is the hub problem and how do you handle it?
- Why only 1 hop?
- What if the query has no recognizable entities?

## 12. Natural Spoken Answers

**"When does the graph actually help?"**
> "Sir, two cases. First, vocabulary mismatch: the query says 'drugs for breast cancer' but the evidence says 'tamoxifen was administered' — no word overlap, but both mention the same normalized entity, so the MENTIONS traversal finds it. Second, neighborhood: a chunk very similar to a great vector hit might itself be phrased differently from the query — the SEMANTIC_SIMILAR hop catches it. When it could hurt is with hub entities like 'cancer' mentioned everywhere — that's why we skip entities with over a thousand mentions."

## 13. Knowledge Bites

- Two graph paths: entity→MENTIONS→chunks; vector hits→SEMANTIC_SIMILAR→chunks.
- Entity path fixes vocabulary mismatch; semantic path exploits chunk neighborhoods.
- 1 hop only — bounded by design on 265k edges.
- Hub cap 1000 skips ultra-common entities (noise control).
- Graph signals are candidates; fusion (0.25/0.15) + cross-encoder decide.
- No GDS in the live path; no graph-aware reranker module — plain Cypher traversals.
- Missing query entities → graceful degradation to vector+semantic.
