# 01 — System Architecture

## 1. What this component does

This document explains the entire Scientific Literature GraphRAG system: how a user's question becomes an evidence-grounded answer with citations. It maps every major layer to the actual files that implement it.

## 2. Why it exists

A professor's first question is always "explain your project architecture." This is the answer — not a textbook diagram, but *this* repository's real structure, file by file.

## 3. Where it lives in the repository

| Layer | Files |
|---|---|
| UI | `app.py` (1,023 lines) |
| Query understanding | `src/sgrag/query/classification.py` |
| Retrieval | `src/sgrag/retrieval/retriever.py`, `vector.py`, `graph.py`, `fusion.py`, `query.py` |
| Reranking | `src/sgrag/retrieval/rerank.py` |
| Generation | `src/sgrag/generation/generator.py`, `prompts.py`, `llm.py`, `citations.py`, `evidence.py` |
| Graph | `src/sgrag/graph/schema.py`, `ingest.py`, `entities.py`, `similarity.py` |
| Data/chunking/embeddings | `src/sgrag/data/`, `src/sgrag/chunking/`, `src/sgrag/embeddings/` |
| Config | `src/sgrag/config/`, `configs/*.yaml`, `.env` (git-ignored) |
| Scripts | `scripts/` (pipeline procedures) |
| Evaluation | `scripts/evaluate_retrieval.py`, `data/evaluation/` |

## 4. Important classes/functions

- **`GroundedGenerator`** (`src/sgrag/generation/generator.py`) — the top-level orchestrator. `answer(query)` runs classify → retrieve → evidence → generate → validate.
- **`HybridRetriever`** (`src/sgrag/retrieval/retriever.py`) — the retrieval orchestrator. `retrieve(query)` runs embed → vector search → entity extraction → graph expansion → fusion → rerank.
- **`classify_query`** (`src/sgrag/query/classification.py`) — rule-based classifier returning `QueryClass` (factual/entity_focused/comparison/multipart/unsupported) plus ≤3 subqueries.
- **`ask_pipeline`** (`app.py`) — thin bridge: `get_generator(rerank)` → `gen.answer(query)`.

## 5. Step-by-step execution (query time)

```text
User types query in app.py
  → ask_pipeline(query)
    → GroundedGenerator.answer(query)
      → classify_query(query)                    [query/classification.py]
      → _retrieve_all(classified)                [per subquery if multipart]
        → HybridRetriever.retrieve(subquery)
          → embed_query()                        [384-dim MiniLM vector]
          → vector_search()                      [Neo4j vector index]
          → extract_query_entities()             [scispaCy BC5CDR]
          → expand_entities()                    [MENTIONS edges, hub-capped]
          → expand_semantic()                    [SEMANTIC_SIMILAR 1-hop]
          → fuse_candidates()                    [weighted/RRF]
          → CrossEncoderReranker.rerank()        [top-50 rescoring]
      → pack_evidence(results)                   [top-8 evidence items]
      → build_messages(query, evidence)          [strict evidence-only prompt]
      → llm.generate(messages)                   [Groq / mock]
      → validate_citations(answer)               [retry-then-strip]
    → render_answer / render_evidence / render_graph_section  [app.py]
```

## 6. Data flow

`str query` → `ClassifiedQuery` → `list[RetrievalResult]` (+ diagnostics dict) → `list[EvidenceItem]` → `list[message dicts]` → `str answer` → `GenerationResult` (answer + evidence + citations + metadata) → Streamlit rendering.

## 7. How it connects to the rest of the system

Everything flows through `GroundedGenerator.answer()`. The UI (`app.py`) never touches Neo4j directly for retrieval — it calls the generator, which owns the retriever, which owns the driver. Configuration flows downward: `.env` → `RetrievalConfig` / `GenConfig` → components. Diagnostics flow upward: per-stage timings and counts bubble back to the UI's "retrieval intelligence" panel.

## 8. Important design decisions

- **Layered, not monolithic:** retrieval has no LLM, generation has no UI, UI has no Cypher. Each layer is independently testable (162 tests).
- **Bounded everything:** vector_top_k=20, entity_top_k=15, hub_degree_cap=1000, 1-hop semantic expansion. The graph has 265k edges; unbounded traversal would explode.
- **Deterministic:** fusion ranking breaks ties by chunk_id; eval sets are deterministically constructed. Reproducibility is a design goal, not an accident.
- **Honest failure modes:** unsupported questions, empty evidence, and unreachable Neo4j each have explicit, user-visible handling — no silent garbage.

## 9. Example from the actual project

Query: *"What drugs are associated with breast cancer treatment?"* → classified as entity-focused → entities extracted ("breast cancer") → vector search finds similar chunks + MENTIONS expansion finds chunks mentioning the entity + semantic expansion finds similar chunks → fused → cross-encoder reranks top 50 → top 8 packed as evidence → LLM answers with `[1]`, `[2]` citations → UI renders answer, evidence cards, and a PyVis graph.

## 10. Common misunderstandings

- **"The LLM does the retrieval."** No — retrieval is pure Neo4j + embeddings + cross-encoder. The LLM only sees the final top-8 evidence.
- **"Graph-aware reranking is a separate module."** Not in this codebase — there is no `graph_aware` code. Graph signals enter through *retrieval expansion* (entity/semantic) and *fusion*, not a separate reranking step. Don't claim otherwise in viva.
- **"GDS runs the retrieval."** No — GDS is installed as a Neo4j plugin and used in `scripts/validate_graph.py`, not in the live retrieval path.

## 11. Professor Viva Questions

- Explain your project architecture.
- Why did you use GraphRAG instead of plain vector search?
- What happens when a user enters a query? (Trace it.)
- Why Neo4j? What does the graph store that a vector DB can't?
- Where does the LLM fit? What does it see?
- What happens if Neo4j is down?

## 12. Natural Spoken Answers

**"Explain your architecture."**
> "Sir, it's a layered pipeline. The user asks a question in the Streamlit app. First we classify the query — factual, entity-focused, comparison, multipart, or unsupported. Then the hybrid retriever runs: it embeds the query, does vector search in Neo4j, extracts biomedical entities with scispaCy, expands along the graph through MENTIONS and semantic-similarity edges, fuses the candidates, and reranks the top 50 with a cross-encoder. The top 8 chunks become evidence. Then we build a strict evidence-only prompt, call the LLM, and validate that every citation actually points to retrieved evidence. The UI renders the answer, the evidence, and a graph visualization."

**"Why GraphRAG?"**
> "Because vector search only finds text that looks similar. Our questions are biomedical — 'which drug treats which disease' is a *relationship*, and relationships live naturally in a graph. We store entities and semantic chunk links in Neo4j, so retrieval can walk from an entity to every chunk mentioning it, even if the wording differs completely."

## 13. Knowledge Bites

- `GroundedGenerator.answer()` is the single entry point for the whole pipeline.
- `HybridRetriever.retrieve()` is the single entry point for retrieval.
- UI never writes Cypher; all Neo4j access goes through `src/sgrag/`.
- Config flows down (`.env` → config objects); diagnostics flow up (to the UI).
- Every stage is bounded: top-k limits and hub caps prevent graph explosion.
- The LLM sees only the top-8 evidence items — never the raw graph.
- Unsupported questions are refused *before* retrieval; empty evidence refuses *before* the LLM call.
- Deterministic ranking (tie-break by chunk_id) makes results reproducible.
- 162 tests cover unit, integration, and end-to-end — layers are independently testable.
- No `graph_aware` module exists; graph signals enter via expansion + fusion.
- GDS is a plugin used in validation, not in live retrieval.
