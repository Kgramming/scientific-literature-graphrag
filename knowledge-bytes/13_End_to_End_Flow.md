# 13 — End-to-End Flow

## 1. What this component does

Traces ONE user query through every stage of the system, naming the actual files and functions at each step. This is the single most important viva document — if you can narrate this, you can answer any architecture question.

## 2. Why it exists

Professors don't ask about isolated modules; they ask "what happens when I type a question?" This file is that answer, end to end, with no hand-waving.

## 3. Where it lives in the repository

This flow touches: `app.py` → `src/sgrag/generation/generator.py` → `src/sgrag/query/classification.py` → `src/sgrag/retrieval/retriever.py` → `vector.py` → `query.py` → `graph.py` → `fusion.py` → `rerank.py` → `src/sgrag/generation/evidence.py` → `prompts.py` → `llm.py` → `citations.py` → back to `app.py`.

## 4. The full trace

**Query:** *"What drugs are associated with breast cancer treatment?"*

```text
① app.py — user clicks "Search literature"
   ask_pipeline(query) → get_generator() [cached] → gen.answer(query)

② generator.py — GroundedGenerator.answer()
   classify_query() → entity_focused, subqueries=[]

③ query/classification.py — rule-based heuristics
   "drugs...treatment" + entity mention → ENTITY_FOCUSED

④ generator.py — _retrieve_all([query])
   → retriever.retrieve("What drugs are associated with breast cancer treatment?")

⑤ retriever.py — HybridRetriever.retrieve(), stage 1-2
   embed_query() → 384-dim vector (all-MiniLM-L6-v2)
   vector_search() → 20 chunks via chunk_embedding_idx (cosine)

⑥ retrieval/query.py — stage 3
   extract_query_entities() → scispaCy → ["disease:breast cancer"]

⑦ retrieval/graph.py — stage 4a
   expand_entities(["disease:breast cancer"]) → ≤15 chunks via MENTIONS
   (hub check: degree < 1000, so expansion proceeds)

⑧ retrieval/graph.py — stage 4b
   expand_semantic(vector_hit_ids) → ≤3 neighbors per hit via SEMANTIC_SIMILAR

⑨ retrieval/fusion.py — stage 5
   merge by chunk_id → weighted 0.6/0.25/0.15 → deterministic ranking
   → ~40 fused candidates

⑩ retriever.py — stage 6-7
   TITLE_LOOKUP (batched) → CrossEncoderReranker.rerank() → top-50 rescored
   → final ranked list + diagnostics {embed_ms, n_vector:20, n_entity:.., ...}

⑪ generator.py — evidence
   pack_evidence(results, top_k=8) → 8 EvidenceItems [1..8]

⑫ generation/prompts.py — prompt
   build_messages(query, evidence, strict=False)
   SYSTEM: "Answer ONLY from the evidence. Cite as [n]." + evidence block

⑬ generation/llm.py — generation
   GroqProvider.generate() → "Tamoxifen [1] and trastuzumab [3] are associated..."

⑭ generation/citations.py — validation
   extract_citations() → [1, 3]; validate against {1..8} → valid ✓

⑮ app.py — rendering
   render_answer() → text + chips; render_evidence() → cards;
   render_intelligence() → timings; render_graph_section() → PyVis
```

Total: ~13 stages, each with a named file and function.

## 5. Step-by-step execution

See the trace above. Note the two short-circuits not shown: `UNSUPPORTED` classification returns before ④; empty evidence returns before ⑫.

## 6. Data flow (shapes)

`str` → `ClassifiedQuery` → `(384-vector, [entity keys])` → 3× `list[chunk]` → `list[RetrievalResult]` → `list[EvidenceItem](8)` → `list[message]` → `str answer` → `GenerationResult` → rendered HTML.

## 7. How it connects to the rest of the system

This *is* the rest of the system, in order. Files 01–12 each deep-dive one numbered stage.

## 8. Important design decisions (recap)

Bounded at every stage; deterministic throughout; LLM last and least-privileged (8 chunks only); honest failures at 4 levels; diagnostics bubble up for observability.

## 9. Example from the actual project

The trace above uses the project's real example query (also in `scripts/ask.py` docs and the README).

## 10. Common misunderstandings

- **Skipping stages in narration.** A viva answer that jumps "query → LLM → answer" misses the entire engineering. Name the stages.
- **"The graph does everything."** The graph enables two of three retrieval signals; vector search and the cross-encoder do heavy lifting too.

## 11. Professor Viva Questions

- Trace a query end to end. (This file.)
- Where is the most compute spent?
- Which stage would you optimize first?
- What happens at each failure point?

## 12. Natural Spoken Answers

**"Explain the project in 2 minutes."**
> "Sir, it's an evidence-grounded biomedical QA system. A user asks a question in our Streamlit app. First we classify the query — this one's entity-focused — using fast rules, no LLM. Then hybrid retrieval runs three searches in parallel against Neo4j: vector search over 53,000 chunk embeddings, entity expansion walking MENTIONS edges from 'breast cancer' to every chunk discussing it, and one hop along semantic-similarity edges from the vector hits. We fuse the candidates with weights, rerank the top 50 with a cross-encoder — that step alone lifted our Recall@5 from 0.945 to 0.985 on 200 PubMedQA questions. The top 8 chunks become evidence. Then we build a strict evidence-only prompt, call the LLM, and validate every citation against the retrieved set — retrying once strictly, stripping anything invalid. The app renders the answer with citation chips, evidence cards, retrieval timings, and a graph visualization. If the question isn't scientific, or there's no evidence, the system refuses honestly instead of hallucinating."

## 13. Knowledge Bites

- 13 stages: UI → classify → retrieve(7 sub-stages) → evidence → prompt → LLM → validate → render.
- Each stage names a file and function — no black boxes.
- Two short-circuits: unsupported (before retrieval), no evidence (before LLM).
- LLM is stage 12 of 13 — least-privileged, 8 chunks only.
- Diagnostics flow upward at every stage to the UI.
- The trace above is narratable in ~2 minutes — practice it.
- Numbers to cite: 53,091 chunks, 9,945 entities, Recall@5 0.9850, MRR 0.9689.
- Failure honesty at 4 levels: unsupported / no evidence / retrieval down / LLM down.
