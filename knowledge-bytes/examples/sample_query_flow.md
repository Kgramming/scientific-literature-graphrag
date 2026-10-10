# Example: Sample Query Flow

**Type:** illustrative walkthrough (uses real component names and real config values; timings are illustrative).

**Query:** *"What drugs are associated with breast cancer treatment?"*

```text
INPUT  (app.py)
  "What drugs are associated with breast cancer treatment?"
  |
  v
CLASSIFY  (src/sgrag/query/classification.py :: classify_query)
  → QueryClass.ENTITY_FOCUSED, subqueries=[], reason="drug/treatment + entity mention"
  |
  v
RETRIEVE  (src/sgrag/retrieval/retriever.py :: HybridRetriever.retrieve)
  ├─ embed_query → vector[384]
  ├─ vector_search(top_k=20) → 20 chunk candidates        [stage 2]
  ├─ extract_query_entities → ["disease:breast cancer"]   [stage 3]
  ├─ expand_entities(top_k=15) → ~12 chunks via MENTIONS   [stage 4a]
  ├─ expand_semantic(top_k=3) → ~30 chunks, 1 hop          [stage 4b]
  ├─ fuse_candidates(0.6/0.25/0.15) → ~45 fused, ranked    [stage 5]
  └─ CrossEncoderReranker(top-50) → final ranking          [stage 7]
  |
  v
EVIDENCE  (src/sgrag/generation/evidence.py :: pack_evidence)
  → top 8 → EvidenceItem[1..8] {pmid, chunk_id, text, title}
  |
  v
PROMPT  (src/sgrag/generation/prompts.py :: build_messages)
  SYSTEM: "Answer ONLY from the evidence below. Cite sources as [n]."
  USER:   [1] <chunk text> (PMID ...) ... [8] <chunk text> + question
  |
  v
GENERATE  (src/sgrag/generation/llm.py :: GroqProvider.generate)
  → "Tamoxifen [1] and trastuzumab [3] are associated with breast cancer
     treatment in the retrieved literature. ..."
  |
  v
VALIDATE  (src/sgrag/generation/citations.py :: validate_citations)
  → citations {1, 3} ⊆ evidence {1..8} → VALID
  |
  v
RENDER  (app.py :: render_answer, render_evidence, render_graph_section)
  → answer + chips [1][3], evidence cards, PyVis graph, diagnostics
```

**Refusal variant** (same pipeline, different query):

```text
INPUT: "What is RAG and how is it better than an LLM?"
  → classify → UNSUPPORTED → return UNSUPPORTED_QUESTION_ANSWER
  → (no retrieval, no embedding, no LLM call)
```
