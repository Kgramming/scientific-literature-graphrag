# 08 — Query Processing and Decomposition

## 1. What this component does

Analyzes the user's raw question *before* retrieval: classifies it into one of five types, and for multi-part questions, splits it into ≤3 subqueries that are each retrieved independently and fused.

## 2. Why it exists

Not all questions want the same retrieval strategy. "What is diabetes?" (factual) needs direct lookup; "Compare metformin vs insulin" (comparison) needs two entity foci; "What causes X, and how is it treated?" (multipart) needs two separate retrievals. Classification routes; decomposition multiplies.

## 3. Where it lives in the repository

- `src/sgrag/query/classification.py` — `QueryClass` enum, `ClassifiedQuery`, `classify_query()`
- `src/sgrag/generation/generator.py` — `_retrieve_all()`: per-subquery retrieval + fusion (the decomposition *execution*)
- `configs/decomposition.yaml` — decomposition settings

## 4. Important classes/functions

- **`QueryClass`** — enum: `factual`, `entity_focused`, `comparison`, `multipart`, `unsupported`.
- **`classify_query(text)`** — **rule-based** (regex/keyword heuristics, no LLM): detects comparison patterns ("X vs Y", "compare"), multipart signals (multiple question marks/clauses), entity mentions, and unsupported inputs (non-scientific, empty, excessively long). Returns `ClassifiedQuery(text, query_class, subqueries, reason)`.
- **`ClassifiedQuery.subqueries`** — for `multipart`, up to 3 sub-question strings; otherwise empty.
- **`GroundedGenerator._retrieve_all(classified)`** — runs `retriever.retrieve(sq)` for each subquery (or the single query), merges by `chunk_id` keeping the best `fused_score`, sums diagnostics. This is where decomposition actually happens at runtime.
- **`UNSUPPORTED` handling** — in `GroundedGenerator.answer()`: unsupported questions return `UNSUPPORTED_QUESTION_ANSWER` *before any retrieval* — no Neo4j, no LLM.

## 5. Step-by-step execution

```text
"What causes Alzheimer's and how is it treated?"
  → classify_query()
    → detects two clauses → QueryClass.MULTIPART
    → subqueries = ["What causes Alzheimer's?", "How is Alzheimer's treated?"]
  → _retrieve_all()
    → retrieve("What causes Alzheimer's?") → results A
    → retrieve("How is Alzheimer's treated?") → results B
    → merge by chunk_id (best fused_score wins) → fused list
  → evidence → generation (as usual)

"What is RAG and how is it better than an LLM?"
  → classify_query() → QueryClass.UNSUPPORTED (non-scientific)
  → return refusal immediately — no retrieval, no LLM call
```

## 6. Data flow

`str query` → `ClassifiedQuery {query_class, subqueries, reason}` → `list[str] queries` (1 or ≤3) → per-query `list[RetrievalResult]` → merged `list[RetrievalResult]`.

## 7. How it connects to the rest of the system

Classification is the *first* step of `GroundedGenerator.answer()`, before any retrieval. Its output selects the retrieval fan-out (`_retrieve_all`) and gates the unsupported path. The query class is also reported in diagnostics and shown in the UI.

## 8. Important design decisions

- **Rule-based, not LLM-based:** classification must be fast, deterministic, and free — an LLM call per query just to classify would add latency and cost. Heuristics are transparent and testable.
- **Bounded decomposition (≤3):** each subquery is a full retrieval pipeline run; unbounded splitting would multiply latency.
- **Merge by best score:** subquery results merge on `chunk_id` keeping the max `fused_score` — a chunk relevant to two subqueries isn't duplicated.
- **Unsupported short-circuit:** refusing before retrieval saves compute and is honest — the system knows what it *doesn't* cover.

## 9. Example from the actual project

"What is RAG and how is it better than an LLM?" → `UNSUPPORTED` → the system returns the unsupported-question answer without touching Neo4j or the LLM. This is the guardrail demo in file 10.

## 10. Common misunderstandings

- **"Decomposition uses an LLM."** No — rule-based splitting. The LLM never sees the raw query until evidence is packed.
- **"Every query is decomposed."** No — only `multipart`. Factual/entity/comparison queries run as single retrievals (comparison currently retrieves once; multi-entity handling is future scope).
- **"Unsupported means the LLM refused."** No — refusal happens *before* retrieval, in Python. The LLM is never called.

## 11. Professor Viva Questions

- How do you handle complex multi-part questions?
- What query types do you support? How are they detected?
- Why rule-based classification instead of an LLM?
- What happens for non-scientific questions?
- Why cap subqueries at 3?

## 12. Natural Spoken Answers

**"How do you handle a question with two parts?"**
> "Sir, we classify every query first with fast rule-based heuristics — no LLM involved. If it's multipart, like 'what causes Alzheimer's and how is it treated', we split it into up to three subqueries, run the full retrieval pipeline for each, and merge the results by chunk ID keeping the best score. And if the question isn't scientific at all, we refuse immediately before any retrieval — that's our guardrail."

## 13. Knowledge Bites

- Five query classes: factual, entity_focused, comparison, multipart, unsupported.
- `classify_query()` is rule-based (regex/heuristics) — fast, deterministic, free.
- Multipart → ≤3 subqueries → independent retrievals → merged by chunk_id.
- `_retrieve_all()` is the decomposition executor in the generator.
- UNSUPPORTED short-circuits before retrieval — no Neo4j, no LLM.
- Bounded at 3: each subquery costs a full pipeline run.
- Query class flows into diagnostics and the UI.
