# 07 — Reranking and Evidence Selection

## 1. What this component does

Takes the fused candidate list (~up to 100 chunks) and produces the final ranking, then selects the top evidence items that will be shown to the LLM. Two sub-steps: cross-encoder rescoring, then top-k evidence packing.

## 2. Why it exists

Retrieval is optimized for *recall* (don't miss anything) — it's fast and coarse. Reranking is optimized for *precision*: a cross-encoder reads the actual (query, chunk) pair deeply, which is too expensive for 53k chunks but perfect for 50 candidates. Evidence selection then cuts to what the LLM can actually use.

## 3. Where it lives in the repository

- `src/sgrag/retrieval/rerank.py` — `Reranker` (Protocol), `NoOpReranker`, `CrossEncoderReranker`, `RerankerError`
- `src/sgrag/generation/evidence.py` — `pack_evidence(results, top_k=8)`
- `src/sgrag/generation/models.py` — `EvidenceItem`
- Wiring: `HybridRetriever.__init__` picks the reranker from `RetrievalConfig.reranker`

## 4. Important classes/functions

- **`CrossEncoderReranker`** — loads `cross-encoder/ms-marco-MiniLM-L-6-v2`; `rerank(query, results)` scores `(query, chunk_text)` pairs for the top-50 pool (`candidate_k`), sorts by `(-reranker_score, chunk_id)`, sets `reranker_score/rank/final_score/final_rank`. The unreranked tail keeps fused order. Deterministic.
- **`NoOpReranker`** — pass-through; `final_score = fused_score`. Used when reranking is disabled or as fallback.
- **`RerankerError`** — raised on model-load/inference failure; the retriever falls back to fusion ranking if `fallback_to_fusion` is set (fail-safe, not fail-dead).
- **`pack_evidence(results, top_k=8)`** — takes the final ranking, keeps the top 8, builds `EvidenceItem`s with `{pmid, chunk_id, text, title, rank, score}` for the prompt.

## 5. Step-by-step execution

```text
fused candidates (≤100, ranked)
  → CrossEncoderReranker.rerank(query, results)
    → pool = top 50; pairs = [(query, chunk.text) for chunk in pool]
    → model.predict(pairs, batch_size=32)   [~seconds on CPU]
    → sort by (-score, chunk_id); assign final_rank
    → tail (51+) keeps fused order
  → pack_evidence(results, top_k=8)
    → 8 EvidenceItems → build_messages() → LLM
```

## 6. Data flow

`list[RetrievalResult]` (fused) → `list[RetrievalResult]` (reranked, with `final_score`) → `list[EvidenceItem]` (top 8). Only the 8 evidence items ever reach the LLM prompt.

## 7. How it connects to the rest of the system

Reranking is stage 7 of `HybridRetriever.retrieve()`; evidence packing is the first step of `GroundedGenerator.answer()` after retrieval. The measured jump (Recall@5 0.9450→0.9850, MRR 0.8623→0.9689) is *this* component's contribution.

## 8. Important design decisions

- **Cross-encoder, not bi-encoder:** a cross-encoder jointly reads query+chunk (no separate embeddings), giving deeper relevance judgments — at the cost of speed, hence only 50 candidates.
- **Bounded pool (50):** full-list rescoring would be slow on CPU; top-50 captures nearly all the gain.
- **Deterministic ordering** even here: `(-reranker_score, chunk_id)`.
- **Graceful degradation:** `RerankerError` → fallback to fused ranking (configurable) — a missing model degrades quality, never crashes the pipeline.
- **Top-8 evidence cap:** LLM context is finite; 8 chunks balance coverage vs. prompt length. Configurable via `evidence_top_k`.

## 9. Example from the actual project

On the 200-question eval set, adding the cross-encoder moved Recall@5 from 0.9450 to **0.9850** and MRR from 0.8623 to **0.9689** (`docs/DEMO_GUIDE.md` §8, `data/evaluation/retrieval_eval_5a.json`). That's the single biggest measured win in the pipeline.

## 10. Common misunderstandings

- **"Reranking and retrieval are the same."** Retrieval = fast coarse recall over 53k chunks (vector index). Reranking = slow precise scoring of 50 candidates (joint query-chunk reading). Different jobs, different costs.
- **"There's a graph-aware reranker."** There isn't — no `graph_aware` code exists. Graph signals influence ranking *before* this step, via expansion and fusion. The reranker itself is pure cross-encoder.
- **"Top-8 is arbitrary."** It's a context-window/cost tradeoff, exposed as `evidence_top_k` — tunable, not magic.

## 11. Professor Viva Questions

- Why rerank after retrieval? What's the difference?
- What is a cross-encoder? How does it differ from a bi-encoder?
- Why only 50 candidates?
- What happens if the reranker model fails to load?
- How do you know reranking helped? (Answer: the measured deltas.)

## 12. Natural Spoken Answers

**"Why rerank? Isn't retrieval enough?"**
> "Sir, retrieval is built for recall — it must be fast over 53,000 chunks, so it uses coarse signals like vector cosine. Reranking is built for precision: the cross-encoder reads each query-chunk pair jointly and deeply, which is too slow for the full corpus but perfect for 50 candidates. On our 200-question eval set, it lifted Recall@5 from 0.9450 to 0.9850 — the biggest single improvement we measured."

## 13. Knowledge Bites

- Retrieval = recall (fast, coarse, 53k chunks); reranking = precision (slow, deep, 50 candidates).
- `CrossEncoderReranker`: ms-marco-MiniLM-L-6-v2, batch 32, deterministic ordering.
- Unreranked tail preserves fused order — no candidate is lost.
- `RerankerError` → configurable fallback to fusion ranking (fail-safe).
- `pack_evidence(top_k=8)`: only 8 items reach the LLM.
- Measured win: Recall@5 +0.0400, MRR +0.1066 on 200 PubMedQA questions.
- No graph-aware reranker module exists — graph signals act earlier (expansion + fusion).
