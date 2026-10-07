# Reference Gap Analysis

**Reference repository:** https://github.com/piyushghadge02/GraphRAG-Scientific-Literature
**Inspected commit:** `06b1ffe8da9b3263f62660e8c446fd872695849c` (2026-10-05)
**Inspection method:** read-only `git clone` into an ephemeral directory (`/tmp`, never copied into this
project). We studied architecture, module organization, configuration, evaluation methodology, and the
README's own stated limitations. **No code, README text, or directory structure was copied.**

This document is the evidence base for our architecture proposal. Every claim about the reference cites
the file or README section where it was observed.

---

## 1. What the reference does well (verified strengths)

| # | Strength | Evidence |
|---|----------|----------|
| S1 | Honest evaluation reporting — refuses to fabricate matches | README §1: "Measured overlap of the dataset pairing is small (**6 verified questions**); only verified matches are ever written, never fabricated." |
| S2 | Clean modular pipeline | `src/chunking/`, `src/graph/`, `src/retrieval/`, `src/generation/`, `src/evaluation/` — ~4.9k lines, single-responsibility modules |
| S3 | Baseline embedding pinned and referenced centrally | `src/config.py` (`BASELINE_EMBEDDING_MODEL`), `src/embeddings.py` cached loader; semantic clustering stays pinned to `all-MiniLM-L6-v2` while retrieval model is configurable |
| S4 | Explainable, normalized fusion | `src/retrieval/fusion.py`: `final = α·vector + β·entity + γ·proximity`, weights rescaled to sum to 1; per-candidate `score_explanation` |
| S5 | Adaptive retrieval with logged decisions | `src/retrieval/adaptive.py`: deterministic rules (top-1 dominance/gap thresholds, multi-concept keyword cues); decision + reasons logged and shown in UI |
| S6 | PageRank on retrieved subgraph with graceful degradation | `src/retrieval/gds_rerank.py`: temporary GDS projection, min-max normalization, blend `(1−w)·combined + w·pagerank`; falls back cleanly when GDS is absent |
| S7 | Provider-agnostic LLM client | `src/generation/llm_client.py`: Groq (key rotation) / OpenAI / Ollama |
| S8 | Same-question comparison harness | `src/evaluation/compare.py`: every system evaluated on identical questions; CSV + PNG artifacts |
| S9 | Entity hygiene pass | `src/graph/entity_extraction.py`: whitespace collapsing, possessive stripping, noise filters, deterministic first-seen type resolution |
| S10 | Viva-grade documentation | `Knowledge Bytes/` — 23 files covering architecture through viva questions |

We adopt the *principles* behind S1, S2, S4, S5, S8, S10 (honesty, modularity, explainability, same-question
ablation, viva docs) while re-engineering the implementations independently.

---

## 2. Assignment coverage vs. reference (observed gaps)

| Assignment requirement | Reference status | Evidence / note |
|---|---|---|
| ~5,000 PubMed abstracts | ✅ Done | `ccdv/pubmed-summarization`, seed-42 sample, `data/raw/abstracts_sample.jsonl` |
| Fixed-token + sentence-boundary + semantic chunking | ⚠️ Partial | All three exist (`src/chunking/`), **but** the submitted graph merges semantic clusters back to **one chunk per article** (`--merge-per-article`, README §3). The semantic pipeline's output is discarded for the indexed graph — exactly the anti-pattern our spec forbids |
| Chunk cohesion analysis / visualization | ❌ Missing | `validate_semantic.py` (79 lines) exists but no cohesion distributions or comparative visualizations were found |
| Neo4j + APOC + GDS, vector index | ✅ Done | Schema `Article→HAS_CHUNK→Chunk→MENTIONS→Entity` (+ optional `SEMANTIC_SIMILAR`), 384D cosine vector index |
| Richer graph schema (authors, methods, topics, typed entities) | ❌ Missing | Schema limited to Article/Chunk/Entity; entities are `{name, type}` from generic `en_core_sci_sm`; no canonicalization, no aliases, plurals deliberately kept distinct |
| Biomedical entity normalization (abbreviations, aliases) | ❌ Missing | `normalize_entity_name` handles case/whitespace/possessives/punctuation only; docstring explicitly declines plurals/synonyms/acronyms |
| Hybrid retrieval (vector + entity + graph) | ✅ Done | Vector pool (3×) → shared-entity expansion (depth ≤ 2) → fusion |
| Cross-encoder reranker | ❌ Missing | "Reranking" = PageRank blend only (`gds_rerank.py`); no second-stage neural reranker anywhere in `src/retrieval/` |
| Query classification + routing | ❌ Missing | `adaptive.py` uses vector-score thresholds + keyword cues ("and", "vs", "relationship between"); no query taxonomy, no per-class routing table |
| Query decomposition | ⚠️ Partial | `src/retrieval/decompose.py` exists: LLM prompt → JSON parse → fixed `top_k_per_subq=3` merge by `combined_score`. **No validation** of sub-questions (duplicates, empties, semantic drift), no fallback when a sub-query retrieves nothing, and decomposition is **not** part of the A–E ablation systems (`src/evaluation/systems.py`) |
| Graph-aware reranking with GDS | ⚠️ Partial | PageRank only; no degree/proximity/entity-overlap/community signals; `γ=0.0` default means graph distance contributes nothing unless manually raised |
| Community / cluster information | ❌ Missing | No Louvain/Leiden or community-based retrieval signal |
| Evidence-first generation + citation mapping | ⚠️ Partial | Citation instructions in prompt (`prompt_templates.py`); no citation-correctness validation (citations are not checked against retrieved chunk IDs), no claim→chunk mapping |
| Faithfulness evaluation | ❌ Missing | Generation eval = ROUGE-L F1 + BERTScore F1 (`distilbert-base-uncased`) only |
| Statistically meaningful eval set | ❌ **Critical gap** | Corpus has **no PubMed IDs** → PubMedQA matching requires verbatim text match → **6 verified questions** (README §1). Yet §12/§17 discuss `--num-questions 200` runs — the comparison harness is sound, but the verified question pool is far too small for any claim |
| Article-level AND chunk-level retrieval metrics | ⚠️ Partial | Article-level Recall@5/10 + MRR only; no chunk-level metrics, no nDCG, no Precision@K (README §19 admits article-level recall "rewards finding the gold article, not the best chunk") |
| Latency / resource evaluation | ❌ Missing | No per-stage timing, no candidate/node/token accounting |
| Config-driven reproducible experiments | ⚠️ Partial | Env-var config in `src/config.py` (no YAML experiment configs); runs are deterministic (seed 42) and artifacts are saved, but there is no one-command experiment with recorded config + model versions + provenance |
| RRF / weight-sensitivity study | ❌ Missing | Single weighted blend; no RRF implementation; no sensitivity analysis over (α, β, γ) |
| Ablation A–G incl. decomposition | ⚠️ Partial | A–E systems (vector-only → adaptive); decomposition toggle exists in Streamlit but is not in the measured ablation matrix |

---

## 3. Weaknesses we will exploit — with the measurable improvement each enables

**W1. The 6-question evaluation bottleneck (biggest weakness).**
Root cause: corpus abstracts carry no PubMed IDs, so gold mappings need verbatim text matching.
Our fix: build the corpus *from* PubMedQA gold PMIDs (fetch the actual abstracts, real PMIDs preserved)
plus PubMed fill to ~5,000 → deterministic PMID-equality matching → potentially **hundreds** of verified
questions instead of 6. Measurable: report verified-question count, match rate, and dropped-question audit.

**W2. Semantic chunking merged back to one chunk/article.**
Our fix: a true 4-strategy comparison where the improved semantic chunker *preserves* sections
(breakpoint detection + min/max size guards), plus intra- vs inter-chunk cohesion visualizations.
Measurable: chunk-count/size distributions, cohesion plots, and retrieval deltas per strategy.

**W3. Generic NER, no normalization.**
Our fix: `en_ner_bc5cdr_md` (DISEASE/CHEMICAL, trained on the BC5CDR biomedical corpus) + canonical
entity nodes (case/whitespace/punct + abbreviation expansion + alias map + surface-form preservation).
Measurable: entity node count before/after dedup, entity-recall contribution in ablations.

**W4. No neural reranker.**
Our fix: `cross-encoder/ms-marco-MiniLM-L-6-v2` second-stage rerank on a bounded candidate pool,
ablated as System E. Measurable: Recall/MRR with and without reranker on identical questions.

**W5. Naive decomposition.**
Our fix: decomposition gated by query classification, with deterministic validators (non-empty, distinct,
bounded count, term-overlap with original, per-sub-query non-empty retrieval fallback). Ablated as
System F. Measurable: multi-hop question subset performance with/without decomposition; degradation
rate on simple questions (must be ~zero).

**W6. Article-level metrics only.**
Our fix: article-level AND chunk-level (weak labels, clearly marked) + nDCG@10 + Precision@5.
Measurable: the metrics themselves, with a documented explanation of what each level means.

**W7. No faithfulness or latency measurement.**
Our fix: citation-correctness, evidence coverage, unsupported-claim rate (NLI, labeled auxiliary) +
per-stage latency tracking and an accuracy-vs-latency analysis. Measurable: faithfulness suite per system,
latency breakdown table.

**W8. No experiment provenance.**
Our fix: YAML configs per system + an experiment runner that stamps every run with config hash, model
versions, dataset manifest, seed, timestamp. One-command reproduction. Measurable: reproducibility —
re-running a config reproduces the CSV exactly.

---

## 4. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| PubMedQA gold abstracts un-fetchable for some PMIDs (rate limits, retractions) | Medium | Batch Entrez `efetch`, local cache, retry; **report** fetch failures in the manifest — never substitute |
| Even with PMID matching, verified set ends up small (<50) | Low–Medium | Fall back to the reference's honest posture: report the limitation, widen CIs, avoid superiority claims; still strictly better than 6 |
| `en_ner_bc5cdr_md` underperforms generic NER on some entity types | Medium | Ablate NER choice; keep generic fallback as a config option; measure, don't assume |
| Cross-encoder too slow on this hardware | Medium | Bound rerank pool (top-50 → top-10); measure latency; make reranker optional per config |
| GDS memory pressure on student machines | Medium | PageRank/Louvain on retrieved subgraph or bounded entity projection only; document heap/pagecache settings |
| No Docker on the build VM (verified 2026-10-07) | Certain (here) | Neo4j runs on the user's Mac via Docker Desktop (Phase 3); all non-Neo4j phases run anywhere |
| Scope creep: "more features" without measurement | Medium | Hard rule — every component enters the *final* system only with a measured delta (see EXPERIMENT_PLAN.md) |

---

## 5. Computational concerns (student-machine budget)

- **Embeddings:** 5,000 abstracts × ~4 chunks × 384D MiniLM on CPU ≈ minutes; persisted to Parquet, never recomputed.
- **NER:** scispaCy `en_ner_bc5cdr_md` via `nlp.pipe` batching ≈ tens of minutes for 20k chunks on CPU; cached per chunk.
- **HDBSCAN** per abstract is trivial; the improved chunker is O(sentences²) cosine — negligible at abstract scale.
- **Neo4j:** the heavy component. 5k articles / ~20k chunks / ~100k+ entity mentions fits comfortably in a
  4 GB Docker allocation; vector index build is the slowest step (minutes). GDS projections are transient.
- **Cross-encoder:** ~50 pairs/query on CPU ≈ 1–3 s; acceptable for a research UI, measured per query.
- **BERTScore:** `distilbert-base-uncased` on ≤ a few hundred answers ≈ minutes; generation eval is offline/batch.
- **LLM:** API-based (Groq/OpenAI/Ollama); cost tracked per experiment (tokens in/out where the API reports them).

Nothing in the plan requires a GPU. Every expensive operation is batched, cached, bounded, or optional —
per the assignment's performance constraints.
