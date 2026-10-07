# Architecture Proposal — Scientific Literature GraphRAG

**Status:** proposal (planning phase). Implementation begins only after approval.
**Design rule:** every major component must be *measurable* — it ships in the final system only if
experiments show a benefit (see `EXPERIMENT_PLAN.md`). Complexity without a measured delta is rejected.

---

## 1. System overview

```text
                        USER QUESTION
                              |
                              v
                    ┌─ Query Understanding ─┐
                    │  classification       │
                    │  (rule-based,         │
                    │   explainable)        │
                    └─────────┬─────────────┘
                              |
                 ┌──────────────┴──────────────┐
                 |                             |
        routing table                  decomposition (gated:
        selects strategy                only for complex classes,
        weights per class               validated sub-questions)
                 |                             |
                 └──────────────┬──────────────┘
                                v
                   Multi-Strategy Retrieval
              ┌──────────┬──────────────┬────────────┐
              │          │              │            │
        Vector search  Entity-aware   Graph       (Community:
        (Neo4j HNSW,    (query ents →  expansion   optional
         cosine)        chunks via    (bounded     GDS signal,
                        MENTIONS)     1–2 hops)   ablated)
              │          │              │            │
              └──────────┴──────┬───────┴────────────┘
                                v
                       Candidate Fusion
                 (RRF or weighted normalized sum;
                  weights from YAML config)
                                v
                    Graph Expansion / Traversal
                 (from fused seeds; hub-guarded)
                                v
                   Cross-Encoder Reranking
                 (ms-marco-MiniLM-L-6-v2, top-N pool)
                                v
                   Graph-Aware Reranking
         final = w1·vector + w2·graph + w3·entity
               + w4·reranker + w5·evidence_quality
              (each term normalized; weights in config;
               per-candidate explanation)
                                v
                      Evidence Selection
                 (top-m within token budget, dedup)
                                v
                     Context Compression
                                v
                    Grounded LLM Generation
                 (evidence-first prompt; citations
                  validated against retrieved chunk IDs;
                  explicit insufficient-evidence fallback)
                                v
                   Citation / Evidence Mapping
                 (claim → chunk_id links for the UI)
                                v
                        Answer + Graph UI
```

---

## 2. Module design

### 2.1 Data pipeline (`src/sgrag/data/`)

- **Corpus construction (deterministic):** load PubMedQA `pqa_labeled` (1,000 questions) → extract gold
  PMIDs per question from context `pmids` → fetch those abstracts (real PMIDs preserved) → sample
  additional PubMed abstracts to reach ~5,000 → write `data/processed/corpus.jsonl` + `manifest.json`
  (source, seed, timestamp, fetch failures). *Why:* PMID-equality matching makes gold mappings
  deterministic and auditable — the direct fix for the reference's 6-question bottleneck.
- **Stages:** `data/raw/` (immutable downloads) → `data/interim/` (chunked, embedded, NER output) →
  `data/processed/` (graph-load-ready) → `data/evaluation/` (question manifests — the only stage
  committed to git, small JSON).
- **Provenance:** every artifact carries `{producer_script, config_hash, seed, timestamp}`.

### 2.2 Chunking (`src/sgrag/chunking/`) — 4 strategies, honestly compared

| Strategy | Method | Purpose |
|---|---|---|
| A. `fixed_token` | 256 WordPiece tokens (MiniLM tokenizer), 32-token overlap | Required baseline |
| B. `sentence_boundary` | scispaCy sentences packed to ≤256 tokens, never split | Required baseline |
| C. `semantic_cluster` | Sentence embeddings + HDBSCAN (`min_cluster_size=2`), noise→singletons | Reproduces reference behavior for comparison |
| D. `semantic_sections` (**ours**) | Adjacent-sentence cosine similarity → breakpoints at low-similarity percentiles → enforce min 2 sentences / max 512 tokens (split oversized at weakest internal boundary, merge tiny with neighbor) → topic label from centroid-proximal terms | Improved chunker that **preserves sections** |

Chunk record: `article_id, pmid, chunk_id, title, strategy, ordinal, sentence_start/end, text,
token_count, embedding_ref, entities (filled post-NER), source metadata`.
**Cohesion analysis:** intra-chunk vs inter-chunk cosine-similarity distributions per strategy
(`analyze_cohesion.py` + plots). If D does not beat C on retrieval, we report that.

### 2.3 Embeddings (`src/sgrag/embeddings/`)

- **Baseline (required):** `all-MiniLM-L6-v2`, 384D — used for everything unless an experiment overrides it.
- **Experiment candidate:** a biomedical sentence embedding (Phase 1 validates candidates, e.g.
  PubMedBERT-derived or BGE variants). Adoption rule: only if retrieval metrics improve on identical
  questions; otherwise the report states the negative result.
- Cached `SentenceTransformer` loader, dimension validation, Parquet persistence (never re-embed).

### 2.4 Entities (`src/sgrag/entities/`)

- **Extractor:** scispaCy `en_ner_bc5cdr_md` — DISEASE and CHEMICAL entities, trained on the BC5CDR
  biomedical corpus. *Why this over generic `en_core_sci_sm`:* PubMed abstracts are biomedical text;
  a domain NER gives typed, reliable entities instead of generic spans. (Ablated in experiments.)
- **Normalization pipeline** (each step unit-tested):
  1. casefold, whitespace collapse, punctuation/quote/bracket cleanup, possessive strip;
  2. abbreviation expansion via parenthetical pattern (`"tumor necrosis factor (TNF)"` → alias link);
  3. curated alias map for frequent variants (versioned file, documented);
  4. conservative plural handling for biomedical terms (measured, not assumed);
  5. noise filters (pure numbers, >150 chars, <2 chars).
- **Canonical node:** `{canonical_name, type, surface_forms[], source_chunk_ids[], extraction_method}`.
  Duplicates merge by canonical key — never raw strings.

### 2.5 Knowledge graph (`src/sgrag/graph/`)

```text
(:Article {pmid, title, year, journal})-[:HAS_CHUNK]->(:Chunk {chunk_id, strategy, ordinal,
    sentence_start, sentence_end, text, token_count, embedding})
(:Chunk)-[:MENTIONS {confidence, extraction_method}]->(:Entity {canonical_name, type, surface_forms})
(:Entity)-[:RELATED_TO {method:"cooccurrence", support, pmi}]->(:Entity)   (thresholded, documented)
(:Article)-[:HAS_TOPIC]->(:Concept {mesh_id, term})                        (MeSH, when source provides it)
```

- **Constraints/indexes:** uniqueness on `Article.pmid`, `Chunk.chunk_id`, `Entity.canonical_name`,
  `Concept.mesh_id`; indexes on `Entity.type`, `Chunk.strategy`.
- **Vector index:** `chunk_embedding`, cosine, 384D (recreated explicitly on model change — never silently).
- **APOC:** batch inserts (`apoc.periodic.iterate`) and conditional merges.
- **GDS:** transient projections only — PageRank over the retrieved subgraph; Louvain/Leiden over a
  bounded entity co-occurrence projection as an *optional, ablated* signal.
- **No O(N²):** `RELATED_TO` built from per-chunk co-occurrence aggregation (linear in mentions);
  similarity edges (if any) use the vector index, never pairwise scans.

### 2.6 Retrieval (`src/sgrag/retrieval/`)

Four strategies, individually callable and ablatable:
1. **Vector:** Neo4j vector index top-k (3× pool for downstream stages).
2. **Entity-aware:** query → same NER/normalization pipeline → `Entity` match → chunks via `MENTIONS`,
   scored by matched-entity weight.
3. **Graph expansion:** from seed chunks/entities, 1–2 hop traversal (`MENTIONS`, `RELATED_TO`,
   `HAS_CHUNK` reverse), hub-guarded (skip entities with degree > cap), bounded fan-out.
4. **Hybrid:** fusion of the above.

**Fusion:** two implemented options, selected in config — **RRF** (k=60) and **weighted normalized sum**.
Weights are config-driven; a weight-sensitivity script reports metric deltas over a grid (no "one true
weighting" assumed).

### 2.7 Query understanding (`src/sgrag/query/`)

- **Classifier:** deterministic rule-based taxonomy — `single_fact, comparison, relationship, causal,
  multi_hop, methodological, evidence_seeking` — from lexical/syntactic cues (documented patterns,
  unit-tested). *Why rule-based:* routing must be explainable; an LLM classifier would be a black box
  deciding the pipeline.
- **Routing table** (in config, shown in UI): e.g. `single_fact → vector-heavy`,
  `relationship → entity+graph`, `multi_hop → decomposition+graph`. Classification + chosen route are
  logged per query.

### 2.8 Query decomposition (`src/sgrag/query/decompose.py`)

LLM proposes sub-questions **only** for complex classes; deterministic validators enforce:
non-empty, ≤4 sub-questions, pairwise distinct, term-overlap with the original (drift guard),
and per-sub-query retrieval must return candidates or the sub-query is dropped with fallback to the
original query. A bad decomposition therefore *cannot* silently destroy retrieval — the worst case is
the original query's results. Ablated as System F, including a "simple-question degradation" check.

### 2.9 Reranking (`src/sgrag/reranking/`)

1. **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` scores (query, chunk) over a bounded pool
   (top-50 → keep 10). *Why this model:* openly available, well-benchmarked on MS MARCO, CPU-feasible.
2. **Graph-aware score:**
   `final = w1·vector + w2·graph_proximity + w3·entity_overlap + w4·reranker + w5·evidence_quality`,
   each term min-max normalized over the candidate pool, weights from config. `graph_proximity` from
   shortest-path distance to query entities; `evidence_quality` from chunk length/entity-density sanity
   signals (documented, small weight). Every candidate carries its component breakdown → UI
   explainability tab.

### 2.10 Generation (`src/sgrag/generation/`)

- **LLM client:** provider-agnostic (`groq` / `openai` / `ollama`), model + key from `SGRAG_*` env vars.
- **Evidence-first prompt:** numbered context chunks, instructions to cite `[C<chunk_id>]`, distinguish
  evidence from inference, and emit an explicit insufficient-evidence statement when coverage is thin.
- **Citation validation:** post-generation check that every cited ID exists in the retrieved set;
  invalid citations are flagged (never silently kept).
- **Claim→chunk mapping:** answer sentences aligned to cited chunks for UI inspection.

### 2.11 Evaluation (`src/sgrag/evaluation/`)

- **Dataset builder:** deterministic PMID matching (see §2.1); manifest with counts + dropped-question audit.
- **Retrieval metrics:** Recall@5, Recall@10, MRR, nDCG@10, Precision@5 — at **article level** (gold =
  question's PMIDs; hit = any chunk from a gold article in top-k) **and chunk level** (weak labels:
  chunks of gold articles; clearly marked as weak). Docs explain the difference and why both matter.
- **Generation metrics:** ROUGE-L, BERTScore (required) + **faithfulness suite**: citation precision
  (cited IDs ⊆ retrieved IDs), evidence coverage (answer entities ∩ cited-chunk entities), unsupported-
  claim rate via NLI (labeled auxiliary — we do not claim it measures truth).
- **Latency tracker:** per-stage ms + candidates traversed + nodes visited + tokens; accuracy-vs-latency
  analysis in the final report.
- **Experiment runner:** `run_experiment.py --config configs/<name>.yaml` → stamps results with config
  hash, model versions, dataset manifest, seed, timestamp → `results/<run_id>/` (metrics CSV, per-question
  details, plots). Re-running reproduces the CSV.

### 2.12 UI (`app/`)

Streamlit, 6 tabs per the assignment: **Ask** (answer + citations + evidence), **Retrieval** (chunks,
scores, ranking explanation, strategy), **Knowledge Graph** (PyVis interactive graph, path highlighting),
**Explainability** (per-chunk vector/entity/graph/reranker/final scores), **Evaluation** (all metrics,
baseline-vs-proposed, ablations), **System** (models, dataset, config, latency). Research-prototype
styling, not a bare form.

### 2.13 Testing (`tests/`)

- `unit/`: chunking, normalization, fusion/RRF, score normalization, classifier, decomposition
  validators, metrics — no services, no keys.
- `integration/`: Neo4j (marked; skipped without a live instance), vector index, traversal, retrieval
  and generation pipelines on a tiny fixture.
- `e2e/`: question→retrieval→graph→answer→citations smoke test (marked; needs LLM key).

---

## 3. Technology choices (with alternatives considered)

| Decision | Choice | Why | Alternative considered |
|---|---|---|---|
| Baseline embedding | `all-MiniLM-L6-v2` (384D) | Assignment-required; fast on CPU | — (required) |
| Stronger embedding (experiment) | Biomedical sentence model, validated in Phase 1 | Domain match may help; adopted only on measured gain | BGE/GTE general models |
| NER | scispaCy `en_ner_bc5cdr_md` | DISEASE/CHEMICAL types trained on biomedical corpus | Generic `en_core_sci_sm` (reference choice); UMLS linker (heavy, API-bound) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Open, benchmarked, CPU-feasible | Larger cross-encoders (too slow on CPU) |
| Graph DB | Neo4j 5 + APOC + GDS (Docker) | Assignment-required; vector index + GDS in one engine | — (required) |
| LLM | Configurable: Groq / OpenAI / Ollama | Cost control; provider swap without code change | Hardcoded single provider |
| UI | Streamlit + PyVis | Assignment allows Streamlit; PyVis for interactive graphs | Gradio (weaker multi-tab research layout) |
| Metrics | rouge-score, bert-score, sklearn | Standard implementations | Custom reimplementations (rejected: use standards) |
| Config | YAML per experiment | Human-readable, diffable, one-command runs | Env-var-only (reference approach; rejected for provenance) |

---

## 4. Open questions (resolved in Phase 0/1, before expensive work)

1. Exact HF corpus source carrying **real PMIDs** for the fill abstracts (candidates: PubMedQA-derived,
   Entrez fetch, or an HF PubMed mirror — validated in Phase 1).
2. Whether source records include **MeSH headings** (decides `HAS_TOPIC` edges).
3. Stronger-embedding candidate shortlist (benchmarked on a 200-abstract pilot).
4. Neo4j host: user's Mac via Docker Desktop (this VM has no Docker — verified) vs. another host.
5. LLM default provider/model + fresh `SGRAG_` API key (secure vault when needed; never in chat/files).
