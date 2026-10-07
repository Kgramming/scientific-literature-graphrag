# Experiment Plan

## 1. Systems under test (ablation matrix)

All systems run on **exactly the same evaluation questions**. Each maps to one YAML config in `configs/`.

| System | Config | Definition — what changes vs. the previous row |
|---|---|---|
| **A** | `baseline.yaml` | Dense vector baseline: fixed-token chunks, `all-MiniLM-L6-v2`, vector-only retrieval, no graph, no reranker |
| **B** | `semantic.yaml` | A + improved semantic chunking (`semantic_sections`) — **isolates chunking** |
| **C** | `graph.yaml` | A + entity-aware retrieval + bounded graph expansion — **isolates expansion** (fixed-token chunks, so chunking is not a confound) |
| **D** | `fusion.yaml` | C + hybrid fusion (RRF / weighted, config weights) + query-class routing — **isolates fusion** |
| **E** | `reranker.yaml` | D + cross-encoder rerank + graph-aware final score — **isolates reranking** |
| **F** | `decomposition.yaml` | E + validated query decomposition for complex classes — **isolates decomposition** |
| **G** | `full.yaml` | F + semantic chunking + community signal (only if ablations justify it) — **proposed system** |

Additional pairwise studies: embedding baseline vs biomedical candidate (on System A); NER generic vs
BC5CDR (entity-count + retrieval delta); fusion weight sensitivity grid; PageRank on/off;
Louvain signal on/off.

## 2. Evaluation dataset construction (deterministic, auditable)

1. Load PubMedQA `pqa_labeled` (1,000 questions); per question, gold PMIDs = context `pmids`.
2. Fetch those abstracts (real PMIDs preserved) + PubMed fill to ~5,000 → `corpus.jsonl` + manifest.
3. Eval set = questions whose **all** gold PMIDs are in the corpus. Everything else is listed in the
   manifest's dropped-question audit with reasons — never silently dropped, never fabricated.
4. Artifacts: `data/evaluation/questions.jsonl`, `data/evaluation/manifest.json`
   (construction script, seed, timestamp, counts, match rate, drops).

Target: hundreds of verified questions (vs. the reference's 6). If the verified set is small, we report
the limitation explicitly and widen confidence intervals — the honest posture, with better numbers.

## 3. Metrics

**Retrieval** (per question, then averaged):
- Recall@5, Recall@10, MRR, nDCG@10, Precision@5
- **Article level:** gold = question's PMIDs; hit = any retrieved chunk from a gold article.
- **Chunk level (weak labels):** gold = chunks belonging to gold articles; marked WEAK in every table.
  Rationale (documented in `docs/evaluation.md`): PubMedQA's gold granularity is the abstract, so
  article-level is the strong metric; chunk-level is diagnostic for chunking comparisons.

**Generation** (vs. PubMedQA `long_answer`):
- ROUGE-L F1, BERTScore F1 (required)
- **Faithfulness suite:** citation precision (cited chunk IDs ⊆ retrieved IDs), evidence coverage
  (answer entities ∩ cited-chunk entities), unsupported-claim rate via NLI (**labeled auxiliary** —
  we do not claim it measures factual truth).

**Latency/resources** (per query): embedding ms, retrieval ms, expansion ms, rerank ms, LLM ms,
total ms; candidates retrieved, graph nodes traversed, tokens in/out. Reported as medians/p90 and an
accuracy-vs-latency plot.

## 4. Statistics & honest-reporting rules

- Paired per-question scores; **Wilcoxon signed-rank** + 95% CI of the mean delta per metric.
- A delta whose CI crosses zero is reported as **inconclusive**, not as an improvement.
- No question cherry-picking: identical question sets across systems; any subset analysis (e.g.
  multi-hop only) is pre-registered here and reported alongside full-set results.
- Negative results are published (e.g. "semantic chunking did not improve Recall@10: Δ=−0.8%,
  95% CI [−2.1, +0.5]").
- Every run directory records: config hash, model versions, dataset manifest hash, seed, timestamp,
  hardware note. `run_experiment.py --config <f>` reproduces the CSV bit-for-bit.

## 5. Research questions → measurements (traceability)

| Question | Measured by |
|---|---|
| Why is GraphRAG better? | G vs A deltas on retrieval + faithfulness |
| When is graph better than vector? | Per-query-class breakdown (C vs A by class) |
| When does graph expansion hurt? | C vs A on single-fact subset; adaptive-routing analysis |
| Does semantic chunking help retrieval? | B vs A (+ cohesion plots) |
| Does query decomposition help multi-hop? | F vs E on multi-hop subset; degradation check on simple |
| Does graph-aware reranking help Recall/MRR? | E vs D |
| Does a stronger reranker improve evidence quality? | E vs D on faithfulness suite |
| Grounded answers? | Citation precision + evidence coverage (all systems) |
| Latency/cost tradeoff? | Per-stage latency table + accuracy-vs-latency plot |

## 6. Artifacts per run

`results/<run_id>/`: `config.yaml` (resolved), `metrics.csv` (per-system), `per_question.jsonl`,
`comparison.md` (deltas + CIs), plots (`recall_at_k.png`, `mrr.png`, `latency.png`, `cohesion.png`),
`provenance.json`. The UI Evaluation tab reads these directly — no hand-copied numbers.
