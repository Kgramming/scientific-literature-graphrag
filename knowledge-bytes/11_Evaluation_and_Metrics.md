# 11 — Evaluation and Metrics

## 1. What this component does

Measures whether the system actually works: retrieval quality on 200 PubMedQA questions, plus the framework for generation quality (ROUGE, BERTScore, faithfulness). Separates *measured* results from *planned* ones.

## 2. Why it exists

"This project works" is a claim; "Recall@5 0.9850 on 200 PubMedQA questions" is evidence. The evaluation framework turns architectural choices (reranking, chunking strategies, embedding models) into numbers a professor can interrogate.

## 3. Where it lives in the repository

- `scripts/evaluate_retrieval.py` — `pmid_ranking()`, `metrics_for()`, `main()`: the retrieval benchmark runner
- `data/evaluation/retrieval_eval_5a.json` — full per-query results (committed)
- `data/evaluation/pilot_pubmedqa_mapping.json` — question→gold-PMID mapping
- `data/evaluation/full_corpus_manifest.json` — corpus provenance (protected local file)
- `docs/EXPERIMENT_PLAN.md` — what each experiment measures (§2: deterministic eval-set construction)
- `docs/DEMO_GUIDE.md` §8 — the reported results table
- `src/sgrag/evaluation/` — currently only `__init__.py` (evaluation *logic* lives in scripts/)

## 4. Important classes/functions

- **`pmid_ranking(results)`** — maps a retrieval result list to an ordered PMID list (article-level ranking from chunk results).
- **`metrics_for(gold_pmid, pmids)`** — computes per-question: Recall@5, Recall@10, MRR, nDCG@10 via PMID-equality (is the gold abstract's PMID in the top-k? at what rank?).
- **`main()`** — runs all 200 questions through the retriever (hybrid vs hybrid+cross-encoder), aggregates metrics, writes `retrieval_eval_5a.json`.

## 5. Step-by-step execution

```text
scripts/evaluate_retrieval.py
  → load 200 PubMedQA questions + gold PMIDs (mapping JSON)
  → for each question:
      → retriever.retrieve(question)   [hybrid]
      → pmid_ranking(results)          [chunk hits → PMID ranking]
      → metrics_for(gold_pmid, pmids)  [R@5, R@10, MRR, nDCG@10]
  → repeat with cross-encoder reranking enabled
  → aggregate → data/evaluation/retrieval_eval_5a.json
```

## 6. Data flow

`200 questions + gold PMIDs` → per-question rankings → per-question metrics → aggregated table → JSON results file → reported in docs.

## 7. How it connects to the rest of the system

Evaluation *uses* the production retriever (`HybridRetriever`) — it measures the real pipeline, not a toy. The manifests in `data/evaluation/` make the eval set deterministic: same questions, same gold answers, every run.

## 8. Important design decisions

- **PMID-equality gold matching:** the gold answer is the source abstract's PMID — objective, no human judgment, no LLM-as-judge bias.
- **Deterministic eval set:** `docs/EXPERIMENT_PLAN.md` §2 describes the construction; reruns are comparable.
- **Ablation systems A–G:** identical questions across configurations isolate each component's contribution.
- **What was ACTUALLY measured:** retrieval metrics on 200 questions (table below). **Planned/not measured here:** generation metrics (ROUGE-L, BERTScore, faithfulness suite) are *framework-described* in docs, not reported as results — do not present them as measured.

## 9. Example from the actual project (measured results)

Question→source-abstract retrieval, 200 PubMedQA questions, PMID-deduplicated ranking:

| Metric | Hybrid | Hybrid + CrossEncoder | Δ |
|---|---|---|---|
| Recall@5 | 0.9450 | **0.9850** | +0.0400 |
| Recall@10 | 0.9750 | 0.9850 | +0.0100 |
| MRR | 0.8623 | **0.9689** | +0.1066 |
| nDCG@10 | 0.8885 | 0.9729 | +0.0844 |

Source: `docs/DEMO_GUIDE.md` §8, `data/evaluation/retrieval_eval_5a.json`.
**Scope honesty:** "This is a retrieval benchmark, not open-domain scientific QA accuracy."

## 10. Common misunderstandings

- **"0.9850 is answer accuracy."** No — it's *retrieval* Recall@5: the gold abstract was in the top 5 for 98.5% of questions. Answer quality is a separate (planned) measurement.
- **"ROUGE/BERTScore were measured."** The framework is described; results are not reported in the repo. Say "planned," not "done."
- **"More metrics = better."** Each metric answers a different question: Recall@k (did we find it?), MRR (how high did it rank?), nDCG@10 (ranking quality with graded relevance).

## 11. Professor Viva Questions

- How did you evaluate your system?
- What is Recall@5? What is MRR?
- Why PMID-equality instead of human judges?
- What did the cross-encoder actually improve, numerically?
- What *didn't* you measure?

## 12. Natural Spoken Answers

**"How did you evaluate?"**
> "Sir, we built a deterministic eval set of 200 PubMedQA questions, each mapped to its gold source abstract by PMID. For each question we run our retriever, convert chunk hits to a PMID ranking, and check: is the gold PMID in the top 5 — that's Recall@5 — and at what rank — that's MRR. Hybrid retrieval alone gave Recall@5 0.9450; adding the cross-encoder reranker lifted it to 0.9850. I should be precise, sir: this measures retrieval, not final answer accuracy — generation metrics like ROUGE are described in our framework but not yet measured."

## 13. Knowledge Bites

- Eval set: 200 PubMedQA questions, PMID-equality gold matching, deterministic.
- `metrics_for()`: Recall@5/10, MRR, nDCG@10 per question.
- Measured: Recall@5 0.9850, MRR 0.9689 (hybrid + cross-encoder).
- Biggest measured win: the cross-encoder (+0.0400 Recall@5, +0.1066 MRR).
- Per-query results committed: `data/evaluation/retrieval_eval_5a.json`.
- Scope honesty: retrieval benchmark, NOT open-domain QA accuracy.
- ROUGE/BERTScore/faithfulness: framework described, results not measured — say "planned."
- Evaluation uses the production `HybridRetriever` — real pipeline, not a toy.
