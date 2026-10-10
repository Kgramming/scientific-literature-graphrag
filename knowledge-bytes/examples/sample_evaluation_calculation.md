# Example: Sample Evaluation Calculation

**Type:** worked example using the *real* metric definitions from `scripts/evaluate_retrieval.py` (`metrics_for`). Numbers below are illustrative; the reported aggregates are real.

**Setup:** one PubMedQA question, gold answer = abstract with PMID `99999999`.

**System output** (PMID ranking from `pmid_ranking()`):

```text
rank 1: PMID 11111111
rank 2: PMID 99999999   ← gold
rank 3: PMID 22222222
rank 4: PMID 33333333
rank 5: PMID 44444444
...
```

**Metrics for this question** (`metrics_for("99999999", pmids)`):

| Metric | Definition | Value here |
|---|---|---|
| Recall@5 | gold in top 5? | **1** (rank 2 ≤ 5) |
| Recall@10 | gold in top 10? | **1** |
| MRR | 1 / (rank of gold) | **0.5** (1/2) |
| nDCG@10 | graded ranking quality | computed from rank positions |

**Aggregation:** average each metric over all 200 questions → the reported table:

| Metric | Hybrid | Hybrid + CrossEncoder |
|---|---|---|
| Recall@5 | 0.9450 | **0.9850** |
| MRR | 0.8623 | **0.9689** |

**Why PMID-equality:** the gold label is objective (the source abstract's PMID) — no human judges, no LLM-as-judge bias. A "hit" is unambiguous.

**Scope honesty** (from `docs/DEMO_GUIDE.md` §8): *"This is a retrieval benchmark, not open-domain scientific QA accuracy."*
