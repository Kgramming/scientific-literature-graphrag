#!/usr/bin/env python3
"""Phase 5A retrieval evaluation: Hybrid WITHOUT vs WITH CrossEncoder rerank.

Protocol (fair comparison):
  - Same questions: PubMedQA pqa_labeled rows whose pubid is in the corpus.
  - Same corpus/config/top-k candidate pool for both arms.
  - Arm A: Phase 4 hybrid retrieval, fused ranking (no reranker).
  - Arm B: identical candidate pool, reordered by CrossEncoder.
  - The ONLY difference between arms is the reranking stage.
  - Gold = the question's own pubid (question -> source-abstract retrieval).
  - Metrics on PMID-deduplicated rankings: Recall@5, Recall@10, MRR, nDCG@10.

Reads data/evaluation/full_pubmedqa_mapping.json (NEVER modifies it) and
writes data/evaluation/retrieval_eval_5a.json (new file).

Usage:
    no_proxy='*' NO_PROXY='*' HF_HUB_OFFLINE=1 \
      SGRAG_NEO4J_PASSWORD=... SGRAG_SCISPACY_MODEL=... \
      .venv/bin/python scripts/evaluate_retrieval.py [--n 200] [--out ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from sgrag.retrieval import HybridRetriever, RetrievalError  # noqa: E402
from sgrag.retrieval.config import load_retrieval_config  # noqa: E402
from sgrag.retrieval.rerank import CrossEncoderReranker, RerankerError  # noqa: E402

PQA_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/datasets--qiaojin--PubMedQA"
      / "snapshots/9001f2853fb87cab8d220904e0de81ac6973b318"
      / "pqa_labeled/train-00000-of-00001.parquet"
)


def pmid_ranking(results) -> list[str]:
    """Dedupe chunk ranking to PMID ranking, preserving first-seen order."""
    seen: list[str] = []
    for r in results:
        if r.pmid not in seen:
            seen.append(r.pmid)
    return seen


def metrics_for(gold_pmid: str, pmids: list[str]) -> dict:
    try:
        rank = pmids.index(gold_pmid) + 1
    except ValueError:
        rank = None
    rr = 1.0 / rank if rank else 0.0
    hit5 = rank is not None and rank <= 5
    hit10 = rank is not None and rank <= 10
    dcg = 0.0
    for i, p in enumerate(pmids[:10], start=1):
        if p == gold_pmid:
            dcg += 1.0 / math.log2(i + 1)
            break
    return {"rank": rank, "rr": rr, "hit@5": hit5, "hit@10": hit10, "ndcg@10": dcg}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5A retrieval evaluation (A vs B)")
    ap.add_argument("--config", default=str(REPO / "configs" / "retrieval.yaml"))
    ap.add_argument("--mapping", default=str(REPO / "data/evaluation/full_pubmedqa_mapping.json"))
    ap.add_argument("--n", type=int, default=200,
                    help="max questions to evaluate (deterministic first-N)")
    ap.add_argument("--out", default=str(REPO / "data/evaluation/retrieval_eval_5a.json"))
    ap.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--candidate-k", type=int, default=50)
    args = ap.parse_args()

    with open(args.mapping, encoding="utf-8") as fh:
        mapping_doc = json.load(fh)
    mapping: dict[str, list[int]] = mapping_doc["mapping"]
    df = pd.read_parquet(PQA_SNAPSHOT)

    # Usable questions: rows whose pubid is in the corpus mapping.
    usable: list[tuple[int, str, str]] = []  # (row_idx, question, gold_pmid)
    skipped: list[dict] = []
    for idx, row in df.iterrows():
        pmid = str(row["pubid"])
        q = str(row["question"] or "").strip()
        if pmid not in mapping:
            skipped.append({"row": int(idx), "reason": "pubid not in corpus mapping"})
        elif not q:
            skipped.append({"row": int(idx), "reason": "empty question text"})
        else:
            usable.append((int(idx), q, pmid))
    questions = usable[: args.n]

    cfg = load_retrieval_config(args.config)
    # Arm A: reranker disabled in the retriever (fused ranking stands).
    cfg_a = dataclasses.replace(
        cfg, reranker=dataclasses.replace(cfg.reranker, enabled=False))
    reranker = CrossEncoderReranker(model=args.reranker_model,
                                   candidate_k=args.candidate_k)

    per_query: list[dict] = []
    t_start = time.perf_counter()
    try:
        with HybridRetriever(cfg_a) as ret:
            idx_info = ret.check_index()
            print(f"[index] {idx_info['name']} {idx_info['state']}")
            for qi, (row_idx, question, gold) in enumerate(questions):
                try:
                    results_a, diag = ret.retrieve(question)
                except RetrievalError as exc:
                    skipped.append({"row": row_idx, "reason": f"retrieval: {exc}"})
                    continue
                pmids_a = pmid_ranking(results_a)
                m_a = metrics_for(gold, pmids_a)
                # Arm B: SAME candidate pool, CrossEncoder reorder.
                t_r = time.perf_counter()
                try:
                    results_b = reranker.rerank(
                        question, [r for r in results_a])
                except RerankerError as exc:
                    skipped.append({"row": row_idx,
                                    "reason": f"reranker: {exc}"})
                    continue
                rerank_ms = (time.perf_counter() - t_r) * 1000
                pmids_b = pmid_ranking(results_b)
                m_b = metrics_for(gold, pmids_b)
                per_query.append({
                    "row_idx": row_idx,
                    "question": question,
                    "gold_pmid": gold,
                    "baseline_pmids": pmids_a[:10],
                    "reranked_pmids": pmids_b[:10],
                    "baseline_chunk_ids": [r.chunk_id for r in results_a[:10]],
                    "reranked_chunk_ids": [r.chunk_id for r in results_b[:10]],
                    "baseline": m_a,
                    "reranked": m_b,
                    "top_reranker_scores": [
                        round(r.reranker_score, 4)
                        for r in results_b[:10] if r.reranker_score is not None],
                    "retrieval_ms": diag["total_ms"],
                    "rerank_ms": round(rerank_ms, 1),
                })
                if (qi + 1) % 25 == 0:
                    print(f"[{qi + 1}/{len(questions)}] ...")
    except RetrievalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    def agg(key):
        vals_a = [q["baseline"][key] for q in per_query]
        vals_b = [q["reranked"][key] for q in per_query]
        mean = lambda v: (sum(v) / len(v)) if v else 0.0  # noqa: E731
        return {"A": round(mean(vals_a), 4), "B": round(mean(vals_b), 4),
                "delta": round(mean(vals_b) - mean(vals_a), 4)}

    report = {
        "protocol": {
            "arm_A": "Phase 4 hybrid retrieval, fused ranking (no reranker)",
            "arm_B": "identical candidate pool reordered by CrossEncoder",
            "gold": "question pubid (question -> source-abstract retrieval)",
            "ranking": "PMID-deduplicated, first-seen order",
            "candidate_k": args.candidate_k,
            "reranker_model": args.reranker_model,
            "retrieval_config": str(args.config),
            "selection": f"deterministic first-N rows of pqa_labeled/train, N={args.n}",
            "mapping_file": str(args.mapping),
        },
        "n_usable_questions": len(usable),
        "n_evaluated": len(per_query),
        "n_skipped": len(skipped),
        "skips": skipped,
        "aggregate": {
            "recall@5": agg("hit@5"),
            "recall@10": agg("hit@10"),
            "mrr": agg("rr"),
            "ndcg@10": agg("ndcg@10"),
        },
        "elapsed_s": round(time.perf_counter() - t_start, 1),
        "per_query": per_query,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[eval] usable={len(usable)} evaluated={len(per_query)} "
          f"skipped={len(skipped)} -> {out}")
    for name, vals in report["aggregate"].items():
        print(f"  {name:10s} A={vals['A']:.4f}  B={vals['B']:.4f}  "
              f"delta={vals['delta']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
