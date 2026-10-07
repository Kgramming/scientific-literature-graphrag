#!/usr/bin/env python3
"""Embedding comparison: baseline vs one biomedical candidate (pilot only).

Experimental control: the ONLY variable is the embedding model. Both models
see the same 200 abstracts, the same sentence splits, the same FIXED S1
semantic chunk boundaries, the same cosine-similarity methodology, and no
model-specific tuning.

Two evaluations:
  1. Coherence — the Phase 2 methodology (intra-chunk vs adjacent
     inter-chunk cosine, gap) recomputed per model on the fixed chunks.
     Caveat: chunk boundaries were produced with the baseline's embeddings,
     giving the baseline a mild home-field advantage here.
  2. Title->sentence retrieval — for each article title (query), rank all
     2293 abstract sentences by cosine similarity. Relevant = sentences from
     the query's own abstract. Labels come from corpus structure only, so
     there is no leakage and no home-field advantage. Metrics: MRR and
     success@k (fraction of titles with a relevant sentence in the top k).

Decision rule: recommend the candidate only on measured, retrieval-relevant
advantage. Negligible difference -> keep all-MiniLM-L6-v2. Inconclusive ->
say so.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgrag.chunking.base import Chunk
from sgrag.chunking.coherence import coherence_for_article, cosine_sim
from sgrag.embeddings import EmbeddingConfig, encode
from sgrag.chunking.sentences import split_sentences
from sgrag.utils.env import sanitize_proxy_env


def title_retrieval(title_embs, sent_embs, relevant_idx, recall_k=(1, 5, 10)):
    """title_embs: (200, d); sent_embs: (2293, d); relevant_idx: list per title
    of sentence indices from the same abstract. Returns MRR, success@k."""
    # Cosine similarity via normalized dot product.
    tn = title_embs / np.linalg.norm(title_embs, axis=1, keepdims=True)
    sn = sent_embs / np.linalg.norm(sent_embs, axis=1, keepdims=True)
    sims = tn @ sn.T  # (200, 2293)
    rr, success = [], {k: 0 for k in recall_k}
    n = sims.shape[0]
    for i in range(n):
        ranking = np.argsort(-sims[i], kind="stable")
        rel = set(relevant_idx[i])
        first = next(r for r, j in enumerate(ranking, start=1) if j in rel)
        rr.append(1.0 / first)
        for k in recall_k:
            if first <= k:
                success[k] += 1
    return {"mrr": float(np.mean(rr)),
            **{f"success@{k}": success[k] / n for k in recall_k},
            "n_queries": n, "n_candidates": sims.shape[1]}


def build_report(results, meta):
    """Assemble the comparison report dict (pure; schema-tested)."""
    report = {"experiment": meta["experiment"], "models": {}}
    for name, r in results.items():
        report["models"][name] = {
            "config": r["config"],
            "embedding_dim": r["provenance"]["embedding_dim"],
            "normalize": r["config"]["normalize"],
            "device": r["config"]["device"],
            "encode_seconds_sentences": r["provenance"]["encode_seconds"],
            "encode_seconds_titles": r["title_provenance"]["encode_seconds"],
            "n_texts": r["provenance"]["n_texts"],
            "cache_hit_sentences": r["provenance"]["cache_hit"],
            "timestamp_utc": r["provenance"]["timestamp_utc"],
            "coherence": r["coherence"],
            "title_retrieval": r["retrieval"],
            "storage_bytes": r["storage_bytes"],
        }
    b, c = report["models"]["baseline"], report["models"]["candidate"]
    report["deltas"] = {
        "coherence_gap": c["coherence"]["gap_mean"] - b["coherence"]["gap_mean"],
        "mrr": c["title_retrieval"]["mrr"] - b["title_retrieval"]["mrr"],
        **{f"success@{k}": c["title_retrieval"][f"success@{k}"]
                            - b["title_retrieval"][f"success@{k}"]
           for k in meta["recall_k"]},
        "encode_time_ratio": (c["encode_seconds_sentences"] /
                              b["encode_seconds_sentences"]
                              if b["encode_seconds_sentences"] else None),
        "dim_ratio": c["embedding_dim"] / b["embedding_dim"],
        "storage_ratio": (c["storage_bytes"] / b["storage_bytes"]
                          if b["storage_bytes"] else None),
    }
    return report


def main():
    sanitize_proxy_env()
    ap = argparse.ArgumentParser(description="Compare baseline vs candidate embeddings.")
    ap.add_argument("--config", default="configs/embedding_comparison.yaml")
    args = ap.parse_args()
    t_start = time.perf_counter()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = Path(cfg["experiment"]["out_dir"])
    res_dir = out_dir
    res_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(Path(cfg["experiment"]["corpus"]), encoding="utf-8")
               if l.strip()]
    chunks = [json.loads(l) for l in open(Path(cfg["experiment"]["chunks"]), encoding="utf-8")
              if l.strip()]
    print(f"[emb] {len(records)} articles, {len(chunks)} fixed chunks")

    # One shared sentence inventory (identical inputs for both models).
    sentences, sent_owner, titles = [], [], []
    for r in records:
        pmid = str(r["pmid"])
        titles.append(r.get("title") or "")
        for s in split_sentences(r.get("abstract") or ""):
            sent_owner.append(pmid)
            sentences.append(s)
    print(f"[emb] {len(sentences)} sentences, {len(titles)} titles")
    assert len(sentences) == 2293, f"expected 2293 pilot sentences, got {len(sentences)}"

    # Fixed chunks grouped per article (boundaries never change).
    by_art = {}
    for ch in chunks:
        by_art.setdefault(ch["pmid"], []).append(Chunk(**ch))
    for v in by_art.values():
        v.sort(key=lambda c: c.ordinal)
    art_emb_slices = {}
    idx_by_pmid = {}
    for i, pmid in enumerate(sent_owner):
        idx_by_pmid.setdefault(pmid, []).append(i)

    results = {}
    for name, m in cfg["models"].items():
        ecfg = EmbeddingConfig(model_name=m["model_name"], normalize=m["normalize"],
                               device=m["device"], batch_size=m["batch_size"],
                               cache_dir=cfg["experiment"]["cache_dir"])
        sent_embs, prov = encode(sentences, ecfg)
        title_embs, title_prov = encode(titles, ecfg)
        dim = sent_embs.shape[1]
        assert sent_embs.shape == (len(sentences), dim)
        assert title_embs.shape == (len(titles), dim)

        # 1. Coherence on fixed chunks (Phase 2 methodology).
        # Chunk sentence indices are already 0-based per article, matching
        # the per-article embedding slice order — no rebasing needed.
        gaps, intras, inters = [], [], []
        for pmid, chs in by_art.items():
            idxs = idx_by_pmid[pmid]
            emb = sent_embs[idxs]
            r = coherence_for_article(chs, emb)
            if r["intra_mean"] == r["intra_mean"]:
                intras.append(r["intra_mean"])
            if r["inter_mean"] == r["inter_mean"]:
                inters.append(r["inter_mean"])
            if r["gap"] == r["gap"]:
                gaps.append(r["gap"])
        coherence = {"intra_chunk_cosine_mean": float(np.mean(intras)),
                     "inter_chunk_cosine_mean": float(np.mean(inters)),
                     "gap_mean": float(np.mean(gaps)),
                     "n_articles": len(gaps)}

        # 2. Title -> sentence retrieval (structure-derived labels).
        rel = [idx_by_pmid[str(r["pmid"])] for r in records]
        retrieval = title_retrieval(title_embs, sent_embs, rel,
                                    tuple(cfg["evaluation"]["title_retrieval"]["recall_k"]))

        storage = Path(ecfg.cache_dir).glob(f"{ecfg.cache_slug()}.*.npy")
        storage_bytes = sum(p.stat().st_size for p in storage)

        results[name] = {"config": ecfg.to_dict(), "provenance": prov,
                         "title_provenance": title_prov, "coherence": coherence,
                         "retrieval": retrieval, "storage_bytes": storage_bytes}
        print(f"[emb] {name}: dim={dim} intra={coherence['intra_chunk_cosine_mean']:.4f} "
              f"inter={coherence['inter_chunk_cosine_mean']:.4f} gap={coherence['gap_mean']:+.4f} "
              f"mrr={retrieval['mrr']:.4f} enc={prov['encode_seconds']:.1f}s")

    meta = {"experiment": cfg["experiment"]["name"],
            "recall_k": cfg["evaluation"]["title_retrieval"]["recall_k"],
            "chunks_file": cfg["experiment"]["chunks"],
            "total_seconds": time.perf_counter() - t_start}
    report = build_report(results, meta)
    json.dump(report, open(res_dir / "embedding_comparison.json", "w", encoding="utf-8"), indent=2)
    print(f"[emb] wrote {res_dir / 'embedding_comparison.json'}")

    # Plot: coherence gap + MRR, baseline vs candidate.
    names = ["baseline", "candidate"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    gaps = [report["models"][n]["coherence"]["gap_mean"] for n in names]
    mrrs = [report["models"][n]["title_retrieval"]["mrr"] for n in names]
    ax1.bar(names, gaps)
    ax1.set_ylabel("coherence gap (intra - inter)")
    ax1.set_title("Coherence gap on fixed S1 chunks")
    for i, g in enumerate(gaps):
        ax1.text(i, g + 0.005, f"{g:+.3f}", ha="center", fontsize=10)
    ax2.bar(names, mrrs)
    ax2.set_ylabel("MRR (title -> sentence)")
    ax2.set_title("Title retrieval MRR (n=200 queries)")
    for i, v in enumerate(mrrs):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    fig.suptitle("Embedding comparison: baseline vs biomedical candidate (pilot, n=200)")
    fig.tight_layout()
    fig.savefig(res_dir / "embedding_comparison.png", dpi=130)
    plt.close(fig)
    print(f"[emb] wrote {res_dir / 'embedding_comparison.png'}")


if __name__ == "__main__":
    main()
