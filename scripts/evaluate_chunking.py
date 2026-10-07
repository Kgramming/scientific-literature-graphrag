#!/usr/bin/env python3
"""Evaluate the three chunking strategies on the pilot corpus.

Metrics per strategy: chunk counts, chunks/article, char/token length
distribution, sentence-integrity violations, empty/invalid chunks,
semantic coherence (intra vs inter cosine via MiniLM), plus clustering
diagnostics for the semantic strategy. Writes chunking_eval.json and two
plots. Same yardstick for every strategy — no claim is made from the
sophistication of a method, only from the numbers.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgrag.chunking.evaluate import strategy_metrics
from sgrag.chunking.sentences import split_sentences
from sgrag.utils.env import sanitize_proxy_env


def main():
    sanitize_proxy_env()
    ap = argparse.ArgumentParser(description="Evaluate chunking strategies.")
    ap.add_argument("--config", default="configs/chunking_pilot.yaml")
    ap.add_argument("--cache-dir", default=None,
                    help="disk cache for sentence embeddings (sgrag.embeddings); "
                         "shares entries with chunk_corpus.py --cache-dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = Path(cfg["experiment"]["out_dir"])
    res_dir = Path(cfg["evaluation"]["out_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    strategies = list(cfg["strategies"].keys())

    records = {str(json.loads(l)["pmid"]): json.loads(l)
               for l in open(Path(cfg["experiment"]["corpus"]), encoding="utf-8")
               if l.strip()}
    chunks = {}
    for s in strategies:
        path = out_dir / f"chunks_{s}.jsonl"
        chunks[s] = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        print(f"[eval] {s}: {len(chunks[s])} chunks")

    # Sentence embeddings per article (computed once, shared by all strategies).
    model_name = cfg["evaluation"]["coherence_model"]
    if args.cache_dir:
        from sgrag.embeddings import EmbeddingConfig, encode as cached_encode
        ecfg = EmbeddingConfig(model_name=model_name, cache_dir=args.cache_dir)
        model = None
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
    art_sentences, art_embs = {}, {}
    for pmid, rec in records.items():
        sents = split_sentences(rec.get("abstract") or "")
        art_sentences[pmid] = sents
        if not sents:
            art_embs[pmid] = np.zeros((0, 1))
        elif args.cache_dir:
            art_embs[pmid], _ = cached_encode(sents, ecfg)
        else:
            art_embs[pmid] = np.asarray(model.encode(sents, show_progress_bar=False))

    report = {"strategies": {}, "config": cfg["strategies"],
              "coherence_model": model_name, "n_articles": len(records)}
    token_series = {}

    for s in strategies:
        m = strategy_metrics(chunks[s], records, art_sentences, art_embs)
        report["strategies"][s] = m
        token_series[s] = [c["token_count"] for c in chunks[s]]

    # Semantic-clustering diagnostics.
    info_path = out_dir / "chunks_semantic_cluster_info.json"
    if info_path.exists():
        diag = json.load(open(info_path, encoding="utf-8"))
        report["semantic_diagnostics"] = {
            "embedding_model": cfg["strategies"]["semantic_cluster"]["embedding_model"],
            "clustering": "hdbscan",
            "min_cluster_size": cfg["strategies"]["semantic_cluster"]["min_cluster_size"],
            "metric": cfg["strategies"]["semantic_cluster"]["metric"],
            "total_clusters": sum(d.get("n_clusters", 0) for d in diag),
            "mean_clusters_per_article": float(np.mean([d.get("n_clusters", 0) for d in diag])),
            "total_noise_singletons": sum(d.get("n_noise", 0) for d in diag),
            "n_fallbacks": sum(1 for d in diag if d.get("fallback")),
            "fallback_pmids": [d["pmid"] for d in diag if d.get("fallback")],
        }

    json.dump(report, open(res_dir / "chunking_eval.json", "w", encoding="utf-8"), indent=2)
    print(f"[eval] wrote {res_dir / 'chunking_eval.json'}")

    # Plot 1: token-length distributions.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([token_series[s] for s in strategies], tick_labels=strategies,
               showfliers=False)
    ax.set_ylabel("tokens per chunk")
    ax.set_title("Chunk size distribution by strategy (pilot, n=200 abstracts)")
    fig.tight_layout()
    fig.savefig(res_dir / "chunk_size_dist.png", dpi=130)
    plt.close(fig)

    # Plot 2: coherence — intra vs inter cosine per strategy.
    # Gap = mean of per-article (intra - inter), matching chunking_eval.json.
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(strategies))
    def _cmean(s, k):
        v = report["strategies"][s]["coherence"][k]
        return v if v is not None else 0.0
    intra_m = [_cmean(s, "intra_chunk_cosine_mean") for s in strategies]
    inter_m = [_cmean(s, "inter_chunk_cosine_mean") for s in strategies]
    gap_m = [_cmean(s, "gap_mean") for s in strategies]
    w = 0.35
    ax.bar(x - w / 2, intra_m, w, label="intra-chunk cosine (higher = tighter)")
    ax.bar(x + w / 2, inter_m, w, label="inter-chunk cosine (lower = cleaner cuts)")
    for i, s in enumerate(strategies):
        ax.text(i, max(intra_m[i], inter_m[i]) + 0.01, f"gap {gap_m[i]:+.3f}",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("mean cosine similarity")
    ax.set_title("Semantic coherence by chunking strategy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(res_dir / "coherence.png", dpi=130)
    plt.close(fig)
    print(f"[eval] wrote plots to {res_dir}")


if __name__ == "__main__":
    main()
