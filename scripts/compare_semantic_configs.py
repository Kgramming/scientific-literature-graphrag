#!/usr/bin/env python3
"""Controlled refinement experiment: semantic clustering configuration.

Compares exactly TWO semantic configurations on the SAME 200 abstracts:
  S1 (current): HDBSCAN(min_cluster_size=2) on raw MiniLM embeddings
                (euclidean geometry)
  S2 (alternative): HDBSCAN(min_cluster_size=2) on L2-normalized MiniLM
                embeddings (cosine/angular geometry)

Everything else is identical: the required all-MiniLM-L6-v2 embeddings are
encoded ONCE and reused for both variants, sentence order and provenance are
preserved, and the max_tokens=256 cap is enforced for both (bug fix, not a
tuning choice — see semantic_cluster._enforce_max_tokens).

Rationale for S2: sentence-transformer embeddings are compared with cosine
similarity everywhere else in this pipeline (including our own coherence
metric). Euclidean distance on raw embeddings conflates vector magnitude
with direction; L2-normalization isolates the angular (semantic) difference
that HDBSCAN should cluster on. For unit vectors ||a-b||^2 = 2(1-cos), so
this is exactly cosine-geometry clustering (the installed HDBSCAN build
rejects metric="cosine"). This is a single principled comparison, not
tuning: no parameter is adjusted after seeing the outcome.
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

from sgrag.chunking import semantic_cluster
from sgrag.chunking.base import SemanticClusterConfig
from sgrag.chunking.evaluate import strategy_metrics
from sgrag.chunking.sentences import split_sentences
from sgrag.utils.env import sanitize_proxy_env

VARIANTS = {
    # (label, extra config overrides)
    "S1_current": {},
    "S2_alternative": {"normalize_embeddings": True},
}


def main():
    sanitize_proxy_env()
    ap = argparse.ArgumentParser(description="Compare two semantic chunking configs.")
    ap.add_argument("--config", default="configs/chunking_pilot.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    base = cfg["strategies"]["semantic_cluster"]
    out_dir = Path(cfg["experiment"]["out_dir"])
    res_dir = Path(cfg["evaluation"]["out_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(Path(cfg["experiment"]["corpus"]), encoding="utf-8")
               if l.strip()]
    rec_by_pmid = {str(r["pmid"]): r for r in records}
    print(f"[compare] {len(records)} articles")

    # Encode once; reuse the identical embedding matrix for both variants.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(base["embedding_model"])
    emb_by_pmid, sent_by_pmid = {}, {}
    for r in records:
        pmid = str(r["pmid"])
        sents = split_sentences(r.get("abstract") or "")
        sent_by_pmid[pmid] = sents
        emb_by_pmid[pmid] = np.asarray(model.encode(sents, show_progress_bar=False))
    art_embs = {p: e for p, e in emb_by_pmid.items()}

    comparison = {"variants": {}, "rationale": __doc__.split("\n\n")[2]}
    token_series = {}

    for label, overrides in VARIANTS.items():
        s_cfg = SemanticClusterConfig(**{**base, **overrides})
        geom = "cosine" if s_cfg.normalize_embeddings else "euclidean"
        tag = f"semantic_{label.lower()}"
        out_path = out_dir / f"chunks_{tag}.jsonl"
        diag = []
        n_chunks = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                pmid = str(r["pmid"])
                def embed_fn(sents, _pmid=pmid):
                    assert len(sents) == len(emb_by_pmid[_pmid])
                    return emb_by_pmid[_pmid]
                chunks, info = semantic_cluster.chunk_record(
                    r, config=s_cfg, embed_fn=embed_fn)
                info["pmid"] = pmid
                diag.append(info)
                for ch in chunks:
                    f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
                n_chunks += len(chunks)
        diag_path = out_dir / f"chunks_{tag}_info.json"
        json.dump(diag, open(diag_path, "w", encoding="utf-8"), indent=2)

        chunks = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
        m = strategy_metrics(chunks, rec_by_pmid, sent_by_pmid, art_embs)
        token_series[label] = [c["token_count"] for c in chunks]
        n_noise = sum(d.get("n_noise", 0) for d in diag)
        n_sent = sum(d.get("n_sentences", 0) for d in diag)
        comparison["variants"][label] = {
            "geometry": geom,
            "config": {k: getattr(s_cfg, k) for k in
                       ("embedding_model", "clustering", "min_cluster_size",
                        "metric", "normalize_embeddings", "min_sentences",
                        "max_tokens")},
            "n_chunks": m["n_chunks"],
            "chunks_per_article": m["chunks_per_article"],
            "token_length": m["token_length"],
            "sentence_integrity_violations": m["sentence_integrity_violations"],
            "empty_chunks": m["empty_chunks"],
            "invalid_chunks": m["invalid_chunks"],
            "coverage_exact_once_articles": m["coverage_exact_once_articles"],
            "coherence": m["coherence"],
            "total_clusters": sum(d.get("n_clusters", 0) for d in diag),
            "noise_singletons": n_noise,
            "singleton_rate": n_noise / n_sent if n_sent else 0.0,
            "n_fallbacks": sum(1 for d in diag if d.get("fallback")),
            "n_size_splits": sum(d.get("n_size_splits", 0) for d in diag),
        }
        c = m["coherence"]
        print(f"[compare] {label} ({geom}): {m['n_chunks']} chunks, "
              f"singletons {n_noise}/{n_sent} ({n_noise/n_sent:.1%}), "
              f"max_tok {m['token_length']['max']:.0f}, "
              f"intra {c['intra_chunk_cosine_mean']:.4f} "
              f"inter {c['inter_chunk_cosine_mean']:.4f} "
              f"gap {c['gap_mean']:+.4f}")

    json.dump(comparison,
              open(res_dir / "semantic_comparison.json", "w", encoding="utf-8"), indent=2)
    print(f"[compare] wrote {res_dir / 'semantic_comparison.json'}")

    # Plot: coherence + size distributions, S1 vs S2.
    labels = list(VARIANTS.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))
    intra = [comparison["variants"][l]["coherence"]["intra_chunk_cosine_mean"] for l in labels]
    inter = [comparison["variants"][l]["coherence"]["inter_chunk_cosine_mean"] for l in labels]
    gaps = [comparison["variants"][l]["coherence"]["gap_mean"] for l in labels]
    w = 0.35
    ax1.bar(x - w / 2, intra, w, label="intra-chunk cosine")
    ax1.bar(x + w / 2, inter, w, label="inter-chunk cosine")
    for i, g in enumerate(gaps):
        ax1.text(i, max(intra[i], inter[i]) + 0.01, f"gap {g:+.3f}", ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\n({comparison['variants'][l]['geometry']})" for l in labels])
    ax1.set_ylabel("mean cosine similarity")
    ax1.set_title("Coherence: current vs alternative")
    ax1.legend()
    ax2.boxplot([token_series[l] for l in labels], tick_labels=labels, showfliers=False)
    ax2.set_ylabel("tokens per chunk")
    ax2.set_title("Chunk size distribution")
    fig.suptitle("Semantic clustering: controlled configuration comparison (n=200)")
    fig.tight_layout()
    fig.savefig(res_dir / "semantic_comparison.png", dpi=130)
    plt.close(fig)
    print(f"[compare] wrote {res_dir / 'semantic_comparison.png'}")


if __name__ == "__main__":
    main()
