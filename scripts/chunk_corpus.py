#!/usr/bin/env python3
"""Run the three chunking strategies over the pilot corpus.

Reads data/processed/pilot_corpus.jsonl, writes one JSONL per strategy to
the configured out_dir, plus per-article clustering diagnostics for the
semantic strategy. Deterministic for fixed_token and sentence_boundary;
semantic_cluster is deterministic given the embedding model (HDBSCAN has no
random component).
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgrag.chunking import CONFIGS, STRATEGIES
from sgrag.chunking.semantic_cluster import make_embed_fn as semantic_embed_fn
from sgrag.utils.env import sanitize_proxy_env


def main():
    sanitize_proxy_env()
    ap = argparse.ArgumentParser(description="Chunk the pilot corpus (3 strategies).")
    ap.add_argument("--config", default="configs/chunking_pilot.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    corpus_path = Path(cfg["experiment"]["corpus"])
    out_dir = Path(cfg["experiment"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(corpus_path, encoding="utf-8") if l.strip()]
    print(f"[chunk] {len(records)} articles from {corpus_path}")

    for strategy, fn in STRATEGIES.items():
        s_cfg = CONFIGS[strategy](**cfg["strategies"][strategy])
        out_path = out_dir / f"chunks_{strategy}.jsonl"
        diag = []
        n_chunks = 0
        # Load the embedding model once for the semantic strategy.
        embed_fn = (semantic_embed_fn(s_cfg.embedding_model)
                    if strategy == "semantic_cluster" else None)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                if strategy == "semantic_cluster":
                    chunks, info = fn(rec, config=s_cfg, embed_fn=embed_fn)
                    info["pmid"] = str(rec["pmid"])
                    diag.append(info)
                else:
                    chunks = fn(rec, config=s_cfg)
                for ch in chunks:
                    f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
                n_chunks += len(chunks)
        print(f"[chunk] {strategy}: {n_chunks} chunks -> {out_path}")
        if strategy == "semantic_cluster":
            diag_path = out_dir / "chunks_semantic_cluster_info.json"
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(diag, f, indent=2)
            n_fallback = sum(1 for d in diag if d.get("fallback"))
            n_clusters = sum(d.get("n_clusters", 0) for d in diag)
            print(f"[chunk] semantic diagnostics: {n_clusters} clusters total, "
                  f"{n_fallback} fallbacks -> {diag_path}")


if __name__ == "__main__":
    main()
