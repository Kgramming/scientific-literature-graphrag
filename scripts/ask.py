#!/usr/bin/env python3
"""Phase 5G CLI: full GraphRAG pipeline without the UI.

    query -> classify -> hybrid retrieval -> rerank -> evidence pack
           -> LLM generation -> citation validation -> answer

Usage:
    no_proxy='*' NO_PROXY='*' HF_HUB_OFFLINE=1 \\
      SGRAG_NEO4J_PASSWORD=... SGRAG_SCISPACY_MODEL=... \\
      .venv/bin/python scripts/ask.py --query "..." [--no-rerank] [--json]

LLM provider via SGRAG_LLM_PROVIDER (default "mock"; "groq" needs
SGRAG_LLM_API_KEY). Never prints API keys.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sgrag.generation import GroundedGenerator, load_generation_config  # noqa: E402
from sgrag.retrieval import HybridRetriever  # noqa: E402
from sgrag.retrieval.config import load_retrieval_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="GraphRAG ask CLI (5G)")
    ap.add_argument("--query", required=True)
    ap.add_argument("--config", default=str(REPO / "configs" / "retrieval.yaml"))
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = load_retrieval_config(args.config)
    # Reranking ON by default in the final product CLI (measured: MRR
    # 0.8623 -> 0.9689); --no-rerank restores fused-only ranking.
    if not args.no_rerank:
        cfg = dataclasses.replace(
            cfg, reranker=dataclasses.replace(cfg.reranker, enabled=True))
    gen_cfg = load_generation_config()
    gen = GroundedGenerator(gen_config=gen_cfg,
                            retriever=HybridRetriever(cfg))
    try:
        result = gen.answer(args.query)
    finally:
        gen.close()

    print(f"[provider] {result.provider} model={result.model}")
    print(f"[class] {result.retrieval_metadata.get('query_class', '-')}")
    print(f"[entities] {result.retrieval_metadata.get('query_entities') or '-'}")
    print(f"[evidence] {len(result.evidence)} chunks packed")
    md = result.retrieval_metadata
    print(f"[latency] retrieval={md.get('retrieval_ms')}ms "
          f"rerank={md.get('rerank_ms')}ms "
          f"generation={result.generation_metadata.get('generation_ms')}ms")
    print("\n=== ANSWER ===")
    print(result.answer)
    if result.citations:
        print("\n=== CITATIONS ===")
        for c in result.citations:
            print(f"  [PMID:{c.pmid}]"
                  + (f" [CHUNK:{c.chunk_id}]" if c.chunk_id else "")
                  + (" (INVALID)" if not c.valid else ""))
    cv = result.citation_validation
    print(f"\n[citation validation] valid={cv.valid} "
          f"regenerated={cv.regenerated} n_invalid={len(cv.invalid)}")
    for w in result.warnings:
        print(f"[warning] {w}")
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
