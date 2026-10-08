#!/usr/bin/env python3
"""Phase 4 smoke-test CLI: run hybrid retrieval against the Phase 3B graph.

Usage:
    no_proxy='*' NO_PROXY='*' HF_HUB_OFFLINE=1 \
      SGRAG_NEO4J_PASSWORD=... SGRAG_SCISPACY_MODEL=... \
      .venv/bin/python scripts/retrieve.py --query "..." [--top 5] [--json]

Never rebuilds the graph and never touches evaluation files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Offline-first: the VM egress proxy breaks HF hub HTTP; models are cached locally.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sgrag.retrieval import HybridRetriever, RetrievalError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 hybrid retrieval smoke test")
    ap.add_argument("--query", required=True, help="user query")
    ap.add_argument("--config", default=str(REPO / "configs" / "retrieval.yaml"))
    ap.add_argument("--top", type=int, default=5, help="results to print")
    ap.add_argument("--json", action="store_true", help="emit full JSON")
    args = ap.parse_args()

    try:
        with HybridRetriever(args.config) as ret:
            idx = ret.check_index()
            print(f"[index] {idx['name']} type={idx['type']} state={idx['state']}")
            results, diag = ret.retrieve(args.query)
    except RetrievalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[query] {diag['query']}")
    print(f"[entities] {diag['query_entities'] or '(none -> vector-only fallback)'}")
    print(
        f"[counts] vector={diag['n_vector']} entity={diag['n_entity']} "
        f"semantic={diag['n_semantic']} fused={diag['n_fused']} "
        f"multi_source={diag['n_multi_source']}"
    )
    print(
        f"[latency] embed={diag['embed_ms']}ms vector={diag['vector_ms']}ms "
        f"entity={diag['entity_ms']}ms fusion={diag['fusion_ms']}ms "
        f"total={diag['total_ms']}ms"
    )
    for r in results[: args.top]:
        vs = f"{r.vector_score:.4f}" if r.vector_score is not None else "-"
        gs = f"{r.graph_score:.4f}" if r.graph_score is not None else "-"
        ss = f"{r.semantic_score:.4f}" if r.semantic_score is not None else "-"
        print(f"\n#{r.rank} fused={r.fused_score:.4f} "
              f"[vec={vs} ent={gs} sem={ss}] sources={','.join(r.retrieval_sources)}")
        print(f"   pmid={r.pmid} chunk={r.chunk_id}")
        print(f"   title={r.title[:110]}")
        print(f"   entities={r.matched_entities or '-'}")
        print(f"   text={r.text[:220].replace(chr(10), ' ')}...")
    if args.json:
        print(json.dumps({"diagnostics": diag,
                          "results": [r.to_dict() for r in results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
