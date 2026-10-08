#!/usr/bin/env python3
"""Build the Neo4j knowledge graph (Phase 3B).

Pipeline:
  corpus JSONL -> Article nodes -> Chunk nodes (+ MiniLM embeddings)
  -> scispaCy bc5cdr entity extraction -> Entity nodes -> MENTIONS
  -> top-k SEMANTIC_SIMILAR edges

Idempotent: every write is a MERGE, so re-running duplicates nothing.
Use --reset to wipe the graph first (schema objects are kept).

Embeddings are cached at data/interim/graph_chunk_embeddings.npy (gitignored)
keyed by chunk count + first/last chunk_id fingerprint.

Usage:
  SGRAG_NEO4J_PASSWORD=... .venv/bin/python scripts/build_graph.py \
      --config configs/graph_build.yaml [--reset] [--limit 50]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sgrag.graph.config import load_graph_config  # noqa: E402
from sgrag.graph.ingest import GraphStore  # noqa: E402
from sgrag.graph.schema import build_article_params, build_chunk_params  # noqa: E402
from sgrag.graph.similarity import top_k_neighbors  # noqa: E402

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_CACHE = REPO / "data" / "interim" / "graph_chunk_embeddings.npy"
EMBEDDING_META = REPO / "data" / "interim" / "graph_chunk_embeddings.meta.json"


def read_jsonl(path: Path, limit: int | None = None):
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def fingerprint_chunks(chunks: list[dict]) -> str:
    h = hashlib.sha256()
    h.update(str(len(chunks)).encode())
    h.update(chunks[0]["chunk_id"].encode())
    h.update(chunks[-1]["chunk_id"].encode())
    return h.hexdigest()[:16]


def resolve_embedding_model(model_name: str) -> str:
    """Resolve a local HF-hub snapshot path to avoid network on this VM.

    The egress proxy breaks huggingface_hub's HTTP client; the model is
    already cached under ~/.cache/huggingface/hub, so load it directly.
    """
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_name.replace("/", "--")
    snap_dir = cache / slug / "snapshots"
    if snap_dir.is_dir():
        snaps = sorted(p for p in snap_dir.iterdir() if p.is_dir())
        if snaps:
            return str(snaps[0])
    return model_name


def load_or_encode_embeddings(chunk_ids: list[str], texts: list[str]) -> np.ndarray:
    """Encode chunk texts with MiniLM, reusing the on-disk cache when valid."""
    fp = fingerprint_chunks([{"chunk_id": c} for c in chunk_ids])
    if EMBEDDING_CACHE.exists() and EMBEDDING_META.exists():
        meta = json.loads(EMBEDDING_META.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fp and meta.get("model") == EMBEDDING_MODEL:
            arr = np.load(EMBEDDING_CACHE)
            if arr.shape[0] == len(chunk_ids):
                print(f"  embedding cache hit ({arr.shape[0]} x {arr.shape[1]})")
                return arr
    from sentence_transformers import SentenceTransformer

    model_path = resolve_embedding_model(EMBEDDING_MODEL)
    model = SentenceTransformer(model_path)
    print(f"  encoding {len(texts)} chunk texts with {EMBEDDING_MODEL} ...")
    t0 = time.time()
    arr = model.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
    print(f"  encoded in {time.time() - t0:.1f}s")
    EMBEDDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDING_CACHE, arr.astype(np.float32))
    EMBEDDING_META.write_text(
        json.dumps({"fingerprint": fp, "model": EMBEDDING_MODEL, "count": len(chunk_ids)}),
        encoding="utf-8",
    )
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Neo4j knowledge graph")
    ap.add_argument("--config", default=str(REPO / "configs" / "graph_build.yaml"))
    ap.add_argument("--reset", action="store_true", help="wipe all nodes/relationships first")
    ap.add_argument("--limit", type=int, default=None, help="process first N articles only")
    ap.add_argument("--skip-similarity", action="store_true")
    args = ap.parse_args()

    cfg = load_graph_config(args.config)
    timings: dict[str, float] = {}
    overall_t0 = time.time()

    corpus_path = REPO / "data" / "processed" / "full_corpus.jsonl"
    chunk_path = REPO / "data" / "processed" / "full_chunks" / "chunks_semantic_cluster.jsonl"
    strategy = cfg.ingestion.chunk_strategy
    if strategy != "semantic_cluster":
        raise SystemExit(f"refusing: graph_build.yaml pins semantic_cluster, got {strategy}")

    articles = list(read_jsonl(corpus_path, args.limit))
    pmids = {str(a["pmid"]) for a in articles}
    chunks = [c for c in read_jsonl(chunk_path) if str(c["pmid"]) in pmids]
    print(f"articles: {len(articles)}, chunks ({strategy}): {len(chunks)}")
    chunk_ids = [c["chunk_id"] for c in chunks]

    with GraphStore(cfg) as store:
        if args.reset:
            t0 = time.time()
            store.clear_graph()
            timings["reset"] = time.time() - t0

        t0 = time.time()
        store.ensure_schema()
        timings["schema"] = time.time() - t0
        print("schema ensured (constraints, indexes, vector index)")

        t0 = time.time()
        n = store.upsert_articles([build_article_params(a) for a in articles])
        timings["articles"] = time.time() - t0
        print(f"articles upserted: {n}")

        t0 = time.time()
        embeddings = load_or_encode_embeddings(chunk_ids, [c["text"] for c in chunks])
        timings["embeddings"] = time.time() - t0
        assert embeddings.shape == (len(chunks), 384), embeddings.shape

        t0 = time.time()
        chunk_rows_done = 0
        # Stream chunks in batches: materializing all 53k embedding lists at
        # once costs ~700MB RSS on top of everything else.
        for start in range(0, len(chunks), 1000):
            batch = chunks[start:start + 1000]
            rows = [
                build_chunk_params(c, embeddings[i].astype(float).tolist())
                for i, c in enumerate(batch, start=start)
            ]
            chunk_rows_done += store.upsert_chunks(rows)
            del rows
        timings["chunks"] = time.time() - t0
        print(f"chunks upserted: {chunk_rows_done}")
        del batch

        t0 = time.time()
        ent_cfg = cfg.entities
        # SGRAG_SCISPACY_MODEL may point at an extracted model directory
        # (the legacy bc5cdr wheel does not build on Python 3.12).
        spacy_model = os.environ.get("SGRAG_SCISPACY_MODEL", ent_cfg.model)
        # Streaming extraction: NER over 53k chunks in batches, spilling
        # mentions to a JSONL file instead of holding all mentions in RAM.
        from collections import Counter, defaultdict

        from sgrag.graph.entities import BiomedicalEntityExtractor, normalized_key
        from sgrag.graph.ingest import build_entity_rows

        extractor = BiomedicalEntityExtractor(
            model=spacy_model, allowed_types=ent_cfg.allowed_types)
        key_surfaces: dict[str, Counter] = defaultdict(Counter)
        key_types: dict[str, str] = {}
        out_dir = REPO / "results" / "graph_build"
        out_dir.mkdir(parents=True, exist_ok=True)
        mentions_path = out_dir / "mentions.jsonl"
        n_mentions = 0
        with open(mentions_path, "w", encoding="utf-8") as out:
            for start in range(0, len(chunks), 2000):
                batch = chunks[start:start + 2000]
                per_chunk = extractor.extract_batch(
                    [c["text"] for c in batch], batch_size=ent_cfg.batch_size)
                for c, ents in zip(batch, per_chunk):
                    for e in ents:
                        key = normalized_key(e.entity_type, e.surface_form)
                        key_surfaces[key][e.surface_form] += 1
                        key_types[key] = e.entity_type
                        out.write(json.dumps({
                            "chunk_id": c["chunk_id"],
                            "normalized_key": key,
                            "surface_form": e.surface_form,
                            "char_start": e.char_start,
                            "char_end": e.char_end,
                        }) + "\n")
                        n_mentions += 1
                del per_chunk, batch
        del extractor
        timings["extraction"] = time.time() - t0
        print(f"extracted {n_mentions} mentions -> {mentions_path.name}")

        t0 = time.time()
        entity_rows = build_entity_rows(key_surfaces, key_types)
        del key_surfaces, key_types
        n_e = store.upsert_entities(entity_rows)
        del entity_rows
        # stream mentions JSONL in batches
        n_m = 0
        m_batch: list[dict] = []
        with open(mentions_path, encoding="utf-8") as fh:
            for line in fh:
                m_batch.append(json.loads(line))
                if len(m_batch) >= 2000:
                    n_m += store.upsert_mentions(m_batch)
                    m_batch = []
            if m_batch:
                n_m += store.upsert_mentions(m_batch)
        timings["entity_writes"] = time.time() - t0
        print(f"entities: {n_e} nodes, mentions: {n_m} rels")

        n_s = 0
        if not args.skip_similarity:
            t0 = time.time()
            sim_cfg = cfg.similarity
            edges = top_k_neighbors(chunk_ids, embeddings, top_k=sim_cfg.top_k,
                                    model=sim_cfg.embedding_model)
            timings["similarity_compute"] = time.time() - t0
            t0 = time.time()
            n_s = store.upsert_similar(edges)
            timings["similarity_writes"] = time.time() - t0
            print(f"SEMANTIC_SIMILAR edges upserted: {n_s}")

        counts = store.counts()

    timings["total"] = time.time() - overall_t0
    report = {
        "config": args.config,
        "chunk_strategy": strategy,
        "articles": len(articles),
        "chunks": len(chunks),
        "entities": n_e,
        "mentions": n_m,
        "similar_edges": n_s,
        "timings_sec": {k: round(v, 2) for k, v in timings.items()},
        "db_counts": counts,
    }
    out = REPO / "results" / "graph_build"
    out.mkdir(parents=True, exist_ok=True)
    (out / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
