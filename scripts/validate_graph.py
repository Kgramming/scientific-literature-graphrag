#!/usr/bin/env python3
"""Validate the Phase 3B Neo4j knowledge graph.

Checks: node/relationship counts, chunk<->article integrity, orphans,
duplicate IDs/keys, embedding dimensions, vector index ONLINE, APOC/GDS
availability, and that SEMANTIC_SIMILAR edges respect the configured top-k.

Usage:
  SGRAG_NEO4J_PASSWORD=... .venv/bin/python scripts/validate_graph.py \
      --config configs/graph_build.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sgrag.graph.config import load_graph_config  # noqa: E402
from sgrag.graph.ingest import GraphStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the Neo4j knowledge graph")
    ap.add_argument("--config", default=str(REPO / "configs" / "graph_build.yaml"))
    args = ap.parse_args()
    cfg = load_graph_config(args.config)

    checks: dict[str, dict] = {}
    ok = True

    def record(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        checks[name] = {"passed": passed, "detail": detail}
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

    with GraphStore(cfg) as store:
        counts = store.counts()
        print("counts:", json.dumps(counts))
        record("nonzero_nodes", all(counts[k] > 0 for k in ("articles", "chunks", "entities")),
               str({k: counts[k] for k in ("articles", "chunks", "entities")}))
        record("nonzero_rels", all(counts[k] > 0 for k in ("has_chunk", "mentions", "similar")),
               str({k: counts[k] for k in ("has_chunk", "mentions", "similar")}))

        q = store.query(
            "MATCH (c:Chunk) WHERE NOT EXISTS { MATCH (:Article)-[:HAS_CHUNK]->(c) } "
            "RETURN count(c) AS n")[0]["n"]
        record("no_orphan_chunks", q == 0, f"orphans={q}")

        q = store.query(
            "MATCH (c:Chunk) WHERE NOT EXISTS { MATCH (a:Article {pmid: c.pmid}) } "
            "RETURN count(c) AS n")[0]["n"]
        record("chunk_pmids_valid", q == 0, f"invalid={q}")

        q = store.query(
            "MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity) "
            "WHERE NOT EXISTS { MATCH (:Article)-[:HAS_CHUNK]->(c) } RETURN count(r) AS n")[0]["n"]
        record("mention_sources_valid", q == 0, f"invalid={q}")

        for label, prop in (("Article", "pmid"), ("Chunk", "chunk_id"), ("Entity", "normalized_key")):
            q = store.query(
                f"MATCH (n:{label}) WITH n.{prop} AS k, count(*) AS c "
                f"WHERE c > 1 RETURN count(*) AS n")[0]["n"]
            record(f"no_duplicate_{label.lower()}_{prop}", q == 0, f"dup_groups={q}")

        q = store.query(
            "MATCH (c:Chunk) WITH c LIMIT 5 RETURN c.embedding AS emb")
        dims = {len(r["emb"]) for r in q if r["emb"]}
        record("embedding_dimensions", dims == {384}, f"dims={sorted(dims)}")

        idx = store.query("SHOW INDEXES YIELD name, type, state WHERE type = 'VECTOR'")
        online = [r for r in idx if r["name"] == cfg.vector_index.name and r["state"] == "ONLINE"]
        record("vector_index_online", len(online) == 1,
               f"name={cfg.vector_index.name} state={[r['state'] for r in idx if r['name'] == cfg.vector_index.name]}")

        try:
            apoc = store.query("RETURN apoc.version() AS v")[0]["v"]
            record("apoc_available", True, f"v{apoc}")
        except Exception as exc:  # noqa: BLE001
            record("apoc_available", False, str(exc)[:120])
        try:
            gds = store.query("RETURN gds.version() AS v")[0]["v"]
            record("gds_available", True, f"v{gds}")
        except Exception as exc:  # noqa: BLE001
            record("gds_available", False, str(exc)[:120])

        deg = store.query(
            "MATCH (c:Chunk) OPTIONAL MATCH (c)-[r:SEMANTIC_SIMILAR]->() "
            "WITH c, count(r) AS d RETURN "
            "count(*) AS n, avg(d) AS avg_d, max(d) AS max_d, "
            "sum(CASE WHEN d = 0 THEN 1 ELSE 0 END) AS zero_d")
        d = deg[0]
        k = cfg.similarity.top_k
        record("similarity_topk_bounded", d["max_d"] <= k,
               f"avg={d['avg_d']:.2f} max={d['max_d']} zero_degree={d['zero_d']}")
        record("similarity_edges_expected",
               counts["similar"] == counts["chunks"] * k,
               f"{counts['similar']} vs {counts['chunks']}*{k}")

        art_chunks = store.query(
            "MATCH (a:Article) OPTIONAL MATCH (a)-[:HAS_CHUNK]->(c:Chunk) "
            "WITH a, count(c) AS n RETURN "
            "count(*) AS articles, min(n) AS min_c, max(n) AS max_c, "
            "sum(CASE WHEN n = 0 THEN 1 ELSE 0 END) AS empty")
        ac = art_chunks[0]
        record("articles_have_chunks", ac["empty"] == 0,
               f"min={ac['min_c']} max={ac['max_c']} empty={ac['empty']}")

        ent_deg = store.query(
            "MATCH (e:Entity) OPTIONAL MATCH ()-[r:MENTIONS]->(e) "
            "WITH e, count(r) AS d RETURN count(*) AS n, avg(d) AS avg_d, max(d) AS max_d")
        ed = ent_deg[0]
        print(f"entity mention degree: n={ed['n']} avg={ed['avg_d']:.2f} max={ed['max_d']}")

        type_counts = store.query(
            "MATCH (e:Entity) RETURN e.entity_type AS t, count(*) AS n ORDER BY n DESC")
        print("entity types:", [(r["t"], r["n"]) for r in type_counts])

    out = REPO / "results" / "graph_build"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.json").write_text(
        json.dumps({"passed": ok, "counts": counts, "checks": checks}, indent=2), encoding="utf-8")
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
