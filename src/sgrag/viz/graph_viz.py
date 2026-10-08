"""Bounded evidence-graph visualization (Phase 5D).

Builds a small query-specific graph from ACTUAL retrieved evidence:
  query entities -> MENTIONS -> evidence chunks -> SEMANTIC_SIMILAR -> chunks

Never the 53k-node graph. Hard-capped node/edge counts. Renders with PyVis;
on any PyVis failure returns structured node/edge tables as a clean fallback.
"""

from __future__ import annotations

MAX_VIZ_NODES = 60
MAX_VIZ_EDGES = 120

SEM_EDGES_QUERY = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})-[r:SEMANTIC_SIMILAR]->(n:Chunk)
WHERE n.chunk_id IN $chunk_ids
RETURN cid AS src, n.chunk_id AS dst, r.score AS score
ORDER BY score DESC
"""


def build_evidence_graph(run_query_fn, query_entities: list[str],
                         evidence: list) -> dict:
    """Build a bounded evidence graph.

    run_query_fn: callable(cypher, **params) -> rows (Neo4j read).
    evidence: list of EvidenceItem (or RetrievalResult) with chunk_id, pmid,
              title, matched_entities.
    Returns {nodes: [...], edges: [...]} with hard size caps.
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_node(nid, label, kind, title=""):
        if nid not in nodes and len(nodes) < MAX_VIZ_NODES:
            nodes[nid] = {"id": nid, "label": label, "kind": kind,
                          "title": title}

    for ent in (query_entities or [])[:10]:
        add_node(f"ent:{ent}", ent.split(":", 1)[-1][:24], "entity",
                 f"Query entity {ent}")
    for ev in evidence[:20]:
        cid = ev.chunk_id
        add_node(f"chunk:{cid}", cid[-14:], "chunk",
                 f"{ev.title[:80]} (PMID:{ev.pmid})")
        for ent in (ev.matched_entities or [])[:5]:
            eid = f"ent:{ent}"
            add_node(eid, ent.split(":", 1)[-1][:24], "entity")
            if len(edges) < MAX_VIZ_EDGES:
                edges.append({"src": eid, "dst": f"chunk:{cid}",
                              "label": "MENTIONS"})
    # semantic edges among evidence chunks (from the real graph)
    chunk_ids = [ev.chunk_id for ev in evidence[:20]]
    if chunk_ids:
        try:
            rows = run_query_fn(SEM_EDGES_QUERY, chunk_ids=chunk_ids)
            for row in rows:
                src, dst = f"chunk:{row['src']}", f"chunk:{row['dst']}"
                if src in nodes and dst in nodes and len(edges) < MAX_VIZ_EDGES:
                    edges.append({"src": src, "dst": dst,
                                  "label": f"SIMILAR {row['score']:.2f}"})
        except Exception:
            pass  # viz must never break the app; fallback tables still work
    # dedupe edges
    seen, deduped = set(), []
    for e in edges:
        key = (e["src"], e["dst"], e["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return {"nodes": list(nodes.values()), "edges": deduped,
            "truncated": len(nodes) >= MAX_VIZ_NODES}


def render_pyvis_html(graph: dict, height: str = "420px") -> str:
    """Render the evidence graph to a PyVis HTML string.

    Raises on PyVis failure so callers can fall back to tables.
    """
    from pyvis.network import Network

    net = Network(height=height, width="100%", directed=True,
                  bgcolor="#0f172a", font_color="#e2e8f0")
    colors = {"entity": "#f59e0b", "chunk": "#38bdf8"}
    for n in graph["nodes"]:
        net.add_node(n["id"], label=n["label"], title=n.get("title", ""),
                     color=colors.get(n["kind"], "#94a3b8"))
    for e in graph["edges"]:
        net.add_edge(e["src"], e["dst"], title=e["label"],
                     label=e["label"] if "SIMILAR" not in e["label"] else "")
    net.set_options("""
    {"physics": {"barnesHut": {"gravitationalConstant": -4000}},
     "interaction": {"hover": true}}
    """)
    return net.generate_html()
