"""Unit tests: bounded evidence-graph visualization."""

from sgrag.generation.models import EvidenceItem
from sgrag.viz import (
    MAX_VIZ_EDGES,
    MAX_VIZ_NODES,
    build_evidence_graph,
    render_pyvis_html,
)


def _ev(cid, ents=()):
    return EvidenceItem(chunk_id=cid, pmid="111", title="T", text="x",
                        matched_entities=list(ents))


def test_graph_from_evidence():
    ev = [_ev("c1", ["DISEASE:a"]), _ev("c2", ["DISEASE:a", "DISEASE:b"])]
    g = build_evidence_graph(lambda *a, **k: [], ["DISEASE:a"], ev)
    kinds = {n["kind"] for n in g["nodes"]}
    assert kinds == {"entity", "chunk"}
    assert any(e["label"] == "MENTIONS" for e in g["edges"])


def test_semantic_edges_from_real_query_fn():
    def run(cypher, **params):
        assert "SEMANTIC_SIMILAR" in cypher
        return [{"src": "c1", "dst": "c2", "score": 0.8}]

    g = build_evidence_graph(run, [], [_ev("c1"), _ev("c2")])
    assert any("SIMILAR" in e["label"] for e in g["edges"])


def test_graph_size_caps():
    ev = [_ev(f"c{i}", [f"DISEASE:e{i}", f"CHEMICAL:f{i}"]) for i in range(50)]
    g = build_evidence_graph(lambda *a, **k: [], [], ev)
    assert len(g["nodes"]) <= MAX_VIZ_NODES
    assert len(g["edges"]) <= MAX_VIZ_EDGES


def test_edge_deduplication():
    ev = [_ev("c1", ["DISEASE:a"])]
    g = build_evidence_graph(lambda *a, **k: [], ["DISEASE:a"], ev)
    keys = [(e["src"], e["dst"], e["label"]) for e in g["edges"]]
    assert len(keys) == len(set(keys))


def test_pyvis_renders_html():
    g = build_evidence_graph(lambda *a, **k: [], ["DISEASE:a"], [_ev("c1", ["DISEASE:a"])])
    html = render_pyvis_html(g)
    assert "<html" in html.lower() and "c1" in html


def test_viz_failure_does_not_raise():
    def boom(*a, **k):
        raise RuntimeError("db down")

    g = build_evidence_graph(boom, ["DISEASE:a"], [_ev("c1", ["DISEASE:a"])])
    assert g["nodes"]  # node/edge tables still returned as fallback
