"""Scientific Literature GraphRAG — Streamlit demo app (Phase 5D).

Launch:
    .venv/bin/streamlit run app.py

Requires Neo4j running with the Phase 3B graph. LLM generation uses
SGRAG_LLM_PROVIDER (default "mock" — clearly labeled; set to "groq" with
SGRAG_LLM_API_KEY for live answers).
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sgrag.generation import GroundedGenerator, load_generation_config  # noqa: E402
from sgrag.retrieval import RetrievalError  # noqa: E402
from sgrag.viz import build_evidence_graph, render_pyvis_html  # noqa: E402

log = logging.getLogger("sgrag.app")

EXAMPLES = [
    "What are the risk factors associated with Alzheimer's disease?",
    "What drugs are associated with breast cancer treatment?",
    "How does inflammation affect cardiovascular disease?",
]

CORPUS_STATS = {
    "articles": "5,000",
    "chunks": "53,091",
    "entities": "9,945",
    "semantic edges": "265,455",
}


@st.cache_resource(show_spinner=False)
def get_generator(rerank_enabled: bool) -> GroundedGenerator:
    """Cached: heavy model loads (MiniLM, scispaCy, CrossEncoder) happen once."""
    import dataclasses

    from sgrag.retrieval import HybridRetriever
    from sgrag.retrieval.config import load_retrieval_config

    cfg = load_retrieval_config("configs/retrieval.yaml")
    cfg = dataclasses.replace(
        cfg, reranker=dataclasses.replace(cfg.reranker, enabled=rerank_enabled))
    return GroundedGenerator(gen_config=load_generation_config(),
                             retriever=HybridRetriever(cfg))


def ask_pipeline(query: str, rerank: bool):
    gen = get_generator(rerank)
    return gen.answer(query), gen


# ---------------------------------------------------------------- page ----
st.set_page_config(page_title="Scientific Literature GraphRAG", layout="wide")
st.title("Scientific Literature GraphRAG")
st.caption("Evidence-grounded scientific literature search using hybrid "
           "vector + knowledge graph retrieval.")

gen_cfg = load_generation_config()
provider_badge = ("LIVE (Groq)" if gen_cfg.provider == "groq"
                  else "MOCK (no API key — answers are illustrative)")

with st.sidebar:
    st.header("System")
    for k, v in CORPUS_STATS.items():
        st.metric(k, v)
    st.divider()
    st.write(f"**Embedding:** all-MiniLM-L6-v2")
    st.write(f"**Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2")
    st.write(f"**LLM:** {provider_badge}")
    st.write(f"**LLM model:** `{gen_cfg.model}`")
    rerank_on = st.toggle("CrossEncoder reranking", value=True)
    st.divider()
    if st.button("About this system"):
        st.session_state["show_about"] = not st.session_state.get("show_about")

if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""
query = st.text_area("Ask a biomedical question", height=100,
                     placeholder=EXAMPLES[0], key="query_text")
cols = st.columns(4)
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(f"Example {i + 1}", use_container_width=True):
        st.session_state["query_text"] = ex
        st.rerun()
ask = st.button("Ask", type="primary", use_container_width=True)

if st.session_state.get("show_about"):
    st.header("About")
    st.markdown("""
1. **Corpus** — 5,000 PubMed abstracts (PubMedQA-derived), audited.
2. **Semantic chunking** — MiniLM + HDBSCAN sentence clustering (53,091 chunks).
3. **Knowledge graph** — Neo4j: Article/Chunk/Entity nodes, MENTIONS + top-5 SEMANTIC_SIMILAR edges.
4. **Hybrid retrieval** — vector search + entity-aware graph expansion, weighted fusion.
5. **CrossEncoder reranking** — ms-marco-MiniLM-L-6-v2 reorders the top-50 candidates.
6. **Grounded generation** — LLM answers ONLY from retrieved evidence, with [PMID:...] citations.
7. **Citation validation** — every citation checked against the evidence; fabricated ones removed.

```
query -> classify -> hybrid retrieval -> fusion -> rerank
      -> evidence pack -> LLM -> citation validation -> answer
```
""")

if ask:
    if not (query or "").strip():
        st.warning("Please enter a question first.")
        st.stop()
    if len(query) > 2000:
        st.warning("That question is too long (over 2000 characters). "
                   "Please shorten it.")
        st.stop()
    try:
        with st.spinner("Retrieving evidence and generating an answer…"):
            result, gen = ask_pipeline(query.strip(), rerank_on)
    except RetrievalError as exc:
        st.error("Could not reach the literature database. "
                 "Make sure Neo4j is running (`bin/neo4j start`), then retry.")
        log.error("retrieval failed: %s", traceback.format_exc())
        st.stop()
    except Exception:  # noqa: BLE001 - never expose stack traces
        st.error("Something went wrong while answering. "
                 "Details were written to the application log.")
        log.error("pipeline failed: %s", traceback.format_exc())
        st.stop()

    st.header("Answer")
    st.markdown(result.answer)
    if result.citation_validation and not result.citation_validation.valid:
        st.warning("Some citations could not be verified against the evidence "
                   "and were removed.")
    for w in result.warnings:
        st.warning(w)
    if result.citations:
        st.subheader("Citations")
        for c in result.citations:
            st.code(f"[PMID:{c.pmid}]" +
                    (f" [CHUNK:{c.chunk_id}]" if c.chunk_id else ""))

    st.header("Retrieved evidence")
    for ev in result.evidence:
        with st.expander(f"PMID:{ev.pmid} — {ev.title[:90]}"):
            st.write(f"**Chunk:** `{ev.chunk_id}`")
            rs = (f"{ev.reranker_score:.4f}" if ev.reranker_score is not None
                  else "—")
            st.write(f"**Reranker score:** {rs} · "
                     f"**Fused score:** {ev.fused_score:.4f}")
            st.write(f"**Sources:** {', '.join(ev.retrieval_sources) or '—'}")
            st.write(f"**Matched entities:** "
                     f"{', '.join(ev.matched_entities) or '—'}")
            st.write(ev.text)

    st.header("Retrieval transparency")
    md = result.retrieval_metadata or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("vector candidates", md.get("n_vector", "—"))
    c2.metric("entity candidates", md.get("n_entity", "—"))
    c3.metric("fused candidates", md.get("n_fused", "—"))
    c4.metric("query class", md.get("query_class", "—"))
    st.write(f"**Query entities:** {', '.join(md.get('query_entities') or ['—'])}")
    lat = []
    for k, label in (("retrieval_ms", "retrieval"), ("rerank_ms", "reranking"),
                     ("generation_ms", "generation")):
        v = (md.get(k) or
             result.generation_metadata.get(k))
        if v is not None:
            lat.append(f"{label}: {v} ms")
    if lat:
        st.write("**Latency** — " + " · ".join(lat))

    st.header("Evidence graph")
    try:
        graph = build_evidence_graph(
            gen.retriever._run, md.get("query_entities") or [], result.evidence)
        try:
            html = render_pyvis_html(graph)
            components.html(html, height=440)
        except Exception:
            st.info("Interactive graph unavailable — showing relationships "
                    "as a table instead.")
            st.table(graph["edges"])
        if graph.get("truncated"):
            st.caption("Graph truncated to 60 nodes for readability.")
    except Exception:  # noqa: BLE001
        st.info("Evidence graph could not be built for this query.")
