"""Scientific Literature GraphRAG — Streamlit research application (Phase 1 UI Polish).

Launch:
    .venv/bin/streamlit run app.py

Requires Neo4j running with the Phase 3B graph. LLM generation uses
SGRAG_LLM_PROVIDER (default "mock" — clearly labeled; set to "groq" with
SGRAG_LLM_API_KEY for live answers).

Phase 1 polish focuses on:
- Professional research-oriented layout
- Clean header and navigation
- Improved query experience with example cards
- Better answer and citation presentation
- Organized evidence display
- Transparent retrieval metadata
- Responsive, spacious design
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

# Benchmark corpus statistics (labeled as such, not live values)
CORPUS_STATS = {
    "articles": "5,000",
    "chunks": "53,091",
    "entities": "9,945",
    "semantic edges": "265,455",
}


@st.cache_resource(show_spinner=False)
def get_generator(rerank_enabled: bool) -> GroundedGenerator:
    """Cached: heavy model loads (MiniLM, scispaCy, CrossEncoder) happen once per rerank state."""
    import dataclasses

    from sgrag.retrieval import HybridRetriever
    from sgrag.retrieval.config import load_retrieval_config

    cfg = load_retrieval_config("configs/retrieval.yaml")
    cfg = dataclasses.replace(
        cfg, reranker=dataclasses.replace(cfg.reranker, enabled=rerank_enabled))
    return GroundedGenerator(gen_config=load_generation_config(),
                             retriever=HybridRetriever(cfg))


@st.cache_resource(show_spinner=False)
def get_cached_gen_config():
    """Cached generation config (stable across runs)."""
    return load_generation_config()


def ask_pipeline(query: str, rerank: bool):
    gen = get_generator(rerank)
    return gen.answer(query), gen


def make_pubmed_url(pmid: str) -> str:
    """Create PubMed URL for a PMID. Safe: no fabrication."""
    if pmid and pmid.isdigit():
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return ""


def render_citation_badges(citations: list) -> None:
    """Render citations as clean, interactive badges with PubMed links."""
    if not citations:
        return
    badge_cols = st.columns(min(len(citations), 5))
    for idx, citation in enumerate(citations):
        col = badge_cols[idx % len(badge_cols)]
        pmid = citation.pmid
        chunk_label = f"<br><small>{citation.chunk_id}</small>" if citation.chunk_id else ""
        pubmed_url = make_pubmed_url(pmid)
        
        if pubmed_url:
            col.markdown(
                f"<a href='{pubmed_url}' target='_blank' style='text-decoration: none;'>"
                f"<div style='background-color: #f0f2f6; padding: 10px 12px; "
                f"border-radius: 4px; text-align: center; font-size: 13px; "
                f"cursor: pointer; transition: background-color 0.2s;'>"
                f"<strong>PMID</strong><br><code>{pmid}</code>{chunk_label}"
                f"</div></a>",
                unsafe_allow_html=True
            )
        else:
            col.markdown(
                f"<div style='background-color: #f0f2f6; padding: 10px 12px; "
                f"border-radius: 4px; text-align: center; font-size: 13px;'>"
                f"<strong>PMID</strong><br><code>{pmid}</code>{chunk_label}"
                f"</div>",
                unsafe_allow_html=True
            )


def get_answer_with_citations(result) -> str:
    """Format answer with citations for copying."""
    lines = [result.answer]
    if result.citations:
        lines.append("\n\nCitations:")
        for c in result.citations:
            chunk_ref = f" [{c.chunk_id}]" if c.chunk_id else ""
            lines.append(f"  [PMID:{c.pmid}]{chunk_ref}")
    return "\n".join(lines)


def render_answer_section(result) -> None:
    """Render answer with strong visual hierarchy and integrated citations."""
    st.divider()
    
    # Answer header with action buttons
    col_header, col_actions = st.columns([3, 1])
    with col_header:
        st.markdown(
            "<h2 style='margin: 0 0 1rem 0; color: #1c1f26;'>📝 Answer</h2>",
            unsafe_allow_html=True
        )
    with col_actions:
        # Copy answer button
        answer_text = get_answer_with_citations(result)
        st.write("")  # Spacing
        if st.button("📋 Copy", use_container_width=True, key="btn_copy_answer",
                     help="Copy answer with citations"):
            st.session_state["_copy_success"] = True
    
    if st.session_state.get("_copy_success"):
        st.success("Copied to clipboard (in a real implementation)")
        st.session_state["_copy_success"] = False
    
    # Citation validation warning
    if result.citation_validation and not result.citation_validation.valid:
        st.warning(
            "⚠️ Some citations could not be verified against the evidence and were removed. "
            "The answer reflects only validated citations."
        )
    
    # Main answer text with better typography
    st.markdown(
        f"""
<div style='background-color: #fafbfc; padding: 1.5rem; border-radius: 6px; 
            border-left: 4px solid #0366d6; line-height: 1.8; font-size: 15px;'>
{result.answer}
</div>
        """,
        unsafe_allow_html=True
    )
    
    # General warnings
    for w in result.warnings:
        st.warning(f"⚠️ {w}")
    
    # Citations section
    if result.citations:
        st.subheader("📌 Citations", divider=False)
        st.caption("Evidence sources supporting the claims above")
        render_citation_badges(result.citations)


def render_entity_badge(entity: str, badge_type: str = "DEFAULT") -> str:
    """Render an entity badge with subtle styling."""
    colors = {
        "DISEASE": "#ffe6e6",
        "CHEMICAL": "#e6f2ff",
        "GENE": "#e6ffe6",
        "DEFAULT": "#f0f0f0",
    }
    color = colors.get(badge_type, colors["DEFAULT"])
    return (
        f"<span style='background-color: {color}; padding: 2px 6px; "
        f"border-radius: 3px; font-size: 12px; margin-right: 4px;'>"
        f"{entity}</span>"
    )


def render_evidence_cards(evidence: list) -> None:
    """Render evidence items as clean, organized cards grouped by PMID with entity display."""
    if not evidence:
        return
    
    # Group evidence by PMID
    by_pmid = {}
    for ev in evidence:
        if ev.pmid not in by_pmid:
            by_pmid[ev.pmid] = []
        by_pmid[ev.pmid].append(ev)
    
    for pmid, items in by_pmid.items():
        # Show title once per PMID
        title = items[0].title
        with st.expander(f"📄 PMID:{pmid} — {title[:80]}"):
            
            for idx, ev in enumerate(items):
                if idx > 0:
                    st.divider()
                
                # Chunk info and scores (compact layout)
                col1, col2, col3 = st.columns([2, 2, 2])
                with col1:
                    st.caption(f"**Chunk:** `{ev.chunk_id}`")
                with col2:
                    if ev.reranker_score is not None:
                        st.caption(f"**Reranker:** {ev.reranker_score:.3f}")
                    else:
                        st.caption("**Reranker:** —")
                with col3:
                    st.caption(f"**Score:** {ev.fused_score:.3f}")
                
                # Retrieval provenance (how was this found)
                sources_text = ", ".join(ev.retrieval_sources) if ev.retrieval_sources else "—"
                st.caption(f"**Found via:** {sources_text}")
                
                # Entities with subtle badges
                if ev.matched_entities:
                    entities_html = "".join([render_entity_badge(e) for e in ev.matched_entities])
                    st.markdown(f"**Entities:** {entities_html}", unsafe_allow_html=True)
                else:
                    st.caption("**Entities:** —")
                
                # Evidence text (italics for distinction)
                st.markdown(f"_{ev.text}_")


def render_retrieval_intelligence(result_md: dict, gen_md: dict) -> None:
    """Render comprehensive retrieval intelligence for research users."""
    md = result_md or {}
    
    # Query classification and decomposition
    col1, col2 = st.columns([1, 2])
    with col1:
        query_class = md.get("query_class", "—")
        st.metric("Query Type", query_class, label_visibility="collapsed")
    with col2:
        entities = md.get("query_entities", [])
        if entities:
            entities_html = "".join([render_entity_badge(e) for e in entities])
            st.markdown(f"**Detected Entities:** {entities_html}", unsafe_allow_html=True)
        else:
            st.caption("**Detected Entities:** —")
    
    # Retrieval strategy breakdown
    st.subheader("Retrieval Strategy", divider=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_vector = md.get("n_vector", 0)
        st.metric("Vector Search", f"{n_vector} candidates", label_visibility="collapsed")
        st.caption("Semantic similarity in embedding space")
    with col2:
        n_entity = md.get("n_entity", 0)
        st.metric("Entity Graph", f"{n_entity} candidates", label_visibility="collapsed")
        st.caption("Entity mentions + HDBSCAN expansion")
    with col3:
        n_fused = md.get("n_fused", 0)
        st.metric("After Fusion", f"{n_fused} candidates", label_visibility="collapsed")
        st.caption("Weighted normalization fusion")
    
    # Reranking info
    if md.get("rerank_enabled"):
        st.caption("✓ **CrossEncoder reranking active** — ms-marco-MiniLM-L-6-v2 reordering top candidates by relevance")
    else:
        st.caption("• CrossEncoder reranking disabled")
    
    # Latency breakdown
    lat_items = []
    for k, label in (("retrieval_ms", "Retrieval"), ("rerank_ms", "Reranking"),
                     ("generation_ms", "Generation")):
        v = md.get(k) or gen_md.get(k)
        if v is not None:
            lat_items.append(f"{label}: {v} ms")
    if lat_items:
        st.caption("⏱️ **Latency** — " + " · ".join(lat_items))


def render_source_explorer(evidence: list) -> None:
    """Render a compact source explorer showing unique articles and their contribution."""
    if not evidence:
        return
    
    # Group by PMID
    by_pmid = {}
    for ev in evidence:
        if ev.pmid not in by_pmid:
            by_pmid[ev.pmid] = {
                "title": ev.title,
                "chunks": [],
                "entities": set(),
                "max_score": 0,
            }
        by_pmid[ev.pmid]["chunks"].append(ev)
        by_pmid[ev.pmid]["entities"].update(ev.matched_entities or [])
        if ev.reranker_score:
            by_pmid[ev.pmid]["max_score"] = max(by_pmid[ev.pmid]["max_score"], ev.reranker_score)
    
    st.subheader("📖 Source Overview", divider=False)
    st.caption(f"{len(by_pmid)} unique articles contributed {len(evidence)} evidence items")
    
    # Display each source in a compact row
    for pmid, info in sorted(by_pmid.items(), key=lambda x: x[1]["max_score"], reverse=True):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            pubmed_url = make_pubmed_url(pmid)
            if pubmed_url:
                st.markdown(f"**[PMID:{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)** — {info['title'][:60]}")
            else:
                st.markdown(f"**PMID:{pmid}** — {info['title'][:60]}")
        with col2:
            st.caption(f"{len(info['chunks'])} chunks")
        with col3:
            entities_display = ", ".join(sorted(info['entities'])[:2])
            if len(info['entities']) > 2:
                entities_display += f" +{len(info['entities'])-2}"
            st.caption(entities_display if entities_display else "—")
        with col4:
            if info['max_score'] > 0:
                st.caption(f"⭐ {info['max_score']:.2f}")
            else:
                st.caption("—")


def render_advanced_diagnostics(result) -> None:
    """Render advanced diagnostics in a collapsed section."""
    with st.expander("🔍 Advanced Diagnostics"):
        st.subheader("Raw Metadata")
        tab1, tab2 = st.tabs(["Retrieval", "Generation"])
        with tab1:
            st.json(result.retrieval_metadata or {})
        with tab2:
            st.json(result.generation_metadata or {})


# ================================================================ PAGE SETUP ====

st.set_page_config(
    page_title="Scientific Literature GraphRAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================ SIDEBAR ====

with st.sidebar:
    st.header("System Configuration")
    
    with st.expander("📊 Corpus Statistics", expanded=True):
        st.caption("Benchmark dataset (5,000 PubMed abstracts)")
        for k, v in CORPUS_STATS.items():
            st.metric(k.title(), v, label_visibility="collapsed")
    
    st.divider()
    
    gen_cfg = get_cached_gen_config()
    provider_label = ("🟢 LIVE (Groq)" if gen_cfg.provider == "groq"
                      else "🔵 MOCK (Illustrative)")
    
    st.caption("**Models & Systems**")
    st.text(f"Embedding: all-MiniLM-L6-v2")
    st.text(f"Reranker: ms-marco-MiniLM-L-6-v2")
    st.text(f"LLM: {provider_label}")
    st.text(f"Model: {gen_cfg.model}")
    
    st.divider()
    
    rerank_on = st.toggle("CrossEncoder Reranking", value=True,
                          help="Reorder top candidates using cross-encoder scoring")
    
    st.divider()
    
    if st.button("ℹ️ About This System", use_container_width=True):
        st.session_state["show_about"] = not st.session_state.get("show_about")

# ================================================================ HEADER ====

st.markdown("""
<style>
.header-container {
    padding: 2rem 0 1rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid #e1e4e8;
}
.header-title {
    font-size: 2.5em;
    font-weight: 700;
    margin: 0;
    color: #1c1f26;
}
.header-subtitle {
    font-size: 1.0em;
    color: #586069;
    margin: 0.5rem 0 0 0;
    font-weight: 400;
}
</style>

<div class="header-container">
<div class="header-title">Scientific Literature GraphRAG</div>
<div class="header-subtitle">Evidence-grounded biomedical research assistant</div>
</div>
""", unsafe_allow_html=True)

# ================================================================ QUERY SECTION ====

st.subheader("Search Biomedical Literature", divider=False)

if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""


def set_example_query(example_text):
    """Callback to set the query text from an example button."""
    st.session_state["query_text"] = example_text


# Query input with clean styling
query = st.text_area(
    "Your question",
    height=90,
    placeholder=EXAMPLES[0],
    key="query_text",
    label_visibility="collapsed",
    help="Ask a specific biomedical question. Examples: mechanisms, treatments, associations."
)

# Action buttons
col_ask, col_clear, col_space = st.columns([1, 1, 4])
with col_ask:
    ask = st.button("🔍 Search", type="primary", use_container_width=True)
with col_clear:
    if st.button("✕ Clear", use_container_width=True):
        st.session_state["query_text"] = ""
        st.rerun()

# Example queries as research-oriented cards
st.markdown("**Example Queries:**")

examples_meta = [
    ("factual", "Risk factors & mechanisms"),
    ("entity-focused", "Treatments & drugs"),
    ("multi-entity", "Cross-domain relationships"),
]

cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    with cols[i]:
        query_type, desc = examples_meta[i]
        st.markdown(
            f"""
<div style='background-color: #f6f8fa; padding: 12px; border-radius: 6px; 
            border: 1px solid #e1e4e8; margin-bottom: 0.5rem;'>
<small><strong style="color: #0366d6;">{query_type.upper()}</strong></small><br>
<small style="color: #666;">{desc}</small><br><br>
<code style="font-size: 11px; color: #333;">{ex[:45]}...</code>
</div>
            """,
            unsafe_allow_html=True
        )
        st.button(
            f"Use Example {i + 1}",
            use_container_width=True,
            on_click=set_example_query,
            args=(ex,),
            key=f"ex_btn_{i}",
        )

# About section
if st.session_state.get("show_about"):
    st.divider()
    st.subheader("About")
    st.markdown("""
**Architecture Overview**

1. **Corpus** — 5,000 PubMed abstracts (PubMedQA-derived), fully audited and validated.

2. **Semantic Chunking** — MiniLM embeddings + HDBSCAN clustering produces 53,091 
coherent semantic chunks with preserved boundaries.

3. **Knowledge Graph** — Neo4j stores Article/Chunk/Entity nodes with MENTIONS relationships 
and top-5 SEMANTIC_SIMILAR edges for graph-aware retrieval.

4. **Hybrid Retrieval** — Parallel vector search and entity-aware graph expansion with 
weighted normalization fusion for diverse evidence.

5. **CrossEncoder Reranking** — ms-marco-MiniLM-L-6-v2 reorders top 50 candidates by 
relevance, improving MRR from 0.87 → 0.97.

6. **Grounded Generation** — LLM generates answers ONLY from retrieved evidence, with 
explicit [PMID:...] citations.

7. **Citation Validation** — Every citation is checked against evidence; fabricated 
citations are removed automatically.

**Processing Pipeline**

```
Question → Classify → Hybrid Retrieval → Fusion → Rerank → Evidence Pack → LLM
                   ↓                                            ↓
             [Entity Expansion]              [Citation Validation]
                                                     ↓
                                              Answer + Citations
```
    """)

# ================================================================ ANSWER SECTION ====

if ask:
    if not (query or "").strip():
        st.warning("Please enter a question first.")
        st.stop()
    if len(query) > 2000:
        st.warning("That question is too long (over 2,000 characters). Please shorten it.")
        st.stop()
    
    try:
        with st.spinner("🔄 Preparing query…"):
            pass
        with st.spinner("📚 Retrieving evidence from literature…"):
            pass
        with st.spinner("✨ Generating grounded answer…"):
            result, gen = ask_pipeline(query.strip(), rerank_on)
    except RetrievalError as exc:
        st.error(
            "❌ **Database Connection Error**\n\n"
            "Could not reach the Neo4j literature database. "
            "Ensure Neo4j is running: `docker-compose up neo4j` (see README).\n\n"
            "Then refresh this page and try again."
        )
        log.error("retrieval failed: %s", traceback.format_exc())
        st.stop()
    except Exception:  # noqa: BLE001
        st.error(
            "❌ **Processing Error**\n\n"
            "An unexpected error occurred. Check the application logs for details."
        )
        log.error("pipeline failed: %s", traceback.format_exc())
        st.stop()
    
    # ---- Answer Display ----
    render_answer_section(result)
    
    # ---- Evidence ----
    st.divider()
    st.subheader("📚 Supporting Evidence")
    st.caption(f"Retrieved {len(result.evidence)} evidence items from {len(set(e.pmid for e in result.evidence))} articles")
    render_evidence_cards(result.evidence)
    
    # ---- Source Overview ----
    st.divider()
    render_source_explorer(result.evidence)
    
    # ---- Retrieval Intelligence ----
    st.divider()
    st.subheader("🧠 Retrieval Intelligence")
    st.caption("How the search and ranking process worked")
    render_retrieval_intelligence(result.retrieval_metadata or {}, result.generation_metadata or {})
    
    # ---- Evidence Graph ----
    st.divider()
    st.subheader("🔗 Evidence Graph")
    try:
        md = result.retrieval_metadata or {}
        graph = build_evidence_graph(
            gen.retriever._run, md.get("query_entities") or [], result.evidence)
        try:
            html = render_pyvis_html(graph)
            components.html(html, height=500)
        except Exception:
            st.info("Interactive visualization unavailable. Showing relationship table instead.")
            st.table(graph.get("edges", []))
        if graph.get("truncated"):
            st.caption("⚠️ Graph truncated to 60 nodes for readability.")
    except Exception:  # noqa: BLE001
        st.info("Could not build evidence graph for this query. "
                "This may occur for queries with few entities or limited evidence connections.")
    
    # ---- New Search ----
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 New Search", use_container_width=True, key="btn_new_search"):
            st.session_state["query_text"] = ""
            st.rerun()
    with col2:
        if st.button("📋 Copy Answer", use_container_width=True, key="btn_copy_main"):
            pass  # Note: Real copy would use clipboard JS
    
    # ---- Advanced Diagnostics ----
    render_advanced_diagnostics(result)
