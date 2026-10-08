"""Scientific Literature GraphRAG — research application.

"Ink & Amber" editorial design: warm paper background, serif display type,
amber accent. Same backend as the Kiro baseline (hybrid retrieval, rerank,
grounded generation, citation validation); all functionality preserved.

Launch:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import sys
import textwrap
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
    ("Risk factors & mechanisms",
     "What are the risk factors associated with Alzheimer's disease?"),
    ("Treatments & drugs",
     "What drugs are associated with breast cancer treatment?"),
    ("Cross-domain relationships",
     "How does inflammation affect cardiovascular disease?"),
]

CORPUS_STATS = [
    ("Articles", "5,000"),
    ("Chunks", "53,091"),
    ("Entities", "9,945"),
    ("Semantic edges", "265,455"),
]

# ============================================================ DESIGN SYSTEM ====

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Inter:wght@400;500;600&display=swap');

/* base */
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { max-width: 1080px; padding-top: 2rem; }

/* hero */
.hero { margin: 0.5rem 0 1.75rem 0; }
.hero h1 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-weight: 700; font-size: 2.6rem; line-height: 1.15;
    color: #1C1E21; margin: 0;
}
.hero p {
    font-size: 1.02rem; color: #5B5F66; margin: 0.6rem 0 0 0; max-width: 640px;
}
.pill-row { display: flex; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap; }
.pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
    padding: 0.3rem 0.7rem; border-radius: 999px;
    background: #FFFFFF; border: 1px solid #E5DFD3; color: #4A4E54;
}
.pill .dot { width: 7px; height: 7px; border-radius: 50%; background: #16A34A; }
.pill.mock .dot { background: #B45309; }

/* cards */
.card {
    background: #FFFFFF; border: 1px solid #E8E3D8; border-radius: 14px;
    padding: 1.35rem 1.5rem; margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(28,30,33,0.04), 0 8px 24px -18px rgba(28,30,33,0.25);
}
.card h3 { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem;
           margin: 0 0 0.75rem 0; color: #1C1E21; }
.answer-card { border-left: 4px solid #B45309; }
.answer-card .answer-text { font-size: 0.98rem; line-height: 1.75; color: #24272C; }
.answer-card .answer-text p { margin: 0 0 0.75rem 0; }

/* citation chips */
.cite-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.9rem; }
.cite-row a.cite-chip, .cite-row span.cite-chip {
    display: inline-flex; align-items: baseline; gap: 0.3rem;
    background: #1E3A5F; color: #F8F6F1 !important; border-radius: 8px;
    padding: 0.35rem 0.65rem; font-size: 0.78rem; font-weight: 600;
    text-decoration: none !important;
    transition: transform .12s ease, box-shadow .12s ease;
}
.cite-row a.cite-chip *, .cite-row span.cite-chip * { color: #F8F6F1 !important; }
.cite-chip:hover { transform: translateY(-1px); box-shadow: 0 4px 10px -4px rgba(30,58,95,.5); }
.cite-chip .chunk { font-weight: 400; opacity: 0.75; font-size: 0.7rem; }
.cite-chip .ext { opacity: 0.6; font-size: 0.7rem; }

/* entity pills */
.ent { display: inline-block; font-size: 0.72rem; font-weight: 600;
       padding: 0.18rem 0.55rem; border-radius: 999px; margin: 0 0.3rem 0.3rem 0;
       letter-spacing: 0.01em; }
.ent-disease { background: #FDECEA; color: #9B2C2C; border: 1px solid #F5C6C0; }
.ent-chemical { background: #E8F1FD; color: #1E4FA3; border: 1px solid #C4DBF7; }
.ent-other { background: #F1EDE6; color: #5B5F66; border: 1px solid #E0DACA; }

/* evidence */
.ev-head { display: flex; justify-content: space-between; align-items: baseline;
           gap: 1rem; margin-bottom: 0.35rem; }
.ev-title { font-weight: 600; font-size: 0.95rem; color: #1C1E21; }
.ev-pmid { font-size: 0.78rem; font-weight: 600; color: #B45309; white-space: nowrap; }
.ev-meta { font-size: 0.76rem; color: #6B7078; margin: 0.35rem 0; }
.ev-text { font-size: 0.9rem; line-height: 1.65; color: #33373D;
           border-left: 2px solid #EDE7D9; padding-left: 0.9rem; margin-top: 0.6rem; }
.scorebar { height: 5px; border-radius: 3px; background: #EFEAE0; margin-top: 0.25rem;
            overflow: hidden; max-width: 220px; }
.scorebar > div { height: 100%; border-radius: 3px;
                  background: linear-gradient(90deg, #B45309, #F59E0B); }

/* stat cards */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem;
             margin: 1rem 0 1.5rem 0; }
.stat { background: #FFFFFF; border: 1px solid #E8E3D8; border-radius: 12px;
        padding: 0.9rem 1rem; }
.stat .v { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.45rem;
           font-weight: 700; color: #1C1E21; }
.stat .k { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
           text-transform: uppercase; color: #8A8F98; margin-top: 0.15rem; }
@media (max-width: 700px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } }

/* example cards */
.ex-card { background: #FFFFFF; border: 1px solid #E8E3D8; border-radius: 12px;
           padding: 0.9rem 1rem; height: 100%;
           transition: border-color .15s ease, box-shadow .15s ease; }
.ex-card:hover { border-color: #D9A441; box-shadow: 0 6px 18px -12px rgba(180,83,9,.45); }
.ex-tag { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
          text-transform: uppercase; color: #B45309; }
.ex-q { font-size: 0.85rem; color: #33373D; margin-top: 0.35rem; line-height: 1.5; }

/* section titles */
.sec-title { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.45rem;
             font-weight: 700; color: #1C1E21; margin: 2.2rem 0 0.25rem 0; }
.sec-sub { font-size: 0.85rem; color: #6B7078; margin-bottom: 1rem; }

/* sidebar */
section[data-testid="stSidebar"] { background: #F4F0E8; }
.side-stat { display: flex; justify-content: space-between; align-items: baseline;
             padding: 0.3rem 0; border-bottom: 1px dashed #E0DACA; font-size: 0.85rem; }
.side-stat b { font-family: 'Source Serif 4', Georgia, serif; font-size: 1rem; }

/* footer */
.footer { text-align: center; color: #9AA0A8; font-size: 0.75rem;
          margin: 3rem 0 1rem 0; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ============================================================ BACKEND (unchanged) ====

@st.cache_resource(show_spinner=False)
def get_generator(rerank_enabled: bool) -> GroundedGenerator:
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
    return load_generation_config()


def ask_pipeline(query: str, rerank: bool):
    gen = get_generator(rerank)
    return gen.answer(query), gen


def make_pubmed_url(pmid: str) -> str:
    if pmid and pmid.isdigit():
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return ""


def _shorten(text: str, width: int) -> str:
    """Truncate at a word boundary (never mid-word) for titles."""
    return textwrap.shorten(text or "Untitled", width=width, placeholder="…")


def entity_pill(entity: str) -> str:
    kind = "other"
    label = entity
    if ":" in entity:
        prefix, rest = entity.split(":", 1)
        label = rest.strip()
        if prefix.upper() == "DISEASE":
            kind = "disease"
        elif prefix.upper() == "CHEMICAL":
            kind = "chemical"
    safe = label.replace("<", "&lt;").replace(">", "&gt;")
    return f'<span class="ent ent-{kind}">{safe}</span>'


def copy_to_clipboard(text: str):
    """Real clipboard copy via a hidden component (user-gesture triggered)."""
    components.html(
        f"""<script>
        navigator.clipboard.writeText({json.dumps(text)}).then(
          () => {{ window.parent.postMessage({{type: 'sgrag-copied'}}, '*'); }},
          () => {{}}
        );
        </script>""",
        height=0,
    )


# ============================================================ RENDER ====

def render_hero(provider_label: str, is_mock: bool):
    pill_cls = "pill mock" if is_mock else "pill"
    # NOTE: the accent rule uses inline styles — Streamlit's DOM handling can
    # suppress stylesheet-only painting on this particular element.
    st.markdown(f"""
<div class="hero">
  <h1>Scientific Literature<br>GraphRAG</h1>
  <p>Evidence-grounded answers from 5,000 biomedical abstracts — hybrid
     vector&nbsp;+&nbsp;knowledge-graph retrieval, cross-encoder reranking,
     and validated citations.</p>
  <div class="pill-row">
    <span class="pill"><span class="dot"></span>Neo4j connected</span>
    <span class="{pill_cls}"><span class="dot"></span>{provider_label}</span>
    <span class="pill">53,091 chunks · 9,945 entities</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_answer(result) -> None:
    st.markdown('<div class="sec-title">Answer</div>'
                '<div class="sec-sub">Grounded in the retrieved evidence below</div>',
                unsafe_allow_html=True)
    if result.citation_validation and not result.citation_validation.valid:
        st.warning("Some citations could not be verified against the evidence "
                   "and were removed.")
    for w in result.warnings:
        st.warning(w)

    paras = [p for p in result.answer.split("\n") if p.strip()]
    body = "".join(f"<p>{p}</p>" for p in paras)
    st.markdown(f'<div class="card answer-card"><div class="answer-text">{body}</div></div>',
                unsafe_allow_html=True)

    if result.citations:
        chips = []
        for c in result.citations:
            url = make_pubmed_url(c.pmid)
            chunk = f'<span class="chunk">{c.chunk_id}</span>' if c.chunk_id else ""
            ext = '<span class="ext">↗</span>' if url else ""
            inner = (f'<span>PMID:{c.pmid}</span>{chunk}{ext}')
            if url:
                chips.append(f'<a class="cite-chip" href="{url}" target="_blank">{inner}</a>')
            else:
                chips.append(f'<span class="cite-chip">{inner}</span>')
        st.markdown('<div class="cite-row">' + "".join(chips) + "</div>",
                    unsafe_allow_html=True)

    if st.button("⧉ Copy answer with citations", key="btn_copy"):
        lines = [result.answer, "", "Citations:"]
        for c in result.citations:
            lines.append(f"  [PMID:{c.pmid}]"
                         + (f" [{c.chunk_id}]" if c.chunk_id else "")
                         + (f" {make_pubmed_url(c.pmid)}" if make_pubmed_url(c.pmid) else ""))
        copy_to_clipboard("\n".join(lines))
        st.toast("Answer copied to clipboard", icon="⧉")


def render_evidence(evidence: list) -> None:
    st.markdown('<div class="sec-title">Supporting evidence</div>'
                f'<div class="sec-sub">{len(evidence)} chunks from '
                f'{len({e.pmid for e in evidence})} articles, ranked by the cross-encoder</div>',
                unsafe_allow_html=True)
    by_pmid: dict = {}
    for ev in evidence:
        by_pmid.setdefault(ev.pmid, []).append(ev)
    for pmid, items in by_pmid.items():
        url = make_pubmed_url(pmid)
        head = (f'<a href="{url}" target="_blank" style="text-decoration:none;color:#B45309;">'
                f'PMID:{pmid} ↗</a>' if url else f'PMID:{pmid}')
        with st.expander(_shorten(items[0].title, 90), expanded=False):
            st.markdown(f'<div class="ev-pmid">{head}</div>', unsafe_allow_html=True)
            for ev in items:
                rs = ev.reranker_score or 0.0
                pct = max(0, min(100, int(rs / 8 * 100))) if rs else 0
                pills = "".join(entity_pill(e) for e in (ev.matched_entities or []))
                srcs = ", ".join(ev.retrieval_sources) or "—"
                st.markdown(f"""
<div class="ev-head">
  <span class="ev-meta"><code>{ev.chunk_id}</code> · via {srcs}</span>
  <span class="ev-meta">rerank <b>{rs:.2f}</b> · fused <b>{ev.fused_score:.2f}</b></span>
</div>
<div class="scorebar"><div style="width:{pct}%"></div></div>
<div class="ev-meta" style="margin-top:0.5rem;">{pills if pills else "no entities matched"}</div>
<div class="ev-text">{ev.text}</div>
""", unsafe_allow_html=True)


def render_source_overview(evidence: list) -> None:
    st.markdown('<div class="sec-title">Sources</div>'
                '<div class="sec-sub">Unique articles behind the evidence</div>',
                unsafe_allow_html=True)
    by_pmid: dict = {}
    for ev in evidence:
        d = by_pmid.setdefault(ev.pmid, {"title": ev.title, "n": 0, "best": 0.0})
        d["n"] += 1
        d["best"] = max(d["best"], ev.reranker_score or 0.0)
    rows = sorted(by_pmid.items(), key=lambda kv: -kv[1]["best"])
    html = ['<div class="card" style="padding:0.6rem 1.2rem;">']
    for pmid, d in rows:
        url = make_pubmed_url(pmid)
        link = (f'<a href="{url}" target="_blank" style="color:#B45309;font-weight:600;'
                f'text-decoration:none;">PMID:{pmid} ↗</a>' if url else f"PMID:{pmid}")
        html.append(
            f'<div class="side-stat"><span>{link} — {_shorten(d["title"], 75)}</span>'
            f"<span><b>{d['n']}</b> chunk{'s' if d['n'] != 1 else ''} · "
            f"{d['best']:.2f}</span></div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_intelligence(md: dict, gen_md: dict) -> None:
    st.markdown('<div class="sec-title">Retrieval intelligence</div>'
                '<div class="sec-sub">How this answer was produced</div>',
                unsafe_allow_html=True)
    cards = [
        ("Vector", str(md.get("n_vector", "—")), "semantic similarity"),
        ("Entity graph", str(md.get("n_entity", "—")), "MENTIONS expansion"),
        ("Semantic", str(md.get("n_semantic", "—")), "SIMILAR 1-hop"),
        ("Fused", str(md.get("n_fused", "—")), "final candidates"),
    ]
    st.markdown('<div class="stat-grid">' + "".join(
        f'<div class="stat"><div class="v">{v}</div><div class="k">{k}</div>'
        f'<div style="font-size:0.72rem;color:#8A8F98;">{s}</div></div>'
        for k, v, s in cards) + "</div>", unsafe_allow_html=True)

    ents = md.get("query_entities") or []
    cls = md.get("query_class", "—")
    cols = st.columns([1, 2])
    with cols[0]:
        st.markdown(f'<div class="card"><div class="k" style="font-size:0.72rem;'
                    f'font-weight:600;letter-spacing:0.06em;color:#8A8F98;">QUERY TYPE</div>'
                    f'<div style="font-weight:600;margin-top:0.2rem;">{cls}</div></div>',
                    unsafe_allow_html=True)
    with cols[1]:
        pills = "".join(entity_pill(e) for e in ents) or \
            '<span style="color:#8A8F98;font-size:0.85rem;">none — vector-only retrieval</span>'
        st.markdown(f'<div class="card"><div style="font-size:0.72rem;font-weight:600;'
                    f'letter-spacing:0.06em;color:#8A8F98;">DETECTED ENTITIES</div>'
                    f'<div style="margin-top:0.4rem;">{pills}</div></div>',
                    unsafe_allow_html=True)

    lat = []
    for k, label in (("retrieval_ms", "Retrieval"), ("rerank_ms", "Rerank"),
                     ("generation_ms", "Generation")):
        v = md.get(k) if md.get(k) is not None else gen_md.get(k)
        if v is not None:
            lat.append(f"<b>{v}</b>&nbsp;ms {label.lower()}")
    if lat:
        st.markdown(f'<div style="font-size:0.82rem;color:#6B7078;">'
                    f'Latency:&nbsp;{" &nbsp;·&nbsp; ".join(lat)}</div>',
                    unsafe_allow_html=True)


def render_graph_section(result, gen) -> None:
    st.markdown('<div class="sec-title">Evidence graph</div>'
                '<div class="sec-sub">Query entities → evidence chunks → similar chunks</div>',
                unsafe_allow_html=True)
    try:
        md = result.retrieval_metadata or {}
        graph = build_evidence_graph(
            gen.retriever._run, md.get("query_entities") or [], result.evidence)
        try:
            html = render_pyvis_html(graph)
            st.markdown('<div class="card" style="padding:0.5rem;">', unsafe_allow_html=True)
            components.html(html, height=460)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            st.info("Interactive graph unavailable — relationships as a table:")
            st.table(graph.get("edges", []))
        if graph.get("truncated"):
            st.caption("Graph truncated to 60 nodes for readability.")
    except Exception:  # noqa: BLE001
        st.info("Could not build the evidence graph for this query.")


def render_diagnostics(result) -> None:
    with st.expander("Advanced diagnostics"):
        t1, t2 = st.tabs(["Retrieval", "Generation"])
        with t1:
            st.json(result.retrieval_metadata or {})
        with t2:
            st.json(result.generation_metadata or {})


def render_about():
    st.markdown('<div class="sec-title">About this system</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">

**1. Corpus** — 5,000 PubMed abstracts (PubMedQA-derived), fully audited.

**2. Semantic chunking** — MiniLM embeddings + HDBSCAN clustering → 53,091 coherent chunks.

**3. Knowledge graph** — Neo4j: Article / Chunk / Entity nodes, `MENTIONS` and top-5
`SEMANTIC_SIMILAR` edges.

**4. Hybrid retrieval** — vector search + entity-aware graph expansion, weighted fusion.

**5. CrossEncoder reranking** — `ms-marco-MiniLM-L-6-v2` reorders the top-50 candidates
(measured: MRR 0.86 → 0.97 on 200 PubMedQA questions).

**6. Grounded generation** — the LLM answers *only* from retrieved evidence with
`[PMID:…]` citations, and declines when evidence is insufficient.

**7. Citation validation** — every citation is checked against the evidence;
fabricated citations trigger one stricter regeneration, then removal with a warning.

```
query → classify → hybrid retrieval → fusion → rerank
      → evidence pack → LLM → citation validation → answer
```
</div>
""", unsafe_allow_html=True)


def _set_query(text: str):
    """Callback: runs before the script body, so the widget key may be set."""
    st.session_state["query_text"] = text


def _toggle_about():
    st.session_state["show_about"] = not st.session_state.get("show_about")


# ============================================================ PAGE ====

st.set_page_config(page_title="Scientific Literature GraphRAG", layout="wide")
inject_css()

# ---- sidebar ----
with st.sidebar:
    st.markdown("### System")
    gen_cfg = get_cached_gen_config()
    is_mock = gen_cfg.provider != "groq"
    stats_html = "".join(
        f'<div class="side-stat"><span>{k}</span><b>{v}</b></div>'
        for k, v in CORPUS_STATS)
    st.markdown(stats_html, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Models**")
    st.caption(f"Embedding · `all-MiniLM-L6-v2`")
    st.caption(f"Reranker · `ms-marco-MiniLM-L-6-v2`")
    st.caption(f"LLM · `{'mock (illustrative)' if is_mock else gen_cfg.model}`")
    st.markdown("---")
    rerank_on = st.toggle("CrossEncoder reranking", value=True,
                          help="Reorder top candidates with the cross-encoder")
    st.markdown("---")
    st.button("About this system", use_container_width=True,
              on_click=_toggle_about)

# ---- hero ----
render_hero("Mock LLM · illustrative answers" if is_mock else f"Groq · {gen_cfg.model}",
            is_mock)

# ---- query ----
if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""
st.text_area("Your question", height=96, key="query_text",
             placeholder="Ask about mechanisms, treatments, associations…",
             label_visibility="collapsed")

c1, c2, _ = st.columns([1.6, 1, 4.4])
with c1:
    ask = st.button("Search literature", type="primary", use_container_width=True)
with c2:
    st.button("Clear", use_container_width=True, on_click=_set_query, args=("",))

st.markdown('<div class="sec-sub" style="margin-top:1.25rem;">Try an example</div>',
            unsafe_allow_html=True)
cols = st.columns(3)
for i, (tag, ex) in enumerate(EXAMPLES):
    with cols[i]:
        st.markdown(f'<div class="ex-card"><div class="ex-tag">{tag}</div>'
                    f'<div class="ex-q">{ex}</div></div>', unsafe_allow_html=True)
        st.button("Use this question", key=f"ex_{i}", use_container_width=True,
                  on_click=_set_query, args=(ex,))

if st.session_state.get("show_about"):
    render_about()

# ---- answer flow ----
if ask:
    query = (st.session_state.get("query_text") or "").strip()
    if not query:
        st.warning("Please enter a question first.")
        st.stop()
    if len(query) > 2000:
        st.warning("That question is too long (over 2,000 characters). Please shorten it.")
        st.stop()
    try:
        with st.status("Working through the literature…", expanded=True) as status:
            status.write("Classifying query and retrieving evidence…")
            result, gen = ask_pipeline(query, rerank_on)
            status.write("Validating citations…")
            status.update(label="Done", state="complete", expanded=False)
    except RetrievalError:
        st.error("**Database unreachable.** Start Neo4j (`bin/neo4j start`), "
                 "check `SGRAG_NEO4J_PASSWORD`, then retry.")
        log.error("retrieval failed:\n%s", traceback.format_exc())
        st.stop()
    except Exception:  # noqa: BLE001
        st.error("Something went wrong while answering. Details are in the app log.")
        log.error("pipeline failed:\n%s", traceback.format_exc())
        st.stop()

    # The generator degrades gracefully: a retrieval failure comes back as a
    # result carrying a "retrieval error:" warning (not a raised exception).
    # Present that as a clean, actionable error instead of raw socket details.
    retrieval_failed = any(
        str(w).startswith("retrieval error:") for w in result.warnings)

    if retrieval_failed:
        st.error("**Database unreachable.** The literature database (Neo4j) "
                 "could not be reached.\n\n"
                 "Start Neo4j (`bin/neo4j start`), check `SGRAG_NEO4J_PASSWORD`, "
                 "then try again.")
    else:
        render_answer(result)
        if result.evidence:
            render_evidence(result.evidence)
            render_source_overview(result.evidence)
        render_intelligence(result.retrieval_metadata or {},
                            result.generation_metadata or {})
        if result.evidence:
            render_graph_section(result, gen)
        render_diagnostics(result)

    st.button("↺ New search", key="btn_new", on_click=_set_query, args=("",))

st.markdown('<div class="footer">Scientific Literature GraphRAG · evidence-grounded '
            "biomedical QA · Neo4j + MiniLM + CrossEncoder</div>",
            unsafe_allow_html=True)
