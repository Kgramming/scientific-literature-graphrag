# 12 — Streamlit UI and Graph Visualization

## 1. What this component does

`app.py` is the entire frontend: query input, answer rendering with citations, evidence viewer, PyVis knowledge-graph visualization, retrieval diagnostics, live corpus stats, and the 9-theme picker. It's the only thing a demo audience ever sees.

## 2. Why it exists

A pipeline without a UI is a script. The Streamlit app makes the system *demonstrable*: a professor can type a question and watch evidence-grounded answering happen, inspect the retrieval intelligence, and explore the graph — no terminal needed.

## 3. Where it lives in the repository

- `app.py` (1,023 lines) — everything UI
- `src/sgrag/viz/graph_viz.py` — `build_evidence_graph()`, `render_pyvis_html()`: PyVis network construction
- `.streamlit/config.toml` — base theme + `toolbarMode = "minimal"` (hides Streamlit's Deploy toolbar, official mechanism)

## 4. Important classes/functions

- **`ask_pipeline(query, rerank)`** — `get_generator(rerank)` → `gen.answer(query)`; the single bridge from UI to backend.
- **`get_generator()`** — `@st.cache_resource`: builds the `GroundedGenerator` once, reused across reruns (no model reloads per query).
- **`get_graph_stats()`** — 5-minute-cached live Neo4j counts; honest em-dash fallback when unreachable.
- **Render functions:** `render_hero()`, `render_answer()`, `render_evidence()`, `render_source_overview()`, `render_intelligence()`, `render_graph_section()`, `render_diagnostics()`, `render_about()`.
- **`build_evidence_graph()`** — builds nodes/edges from evidence + query entities for PyVis; table fallback if PyVis fails.
- **Theme system:** `THEMES` dict → `--th-*` CSS variables → one generated stylesheet; `st.session_state["theme_name"]` persists across reruns; the popover picker (9 themes) sets it with instant apply.
- **Session state:** `last_result`/`last_gen` persist completed results across reruns and theme switches; `_clear_search()` resets.

## 5. Step-by-step execution (one query)

```text
user types query → clicks "Search literature"
  → ask_pipeline(query) → gen.answer(query) → GenerationResult
  → st.session_state["last_result"] = result   [persists across reruns]
  → render_answer(): answer text + [n] citation chips
  → render_evidence(): evidence cards (rank, PMID, score)
  → render_source_overview(): per-article grouping
  → render_intelligence(): per-stage ms, candidate counts, query entities
  → render_graph_section(): PyVis network (evidence chunks + entities)
  → render_diagnostics(): warnings, model info
```

## 6. Data flow

`GenerationResult` (answer + evidence + diagnostics + metadata) → render functions → Streamlit components. The UI never queries Neo4j for retrieval — only `get_graph_stats()` touches the DB directly (counts only).

## 7. How it connects to the rest of the system

`app.py` imports from `src/sgrag` (`GroundedGenerator`, viz). It owns *zero* retrieval/generation logic — it's a presentation layer over `ask_pipeline()`. Theme CSS is injected per-render via `inject_css()`.

## 8. Important design decisions

- **Cached generator:** `@st.cache_resource` means the cross-encoder and spaCy models load once, not per query.
- **Results in session state:** completed results survive reruns (theme switches don't wipe answers).
- **Single-scroll page:** no nested scroll containers; the browser page is the only scroll context.
- **Honest states:** Neo4j unreachable → em-dashes + notice (never stale numbers); mock LLM → labeled "illustrative answers."
- **9 themes, one architecture:** `THEMES` dict → CSS variables; the popover picker replaced the native selectbox after its dropdown clipped in small viewports.

## 9. Example from the actual project

Query *"What are the risk factors associated with Alzheimer's disease?"* → answer with `[1][2][3]` chips → evidence cards showing chunk text + PMID + scores → "retrieval intelligence" expander with per-stage timings → PyVis graph with entity nodes linked to evidence chunks → theme switchable mid-session without losing results.

## 10. Common misunderstandings

- **"The UI does retrieval."** No — `ask_pipeline()` delegates entirely to `GroundedGenerator`. The UI renders; it doesn't retrieve.
- **"Themes change the pipeline."** No — themes are pure CSS. Retrieval/generation are identical under every theme.
- **"PyVis is the graph database."** No — PyVis is a *visualization* of a subgraph of evidence; Neo4j is the database.

## 11. Professor Viva Questions

- Walk me through your UI.
- How does the frontend talk to the backend?
- What is session state? Why do you need it?
- How do the live corpus stats work? What if Neo4j is down?
- Why 9 themes? How does theme switching work technically?

## 12. Natural Spoken Answers

**"How does your UI connect to the backend?"**
> "Sir, app.py is purely a presentation layer. When you click Search, it calls ask_pipeline(), which gets a cached GroundedGenerator and calls answer(). The returned GenerationResult — answer, evidence, citations, diagnostics — is stored in Streamlit session state and rendered by dedicated functions: answer with citation chips, evidence cards, a retrieval-intelligence panel with per-stage timings, and a PyVis graph of the evidence. The UI never writes Cypher for retrieval; the only direct database call is get_graph_stats(), which just counts nodes for the sidebar."

## 13. Knowledge Bites

- `app.py` = presentation only; `ask_pipeline()` bridges to `GroundedGenerator`.
- `@st.cache_resource` on the generator — models load once.
- Results persist in `st.session_state` across reruns and theme switches.
- Render functions: hero, answer, evidence, sources, intelligence, graph, diagnostics.
- PyVis visualizes the evidence subgraph; table fallback if it fails.
- `get_graph_stats()`: 5-min cached live counts; honest fallback when Neo4j is down.
- 9 themes via `THEMES` → `--th-*` CSS variables; popover picker, instant apply.
- Single-scroll page; no nested scroll regions.
- `.streamlit/config.toml`: official `toolbarMode="minimal"` hides the Deploy toolbar.
