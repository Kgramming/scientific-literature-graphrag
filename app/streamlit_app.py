"""Scientific Literature GraphRAG — Streamlit research dashboard (scaffold).

Tabs (per assignment): Ask · Retrieval · Knowledge Graph · Explainability · Evaluation · System.
Implemented in Phase 12; this placeholder keeps the entry point stable.
"""
import streamlit as st

st.set_page_config(page_title="Scientific Literature GraphRAG", layout="wide")
st.title("Scientific Literature GraphRAG")
st.info(
    "Planning phase (Phase 0): the research dashboard is implemented in Phase 12. "
    "See docs/ARCHITECTURE_PROPOSAL.md §2.12 for the tab specification."
)
