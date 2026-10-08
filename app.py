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

STAT_LABELS = ("Articles", "Chunks", "Entities", "Semantic edges")

# ============================================================ THEME SYSTEM ====
# Centralized palettes. Every theme intentionally defines all UI colors so
# each one is coherent and readable on its own — no color inversion.
# CSS is generated from the selected palette via CSS variables.

DEFAULT_THEME = "Ink & Amber"

THEMES: dict[str, dict[str, str]] = {
    "Ink & Amber": {
        "page_bg": "#FAFAF7", "sidebar_bg": "#F4F0E8", "sidebar_border": "#E5DFD3",
        "card_bg": "#FFFFFF", "card_border": "#E8E3D8",
        "heading": "#1C1E21", "text": "#24272C", "muted": "#6B7078", "faint": "#9AA0A8",
        "accent": "#B45309", "accent_hover": "#92400E", "on_accent": "#FFFFFF",
        "accent_soft": "#F7EBD7", "link": "#9A5B0B",
        "btn_sec_bg": "#FFFFFF", "btn_sec_text": "#1C1E21", "btn_sec_border": "#D8D2C4",
        "input_bg": "#FFFFFF", "input_border": "#D8D2C4", "input_text": "#1C1E21",
        "placeholder": "#9AA0A8",
        "chip_bg": "#1E3A5F", "chip_text": "#F8F6F1",
        "pill_border": "#E5DFD3", "pill_text": "#4A4E54",
        "score_track": "#EFEAE0", "score_fill1": "#B45309", "score_fill2": "#F59E0B",
        "code_text": "#8A5A00", "code_bg": "#F4EFE6",
        "err_bg": "#FDECEA", "err_border": "#E8AFA8", "err_text": "#8F1D1D",
        "warn_bg": "#FFF7E6", "warn_border": "#E8C87E", "warn_text": "#7A4E00",
        "info_bg": "#E8F1FD", "info_border": "#A9C8EE", "info_text": "#1E4FA3",
        "ent_d_bg": "#FDECEA", "ent_d_text": "#9B2C2C", "ent_d_border": "#F5C6C0",
        "ent_c_bg": "#E8F1FD", "ent_c_text": "#1E4FA3", "ent_c_border": "#C4DBF7",
        "ent_o_bg": "#F1EDE6", "ent_o_text": "#5B5F66", "ent_o_border": "#E0DACA",
        "dot_ok": "#16A34A", "dot_warn": "#B45309",
    },
    "Dark / Midnight": {
        "page_bg": "#0E131E", "sidebar_bg": "#141A27", "sidebar_border": "#232C40",
        "card_bg": "#18202F", "card_border": "#2A3549",
        "heading": "#F0F3F9", "text": "#D7DDE8", "muted": "#98A2B6", "faint": "#6B7488",
        "accent": "#E8A33D", "accent_hover": "#F2B45C", "on_accent": "#201503",
        "accent_soft": "#2A2113", "link": "#8FB8E8",
        "btn_sec_bg": "#1E2635", "btn_sec_text": "#E6EAF2", "btn_sec_border": "#33405A",
        "input_bg": "#141A27", "input_border": "#33405A", "input_text": "#EDEFF5",
        "placeholder": "#6B7488",
        "chip_bg": "#2E4A73", "chip_text": "#EDF2FA",
        "pill_border": "#2A3549", "pill_text": "#C2CAD8",
        "score_track": "#232C40", "score_fill1": "#E8A33D", "score_fill2": "#F2B45C",
        "code_text": "#E8A33D", "code_bg": "#141A27",
        "err_bg": "#3A1F1F", "err_border": "#7A2E2E", "err_text": "#F5A3A3",
        "warn_bg": "#3A2C14", "warn_border": "#7A5A1E", "warn_text": "#F2C879",
        "info_bg": "#1B2A44", "info_border": "#2E4A73", "info_text": "#9CC0EE",
        "ent_d_bg": "#3A2226", "ent_d_text": "#F2A3A3", "ent_d_border": "#6E3232",
        "ent_c_bg": "#1E2E4A", "ent_c_text": "#9CC0EE", "ent_c_border": "#2E4A73",
        "ent_o_bg": "#232A3A", "ent_o_text": "#B6BECC", "ent_o_border": "#33405A",
        "dot_ok": "#4ADE80", "dot_warn": "#E8A33D",
    },
    "Dracula": {
        "page_bg": "#282A36", "sidebar_bg": "#21222C", "sidebar_border": "#44475A",
        "card_bg": "#343642", "card_border": "#44475A",
        "heading": "#F8F8F2", "text": "#E8E8E2", "muted": "#9DA0B0", "faint": "#767A8F",
        "accent": "#BD93F9", "accent_hover": "#D0AEFB", "on_accent": "#282A36",
        "accent_soft": "#3D3A52", "link": "#8BE9FD",
        "btn_sec_bg": "#343642", "btn_sec_text": "#F8F8F2", "btn_sec_border": "#6272A4",
        "input_bg": "#21222C", "input_border": "#6272A4", "input_text": "#F8F8F2",
        "placeholder": "#767A8F",
        "chip_bg": "#4A3F6B", "chip_text": "#F8F8F2",
        "pill_border": "#44475A", "pill_text": "#D8D8D2",
        "score_track": "#44475A", "score_fill1": "#BD93F9", "score_fill2": "#FF79C6",
        "code_text": "#F1FA8C", "code_bg": "#21222C",
        "err_bg": "#3D2A33", "err_border": "#8A2E3E", "err_text": "#FF9EAC",
        "warn_bg": "#3D3423", "warn_border": "#8A6D1F", "warn_text": "#F1FA8C",
        "info_bg": "#26394D", "info_border": "#2E5A7A", "info_text": "#8BE9FD",
        "ent_d_bg": "#3D2A33", "ent_d_text": "#FF9EAC", "ent_d_border": "#6E2E3E",
        "ent_c_bg": "#26394D", "ent_c_text": "#8BE9FD", "ent_c_border": "#2E5A7A",
        "ent_o_bg": "#343642", "ent_o_text": "#BDC0CC", "ent_o_border": "#4C4F63",
        "dot_ok": "#50FA7B", "dot_warn": "#FFB86C",
    },
    "Nord": {
        "page_bg": "#2E3440", "sidebar_bg": "#272C36", "sidebar_border": "#434C5E",
        "card_bg": "#3B4252", "card_border": "#4C566A",
        "heading": "#ECEFF4", "text": "#D8DEE9", "muted": "#9BA3B4", "faint": "#6E7687",
        "accent": "#88C0D0", "accent_hover": "#9BD0DE", "on_accent": "#2E3440",
        "accent_soft": "#36414F", "link": "#81A1C1",
        "btn_sec_bg": "#3B4252", "btn_sec_text": "#ECEFF4", "btn_sec_border": "#4C566A",
        "input_bg": "#272C36", "input_border": "#4C566A", "input_text": "#ECEFF4",
        "placeholder": "#6E7687",
        "chip_bg": "#434C5E", "chip_text": "#ECEFF4",
        "pill_border": "#4C566A", "pill_text": "#C9D1DF",
        "score_track": "#434C5E", "score_fill1": "#88C0D0", "score_fill2": "#8FBCBB",
        "code_text": "#EBCB8B", "code_bg": "#272C36",
        "err_bg": "#40282D", "err_border": "#7A3A42", "err_text": "#E89AA2",
        "warn_bg": "#4A3D24", "warn_border": "#7A6234", "warn_text": "#EBCB8B",
        "info_bg": "#2C3E50", "info_border": "#4A5E7A", "info_text": "#9BD0DE",
        "ent_d_bg": "#40282D", "ent_d_text": "#E89AA2", "ent_d_border": "#7A3A42",
        "ent_c_bg": "#2C3E50", "ent_c_text": "#9BD0DE", "ent_c_border": "#4A5E7A",
        "ent_o_bg": "#3B4252", "ent_o_text": "#B4BCCB", "ent_o_border": "#4C566A",
        "dot_ok": "#A3BE8C", "dot_warn": "#EBCB8B",
    },
    "One Dark": {
        "page_bg": "#282C34", "sidebar_bg": "#21252B", "sidebar_border": "#3E4451",
        "card_bg": "#2E333D", "card_border": "#3E4451",
        "heading": "#E8EAEF", "text": "#ABB2BF", "muted": "#8B909B", "faint": "#636D83",
        "accent": "#61AFEF", "accent_hover": "#7DBBF2", "on_accent": "#1B1E24",
        "accent_soft": "#2C3644", "link": "#7DBBF2",
        "btn_sec_bg": "#2E333D", "btn_sec_text": "#E8EAEF", "btn_sec_border": "#4A5160",
        "input_bg": "#21252B", "input_border": "#4A5160", "input_text": "#E8EAEF",
        "placeholder": "#636D83",
        "chip_bg": "#3E4451", "chip_text": "#E8EAEF",
        "pill_border": "#3E4451", "pill_text": "#C3C9D4",
        "score_track": "#3E4451", "score_fill1": "#61AFEF", "score_fill2": "#56B6C2",
        "code_text": "#E5C07B", "code_bg": "#21252B",
        "err_bg": "#3D2428", "err_border": "#7A3540", "err_text": "#E8939C",
        "warn_bg": "#3D3220", "warn_border": "#7A5F2E", "warn_text": "#E5C07B",
        "info_bg": "#24344A", "info_border": "#3A5A80", "info_text": "#7DBBF2",
        "ent_d_bg": "#3D2428", "ent_d_text": "#E8939C", "ent_d_border": "#7A3540",
        "ent_c_bg": "#24344A", "ent_c_text": "#7DBBF2", "ent_c_border": "#3A5A80",
        "ent_o_bg": "#2E333D", "ent_o_text": "#AEB4C0", "ent_o_border": "#3E4451",
        "dot_ok": "#98C379", "dot_warn": "#D19A66",
    },
    "Gruvbox": {
        "page_bg": "#282828", "sidebar_bg": "#232323", "sidebar_border": "#504945",
        "card_bg": "#32302F", "card_border": "#504945",
        "heading": "#FBF1C7", "text": "#EBDBB2", "muted": "#A89A82", "faint": "#7C6F64",
        "accent": "#FE8019", "accent_hover": "#FF9A3C", "on_accent": "#282828",
        "accent_soft": "#3D3226", "link": "#83A598",
        "btn_sec_bg": "#32302F", "btn_sec_text": "#FBF1C7", "btn_sec_border": "#665C54",
        "input_bg": "#232323", "input_border": "#665C54", "input_text": "#FBF1C7",
        "placeholder": "#7C6F64",
        "chip_bg": "#504945", "chip_text": "#FBF1C7",
        "pill_border": "#504945", "pill_text": "#D5C4A1",
        "score_track": "#504945", "score_fill1": "#FE8019", "score_fill2": "#FABD2F",
        "code_text": "#FABD2F", "code_bg": "#232323",
        "err_bg": "#3D2222", "err_border": "#7A3232", "err_text": "#FB9A94",
        "warn_bg": "#3D3220", "warn_border": "#7A5F2E", "warn_text": "#FABD2F",
        "info_bg": "#27333A", "info_border": "#4A5E62", "info_text": "#8EC0C8",
        "ent_d_bg": "#3D2222", "ent_d_text": "#FB9A94", "ent_d_border": "#7A3232",
        "ent_c_bg": "#27333A", "ent_c_text": "#8EC0C8", "ent_c_border": "#4A5E62",
        "ent_o_bg": "#32302F", "ent_o_text": "#C4B49A", "ent_o_border": "#504945",
        "dot_ok": "#B8BB26", "dot_warn": "#FABD2F",
    },
    "Solarized Dark": {
        "page_bg": "#002B36", "sidebar_bg": "#073642", "sidebar_border": "#1A4E5E",
        "card_bg": "#0B3842", "card_border": "#1D5566",
        "heading": "#FDF6E3", "text": "#B0BEC0", "muted": "#839496", "faint": "#5E7A82",
        "accent": "#268BD2", "accent_hover": "#4AA3E0", "on_accent": "#FDF6E3",
        "accent_soft": "#0E3A4A", "link": "#2AA198",
        "btn_sec_bg": "#0B3842", "btn_sec_text": "#FDF6E3", "btn_sec_border": "#2A6B7E",
        "input_bg": "#073642", "input_border": "#2A6B7E", "input_text": "#FDF6E3",
        "placeholder": "#5E7A82",
        "chip_bg": "#155066", "chip_text": "#FDF6E3",
        "pill_border": "#1D5566", "pill_text": "#B0BEC0",
        "score_track": "#0E4A5A", "score_fill1": "#268BD2", "score_fill2": "#2AA198",
        "code_text": "#B58900", "code_bg": "#073642",
        "err_bg": "#3D1F1F", "err_border": "#7A2E2E", "err_text": "#E08A8A",
        "warn_bg": "#3D3218", "warn_border": "#7A6220", "warn_text": "#D9B44A",
        "info_bg": "#123A4A", "info_border": "#1F5A70", "info_text": "#5AC0C8",
        "ent_d_bg": "#3D1F22", "ent_d_text": "#E08A8A", "ent_d_border": "#7A2E35",
        "ent_c_bg": "#123A4A", "ent_c_text": "#5AC0C8", "ent_c_border": "#1F5A70",
        "ent_o_bg": "#0B3842", "ent_o_text": "#93A5A8", "ent_o_border": "#1D5566",
        "dot_ok": "#859900", "dot_warn": "#B58900",
    },
    "Catppuccin Mocha": {
        "page_bg": "#1E1E2E", "sidebar_bg": "#181825", "sidebar_border": "#313244",
        "card_bg": "#262638", "card_border": "#45475A",
        "heading": "#E8ECF8", "text": "#CDD6F4", "muted": "#A6ADC8", "faint": "#6C7086",
        "accent": "#FAB387", "accent_hover": "#FBC89D", "on_accent": "#1E1E2E",
        "accent_soft": "#3A2E22", "link": "#89B4FA",
        "btn_sec_bg": "#262638", "btn_sec_text": "#E8ECF8", "btn_sec_border": "#585B70",
        "input_bg": "#181825", "input_border": "#585B70", "input_text": "#E8ECF8",
        "placeholder": "#6C7086",
        "chip_bg": "#45475A", "chip_text": "#E8ECF8",
        "pill_border": "#45475A", "pill_text": "#BAC2DE",
        "score_track": "#313244", "score_fill1": "#FAB387", "score_fill2": "#F5C2E7",
        "code_text": "#F9E2AF", "code_bg": "#181825",
        "err_bg": "#3D2430", "err_border": "#7A3048", "err_text": "#F5A3B5",
        "warn_bg": "#3D3220", "warn_border": "#7A5C2A", "warn_text": "#F9E2AF",
        "info_bg": "#26314D", "info_border": "#3A4E7A", "info_text": "#89B4FA",
        "ent_d_bg": "#3D2430", "ent_d_text": "#F5A3B5", "ent_d_border": "#7A3048",
        "ent_c_bg": "#26314D", "ent_c_text": "#89B4FA", "ent_c_border": "#3A4E7A",
        "ent_o_bg": "#262638", "ent_o_text": "#A6ADC8", "ent_o_border": "#45475A",
        "dot_ok": "#A6E3A1", "dot_warn": "#FAB387",
    },
    "High Contrast": {
        "page_bg": "#000000", "sidebar_bg": "#0D0D0D", "sidebar_border": "#8A8A8A",
        "card_bg": "#0D0D0D", "card_border": "#FFFFFF",
        "heading": "#FFFFFF", "text": "#FFFFFF", "muted": "#E8E8E8", "faint": "#CFCFCF",
        "accent": "#FFD500", "accent_hover": "#FFE14D", "on_accent": "#000000",
        "accent_soft": "#332D00", "link": "#FFFF00",
        "btn_sec_bg": "#0D0D0D", "btn_sec_text": "#FFFFFF", "btn_sec_border": "#FFFFFF",
        "input_bg": "#000000", "input_border": "#FFFFFF", "input_text": "#FFFFFF",
        "placeholder": "#CFCFCF",
        "chip_bg": "#FFD500", "chip_text": "#000000",
        "pill_border": "#FFFFFF", "pill_text": "#FFFFFF",
        "score_track": "#333333", "score_fill1": "#FFD500", "score_fill2": "#FFE14D",
        "code_text": "#FFFF00", "code_bg": "#0D0D0D",
        "err_bg": "#1A0000", "err_border": "#FF5555", "err_text": "#FFFFFF",
        "warn_bg": "#1A1A00", "warn_border": "#FFD500", "warn_text": "#FFFFFF",
        "info_bg": "#001A33", "info_border": "#4DA6FF", "info_text": "#FFFFFF",
        "ent_d_bg": "#1A0000", "ent_d_text": "#FFFFFF", "ent_d_border": "#FF5555",
        "ent_c_bg": "#001A33", "ent_c_text": "#FFFFFF", "ent_c_border": "#4DA6FF",
        "ent_o_bg": "#0D0D0D", "ent_o_text": "#FFFFFF", "ent_o_border": "#FFFFFF",
        "dot_ok": "#00FF66", "dot_warn": "#FFD500",
    },
}


def theme_palette(name: str) -> dict[str, str]:
    """Return the palette for a theme name, falling back to the default."""
    return THEMES.get(name, THEMES[DEFAULT_THEME])


def build_css(p: dict[str, str]) -> str:
    """Generate the full <style> block from a theme palette.

    Palettes are exposed as CSS variables (--th-*) so the stylesheet stays a
    single template shared by all themes.
    """
    var_defs = "".join(f"--th-{k.replace('_', '-')}: {v};\n" for k, v in p.items())
    return (
        "<style>\n"
        "@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Inter:wght@400;500;600&display=swap');\n"
        ":root {\n" + var_defs + "  --primary-color: var(--th-accent);\n}\n"
        """
/* base */
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background-color: var(--th-page-bg); }
.block-container { max-width: 1080px; padding-top: 2rem; }
hr { border-color: var(--th-card-border); }

/* hero */
.hero { margin: 0.5rem 0 1.75rem 0; }
.hero h1 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-weight: 700; font-size: 2.6rem; line-height: 1.15;
    color: var(--th-heading); margin: 0;
}
.stMarkdown .hero p {
    font-size: 1.02rem; color: var(--th-muted); margin: 0.6rem 0 0 0; max-width: 640px;
}
.pill-row { display: flex; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap; }
.pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
    padding: 0.3rem 0.7rem; border-radius: 999px;
    background: var(--th-card-bg); border: 1px solid var(--th-pill-border);
    color: var(--th-pill-text);
}
.pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--th-dot-ok); }
.pill.mock .dot { background: var(--th-dot-warn); }

/* cards */
.card {
    background: var(--th-card-bg); border: 1px solid var(--th-card-border);
    border-radius: 14px; padding: 1.35rem 1.5rem; margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06), 0 8px 24px -18px rgba(0,0,0,0.35);
}
.card h3 { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem;
           margin: 0 0 0.75rem 0; color: var(--th-heading); }
.card p, .card li { color: var(--th-text); }
.card strong { color: var(--th-heading); }
.answer-card { border-left: 4px solid var(--th-accent); }
.answer-card .answer-text { font-size: 0.98rem; line-height: 1.75; color: var(--th-text); }
.answer-card .answer-text p { margin: 0 0 0.75rem 0; }

/* citation chips */
.cite-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.9rem; }
.cite-row a.cite-chip, .cite-row span.cite-chip {
    display: inline-flex; align-items: baseline; gap: 0.3rem;
    background: var(--th-chip-bg); color: var(--th-chip-text) !important;
    border-radius: 8px; padding: 0.35rem 0.65rem;
    font-size: 0.78rem; font-weight: 600; text-decoration: none !important;
    transition: transform .12s ease, box-shadow .12s ease;
}
.cite-row a.cite-chip *, .cite-row span.cite-chip * { color: var(--th-chip-text) !important; }
.cite-chip:hover { transform: translateY(-1px); box-shadow: 0 4px 10px -4px rgba(0,0,0,0.5); }
.cite-chip .chunk { font-weight: 400; opacity: 0.75; font-size: 0.7rem; }
.cite-chip .ext { opacity: 0.6; font-size: 0.7rem; }

/* entity pills */
.ent { display: inline-block; font-size: 0.72rem; font-weight: 600;
       padding: 0.18rem 0.55rem; border-radius: 999px; margin: 0 0.3rem 0.3rem 0;
       letter-spacing: 0.01em; }
.ent-disease { background: var(--th-ent-d-bg); color: var(--th-ent-d-text);
               border: 1px solid var(--th-ent-d-border); }
.ent-chemical { background: var(--th-ent-c-bg); color: var(--th-ent-c-text);
                border: 1px solid var(--th-ent-c-border); }
.ent-other { background: var(--th-ent-o-bg); color: var(--th-ent-o-text);
             border: 1px solid var(--th-ent-o-border); }

/* evidence */
.ev-head { display: flex; justify-content: space-between; align-items: baseline;
           gap: 1rem; margin-bottom: 0.35rem; }
.ev-title { font-weight: 600; font-size: 0.95rem; color: var(--th-heading); }
.ev-pmid { font-size: 0.78rem; font-weight: 600; color: var(--th-accent); white-space: nowrap; }
.ev-meta { font-size: 0.76rem; color: var(--th-muted); margin: 0.35rem 0; }
.ev-text { font-size: 0.9rem; line-height: 1.65; color: var(--th-text);
           border-left: 2px solid var(--th-card-border); padding-left: 0.9rem;
           margin-top: 0.6rem; }
.scorebar { height: 5px; border-radius: 3px; background: var(--th-score-track);
            margin-top: 0.25rem; overflow: hidden; max-width: 220px; }
.scorebar > div { height: 100%; border-radius: 3px;
                  background: linear-gradient(90deg, var(--th-score-fill1),
                                                     var(--th-score-fill2)); }

/* stat cards */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem;
             margin: 1rem 0 1.5rem 0; }
.stat { background: var(--th-card-bg); border: 1px solid var(--th-card-border);
        border-radius: 12px; padding: 0.9rem 1rem; }
.stat .v { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.45rem;
           font-weight: 700; color: var(--th-heading); }
.stat .k { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
           text-transform: uppercase; color: var(--th-faint); margin-top: 0.15rem; }
.stat .s { font-size: 0.72rem; color: var(--th-faint); }
@media (max-width: 700px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } }

/* example cards */
.ex-card { background: var(--th-card-bg); border: 1px solid var(--th-card-border);
           border-radius: 12px; padding: 0.9rem 1rem; height: 100%;
           transition: border-color .15s ease, box-shadow .15s ease; }
.ex-card:hover { border-color: var(--th-accent); }
.ex-tag { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
          text-transform: uppercase; color: var(--th-accent); }
.ex-q { font-size: 0.85rem; color: var(--th-text); margin-top: 0.35rem; line-height: 1.5; }

/* section titles */
.sec-title { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.45rem;
             font-weight: 700; color: var(--th-heading); margin: 2.2rem 0 0.25rem 0; }
.sec-sub { font-size: 0.85rem; color: var(--th-muted); margin-bottom: 1rem; }

/* links and small labels */
.pmid-link { color: var(--th-link); font-weight: 600; text-decoration: none; }
.pmid-link:hover { text-decoration: underline; }
.lbl { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
       text-transform: uppercase; color: var(--th-faint); }
.lat-line { font-size: 0.82rem; color: var(--th-muted); }
.lat-line b { color: var(--th-text); }

/* sidebar */
section[data-testid="stSidebar"] { background: var(--th-sidebar-bg); }
.side-stat { display: flex; justify-content: space-between; align-items: baseline;
             padding: 0.3rem 0; border-bottom: 1px dashed var(--th-sidebar-border);
             font-size: 0.85rem; color: var(--th-text); }
.side-stat b { font-family: 'Source Serif 4', Georgia, serif; font-size: 1rem;
               color: var(--th-heading); }
.side-stat a { color: var(--th-link); }

/* ---- native Streamlit widgets ---- */
button[data-testid="stBaseButton-secondary"] {
    background: var(--th-btn-sec-bg); color: var(--th-btn-sec-text);
    border: 1px solid var(--th-btn-sec-border); border-radius: 10px;
}
button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--th-accent); color: var(--th-accent);
    background: var(--th-btn-sec-bg);
}
button[data-testid="stBaseButton-secondary"]:focus-visible {
    outline: 2px solid var(--th-accent); outline-offset: 2px;
}
button[data-testid="stBaseButton-primary"] {
    background: var(--th-accent); color: var(--th-on-accent);
    border: 1px solid var(--th-accent); border-radius: 10px;
}
button[data-testid="stBaseButton-primary"]:hover {
    background: var(--th-accent-hover); border-color: var(--th-accent-hover);
    color: var(--th-on-accent);
}
button[data-testid="stBaseButton-primary"]:focus-visible {
    outline: 2px solid var(--th-accent-hover); outline-offset: 2px;
}
div[data-testid="stTextArea"] textarea {
    background: var(--th-input-bg); border: 1px solid var(--th-input-border);
    color: var(--th-input-text); border-radius: 10px;
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: var(--th-placeholder); opacity: 1;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--th-accent); box-shadow: 0 0 0 1px var(--th-accent);
    outline: none;
}
div[data-testid="stSelectbox"] [role="combobox"] {
    background: var(--th-input-bg); border: 1px solid var(--th-input-border);
    border-radius: 10px;
}
div[data-testid="stSelectbox"] [role="combobox"] > div {
    background: transparent;
}
div[data-testid="stSelectbox"] [role="combobox"] input,
div[data-testid="stSelectbox"] [role="combobox"] span {
    color: var(--th-input-text);
}
div[data-testid="stSelectbox"] [role="combobox"]:focus-within {
    border-color: var(--th-accent); box-shadow: 0 0 0 1px var(--th-accent);
}
div[data-testid="stWidgetLabel"] label p, div[data-testid="stWidgetLabel"] p {
    color: var(--th-text);
}
div[data-testid="stToggle"] label { color: var(--th-text); }
div[data-testid="stExpander"] {
    background: var(--th-card-bg); border: 1px solid var(--th-card-border);
    border-radius: 12px;
}
div[data-testid="stExpander"] summary span { color: var(--th-text); }
div[data-testid="stAlert"] { border-radius: 12px; }
div[data-testid="stAlertContainer"] { background: transparent; border: none; padding: 0; }
div[data-testid="stAlertContentError"] {
    background: var(--th-err-bg); border: 1px solid var(--th-err-border);
    border-radius: 12px; padding: 0.9rem 1.1rem;
}
div[data-testid="stAlertContentError"] p { color: var(--th-err-text); }
div[data-testid="stAlertContentWarning"] {
    background: var(--th-warn-bg); border: 1px solid var(--th-warn-border);
    border-radius: 12px; padding: 0.9rem 1.1rem;
}
div[data-testid="stAlertContentWarning"] p { color: var(--th-warn-text); }
div[data-testid="stAlertContentInfo"] {
    background: var(--th-info-bg); border: 1px solid var(--th-info-border);
    border-radius: 12px; padding: 0.9rem 1.1rem;
}
div[data-testid="stAlertContentInfo"] p { color: var(--th-info-text); }
div[data-testid="stStatusWidget"] {
    background: var(--th-card-bg); border: 1px solid var(--th-card-border);
    border-radius: 12px;
}
div[data-testid="stStatusWidget"] p { color: var(--th-text); }
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p {
    color: var(--th-muted);
}
.stMarkdown a { color: var(--th-link); }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: var(--th-heading); }
.stMarkdown p, .stMarkdown li { color: var(--th-text); }
.stMarkdown strong { color: var(--th-heading); }
.stMarkdown code, [data-testid="stCaptionContainer"] code,
section[data-testid="stSidebar"] code {
    color: var(--th-code-text); background: var(--th-code-bg);
    border-radius: 4px; padding: 0.1em 0.35em;
}
code { color: var(--th-code-text); background: var(--th-code-bg);
       border-radius: 4px; padding: 0.1em 0.35em; }
pre { background: var(--th-code-bg); border: 1px solid var(--th-card-border);
      border-radius: 10px; }
pre code { background: transparent; padding: 0; }

/* footer */
.footer { text-align: center; color: var(--th-faint); font-size: 0.75rem;
          margin: 3rem 0 1rem 0; }
</style>
"""
    )


def inject_css(theme_name: str) -> None:
    st.markdown(build_css(theme_palette(theme_name)), unsafe_allow_html=True)





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


@st.cache_resource(show_spinner=False, ttl=300)
def get_graph_stats() -> dict | None:
    """Live corpus counts from Neo4j; None when the database is unreachable.

    Uses the project's existing retrieval configuration
    (configs/retrieval.yaml + SGRAG_NEO4J_* env) — the same connection the
    retriever itself uses — without initializing the heavyweight retrieval
    pipeline (no embedding/reranker model loads). Results are cached for
    5 minutes so the sidebar doesn't hit the database on every rerun.
    """
    from neo4j import GraphDatabase

    from sgrag.retrieval.config import load_retrieval_config

    try:
        cfg = load_retrieval_config("configs/retrieval.yaml")
        driver = GraphDatabase.driver(
            cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))
        driver.verify_connectivity()
        with driver.session(database=cfg.neo4j_database) as session:
            def _count(cypher: str) -> int:
                rec = session.run(cypher).single()
                return int(rec["n"]) if rec else 0

            stats = {
                "Articles": _count("MATCH (a:Article) RETURN count(a) AS n"),
                "Chunks": _count("MATCH (c:Chunk) RETURN count(c) AS n"),
                "Entities": _count("MATCH (e:Entity) RETURN count(e) AS n"),
                "Semantic edges": _count(
                    "MATCH ()-[r:SEMANTIC_SIMILAR]->() RETURN count(r) AS n"),
            }
        driver.close()
        return stats
    except Exception:  # noqa: BLE001 - any failure means "stats unavailable"
        log.warning("live graph stats unavailable", exc_info=True)
        return None


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

def render_hero(provider_label: str, is_mock: bool, neo4j_ok: bool,
                corpus_label: str | None):
    pill_cls = "pill mock" if is_mock else "pill"
    neo4j_pill = ('<span class="pill"><span class="dot"></span>Neo4j connected</span>'
                 if neo4j_ok else
                 '<span class="pill mock"><span class="dot"></span>Neo4j unreachable</span>')
    corpus_pill = (f'<span class="pill">{corpus_label}</span>'
                   if corpus_label else
                   '<span class="pill">corpus stats unavailable</span>')
    # NOTE: the accent rule uses inline styles — Streamlit's DOM handling can
    # suppress stylesheet-only painting on this particular element.
    st.markdown(f"""
<div class="hero">
  <h1>Scientific Literature<br>GraphRAG</h1>
  <p>Evidence-grounded answers from 5,000 biomedical abstracts — hybrid
     vector&nbsp;+&nbsp;knowledge-graph retrieval, cross-encoder reranking,
     and validated citations.</p>
  <div class="pill-row">
    {neo4j_pill}
    <span class="{pill_cls}"><span class="dot"></span>{provider_label}</span>
    {corpus_pill}
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
        head = (f'<a class="pmid-link" href="{url}" target="_blank">'
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
        link = (f'<a class="pmid-link" href="{url}" target="_blank">PMID:{pmid} ↗</a>'
                if url else f"PMID:{pmid}")
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
        f'<div class="s">{s}</div></div>'
        for k, v, s in cards) + "</div>", unsafe_allow_html=True)

    ents = md.get("query_entities") or []
    cls = md.get("query_class", "—")
    cols = st.columns([1, 2])
    with cols[0]:
        st.markdown(f'<div class="card"><div class="lbl">QUERY TYPE</div>'
                    f'<div style="font-weight:600;margin-top:0.2rem;color:var(--th-text);">{cls}</div></div>',
                    unsafe_allow_html=True)
    with cols[1]:
        pills = "".join(entity_pill(e) for e in ents) or \
            '<span style="color:var(--th-muted);font-size:0.85rem;">none — vector-only retrieval</span>'
        st.markdown(f'<div class="card"><div class="lbl">DETECTED ENTITIES</div>'
                    f'<div style="margin-top:0.4rem;">{pills}</div></div>',
                    unsafe_allow_html=True)

    lat = []
    for k, label in (("retrieval_ms", "Retrieval"), ("rerank_ms", "Rerank"),
                     ("generation_ms", "Generation")):
        v = md.get(k) if md.get(k) is not None else gen_md.get(k)
        if v is not None:
            lat.append(f"<b>{v}</b>&nbsp;ms {label.lower()}")
    if lat:
        st.markdown(f'<div class="lat-line">'
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

# Theme is read from session state (persisted across reruns); the selectbox
# in the sidebar updates it and the new CSS applies on the next render —
# no restart required.
theme_name = st.session_state.get("theme_name", DEFAULT_THEME)
if theme_name not in THEMES:
    theme_name = DEFAULT_THEME
inject_css(theme_name)

# ---- sidebar ----
with st.sidebar:
    st.markdown("### System")
    graph_stats = get_graph_stats()
    if graph_stats is not None:
        stats_html = "".join(
            f'<div class="side-stat"><span>{k}</span><b>{v:,}</b></div>'
            for k, v in graph_stats.items())
    else:
        # Honest fallback: em-dashes instead of stale hard-coded numbers.
        stats_html = "".join(
            f'<div class="side-stat"><span>{k}</span><b>—</b></div>'
            for k in STAT_LABELS)
    st.markdown(stats_html, unsafe_allow_html=True)
    if graph_stats is None:
        st.caption("Live stats unavailable — Neo4j unreachable.")
    gen_cfg = get_cached_gen_config()
    is_mock = gen_cfg.provider != "groq"
    st.markdown("---")
    st.markdown("**Models**")
    st.caption(f"Embedding · `all-MiniLM-L6-v2`")
    st.caption(f"Reranker · `ms-marco-MiniLM-L-6-v2`")
    st.caption(f"LLM · `{'mock (illustrative)' if is_mock else gen_cfg.model}`")
    st.markdown("---")
    rerank_on = st.toggle("CrossEncoder reranking", value=True,
                          help="Reorder top candidates with the cross-encoder")
    st.selectbox("Theme", options=list(THEMES.keys()), key="theme_name",
                 help="Interface color theme — applies instantly, no restart needed")
    st.markdown("---")
    st.button("About this system", use_container_width=True,
              on_click=_toggle_about)

# ---- hero ----
_corpus_label = (f"{graph_stats['Chunks']:,} chunks · "
                 f"{graph_stats['Entities']:,} entities"
                 if graph_stats is not None else None)
render_hero("Mock LLM · illustrative answers" if is_mock else f"Groq · {gen_cfg.model}",
            is_mock, neo4j_ok=graph_stats is not None,
            corpus_label=_corpus_label)

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
