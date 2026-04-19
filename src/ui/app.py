"""
app.py — PurpleInsight Main Streamlit Application
──────────────────────────────────────────────────
NatWest Code for Purpose Hackathon
Talk to Data — Seamless Self-Service Intelligence

Pages:
    1. Analyse   — main workspace connected to full AI pipeline
    2. How It Works — explains the system to non-technical judges
    3. Data Explorer — browse registered datasets directly
    4. Dataset Registry — add datasets in one central place

Run:
    streamlit run src/ui/app.py
"""

import sys
import os
import base64
import uuid
import time

# ── Path setup — must come before any project imports ─────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from config.dataset_registry import list_registered_datasets, register_dataset, slugify_dataset_id

# ── Core pipeline imports ──────────────────────────────────────────────────────
try:
    from src.core.intent_router    import IntentRouter, UseCase
    from src.core.ambiguity_handler import AmbiguityHandler
    from src.core.nl_to_sql        import NLToSQL
    from src.core.query_engine     import QueryEngine
    from src.core.narrative        import Narrative
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    PIPELINE_ERROR = str(e)


def get_llm_runtime_status() -> dict:
    """Describe the current LLM provider configuration for the UI."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if groq_key:
        return {
            "provider": "Groq",
            "model": "Llama 3.1 70B",
            "env_var": "GROQ_API_KEY",
            "configured": True,
        }

    if gemini_key:
        return {
            "provider": "Gemini",
            "model": "Gemini",
            "env_var": "GEMINI_API_KEY",
            "configured": True,
        }

    return {
        "provider": "Groq",
        "model": "Llama 3.1 70B",
        "env_var": "GROQ_API_KEY",
        "configured": False,
    }


def get_pipeline_issue(pipeline) -> str:
    """Return the most likely reason the pipeline is unavailable."""
    llm_status = get_llm_runtime_status()

    if not PIPELINE_AVAILABLE:
        return PIPELINE_ERROR

    if not llm_status["configured"]:
        return f"Set {llm_status['env_var']} in .env."

    if llm_status["provider"] != "Groq":
        return (
            f"{llm_status['env_var']} is configured, but this app currently uses Groq. "
            "Add GROQ_API_KEY to .env."
        )

    return "Pipeline failed to initialise. Check the error shown in the app for details."

# ── UI component imports ───────────────────────────────────────────────────────
from src.ui.components.trust_panel    import render_trust_panel
from src.ui.components.chart_renderer import render_chart
from src.ui.components.query_input    import render_query_input, DEMO_QUERIES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "PurpleInsight — NatWest",
    page_icon   = os.path.join(ROOT, "assets", "natwest_logo.png"),
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — NatWest purple/white, clean, professional
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Root variables ─────────────────────────────────────────────────────── */
:root {
    --purple-deep:   #42145f;
    --purple-mid:    #6b3fa0;
    --purple-light:  #e8dff5;
    --purple-pale:   #f5f0fb;
    --white:         #ffffff;
    --grey-100:      #f7f7f8;
    --grey-200:      #ebebed;
    --grey-400:      #9a9aaa;
    --grey-700:      #3d3d4a;
    --green:         #1a8754;
    --amber:         #b45309;
    --red:           #b91c1c;
    --font-main:     'DM Sans', sans-serif;
    --font-mono:     'DM Mono', monospace;
}

/* ── Global reset ───────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: var(--font-main) !important; }

/* ── Hide Streamlit chrome ──────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }

/* ── Sidebar styling ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--purple-deep) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 8px !important;
    font-family: var(--font-main) !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.18) !important;
    border-color: rgba(255,255,255,0.35) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(255,255,255,0.22) !important;
    border-color: rgba(255,255,255,0.5) !important;
    font-weight: 600 !important;
}

/* ── Navbar ─────────────────────────────────────────────────────────────── */
.pi-navbar {
    background: var(--purple-deep);
    padding: 0 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.pi-navbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.pi-navbar-logo {
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
}
.pi-navbar-logo img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}
.pi-navbar-title {
    color: white;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
}
.pi-navbar-subtitle {
    color: rgba(255,255,255,0.6);
    font-size: 12px;
    font-weight: 400;
}
.pi-navbar-badge {
    background: rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.85);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
}

/* ── Cards ──────────────────────────────────────────────────────────────── */
.pi-card {
    background: white;
    border: 1px solid var(--grey-200);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 12px 30px rgba(43, 18, 66, 0.04);
}
.pi-card-purple {
    background:
        radial-gradient(circle at top right, rgba(255,255,255,0.55), transparent 32%),
        linear-gradient(135deg, #f7f1fd 0%, #ede2fb 55%, #f8f3ff 100%);
    border: 1px solid var(--purple-light);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 18px;
    box-shadow: 0 14px 40px rgba(66,20,95,0.08);
}
.pi-shell {
    display: grid;
    gap: 18px;
}
.pi-hero {
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.65), transparent 28%),
        linear-gradient(145deg, #42145f 0%, #5e2d8a 52%, #8c63bc 100%);
    border-radius: 24px;
    padding: 28px 28px 24px;
    color: white;
    margin-bottom: 18px;
    overflow: hidden;
    position: relative;
}
.pi-hero::after {
    content: "";
    position: absolute;
    inset: auto -40px -55px auto;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
}
.pi-eyebrow {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.72;
    margin-bottom: 10px;
}
.pi-hero-title {
    font-size: 31px;
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.03em;
    margin-bottom: 10px;
    max-width: 720px;
}
.pi-hero-subtitle {
    font-size: 14px;
    line-height: 1.7;
    max-width: 760px;
    color: rgba(255,255,255,0.84);
}
.pi-hero-meta {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}
.pi-hero-pill {
    background: rgba(255,255,255,0.12);
    color: white;
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 600;
}
.pi-query-shell {
    margin-bottom: 10px;
}
.pi-query-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 12px;
}
.pi-query-title {
    color: var(--grey-700);
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 6px;
}
.pi-query-subtitle {
    color: #6d6a78;
    font-size: 13px;
    line-height: 1.65;
    max-width: 720px;
}
.pi-query-hint {
    color: #7a7687;
    font-size: 12px;
    line-height: 1.5;
    padding-top: 6px;
}
.pi-chip-row {
    margin-bottom: 10px;
}
.pi-workspace-grid {
    display: grid;
    gap: 18px;
}
.pi-panel-title {
    color: var(--grey-700);
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}
.pi-panel-subtitle {
    color: #75717f;
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 2px;
}
.pi-section-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.1em;
    color: #9893a7;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.pi-answer-card {
    background: linear-gradient(180deg, #ffffff 0%, #fcfbfe 100%);
}
.pi-result-query {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #736d82;
    font-size: 12px;
    margin-bottom: 14px;
}
.pi-result-kpis {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}
.pi-mini-stat {
    background: #faf8fd;
    border: 1px solid #eee6f8;
    border-radius: 12px;
    padding: 12px 14px;
}
.pi-mini-stat-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #938da2;
    margin-bottom: 6px;
}
.pi-mini-stat-value {
    font-size: 18px;
    font-weight: 800;
    color: var(--purple-deep);
}
.pi-trust-intro {
    font-size: 12px;
    color: #706b7d;
    line-height: 1.6;
    margin-bottom: 14px;
}
.pi-trust-stat {
    text-align: center;
    padding: 14px 12px;
    border-radius: 12px;
}
.pi-trust-stat-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.pi-trust-stat-value {
    font-size: 20px;
    font-weight: 800;
    margin-top: 5px;
}
.pi-trust-stat-meta {
    font-size: 11px;
    opacity: 0.8;
}
.pi-trust-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    background: #f9f9fb;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 12px;
    color: #42145f;
}
.pi-trust-chip-metric {
    background: #f5f0fb;
    color: #6b3fa0;
}
.pi-history-card {
    background: linear-gradient(180deg, #ffffff 0%, #fcfbfe 100%);
}
.pi-history-item {
    border: 1px solid #efe8f6;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
    background: #fbf9fd;
}
.pi-history-title {
    color: var(--grey-700);
    font-size: 13px;
    font-weight: 700;
    line-height: 1.5;
    margin-bottom: 6px;
}
.pi-history-meta {
    font-size: 11px;
    color: #958ea6;
}

/* ── Query badge ────────────────────────────────────────────────────────── */
.pi-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.pi-badge-change  { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.pi-badge-compare { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
.pi-badge-breakdown { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
.pi-badge-summary { background: var(--purple-light); color: var(--purple-deep); border: 1px solid #c4b0e0; }
.pi-badge-unknown { background: var(--grey-200); color: var(--grey-700); border: 1px solid #ccc; }

/* ── Answer headline ────────────────────────────────────────────────────── */
.pi-headline {
    font-size: 17px;
    font-weight: 700;
    color: var(--grey-700);
    line-height: 1.4;
    margin: 0 0 10px;
    border-left: 3px solid var(--purple-deep);
    padding-left: 12px;
}
.pi-body {
    font-size: 13.5px;
    color: #555;
    line-height: 1.65;
    margin: 0 0 12px;
}

/* ── Key facts ──────────────────────────────────────────────────────────── */
.pi-fact-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid var(--grey-200);
    font-size: 13px;
    color: var(--grey-700);
}
.pi-fact-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--purple-mid);
    margin-top: 6px;
    flex-shrink: 0;
}

/* ── Anomaly alert ──────────────────────────────────────────────────────── */
.pi-anomaly {
    background: #fef9ec;
    border: 1px solid #fcd34d;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #78350f;
    margin-bottom: 12px;
}

/* ── Metric pills ───────────────────────────────────────────────────────── */
.pi-pill {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.9) !important;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 500;
    margin: 2px 3px 2px 0;
}

/* ── Sidebar section label ──────────────────────────────────────────────── */
.pi-sidebar-label {
    font-size: 9px !important;
    font-weight: 800 !important;
    letter-spacing: 0.14em !important;
    color: rgba(255,255,255,0.45) !important;
    text-transform: uppercase !important;
    margin: 20px 0 8px !important;
}

/* ── Loading state ──────────────────────────────────────────────────────── */
.pi-loading {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 20px;
    color: var(--purple-mid);
    font-size: 14px;
}

/* ── How it works cards ─────────────────────────────────────────────────── */
.hiw-step {
    background: white;
    border: 1px solid var(--grey-200);
    border-radius: 12px;
    padding: 20px;
    position: relative;
}
.hiw-step-number {
    width: 32px;
    height: 32px;
    background: var(--purple-deep);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 12px;
}
.hiw-step-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--grey-700);
    margin-bottom: 6px;
}
.hiw-step-body {
    font-size: 13px;
    color: #666;
    line-height: 1.55;
}

/* ── Data explorer table ────────────────────────────────────────────────── */
.pi-table-header {
    background: var(--purple-deep);
    color: white;
    padding: 10px 16px;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 13px;
}

/* ── Primary button override ────────────────────────────────────────────── */
div[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"],
.stButton > button[kind="primary"] {
    background: var(--purple-deep) !important;
    border-color: var(--purple-deep) !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: var(--font-main) !important;
    transition: all 0.2s !important;
}
div[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"]:hover {
    background: var(--purple-mid) !important;
    border-color: var(--purple-mid) !important;
}

/* ── Text input ─────────────────────────────────────────────────────────── */
.stTextInput > div > input {
    border: 1.5px solid var(--grey-200) !important;
    border-radius: 8px !important;
    font-family: var(--font-main) !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > input:focus {
    border-color: var(--purple-deep) !important;
    box-shadow: 0 0 0 3px rgba(66,20,95,0.08) !important;
}

/* ── Selectbox ──────────────────────────────────────────────────────────── */
.stSelectbox > div > div {
    border: 1.5px solid var(--grey-200) !important;
    border-radius: 8px !important;
    font-family: var(--font-main) !important;
}

/* ── Expander ───────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--purple-pale) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: var(--purple-deep) !important;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
.pi-divider {
    border: none;
    border-top: 1px solid var(--grey-200);
    margin: 20px 0;
}

/* ── Ambiguity banner ───────────────────────────────────────────────────── */
.pi-clarify {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: #1e40af;
    margin-bottom: 14px;
}

/* ── Empty state ────────────────────────────────────────────────────────── */
.pi-empty {
    text-align: center;
    padding: 60px 20px;
    color: var(--grey-400);
}
.pi-empty-icon { font-size: 48px; margin-bottom: 12px; }
.pi-empty-title { font-size: 16px; font-weight: 600; color: var(--grey-700); margin-bottom: 6px; }
.pi-empty-sub { font-size: 13px; }

/* ── Stats row ───────────────────────────────────────────────────────────── */
.pi-stat {
    text-align: center;
    padding: 16px;
    background: linear-gradient(180deg, #ffffff 0%, #faf7fd 100%);
    border: 1px solid #ece3f5;
    border-radius: 14px;
    box-shadow: 0 10px 26px rgba(43, 18, 66, 0.04);
}
.pi-stat-value { font-size: 24px; font-weight: 800; color: var(--purple-deep); }
.pi-stat-label { font-size: 11px; color: var(--grey-400); font-weight: 500; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


def get_natwest_logo_data_uri() -> str:
    """Return the NatWest logo as a data URI for HTML rendering."""
    logo_path = os.path.join(ROOT, "assets", "natwest_logo.png")
    with open(logo_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline singleton — cached across sessions
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Load and cache all pipeline components on first call.
    Cached by Streamlit so models load once, not per query.
    """
    if not PIPELINE_AVAILABLE:
        return None

    try:
        metrics_path = os.path.join(ROOT, "src", "semantic", "metrics.yaml")
        data_dir     = os.path.join(ROOT, "data", "raw")

        return {
            "router":    IntentRouter(metrics_yaml_path=metrics_path),
            "ambiguity": AmbiguityHandler(metrics_yaml_path=metrics_path),
            "nl_to_sql": NLToSQL(metrics_yaml_path=metrics_path),
            "engine":    QueryEngine(data_dir=data_dir),
            "narrator":  Narrative(metrics_yaml_path=metrics_path),
        }
    except Exception as e:
        st.error(f"Pipeline load error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Session state init
# ══════════════════════════════════════════════════════════════════════════════

def init_session():
    """Initialise all session state keys on first load."""
    defaults = {
        "chat_history":       [],
        "page":               "How It Works",
        "feedback_counts":    {"positive": 0, "negative": 0},
        "newly_registered":   None,
        "selected_dataset":   None,   # dataset user explicitly picks in Analyse
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def _build_confidence_reason(confidence: float, ambiguity_result, query_result: dict, sql_result: dict) -> str:
    """Return a human-readable explanation of the confidence score."""
    ambiguity_flags = getattr(ambiguity_result, "detected_terms", [])
    if ambiguity_flags:
        return f"Ambiguous terms detected ({', '.join(ambiguity_flags)}) — results may vary by interpretation."
    if not query_result.get("success") or query_result.get("row_count", 0) == 0:
        return "Query returned no rows — filters may be too narrow or the data may not cover this period."
    sql = sql_result.get("sql", "").lower()
    if "like '%" in sql or 'like "%' in sql:
        return "SQL uses a wildcard match (LIKE) which may produce approximate results."
    if not sql_result.get("metric_definitions"):
        return "No registered metric definitions were matched — answer is based on raw column names."
    if confidence > 0.7:
        return "All metric terms resolved via the semantic dictionary with no ambiguity detected."
    return "Intent classified with moderate confidence — the SQL and answer should be reviewed."


def run_pipeline(query: str, pipeline: dict, forced_dataset: str = None) -> dict:
    """
    Run the full PurpleInsight pipeline for a user query.

    Steps:
        1. Ambiguity handler — resolve vague time references
        2. Intent router     — classify into 4 use cases
        3. NL to SQL         — generate DuckDB SQL via Groq
        4. Query engine      — execute SQL against DuckDB
        5. Narrative         — generate plain-English answer via Groq

    Args:
        query:    Natural language user query
        pipeline: Dict of loaded pipeline components

    Returns:
        dict: Full result including answer, chart data, trust trail
    """
    start_time = time.time()

    # Step 1 — ambiguity resolution
    ambiguity_result = pipeline["ambiguity"].analyse(query)
    time_hints       = pipeline["ambiguity"].get_sql_time_hints(ambiguity_result)
    resolution_summary = pipeline["ambiguity"].format_resolution_summary(ambiguity_result)

    # Step 2 — intent classification
    intent_result = pipeline["router"].classify(query)
    datasets      = pipeline["router"].get_dataset_hint(intent_result, query)
    # If user explicitly selected a dataset, put it first
    if forced_dataset:
        datasets = [forced_dataset] + [d for d in datasets if d != forced_dataset]

    # Step 3 — NL to SQL
    resolved_time = list(time_hints.values())[0] if time_hints else None
    sql_result    = pipeline["nl_to_sql"].generate(
        query        = query,
        use_case     = intent_result.use_case,
        datasets     = datasets,
        resolved_time= resolved_time,
    )

    # Step 4 — execute SQL
    if sql_result["valid"] and sql_result["sql"]:
        query_result = pipeline["engine"].execute(sql_result["sql"])
        query_result["datasets_used"] = datasets
    else:
        query_result = {
            "success":   False,
            "data":      [],
            "columns":   [],
            "row_count": 0,
            "error":     sql_result.get("error", "SQL generation failed"),
            "execution_time_ms": 0,
            "datasets_used": datasets,
        }

    # Step 5 — narrative
    narrative_result = pipeline["narrator"].generate(
        query              = query,
        use_case           = intent_result.use_case,
        query_result       = query_result,
        metric_definitions = sql_result.get("metric_definitions", []),
    )

    total_ms = round((time.time() - start_time) * 1000)

    return {
        # Core answer
        "headline":   narrative_result["headline"],
        "narrative":  narrative_result["narrative"],
        "key_facts":  narrative_result["key_facts"],
        "anomalies":  narrative_result["anomalies"],
        "success":    narrative_result["success"],

        # Use case info
        "use_case":   intent_result.use_case,
        "confidence": intent_result.confidence,

        # Ambiguity
        "needs_clarification":    ambiguity_result.needs_clarification,
        "clarification_question": ambiguity_result.clarification_question,
        "resolution_summary":     resolution_summary,

        # Data
        "data":     query_result.get("data", []),
        "columns":  query_result.get("columns", []),
        "row_count": query_result.get("row_count", 0),
        "query_ms": query_result.get("execution_time_ms", 0),

        # Trust trail
        "trust": {
            "sql":                sql_result.get("sql", ""),
            "valid":              sql_result.get("valid", False),
            "datasets_used":      datasets,
            "metric_definitions": sql_result.get("metric_definitions", []),
            "confidence":         intent_result.confidence,
            "confidence_label":   "High" if intent_result.confidence > 0.7 else "Medium" if intent_result.confidence > 0.45 else "Low",
            "confidence_reason":  _build_confidence_reason(
                intent_result.confidence,
                ambiguity_result,
                query_result,
                sql_result,
            ),
            "use_case_value":     intent_result.use_case.value,
            "intent_method":      intent_result.method,
            "matched_keywords":   intent_result.matched_keywords,
            "ambiguity_flags":    getattr(ambiguity_result, "detected_terms", []),
            "raw_data_exposed":   False,
            "resolution_summary": resolution_summary,
            "execution_ms":       query_result.get("execution_time_ms", 0),
            "total_ms":           total_ms,
        },

        # Chart data (passed to chart_renderer)
        "chart": {
            "query_type": intent_result.use_case.value,
            "data":       query_result.get("data", []),
            "columns":    query_result.get("columns", []),
        },

        "total_ms": total_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Navbar
# ══════════════════════════════════════════════════════════════════════════════

def render_navbar():
    """Render the top navigation bar."""
    logo_data_uri = get_natwest_logo_data_uri()
    st.markdown(f"""
    <div class="pi-navbar">
        <div class="pi-navbar-brand">
            <div class="pi-navbar-logo"><img src="{logo_data_uri}" alt="NatWest logo" /></div>
            <div>
                <div class="pi-navbar-title">PurpleInsight</div>
                <div class="pi-navbar-subtitle">NatWest · Self-Service Banking Intelligence</div>
            </div>
        </div>
        <div class="pi-navbar-badge">Code for Purpose Hackathon 2026</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(pipeline):
    """Render the sidebar: brand, navigation, dataset list, system status."""
    llm_status    = get_llm_runtime_status()
    pipeline_issue = get_pipeline_issue(pipeline)

    with st.sidebar:
        logo_data_uri = get_natwest_logo_data_uri()
        st.markdown(f"""
        <div style="padding:4px 0 20px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <img src="{logo_data_uri}" alt="NatWest logo"
                     style="width:34px;height:34px;object-fit:contain;border-radius:8px;" />
                <div style="font-size:20px;font-weight:800;letter-spacing:-0.5px;">PurpleInsight</div>
            </div>
            <div style="font-size:11px;opacity:0.55;margin-top:2px;">NatWest Banking Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Navigation (logical flow order) ──────────────────────────────────
        st.markdown("<div class='pi-sidebar-label'>NAVIGATION</div>", unsafe_allow_html=True)
        pages = ["How It Works", "Available Data", "Dataset Registry", "Analyse"]
        for page in pages:
            is_active = st.session_state["page"] == page
            if st.button(page, key=f"nav_{page}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state["page"] = page
                st.rerun()

        # ── Available datasets ────────────────────────────────────────────────
        st.markdown("<div class='pi-sidebar-label'>LOADED DATASETS</div>", unsafe_allow_html=True)
        all_datasets = list_registered_datasets()
        for ds in all_datasets:
            dot = "🟢" if ds["exists"] else "🔴"
            st.markdown(
                f"<div style='font-size:11px;opacity:0.8;padding:2px 0;'>"
                f"{dot} {ds['display_name']}</div>",
                unsafe_allow_html=True,
            )

        # ── System status ─────────────────────────────────────────────────────
        st.markdown("<div class='pi-sidebar-label'>SYSTEM STATUS</div>", unsafe_allow_html=True)
        if pipeline:
            ds_count = len([d for d in all_datasets if d["exists"]])
            st.markdown(f"""
            <div style="font-size:12px;opacity:0.8;">
                AI pipeline ready<br>
                {ds_count} dataset{'s' if ds_count != 1 else ''} loaded<br>
                Groq · {llm_status['model']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="font-size:12px;opacity:0.8;">
                Pipeline not connected<br>
                <span style="opacity:0.6;">{pipeline_issue}</span>
            </div>
            """, unsafe_allow_html=True)

        count = len(st.session_state.get("chat_history", []))
        if count > 0:
            st.markdown(
                f"<div style='margin-top:14px;font-size:12px;opacity:0.55;'>"
                f"{count} quer{'y' if count == 1 else 'ies'} this session</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Use case badge
# ══════════════════════════════════════════════════════════════════════════════

def get_badge_html(use_case) -> str:
    """Return coloured badge HTML for a use case."""
    uc = use_case.value if hasattr(use_case, "value") else str(use_case)
    config = {
        "change_analysis": ("pi-badge-change",    "Change Analysis"),
        "compare":         ("pi-badge-compare",   "Comparison"),
        "breakdown":       ("pi-badge-breakdown", "Breakdown"),
        "summarize":       ("pi-badge-summary",   "Summary"),
        "unknown":         ("pi-badge-unknown",   "Analysis"),
    }
    cls, label = config.get(uc, ("pi-badge-unknown", "Analysis"))
    return f"<span class='pi-badge {cls}'>{label}</span>"


# ══════════════════════════════════════════════════════════════════════════════
# Result renderer
# ══════════════════════════════════════════════════════════════════════════════

def render_result(entry: dict):
    """
    Render a complete query result including:
    - Use case badge
    - Headline + narrative
    - Anomaly alerts
    - Key facts
    - Chart
    - Data table
    - Trust trail (expandable)
    - Feedback buttons
    """
    result   = entry["result"]
    query_id = entry["query_id"]

    confidence_label = result.get("trust", {}).get("confidence_label", "Medium")
    confidence_score = round(result.get("confidence", 0.0) * 100)

    datasets_used = result.get("trust", {}).get("datasets_used", [])
    ds_label = next(
        (d["display_name"] for d in list_registered_datasets()
         if d["dataset_id"] == datasets_used[0]) if datasets_used else [],
        datasets_used[0] if datasets_used else "Unknown",
    )

    st.markdown("<div class='pi-card pi-answer-card'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pi-result-query">
        <span style="color:#9a9aaa;">Query</span>
        <strong style="color:#42145f;">{entry['query']}</strong>
        <span style="margin-left:auto; background:#f5f0fb; color:#6b3fa0; border-radius:12px;
                     padding:2px 10px; font-size:11px; font-weight:600; white-space:nowrap;">
            Source: {ds_label}
        </span>
    </div>
    """, unsafe_allow_html=True)

    if result.get("needs_clarification") and result.get("clarification_question"):
        st.markdown(f"""
        <div class='pi-clarify'>
            <strong>Clarification:</strong> {result['clarification_question']}
        </div>
        """, unsafe_allow_html=True)

    # Time resolution note
    if result.get("resolution_summary"):
        st.markdown(f"""
        <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px;
                    padding:8px 14px; font-size:12px; color:#166534; margin-bottom:12px;">
            {result['resolution_summary']}
        </div>
        """, unsafe_allow_html=True)

    badge_html = get_badge_html(result.get("use_case", "unknown"))
    st.markdown(badge_html, unsafe_allow_html=True)

    if result.get("headline"):
        st.markdown(f"<p class='pi-headline'>{result['headline']}</p>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pi-result-kpis">
        <div class="pi-mini-stat">
            <div class="pi-mini-stat-label">Confidence</div>
            <div class="pi-mini-stat-value">{confidence_label} · {confidence_score}%</div>
        </div>
        <div class="pi-mini-stat">
            <div class="pi-mini-stat-label">Rows Returned</div>
            <div class="pi-mini-stat-value">{result.get('row_count', 0)}</div>
        </div>
        <div class="pi-mini-stat">
            <div class="pi-mini-stat-label">Query Runtime</div>
            <div class="pi-mini-stat-value">{result.get('query_ms', 0)}ms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for anomaly in result.get("anomalies", []):
        st.markdown(f"<div class='pi-anomaly'>{anomaly}</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<div class='pi-section-label'>Narrative</div>", unsafe_allow_html=True)

        narrative = result.get("narrative", "")
        if narrative:
            lines = []
            skip_headers = ["HEADLINE:", "DATA SOURCE:", "KEY FACTS:", "TOP CONTRIBUTORS:",
                            "KEY DIFFERENCES:", "THIS PERIOD:", "KEY TAKEAWAY:"]
            for line in narrative.split("\n"):
                if not any(line.strip().startswith(h) for h in skip_headers):
                    lines.append(line)
            clean_narrative = "\n".join(lines).strip()
            st.markdown(f"<p class='pi-body'>{clean_narrative}</p>", unsafe_allow_html=True)

    with col_right:
        if result.get("key_facts"):
            st.markdown("<div class='pi-section-label'>Key Facts</div>", unsafe_allow_html=True)
            for fact in result["key_facts"][:4]:
                st.markdown(f"""
                <div class='pi-fact-row'>
                    <div class='pi-fact-dot'></div>
                    <div>{fact}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    chart_data = result.get("chart", {})
    if chart_data.get("data"):
        st.markdown("<div class='pi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='pi-section-label'>Visual Breakdown</div>", unsafe_allow_html=True)
        render_chart(chart_data)
        st.markdown("</div>", unsafe_allow_html=True)

    # Data table (collapsible)
    if result.get("data") and result.get("columns"):
        with st.expander(f"View raw query results ({result.get('row_count', 0)} rows · {result.get('query_ms', 0)}ms)", expanded=False):
            df = pd.DataFrame(result["data"])
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Query executed in {result.get('query_ms', 0)}ms via DuckDB")

    # Trust trail
    trust = result.get("trust", {})
    if trust:
        render_trust_panel(trust)

    # ── Feedback row ──────────────────────────────────────────────────────────
    st.markdown("<hr class='pi-divider'>", unsafe_allow_html=True)
    fb_key = f"fb_{query_id}"
    fb_val = st.session_state.get(fb_key)

    col_lbl, col_up, col_dn, col_resp = st.columns([2, 1, 1, 3])
    with col_lbl:
        st.markdown(
            "<p style='margin:8px 0;font-size:13px;color:#666;font-weight:500;'>"
            "Was this answer helpful?</p>",
            unsafe_allow_html=True,
        )
    with col_up:
        if st.button("Helpful", key=f"up_{query_id}", use_container_width=True, type="primary"):
            st.session_state[fb_key] = "positive"
            st.session_state["feedback_counts"]["positive"] += 1
            st.rerun()
    with col_dn:
        if st.button("Needs work", key=f"dn_{query_id}", use_container_width=True):
            st.session_state[fb_key] = "negative"
            st.session_state["feedback_counts"]["negative"] += 1
            st.rerun()
    with col_resp:
        if fb_val == "positive":
            st.markdown(
                "<div style='background:#d1fae5;color:#065f46;border-radius:8px;"
                "padding:6px 14px;font-size:12px;font-weight:600;margin-top:2px;'>"
                "Marked as helpful — thank you!</div>",
                unsafe_allow_html=True,
            )
        elif fb_val == "negative":
            st.text_input(
                "What went wrong?",
                placeholder="e.g. wrong date range, metric seemed off…",
                key=f"fb_note_{query_id}",
                label_visibility="collapsed",
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Analyse
# ══════════════════════════════════════════════════════════════════════════════

_DATASET_DEMOS = {
    "regional_revenue": [
        ("Compare regions", "Compare total revenue across all regions last year"),
        ("Revenue breakdown", "Show the breakdown of revenue by product"),
        ("Revenue trend", "Show how revenue changed month by month"),
        ("Top region", "Which region has the highest revenue"),
    ],
    "customer_metrics": [
        ("NPS by region", "Compare NPS scores across all regions"),
        ("Signup trend", "Show new signup trends over time"),
        ("Churn compare", "Compare churn rate across customer segments"),
        ("Complaints", "Show breakdown of complaints by region"),
    ],
    "product_performance": [
        ("Volume compare", "Compare transaction volume across all products"),
        ("Revenue breakdown", "Show breakdown of revenue by product"),
        ("Top product", "Which product has the highest satisfaction score"),
        ("Return rates", "Show return rate by product"),
    ],
    "cost_breakdown": [
        ("Cost by dept", "Show the breakdown of costs by department"),
        ("Dept compare", "Compare total costs across all departments"),
        ("By category", "Show cost breakdown by cost category"),
        ("Highest cost", "Which department has the highest costs"),
    ],
    "weekly_kpis": [
        ("Signup trend", "Show how new signups and churn have changed week by week"),
        ("KPI summary", "Give me a summary of weekly KPI trends"),
        ("NPS trend", "Show the trend in NPS score over time"),
        ("Digital adoption", "Show digital adoption rate trend"),
    ],
}

def _get_demo_queries(dataset_id: str, display_name: str, use_cases: list) -> list:
    """Return 4 demo query (label, query) pairs for a given dataset."""
    if dataset_id in _DATASET_DEMOS:
        return _DATASET_DEMOS[dataset_id]
    demos = []
    uc_templates = {
        "change_analysis": (f"What changed?", f"Why did a metric change in {display_name}?"),
        "compare":         (f"Compare categories", f"Compare the top categories in {display_name}"),
        "breakdown":       (f"Breakdown", f"Show breakdown of {display_name} data by category"),
        "summarize":       (f"Summary", f"Give me a summary of {display_name}"),
    }
    for uc in use_cases:
        if uc in uc_templates:
            demos.append(uc_templates[uc])
    return demos[:4]


def page_analyse(pipeline):
    """Main analysis workspace — dataset selector → query → result."""

    all_datasets = list_registered_datasets()
    available    = [d for d in all_datasets if d["exists"]]
    history      = st.session_state.get("chat_history", [])
    fb           = st.session_state.get("feedback_counts", {})
    total_fb     = fb.get("positive", 0) + fb.get("negative", 0)
    helpful_pct  = round(fb.get("positive", 0) / total_fb * 100) if total_fb > 0 else 0

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="pi-hero">
        <div class="pi-eyebrow">NatWest · AI-Powered Banking Intelligence</div>
        <div class="pi-hero-title">Ask your data anything.</div>
        <div class="pi-hero-subtitle">
            Select a dataset, type your question in plain English, and receive a verified,
            narrated answer with charts and a full audit trail — in seconds.
        </div>
        <div class="pi-hero-meta">
            <span class="pi-hero-pill">Change Analysis</span>
            <span class="pi-hero-pill">Comparison</span>
            <span class="pi-hero-pill">Breakdown</span>
            <span class="pi-hero-pill">Summary</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (str(len(history)), "Queries this session"),
        (str(len(available)), "Datasets available"),
        (f"{helpful_pct}%", "Helpfulness score"),
        ("Ready" if pipeline else "Review", "Pipeline status"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class='pi-stat'>
                <div class='pi-stat-value'>{val}</div>
                <div class='pi-stat-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)

    # ── New dataset notification ───────────────────────────────────────────────
    newly = st.session_state.get("newly_registered")
    if newly:
        col_n, col_x = st.columns([11, 1])
        with col_n:
            st.markdown(f"""
            <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
                        padding:12px 18px;margin-bottom:14px;">
                <span style="font-size:13px;font-weight:700;color:#166534;">
                    New dataset ready: {newly['display_name']}
                </span>
                <span style="font-size:12px;color:#166534;margin-left:8px;">
                    — select it below and start querying.
                </span>
            </div>""", unsafe_allow_html=True)
        with col_x:
            if st.button("×", key="dismiss_notif"):
                st.session_state["newly_registered"] = None
                st.rerun()

    # ── Dataset selector ─────────────────────────────────────────────────────
    if not available:
        st.error("No datasets are available. Go to Dataset Registry to add one.")
        return

    st.markdown("<div class='pi-card'>", unsafe_allow_html=True)
    st.markdown("<div class='pi-panel-title'>1 — Choose a dataset</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pi-panel-subtitle'>Select the dataset you want to query. "
        "Your question will be answered using this data source.</div>",
        unsafe_allow_html=True,
    )

    ds_options   = [d["dataset_id"] for d in available]
    ds_fmt       = {d["dataset_id"]: d["display_name"] for d in available}
    default_idx  = 0
    prev_sel     = st.session_state.get("selected_dataset")
    if prev_sel and prev_sel in ds_options:
        default_idx = ds_options.index(prev_sel)

    selected_id = st.selectbox(
        "Dataset",
        options=ds_options,
        index=default_idx,
        format_func=lambda did: ds_fmt.get(did, did),
        label_visibility="collapsed",
        key="dataset_picker",
    )
    st.session_state["selected_dataset"] = selected_id

    # Show dataset info card
    sel_meta = next((d for d in available if d["dataset_id"] == selected_id), {})
    uc_badges = "".join(
        f"<span style='background:#e8dff5;color:#42145f;border-radius:12px;padding:2px 10px;"
        f"font-size:11px;font-weight:600;margin-right:4px;'>{uc.replace('_',' ').title()}</span>"
        for uc in sel_meta.get("primary_use_cases", [])
    )
    st.markdown(f"""
    <div style="background:#faf8fd;border:1px solid #ece3f5;border-radius:10px;
                padding:12px 16px;margin-top:10px;">
        <div style="font-size:13px;color:#3d3d4a;margin-bottom:4px;">
            <strong>{sel_meta.get('display_name','')}</strong>
            <span style="color:#9a9aaa;font-size:11px;margin-left:8px;">
                {sel_meta.get('category','')}
            </span>
        </div>
        <div style="font-size:12px;color:#666;margin-bottom:8px;">
            {sel_meta.get('description','')}
        </div>
        <div>{uc_badges}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Query input ───────────────────────────────────────────────────────────
    st.markdown("<div class='pi-card'>", unsafe_allow_html=True)
    st.markdown("<div class='pi-panel-title'>2 — Ask your question</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pi-panel-subtitle'>Type a banking question in plain English. "
        "Use the example queries below to get started.</div>",
        unsafe_allow_html=True,
    )

    # Inline demo queries for selected dataset
    demos = _get_demo_queries(
        selected_id,
        sel_meta.get("display_name", selected_id),
        sel_meta.get("primary_use_cases", []),
    )
    if demos:
        st.markdown("<div class='pi-section-label' style='margin-top:8px;'>Example queries</div>",
                    unsafe_allow_html=True)
        demo_cols = st.columns(len(demos))
        for i, (col, (label, q)) in enumerate(zip(demo_cols, demos)):
            with col:
                if st.button(label, key=f"demo_{selected_id}_{label}_{i}", use_container_width=True):
                    st.session_state["auto_query"] = q
                    st.rerun()

    with st.form("main_query_form", clear_on_submit=False):
        query = st.text_input(
            "Ask your data",
            placeholder=f"e.g. {demos[0][1] if demos else 'Ask a question about the data…'}",
            label_visibility="collapsed",
        )
        col_hint, col_btn = st.columns([5, 1.2])
        with col_hint:
            st.markdown(
                "<div class='pi-query-hint'>Questions can include: why did X change, "
                "compare A vs B, show breakdown of X, give me a summary.</div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            submitted = st.form_submit_button("Analyse", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    auto        = st.session_state.pop("auto_query", None)
    final_query = (query.strip() if submitted and query else None) or auto

    if final_query:
        if not pipeline:
            st.error(f"Pipeline not available. {get_pipeline_issue(pipeline)}")
        else:
            with st.spinner(f"Analysing using {sel_meta.get('display_name', selected_id)}…"):
                result = run_pipeline(final_query, pipeline, forced_dataset=selected_id)
            entry = {
                "query":    final_query,
                "result":   result,
                "query_id": str(uuid.uuid4())[:8],
                "dataset":  selected_id,
            }
            st.session_state["chat_history"].append(entry)

    history = st.session_state.get("chat_history", [])

    # ── Result ────────────────────────────────────────────────────────────────
    if not history:
        st.markdown("""
        <div class='pi-card pi-empty'>
            <div class='pi-empty-icon'>💬</div>
            <div class='pi-empty-title'>Ready to answer your banking questions</div>
            <div class='pi-empty-sub'>
                Select a dataset above, pick an example query or write your own, then click Analyse.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(
        "<div style='margin-top:6px;'></div>",
        unsafe_allow_html=True,
    )
    render_result(history[-1])

    if len(history) > 1:
        with st.expander(f"Previous queries this session ({len(history)-1})", expanded=False):
            for entry in reversed(history[:-1]):
                ds_name = ds_fmt.get(entry.get("dataset", ""), entry.get("dataset", ""))
                col_info, col_rerun = st.columns([5, 1])
                with col_info:
                    st.markdown(f"""
                    <div class="pi-history-item">
                        <div class="pi-history-title">{entry['query']}</div>
                        <div class="pi-history-meta">
                            Dataset: {ds_name} ·
                            Use case: {getattr(entry['result'].get('use_case'), 'value', 'analysis')} ·
                            {entry['result'].get('query_ms', 0)}ms
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col_rerun:
                    if st.button("Re-run", key=f"rerun_{entry['query_id']}", use_container_width=True):
                        st.session_state["auto_query"] = entry["query"]
                        st.session_state["selected_dataset"] = entry.get("dataset", selected_id)
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — How It Works
# ══════════════════════════════════════════════════════════════════════════════

def page_how_it_works():
    """Landing page — hero, feature coverage, pipeline walkthrough, use cases."""

    # ── Hero (same as Analyse, since this is now the entry point) ─────────────
    st.markdown("""
    <div class="pi-hero">
        <div class="pi-eyebrow">NatWest · Code for Purpose Hackathon 2026</div>
        <div class="pi-hero-title">PurpleInsight — Talk to Data.</div>
        <div class="pi-hero-subtitle">
            Ask any banking question in plain English. PurpleInsight classifies your intent,
            generates verified SQL, runs it against your data in milliseconds, and returns
            a narrated answer with charts and a full audit trail — no SQL knowledge required.
        </div>
        <div class="pi-hero-meta">
            <span class="pi-hero-pill">NatWest Banking Datasets</span>
            <span class="pi-hero-pill">Groq · Llama 3.1 70B</span>
            <span class="pi-hero-pill">DuckDB In-Memory</span>
            <span class="pi-hero-pill">23 Tests Passing</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA button ────────────────────────────────────────────────────────────
    col_cta, _ = st.columns([2, 5])
    with col_cta:
        if st.button("Start Analysing Data", type="primary", use_container_width=True):
            st.session_state["page"] = "Analyse"
            st.rerun()

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # ── Three pillars ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    for col, (num, title, desc) in zip([c1, c2, c3], [
        ("1", "Clarity",  "Answers in plain English. No SQL, no jargon, no data science degree required."),
        ("2", "Trust",    "Every answer shows the SQL executed, metric definitions applied, and source dataset. Nothing is hidden."),
        ("3", "Speed",    "DuckDB executes queries in milliseconds. Groq Llama generates narrated answers in under 2 seconds."),
    ]):
        with col:
            st.markdown(f"""
            <div class='hiw-step'>
                <div class='hiw-step-number'>{num}</div>
                <div class='hiw-step-title'>{title}</div>
                <div class='hiw-step-body'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    # ── Feature coverage grid ─────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:11px;font-weight:800;letter-spacing:0.12em;color:#9893a7;
                text-transform:uppercase;margin-bottom:14px;">
        What is covered
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("Natural Language → SQL",
         "Type a plain-English question and receive a validated DuckDB SQL query. "
         "The LLM is constrained to SELECT-only with schema-aware prompting.",
         "Groq · Llama 3.3 70B · DuckDB"),
        ("4 Use Case Intent Router",
         "Queries are automatically classified into change_analysis, compare, breakdown, "
         "or summarize — each with a tailored prompt template and output format.",
         "Keyword match → semantic fallback · sentence-transformers"),
        ("Semantic Metric Dictionary",
         "20 NatWest banking metrics (revenue, churn rate, NPS, complaints, etc.) defined "
         "in metrics.yaml. Every query resolves terms through this dictionary before the LLM.",
         "metrics.yaml · 20 metrics · canonical definitions"),
        ("Trust Trail Panel",
         "Every answer shows: SQL executed, metric definitions applied, data source, "
         "confidence level (HIGH/MEDIUM/LOW with reason), and thumbs up/down feedback.",
         "Built in-app · no external dependency"),
        ("Ambiguity Handler",
         "Vague time references like 'last month', 'this quarter', 'recently' are resolved "
         "into concrete date filters before SQL generation — no wrong date ranges.",
         "python-dateutil · zero API cost"),
        ("Auto Chart Selection",
         "Breakdown → stacked bar, Compare → grouped bar or line, "
         "Change Analysis → waterfall, Summarize → KPI cards. All styled in NatWest purple.",
         "Plotly · #42145f theme"),
        ("Dataset Registry",
         "New CSVs can be uploaded and registered through the UI with no code changes. "
         "Schema is auto-detected and the dataset is immediately queryable.",
         "config/datasets.yaml · DuckDB views"),
        ("Zero Raw Data Exposure",
         "Only aggregated query results are ever returned. Row-level data never leaves "
         "the engine — enforced at both SQL validation and engine level.",
         "SELECT + aggregates only · MAX 50 rows"),
        ("LLM Fallback",
         "If the primary LLM (Groq) is unavailable, deterministic local SQL generation "
         "keeps the system running — demo never breaks on connectivity issues.",
         "Local fallback · no downtime"),
        ("FastAPI Backend",
         "Programmatic access via /query, /metrics, /health, and /feedback endpoints — "
         "the full pipeline is accessible as a REST API for integration.",
         "FastAPI · /query · /feedback"),
        ("Streamlit UI",
         "Clean product interface with NatWest branding, dataset selector, "
         "inline example queries, data explorer, and dataset registry.",
         "Streamlit · DM Sans · #42145f"),
        ("23 Tests Passing",
         "Covering intent routing, ambiguity handling, SQL generation, query engine, "
         "trust builder, and dataset registry — ensuring reliability.",
         "pytest · 23 tests · all green"),
    ]

    # Render in a 3-column grid
    for row_start in range(0, len(features), 3):
        cols = st.columns(3)
        for col, feat in zip(cols, features[row_start:row_start + 3]):
            title_f, desc_f, tech_f = feat
            with col:
                st.markdown(f"""
                <div style="background:white;border:1px solid #ece3f5;border-radius:12px;
                            padding:16px 18px;margin-bottom:14px;
                            box-shadow:0 4px 12px rgba(66,20,95,0.05);height:100%;">
                    <div style="font-size:13px;font-weight:700;color:#42145f;margin-bottom:6px;">
                        {title_f}
                    </div>
                    <div style="font-size:12px;color:#555;line-height:1.6;margin-bottom:10px;">
                        {desc_f}
                    </div>
                    <div style="background:#f5f0fb;border-radius:6px;padding:4px 10px;
                                font-size:10px;color:#6b3fa0;font-family:monospace;
                                display:inline-block;">
                        {tech_f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    # ── AI Pipeline step-by-step ──────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:11px;font-weight:800;letter-spacing:0.12em;color:#9893a7;
                text-transform:uppercase;margin-bottom:14px;">
        The AI Pipeline — Step by Step
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1", "Ambiguity Handler",
         "Detects vague time references like 'last month' or 'this week' and resolves them "
         "to exact dates before anything else happens. You never get a wrong date range.",
         "Pure Python · python-dateutil · zero API cost"),
        ("2", "Intent Router",
         "Classifies your question into one of 4 use cases: Change Analysis, Comparison, "
         "Breakdown, or Summary. Uses local sentence-transformer embeddings — no API call.",
         "sentence-transformers · all-MiniLM-L6-v2 · runs locally"),
        ("3", "Semantic Metric Dictionary",
         "Every query is resolved through a YAML metric dictionary before hitting the LLM. "
         "This ensures 'revenue' always means the same thing across all queries and teams.",
         "metrics.yaml · 20 metric definitions · canonical NatWest terms"),
        ("4", "NL to SQL (Groq)",
         "Groq-hosted Llama converts your question into a valid DuckDB SQL query using "
         "metric definitions and table schemas. Only SELECT queries are allowed.",
         "Llama 3.3 70B · DuckDB SQL · schema-aware prompting"),
        ("5", "Query Engine",
         "DuckDB executes the SQL directly in memory. No database server, no cloud. "
         "Raw rows are never returned — only aggregates up to 50 rows.",
         "DuckDB in-memory · MAX 50 rows · aggregates only"),
        ("6", "Narrative (Groq)",
         "Groq Llama converts query results into a plain-English answer using "
         "use-case-specific templates that match the NatWest problem document format.",
         "Llama 3.3 70B · 4 narrative templates · anomaly detection"),
        ("7", "Trust Trail",
         "Every answer is wrapped in a full audit trail: SQL executed, metric definitions "
         "applied, dataset used, confidence level, and raw data exposure status (always none).",
         "Built in-app · no external dependency"),
    ]

    for num, title, desc, tech in steps:
        with st.expander(f"Step {num} — {title}", expanded=(num == "1")):
            col_desc, col_tech = st.columns([3, 1])
            with col_desc:
                st.markdown(
                    f"<p style='font-size:14px;color:#444;line-height:1.6;'>{desc}</p>",
                    unsafe_allow_html=True,
                )
            with col_tech:
                st.markdown(f"""
                <div style="background:#f5f0fb;border-radius:8px;padding:10px 12px;
                            font-size:11px;color:#6b3fa0;font-family:monospace;line-height:1.6;">
                    {tech}
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    # ── 4 use cases with examples ─────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:11px;font-weight:800;letter-spacing:0.12em;color:#9893a7;
                text-transform:uppercase;margin-bottom:14px;">
        The 4 Use Cases — with Example Outputs
    </div>
    """, unsafe_allow_html=True)

    use_cases = [
        ("1", "Change Analysis", "change",
         "Why did revenue drop last month?",
         "Revenue decreased by 20.5% in Feb 2024. The biggest contributor was a 27.6% drop in the South region coinciding with a churn spike from 5.3% to 9.1%."),
        ("2", "Comparison", "compare",
         "Compare North vs South region revenue",
         "North outperforms South by £3.1M (+26.6%) in H1 2024. South was severely impacted in Feb–Apr where revenue fell 27.6%."),
        ("3", "Breakdown", "breakdown",
         "Show the breakdown of costs by department",
         "Technology accounts for 32% of total costs at £25.9M, followed by Operations at 24%. Combined they represent over half of total spend."),
        ("4", "Summary", "summary",
         "Give me a weekly summary of customer metrics",
         "This week: Signups grew by 5%, churn remained stable at 1.8%, NPS improved to 44, and average handle time decreased by 12 seconds."),
    ]

    badge_styles = {
        "change":    ("pi-badge-change",    "#92400e", "#fef3c7"),
        "compare":   ("pi-badge-compare",   "#1e40af", "#dbeafe"),
        "breakdown": ("pi-badge-breakdown", "#065f46", "#d1fae5"),
        "summary":   ("pi-badge-summary",   "#42145f", "#e8dff5"),
    }

    for order, title, uc_key, query_ex, output_ex in use_cases:
        _, text_color, bg_color = badge_styles[uc_key]
        st.markdown(f"""
        <div class='pi-card' style='margin-bottom:10px;'>
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
                <span style='background:{bg_color};color:{text_color};border-radius:20px;
                             padding:3px 12px;font-size:11px;font-weight:700;'>
                    {title}
                </span>
                <span style='font-size:13px;color:#9a9aaa;'>Use Case {order}</span>
            </div>
            <div style='display:flex;gap:16px;flex-wrap:wrap;'>
                <div style='flex:1;min-width:200px;'>
                    <div style='font-size:10px;font-weight:800;letter-spacing:0.1em;
                                color:#9a9aaa;text-transform:uppercase;margin-bottom:4px;'>
                        Example Query
                    </div>
                    <div style='background:#f5f0fb;border-radius:6px;padding:8px 12px;
                                font-size:13px;color:#42145f;font-style:italic;'>
                        "{query_ex}"
                    </div>
                </div>
                <div style='flex:2;min-width:250px;'>
                    <div style='font-size:10px;font-weight:800;letter-spacing:0.1em;
                                color:#9a9aaa;text-transform:uppercase;margin-bottom:4px;'>
                        Example Output
                    </div>
                    <div style='background:#f0fdf4;border-radius:6px;padding:8px 12px;
                                font-size:13px;color:#166534;'>
                        {output_ex}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Security note ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class='pi-card' style='border-left:4px solid #42145f;margin-top:8px;'>
        <div style='font-size:14px;color:#444;line-height:1.65;'>
            <strong>Security &amp; Data Privacy.</strong>
            No raw data is ever exposed by design. The query engine returns only aggregated
            results. Uploaded datasets remain local to this workspace. API keys are stored
            in <code>.env</code> and never committed to version control.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Bottom CTA ────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    col_b, _ = st.columns([2, 5])
    with col_b:
        if st.button("Go to Analyse", key="hiw_cta_bottom", type="primary",
                     use_container_width=True):
            st.session_state["page"] = "Analyse"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════

def page_data_explorer():
    """Browse all registered datasets with full previews and schema info."""

    st.markdown("""
    <div class='pi-card-purple'>
        <div style="font-size:20px;font-weight:800;color:#42145f;margin-bottom:4px;">
            Available Data
        </div>
        <div style="font-size:13px;color:#6b3fa0;line-height:1.6;">
            All datasets currently loaded into PurpleInsight. Built-in datasets are synthetic
            NatWest banking data. Custom datasets you register appear here automatically.
            Click on a dataset tab to explore its schema and preview.
        </div>
    </div>
    """, unsafe_allow_html=True)

    datasets = list_registered_datasets()

    if not datasets:
        st.warning("No datasets registered yet. Go to Dataset Registry to add one.")
        return

    tab_labels = [d["display_name"] for d in datasets]
    tabs = st.tabs(tab_labels)

    for tab, ds in zip(tabs, datasets):
        with tab:
            filepath = ds["file_path"]

            # ── Dataset header ─────────────────────────────────────────────
            uc_badges = "".join(
                f"<span style='background:#e8dff5;color:#42145f;border-radius:12px;"
                f"padding:2px 10px;font-size:11px;font-weight:600;margin-right:4px;'>"
                f"{uc.replace('_',' ').title()}</span>"
                for uc in ds.get("primary_use_cases", [])
            )
            status_color = "#166534" if ds["exists"] else "#991b1b"
            status_text  = "File available" if ds["exists"] else "File missing — re-upload via Dataset Registry"
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;justify-content:space-between;
                        flex-wrap:wrap;gap:10px;margin-bottom:14px;">
                <div>
                    <div style="font-size:16px;font-weight:700;color:#42145f;">
                        {ds['display_name']}
                    </div>
                    <div style="font-size:13px;color:#666;margin:4px 0 8px;">
                        {ds.get('description','')}
                    </div>
                    <div>{uc_badges}</div>
                </div>
                <div style="font-size:12px;color:{status_color};font-weight:600;white-space:nowrap;">
                    {status_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not os.path.exists(filepath):
                st.info("Upload the CSV file in Dataset Registry to make this dataset queryable.")
                continue

            df = pd.read_csv(filepath)

            # ── Stats row ──────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            for col, (val, lbl) in zip([c1, c2, c3, c4], [
                (f"{len(df):,}", "Total rows"),
                (str(len(df.columns)), "Columns"),
                (str(df.isnull().sum().sum()), "Null values"),
                (ds.get("category", "—"), "Category"),
            ]):
                with col:
                    st.markdown(f"""
                    <div class='pi-stat'>
                        <div class='pi-stat-value' style='font-size:18px;'>{val}</div>
                        <div class='pi-stat-label'>{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

            # ── Data preview ───────────────────────────────────────────────
            st.markdown("<div class='pi-section-label'>Data Preview (first 20 rows)</div>",
                        unsafe_allow_html=True)
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)

            # ── Schema + stats in expanders ────────────────────────────────
            col_schema, col_stats = st.columns(2)
            with col_schema:
                with st.expander("Column schema", expanded=False):
                    schema_rows = [
                        {"Column": c, "Type": str(df[c].dtype),
                         "Unique values": df[c].nunique(),
                         "Sample": str(df[c].iloc[0]) if len(df) > 0 else ""}
                        for c in df.columns
                    ]
                    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

            with col_stats:
                num_cols = df.select_dtypes(include=["float64", "int64"]).columns
                if len(num_cols) > 0:
                    with st.expander("Numeric statistics", expanded=False):
                        st.dataframe(df[num_cols].describe().round(2),
                                     use_container_width=True)

            # ── Query shortcut ─────────────────────────────────────────────
            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            if st.button(f"Query {ds['display_name']} in Analyse",
                         key=f"explore_query_{ds['dataset_id']}", type="primary"):
                st.session_state["selected_dataset"] = ds["dataset_id"]
                st.session_state["page"] = "Analyse"
                st.rerun()


def page_dataset_registry():
    """Upload and register datasets in one central place."""
    st.markdown("""
    <div class='pi-card-purple'>
        <div style="font-size:20px; font-weight:800; color:#42145f;">
            Dataset Registry
        </div>
    </div>
    """, unsafe_allow_html=True)

    datasets = list_registered_datasets()
    builtin_ids = {"regional_revenue", "customer_metrics", "product_performance", "cost_breakdown", "weekly_kpis"}

    # ── Built-in datasets ──────────────────────────────────────────────────────
    st.markdown("<div class='pi-card'>", unsafe_allow_html=True)
    st.markdown("<div class='pi-panel-title'>Built-in Demo Datasets</div>", unsafe_allow_html=True)
    st.markdown("<div class='pi-panel-subtitle'>Synthetic NatWest banking datasets — ready to query.</div>", unsafe_allow_html=True)

    builtin_datasets = [d for d in datasets if d["dataset_id"] in builtin_ids]
    for item in builtin_datasets:
        col_info, col_btn = st.columns([5, 1])
        with col_info:
            uc_list = ", ".join(item["primary_use_cases"])
            status_color = "#166534" if item["exists"] else "#991b1b"
            status_text  = "Available" if item["exists"] else "File missing"
            st.markdown(f"""
            <div style="padding:8px 0; border-bottom:1px solid #ebebed;">
                <span style="font-size:13px; font-weight:700; color:#42145f;">{item['display_name']}</span>
                <span style="font-size:11px; color:#9a9aaa; margin-left:8px;">{item['description']}</span>
                <br>
                <span style="font-size:10px; color:#6b3fa0; text-transform:uppercase; letter-spacing:0.08em;">
                    {uc_list}
                </span>
                <span style="font-size:10px; color:{status_color}; margin-left:10px; font-weight:600;">
                    {status_text}
                </span>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            if st.button("Query", key=f"reg_query_{item['dataset_id']}", use_container_width=True):
                first_uc = item.get("primary_use_cases", ["summarize"])[0]
                prefill = {
                    "change_analysis": f"Why did a metric change in {item['display_name']}?",
                    "compare":         f"Compare categories in {item['display_name']}",
                    "breakdown":       f"Show breakdown of {item['display_name']} data",
                    "summarize":       f"Give me a summary of {item['display_name']}",
                }.get(first_uc, f"Show insights from {item['display_name']}")
                st.session_state["auto_query"] = prefill
                st.session_state["page"] = "Analyse"
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Custom datasets ────────────────────────────────────────────────────────
    custom_datasets = [d for d in datasets if d["dataset_id"] not in builtin_ids]
    if custom_datasets:
        st.markdown("<div class='pi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='pi-panel-title'>Your Registered Datasets</div>", unsafe_allow_html=True)
        st.markdown("<div class='pi-panel-subtitle'>Datasets you have added — queryable with all 4 use case features.</div>", unsafe_allow_html=True)

        for item in custom_datasets:
            col_info, col_exp, col_qry = st.columns([5, 0.8, 0.8])
            with col_info:
                uc_list = ", ".join(item["primary_use_cases"])
                status_color = "#166534" if item["exists"] else "#991b1b"
                status_text  = "Available" if item["exists"] else "File missing"
                st.markdown(f"""
                <div style="padding:8px 0; border-bottom:1px solid #ebebed;">
                    <span style="font-size:13px; font-weight:700; color:#42145f;">{item['display_name']}</span>
                    <span style="font-size:11px; color:#9a9aaa; margin-left:8px;">{item['description']}</span>
                    <br>
                    <span style="font-size:10px; color:#6b3fa0; text-transform:uppercase; letter-spacing:0.08em;">
                        {uc_list}
                    </span>
                    <span style="font-size:10px; color:{status_color}; margin-left:10px; font-weight:600;">
                        {status_text}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            with col_exp:
                if st.button("Explore", key=f"reg_explore_{item['dataset_id']}", use_container_width=True):
                    st.session_state["page"] = "Available Data"
                    st.rerun()
            with col_qry:
                if st.button("Query", key=f"reg_cquery_{item['dataset_id']}", use_container_width=True):
                    first_uc = item.get("primary_use_cases", ["summarize"])[0]
                    prefill = {
                        "change_analysis": f"What changed in {item['display_name']}?",
                        "compare":         f"Compare categories in {item['display_name']}",
                        "breakdown":       f"Show breakdown of {item['display_name']} data",
                        "summarize":       f"Give me a summary of {item['display_name']}",
                    }.get(first_uc, f"Show insights from {item['display_name']}")
                    st.session_state["auto_query"] = prefill
                    st.session_state["page"] = "Analyse"
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Add new dataset ────────────────────────────────────────────────────────
    st.markdown("<div class='pi-card'>", unsafe_allow_html=True)
    st.markdown("<div class='pi-panel-title'>Add Dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='pi-panel-subtitle'>Upload a CSV, register it centrally, and make it available to the explorer and AI query pipeline.</div>", unsafe_allow_html=True)

    with st.form("dataset_registry_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("CSV file", type=["csv"])
        display_name = st.text_input("Display name", placeholder="Example: Branch Service Quality")
        description = st.text_area("Description", placeholder="What the dataset contains and how it should be used.")
        category = st.selectbox("Category", ["Revenue", "Customer", "Product", "Cost", "KPI", "Operations", "General"])
        use_cases = st.multiselect(
            "Primary use cases",
            ["change_analysis", "compare", "breakdown", "summarize"],
            default=["summarize"],
        )
        submitted = st.form_submit_button("Save dataset", type="primary")

    if uploaded_file is not None:
        preview_df = pd.read_csv(uploaded_file, nrows=5)
        st.markdown("<div class='pi-section-label'>Preview</div>", unsafe_allow_html=True)
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

    if submitted:
        if not uploaded_file or not display_name.strip() or not description.strip() or not use_cases:
            st.error("Please provide the CSV file, display name, description, and at least one use case.")
        else:
            dataset_id = slugify_dataset_id(display_name)
            raw_dir = os.path.join(ROOT, "data", "raw")
            os.makedirs(raw_dir, exist_ok=True)
            safe_file_name = f"{dataset_id}.csv"
            file_path = os.path.join(raw_dir, safe_file_name)
            with open(file_path, "wb") as handle:
                handle.write(uploaded_file.getbuffer())
            register_dataset(
                dataset_id=dataset_id,
                display_name=display_name.strip(),
                description=description.strip(),
                category=category,
                file_name=safe_file_name,
                primary_use_cases=use_cases,
            )
            # Register the new view in the live pipeline (if loaded) so it's
            # immediately queryable without a full page reload.
            pipeline = load_pipeline()
            if pipeline and "engine" in pipeline:
                pipeline["engine"].register_new_dataset(dataset_id, file_path)

            st.session_state["newly_registered"] = {
                "id": dataset_id,
                "display_name": display_name.strip(),
                "use_cases": use_cases,
            }
            st.success(
                f"Dataset '{display_name.strip()}' registered and loaded. "
                f"Go to the Analyse tab to start querying it."
            )
            col_a, col_b, _ = st.columns([1, 1, 3])
            with col_a:
                if st.button("Go to Analyse", type="primary"):
                    st.session_state["page"] = "Analyse"
                    st.rerun()
            with col_b:
                if st.button("Explore dataset"):
                    st.session_state["page"] = "Available Data"
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main app entry point."""
    init_session()
    pipeline = load_pipeline()

    render_sidebar(pipeline)
    render_navbar()

    page = st.session_state.get("page", "How It Works")

    if page == "How It Works":
        page_how_it_works()
    elif page == "Available Data":
        page_data_explorer()
    elif page == "Dataset Registry":
        page_dataset_registry()
    elif page == "Analyse":
        page_analyse(pipeline)


if __name__ == "__main__":
    main()
