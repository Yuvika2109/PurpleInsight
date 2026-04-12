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

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from config.dataset_registry import list_registered_datasets, register_dataset, slugify_dataset_id

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

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
        "chat_history":    [],
        "active_demo":     None,
        "prefill_query":   "",
        "auto_submit":     False,
        "page":            "Analyse",
        "feedback_counts": {"positive": 0, "negative": 0},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(query: str, pipeline: dict) -> dict:
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
            "intent_method":      intent_result.method,
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
        <div class="pi-navbar-badge">Code for Purpose Hackathon 2025</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(pipeline) -> str | None:
    """
    Render the full sidebar with navigation, demo queries, and metrics glossary.

    Returns:
        str | None: Demo query string if a demo button was clicked
    """
    triggered = None
    llm_status = get_llm_runtime_status()
    pipeline_issue = get_pipeline_issue(pipeline)

    with st.sidebar:
        logo_data_uri = get_natwest_logo_data_uri()
        # Logo + brand
        st.markdown(f"""
        <div style="padding: 4px 0 20px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                <img src="{logo_data_uri}" alt="NatWest logo" style="width:34px; height:34px; object-fit:contain; border-radius:8px;" />
                <div style="font-size:20px; font-weight:800; letter-spacing:-0.5px;">PurpleInsight</div>
            </div>
            <div style="font-size:11px; opacity:0.55; margin-top:2px;">NatWest Banking Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown("<div class='pi-sidebar-label'>NAVIGATION</div>", unsafe_allow_html=True)
        pages = ["Analyse", "How It Works", "Data Explorer", "Dataset Registry"]
        for page in pages:
            is_active = st.session_state["page"] == page
            if st.button(
                page,
                key=f"nav_{page}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["page"] = page
                st.rerun()

        # Demo queries
        st.markdown("<div class='pi-sidebar-label'>TRY A DEMO QUERY</div>", unsafe_allow_html=True)

        demo_map = {
            "Revenue drop analysis": "Why did revenue drop in the South region last month?",
            "Region comparison": "Compare North vs South region revenue for 2024",
            "Cost breakdown": "Show the breakdown of costs by department",
            "Weekly summary": "Give me a weekly summary of customer metrics",
            "Product performance": "Compare Personal Current Account vs Credit Card performance this year",
            "Customer churn analysis": "What caused customer churn to rise in Q3?",
        }

        active = st.session_state.get("active_demo")
        for label, query in demo_map.items():
            is_active = active == label
            if st.button(
                label,
                key=f"demo_{label}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                triggered = query
                st.session_state["active_demo"] = label

        # Metrics glossary
        st.markdown("<div class='pi-sidebar-label'>METRIC GLOSSARY</div>", unsafe_allow_html=True)
        metrics = ["revenue", "churn_rate", "nps_score", "new_signups",
                   "avg_handle_time", "active_customers", "digital_adoption"]
        pills = " ".join(f"<span class='pi-pill'>{m}</span>" for m in metrics)
        st.markdown(pills, unsafe_allow_html=True)

        # Pipeline status
        st.markdown("<div class='pi-sidebar-label'>SYSTEM STATUS</div>", unsafe_allow_html=True)
        if pipeline:
            st.markdown("""
            <div style="font-size:12px; opacity:0.8;">
                AI pipeline ready<br>
                DuckDB connected<br>
                Registered datasets loaded<br>
                Groq Llama active
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div style="font-size:12px; opacity:0.8;">
                    Pipeline not connected<br>
                    <span style="opacity:0.6;">Provider: {llm_status['provider']} ({llm_status['model']})</span><br>
                    <span style="opacity:0.6;">{pipeline_issue}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Query count
        count = len(st.session_state.get("chat_history", []))
        if count > 0:
            st.markdown(f"""
            <div style="margin-top:16px; font-size:12px; opacity:0.6;">
                {count} quer{'y' if count == 1 else 'ies'} this session
            </div>
            """, unsafe_allow_html=True)

    return triggered


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

    st.markdown("<div class='pi-card pi-answer-card'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pi-result-query">
        <span>Query</span>
        <strong style="color:#42145f;">{entry['query']}</strong>
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

    # Feedback row
    st.markdown("<hr class='pi-divider'>", unsafe_allow_html=True)
    col_q, col_up, col_dn, col_resp, _ = st.columns([2.5, 0.4, 0.4, 2, 4])
    fb_key = f"fb_{query_id}"
    fb_val = st.session_state.get(fb_key)

    with col_q:
        st.markdown(
            "<p style='margin:6px 0; font-size:13px; color:#666;'>Was this helpful?</p>",
            unsafe_allow_html=True,
        )
    with col_up:
        if st.button("Helpful", key=f"up_{query_id}"):
            st.session_state[fb_key] = "positive"
            st.session_state["feedback_counts"]["positive"] += 1
            st.rerun()
    with col_dn:
        if st.button("Needs work", key=f"dn_{query_id}"):
            st.session_state[fb_key] = "negative"
            st.session_state["feedback_counts"]["negative"] += 1
            st.rerun()
    with col_resp:
        if fb_val == "positive":
            st.markdown(
                "<span style='background:#d1fae5; color:#065f46; border-radius:20px;"
                "padding:4px 14px; font-size:12px; font-weight:600;'>Marked as helpful</span>",
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

def page_analyse(pipeline):
    """Main chat/analysis page."""

    history = st.session_state.get("chat_history", [])
    fb      = st.session_state.get("feedback_counts", {})
    datasets = list_registered_datasets()
    total_fb = fb.get("positive", 0) + fb.get("negative", 0)
    helpful_pct = round(fb.get("positive", 0) / total_fb * 100) if total_fb > 0 else 0

    llm_status = get_llm_runtime_status()
    pipeline_status = "Ready" if pipeline else "Needs attention"
    pipeline_detail = (
        f"{llm_status['provider']} · {llm_status['model']}" if pipeline
        else get_pipeline_issue(pipeline)
    )

    st.markdown(f"""
    <div class="pi-hero">
        <div class="pi-eyebrow">Self-service banking intelligence</div>
        <div class="pi-hero-title">Ask a business question. Get a clear answer, the chart, and the trust trail in one place.</div>
        <div class="pi-hero-subtitle">
            PurpleInsight turns natural-language questions into trusted analytics across registered business datasets, with clear answers, transparent sources, and a governed trust trail.
        </div>
        <div class="pi-hero-meta">
            <div class="pi-hero-pill">Pipeline: {pipeline_status}</div>
            <div class="pi-hero-pill">Provider: {llm_status['provider']}</div>
            <div class="pi-hero-pill">{pipeline_detail}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='pi-stat'>
            <div class='pi-stat-value'>{len(history)}</div>
            <div class='pi-stat-label'>Queries this session</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='pi-stat'>
            <div class='pi-stat-value'>{len(datasets)}</div>
            <div class='pi-stat-label'>Datasets loaded</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='pi-stat'>
            <div class='pi-stat-value'>
                {helpful_pct}%
            </div>
            <div class='pi-stat-label'>Helpfulness score</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='pi-stat'>
            <div class='pi-stat-value'>{"Ready" if pipeline else "Review"}</div>
            <div class='pi-stat-label'>Pipeline status</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)

    st.markdown("<div class='pi-card'>", unsafe_allow_html=True)
    query = render_query_input()
    st.markdown("</div>", unsafe_allow_html=True)

    # Auto-submit from demo buttons or follow-ups
    auto = st.session_state.pop("auto_query", None)
    final_query = auto or query

    if final_query and final_query.strip():
        if not pipeline:
            st.error(
                "Pipeline not available. "
                f"{get_pipeline_issue(pipeline)} "
                "Also make sure the required dependencies are installed."
            )
        else:
            with st.spinner("Analysing your question..."):
                result = run_pipeline(final_query.strip(), pipeline)

            entry = {
                "query":    final_query.strip(),
                "result":   result,
                "query_id": str(uuid.uuid4())[:8],
            }
            st.session_state["chat_history"].append(entry)

    history = st.session_state.get("chat_history", [])

    if not history:
        st.markdown("""
        <div class='pi-card pi-empty'>
            <div class='pi-empty-icon'>Data</div>
            <div class='pi-empty-title'>Your analysis workspace is ready</div>
            <div class='pi-empty-sub'>
                Start with a query above or use one of the guided examples to see the full flow.<br>
                <span style="color:#b39bd1; margin-top:6px; display:block;">
                    Revenue drivers · Region comparisons · Department cost breakdowns · Weekly KPI summaries
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    latest = history[-1]
    render_result(latest)

    if len(history) > 1:
        with st.expander(f"Recent Query History ({len(history)-1})", expanded=False):
            st.markdown("<div class='pi-card pi-history-card'>", unsafe_allow_html=True)
            for entry in reversed(history[:-1]):
                st.markdown(f"""
                <div class="pi-history-item">
                    <div class="pi-history-title">{entry['query']}</div>
                    <div class="pi-history-meta">Use case: {getattr(entry['result'].get('use_case'), 'value', 'analysis')} · Runtime: {entry['result'].get('query_ms', 0)}ms</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(
                    "Re-run this query",
                    key=f"rerun_{entry['query_id']}",
                    type="secondary",
                ):
                    st.session_state["auto_query"] = entry["query"]
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — How It Works
# ══════════════════════════════════════════════════════════════════════════════

def page_how_it_works():
    """Explains the system architecture for non-technical users and judges."""

    st.markdown("""
    <div class='pi-card-purple'>
        <div style="font-size:22px; font-weight:800; color:#42145f; margin-bottom:6px;">
            How PurpleInsight Works
        </div>
        <div style="font-size:14px; color:#6b3fa0; line-height:1.6;">
            PurpleInsight removes friction between non-technical users and their data.
            Ask a question in plain English — get a trusted, verified answer in seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3 pillars
    st.markdown("### The Three Pillars")
    c1, c2, c3 = st.columns(3)
    pillars = [
        ("1", "Clarity", "Answers in plain English. No SQL. No jargon. No data science degree required."),
        ("2", "Trust", "Every answer shows the SQL that ran, which metrics were used, and the data source. Nothing is hidden."),
        ("3", "Speed", "DuckDB executes queries in milliseconds. Groq-powered Llama generates answers in under 2 seconds."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3], pillars):
        with col:
            st.markdown(f"""
            <div class='hiw-step'>
                <div class='hiw-step-number'>{num}</div>
                <div class='hiw-step-title'>{title}</div>
                <div class='hiw-step-body'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline steps
    st.markdown("### The AI Pipeline — Step by Step")

    steps = [
        ("1", "Ambiguity Handler",
         "Detects vague time references like 'last month' or 'this week' and resolves them to exact dates before anything else happens. You never get a wrong date range.",
         "Pure Python · python-dateutil · zero API cost"),

        ("2", "Intent Router",
         "Classifies your question into one of 4 use cases: Change Analysis, Comparison, Breakdown, or Summary. Uses local sentence-transformer embeddings — fast and free, no API call needed.",
         "sentence-transformers · all-MiniLM-L6-v2 · runs locally"),

        ("3", "Metric Dictionary",
         "Every query is resolved through a YAML metric dictionary before hitting the LLM. This ensures 'revenue' always means the same thing across all queries, teams, and time periods — a NatWest requirement.",
         "metrics.yaml · 20+ metric definitions · canonical NatWest terms"),

        ("4", "NL to SQL (Groq)",
         "Groq-hosted Llama converts your natural language question into a valid DuckDB SQL query, using the metric definitions and table schemas injected into the prompt. Only SELECT queries are allowed.",
         "Llama 3.1 70B · DuckDB SQL · schema-aware prompting"),

        ("5", "Query Engine",
         "DuckDB executes the SQL directly against the CSV datasets in memory. No database server. No cloud. Raw rows are never returned — only aggregates. Results are validated before leaving this layer.",
         "DuckDB in-memory · MAX 50 rows · aggregates only"),

        ("6", "Narrative (Groq)",
         "Groq-hosted Llama converts the query results into a clear, plain-English answer using use-case-specific templates. The output matches the exact format from the NatWest problem document.",
         "Llama 3.1 70B · 4 narrative templates · anomaly detection"),

        ("7", "Trust Trail",
         "Every answer is wrapped in a full trust trail showing: the SQL executed, metric definitions applied, dataset used, confidence level, and whether any raw data was exposed (never).",
         "Built in-app · no external dependency · judge-visible"),
    ]

    for num, title, desc, tech in steps:
        with st.expander(f"Step {num} — {title}", expanded=(num == "1")):
            col_desc, col_tech = st.columns([3, 1])
            with col_desc:
                st.markdown(f"<p style='font-size:14px; color:#444; line-height:1.6;'>{desc}</p>", unsafe_allow_html=True)
            with col_tech:
                st.markdown(f"""
                <div style="background:#f5f0fb; border-radius:8px; padding:10px 12px;
                            font-size:11px; color:#6b3fa0; font-family:monospace; line-height:1.6;">
                    {tech}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 use cases
    st.markdown("### The 4 Use Cases")
    use_cases = [
        ("1", "Change Analysis", "Why did revenue drop last month?",
         "Revenue decreased by 20.5% in Feb 2024. The biggest contributor was a 27.6% drop in the South region coinciding with a churn spike from 5.3% to 9.1%."),
        ("2", "Comparison", "Compare North vs South region revenue",
         "North outperforms South by £3.1M (+26.6%) in H1 2024. South was severely impacted in Feb–Apr where revenue fell 27.6%."),
        ("3", "Breakdown", "Show the breakdown of costs by department",
         "Technology accounts for 32% of total costs at £25.9M, followed by Operations at 24%. Combined they represent over half of total spend."),
        ("4", "Summary", "Give me a weekly summary of customer metrics",
         "This week: Signups grew by 5%, churn remained stable at 1.8%, NPS improved to 44, and average handle time decreased by 12 seconds."),
    ]

    for order, title, query_ex, output_ex in use_cases:
        st.markdown(f"""
        <div class='pi-card' style='margin-bottom:10px;'>
            <div style='font-size:15px; font-weight:700; color:#42145f; margin-bottom:8px;'>
                Use Case {order} · {title}
            </div>
            <div style='display:flex; gap:16px; flex-wrap:wrap;'>
                <div style='flex:1; min-width:200px;'>
                    <div style='font-size:10px; font-weight:800; letter-spacing:0.1em;
                                color:#9a9aaa; text-transform:uppercase; margin-bottom:4px;'>
                        Example Query
                    </div>
                    <div style='background:#f5f0fb; border-radius:6px; padding:8px 12px;
                                font-size:13px; color:#42145f; font-style:italic;'>
                        "{query_ex}"
                    </div>
                </div>
                <div style='flex:2; min-width:250px;'>
                    <div style='font-size:10px; font-weight:800; letter-spacing:0.1em;
                                color:#9a9aaa; text-transform:uppercase; margin-bottom:4px;'>
                        Example Output
                    </div>
                    <div style='background:#f0fdf4; border-radius:6px; padding:8px 12px;
                                font-size:13px; color:#166534;'>
                        {output_ex}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Security note
    st.markdown("### Security & Data Privacy")
    st.markdown("""
    <div class='pi-card' style='border-left:4px solid #42145f;'>
        <div style='font-size:14px; color:#444; line-height:1.65;'>
            <strong>No raw data is ever exposed by design.</strong>
            The query engine is intended to return aggregated results for decision support, while the trust layer surfaces data sources and executed SQL.
            The bundled demo datasets are synthetic and suitable for hackathon presentation.
            Any uploaded datasets remain local to this workspace.
            API keys are stored in <code>.env</code> and never committed to GitHub.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════

def page_data_explorer():
    """Browse registered datasets directly."""

    st.markdown("""
    <div class='pi-card-purple'>
        <div style="font-size:20px; font-weight:800; color:#42145f; margin-bottom:4px;">
            Data Explorer
        </div>
        <div style="font-size:13px; color:#6b3fa0;">
            Browse the registered datasets powering PurpleInsight.
            The bundled demo datasets are generated by <code>scripts/generate_synthetic_data.py</code>.
            Newly added datasets are registered locally and remain inside this workspace.
        </div>
    </div>
    """, unsafe_allow_html=True)

    datasets = list_registered_datasets()

    if not datasets:
        st.warning("No datasets are registered yet. Add one in the Dataset Registry page first.")
        return

    selected = st.selectbox(
        "Select a dataset to explore",
        options=[item["dataset_id"] for item in datasets],
        format_func=lambda dataset_id: next(
            item["display_name"] for item in datasets if item["dataset_id"] == dataset_id
        ),
    )

    selected_dataset = next(item for item in datasets if item["dataset_id"] == selected)
    title = selected_dataset["display_name"]
    desc = selected_dataset["description"]
    use_cases = ", ".join(selected_dataset["primary_use_cases"])
    filepath = selected_dataset["file_path"]

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)

        # Dataset header
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class='pi-stat'>
                <div class='pi-stat-value'>{len(df):,}</div>
                <div class='pi-stat-label'>Total rows</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class='pi-stat'>
                <div class='pi-stat-value'>{len(df.columns)}</div>
                <div class='pi-stat-label'>Columns</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
            <div class='pi-stat'>
                <div class='pi-stat-value'>{df.isnull().sum().sum()}</div>
                <div class='pi-stat-label'>Null values</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='font-size:13px; color:#666; margin:12px 0 16px;'>
            {desc} · Used for: <strong style='color:#42145f;'>{use_cases}</strong>
        </div>
        """, unsafe_allow_html=True)

        # Preview
        st.markdown("<div class='pi-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='pi-table-header'>Preview — {title}</div>", unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Column schema
        with st.expander("Column schema", expanded=False):
            schema_data = []
            for col in df.columns:
                schema_data.append({
                    "Column":   col,
                    "Type":     str(df[col].dtype),
                    "Non-null": df[col].count(),
                    "Unique":   df[col].nunique(),
                    "Sample":   str(df[col].iloc[0]) if len(df) > 0 else "",
                })
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True, hide_index=True)

        # Numeric stats
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        if len(numeric_cols) > 0:
            with st.expander("Numeric statistics", expanded=False):
                st.dataframe(
                    df[numeric_cols].describe().round(2),
                    use_container_width=True,
                )
    else:
        st.error(f"File not found: {filepath}")


def page_dataset_registry():
    """Upload and register datasets in one central place."""
    st.markdown("""
    <div class='pi-card-purple'>
        <div style="font-size:20px; font-weight:800; color:#42145f; margin-bottom:4px;">
            Dataset Registry
        </div>
        <div style="font-size:13px; color:#6b3fa0;">
            Add new datasets in one dedicated place. Files are stored in <code>data/raw</code> and metadata is written to <code>config/datasets.yaml</code>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    datasets = list_registered_datasets()

    st.markdown("<div class='pi-card'>", unsafe_allow_html=True)
    st.markdown("<div class='pi-panel-title'>Registered Datasets</div>", unsafe_allow_html=True)
    registry_rows = [
        {
            "Dataset ID": item["dataset_id"],
            "Display Name": item["display_name"],
            "Category": item["category"],
            "File": item["file"],
            "Available": "Yes" if item["exists"] else "No",
            "Use Cases": ", ".join(item["primary_use_cases"]),
        }
        for item in datasets
    ]
    st.dataframe(pd.DataFrame(registry_rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
            st.success(f"Dataset '{display_name.strip()}' has been added to the registry.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main app entry point."""
    init_session()
    pipeline = load_pipeline()

    # Handle sidebar navigation + demo triggers
    demo_triggered = render_sidebar(pipeline)
    if demo_triggered:
        st.session_state["auto_query"] = demo_triggered
        st.session_state["page"] = "Analyse"
        st.rerun()

    # Navbar
    render_navbar()

    # Page routing
    page = st.session_state.get("page", "Analyse")

    if page == "Analyse":
        page_analyse(pipeline)
    elif page == "How It Works":
        page_how_it_works()
    elif page == "Data Explorer":
        page_data_explorer()
    elif page == "Dataset Registry":
        page_dataset_registry()


if __name__ == "__main__":
    main()
