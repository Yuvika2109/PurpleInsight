"""
query_input.py
--------------
Query input component for PurpleInsight.
Renders the main text input + submit button + suggested query chips.
"""

import streamlit as st

DEMO_QUERIES = [
    "Why did revenue drop in the South region last month?",
    "Compare North vs South region revenue for 2024",
    "Show the breakdown of costs by department",
    "Give me a weekly summary of customer metrics",
    "Compare Personal Current Account vs Credit Card performance",
    "What caused customer churn to rise in Q3?",
]

SUGGESTED_CHIPS = [
    "Revenue drop analysis",
    "Compare regions",
    "Cost breakdown",
    "Weekly summary",
]


def render_query_input() -> str | None:
    """
    Render the main query input bar with submit button and suggested chips.

    Returns:
        str | None: Query string if submitted, else None
    """
    submitted_query = None

    st.markdown("""
    <div class="pi-query-shell">
        <div class="pi-query-header">
            <div>
                <div class="pi-eyebrow">Ask PurpleInsight</div>
                <div class="pi-query-title">Describe the business question in plain English</div>
                <div class="pi-query-subtitle">
                    We classify the intent, generate SQL, run the query in DuckDB, and return a narrated answer with a trust trail.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    chip_queries = {
        "Revenue drop": "Why did revenue drop in the South region last month?",
        "Compare regions": "Compare North vs South region revenue for 2024",
        "Cost breakdown": "Show the breakdown of costs by department",
        "Weekly summary": "Give me a weekly summary of customer metrics",
    }

    st.markdown("<div class='pi-chip-row'>", unsafe_allow_html=True)
    cols = st.columns(len(chip_queries))
    for col, (label, query) in zip(cols, chip_queries.items()):
        with col:
            if st.button(label, key=f"chip_{label}", use_container_width=True):
                st.session_state["auto_query"] = query
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    prefill = st.session_state.pop("prefill_query", "")
    with st.form("main_query_form", clear_on_submit=False):
        query = st.text_input(
            label="Ask your data",
            value=prefill,
            placeholder="Why did revenue drop last month? Compare North vs South. Give me a weekly customer summary.",
            label_visibility="collapsed",
            key="main_query_input",
        )
        col_hint, col_btn = st.columns([4.5, 1.2])
        with col_hint:
            st.markdown(
                "<div class='pi-query-hint'>Try metrics like revenue, churn, NPS, signups, costs, or compare products, regions, and time periods.</div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            submitted = st.form_submit_button(
                "Analyse",
                type="primary",
                use_container_width=True,
            )

    if submitted and query and query.strip():
        submitted_query = query.strip()

    return submitted_query
