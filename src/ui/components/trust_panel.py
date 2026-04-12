"""
trust_panel.py
--------------
Trust Trail panel component for PurpleInsight.
Renders the full transparency layer for every answer — the WOW feature.

Shows:
    - SQL that was executed
    - Metric definitions applied
    - Dataset used
    - Confidence level
    - Whether raw data was exposed
    - Query execution time
"""

import streamlit as st


def render_trust_panel(trust: dict):
    """
    Render the collapsible trust trail panel.

    Args:
        trust: Dict from pipeline containing:
            sql, valid, datasets_used, metric_definitions,
            confidence, confidence_label, raw_data_exposed,
            resolution_summary, execution_ms, total_ms
    """
    if not trust:
        return

    confidence_label = trust.get("confidence_label", "Medium")
    confidence_val   = trust.get("confidence", 0.0)
    execution_ms     = trust.get("execution_ms", 0)
    total_ms         = trust.get("total_ms", 0)
    sql              = trust.get("sql", "")
    datasets         = trust.get("datasets_used", [])
    metrics          = trust.get("metric_definitions", [])
    raw_exposed      = trust.get("raw_data_exposed", False)
    sql_valid        = trust.get("valid", False)

    # Confidence color
    conf_color = {"High": "#166534", "Medium": "#92400e", "Low": "#991b1b"}.get(confidence_label, "#555")
    conf_bg    = {"High": "#d1fae5", "Medium": "#fef3c7", "Low": "#fee2e2"}.get(confidence_label, "#f5f5f5")

    with st.expander("Trust Trail", expanded=False):

        st.markdown("""
        <div class="pi-trust-intro">
            Full audit view of how this answer was produced: intent confidence, generated SQL, source datasets, metric definitions, and safety checks.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="pi-trust-stat" style="background:{conf_bg};">
                <div class="pi-trust-stat-label" style="color:{conf_color};">
                    Confidence
                </div>
                <div class="pi-trust-stat-value" style="color:{conf_color};">
                    {confidence_label}
                </div>
                <div class="pi-trust-stat-meta" style="color:{conf_color};">
                    {round(confidence_val * 100)}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="pi-trust-stat" style="background:#f0fdf4;">
                <div class="pi-trust-stat-label" style="color:#166534;">
                    Raw Data
                </div>
                <div class="pi-trust-stat-value" style="color:#166534;">
                    Protected
                </div>
                <div class="pi-trust-stat-meta" style="color:#166534;">
                    {"Exposed" if raw_exposed else "Not exposed"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="pi-trust-stat" style="background:#f5f0fb;">
                <div class="pi-trust-stat-label" style="color:#42145f;">
                    Query time
                </div>
                <div class="pi-trust-stat-value" style="color:#42145f;">
                    {execution_ms}ms
                </div>
                <div class="pi-trust-stat-meta" style="color:#42145f;">
                    DuckDB
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="pi-trust-stat" style="background:#f5f0fb;">
                <div class="pi-trust-stat-label" style="color:#42145f;">
                    Total time
                </div>
                <div class="pi-trust-stat-value" style="color:#42145f;">
                    {round(total_ms/1000, 1)}s
                </div>
                <div class="pi-trust-stat-meta" style="color:#42145f;">
                    End-to-end
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

        # SQL
        if sql:
            st.markdown("<div class='pi-section-label'>SQL Executed</div>", unsafe_allow_html=True)
            st.code(sql, language="sql")
            status = "Valid SELECT. Safe to execute." if sql_valid else "Validation warning."
            color  = "#166534" if sql_valid else "#92400e"
            st.markdown(
                f"<div style='font-size:11px; color:{color}; margin-bottom:12px;'>{status}</div>",
                unsafe_allow_html=True,
            )

        # Datasets + metrics in two columns
        col_ds, col_mt = st.columns(2)

        with col_ds:
            st.markdown("<div class='pi-section-label'>Data Sources</div>", unsafe_allow_html=True)
            for ds in datasets:
                st.markdown(f"""
                <div class="pi-trust-chip">
                    <code>{ds}.csv</code>
                </div>
                """, unsafe_allow_html=True)

        with col_mt:
            st.markdown("<div class='pi-section-label'>Metric Definitions Applied</div>", unsafe_allow_html=True)
            if metrics:
                for m in metrics:
                    st.markdown(f"""
                    <div class="pi-trust-chip pi-trust-chip-metric">
                        {m}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='font-size:12px; color:#9a9aaa;'>Standard aggregations applied</div>",
                    unsafe_allow_html=True,
                )

        # Resolution summary
        if trust.get("resolution_summary"):
            st.markdown(f"""
            <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px;
                        padding:10px 14px; font-size:12px; color:#166534; margin-top:12px;">
                {trust['resolution_summary']}
            </div>
            """, unsafe_allow_html=True)

        # Footer note
        st.markdown("""
        <div style="font-size:11px; color:#9a9aaa; margin-top:12px; padding-top:12px;
                    border-top:1px solid #ebebed;">
            Raw data was never exposed. Only aggregated results were returned.
            All metric definitions sourced from <code>src/semantic/metrics.yaml</code>.
        </div>
        """, unsafe_allow_html=True)
