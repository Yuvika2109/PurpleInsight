"""
chart_renderer.py
-----------------
Auto-selects and renders the right Plotly chart for each use case.

Use case → chart type:
    change_analysis → line chart with annotated drop zones
    compare         → grouped bar chart
    breakdown       → horizontal bar with % of total
    summarize       → multi-metric line chart
    unknown         → bar chart (default)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ── NatWest brand colours ─────────────────────────────────────────────────────
PURPLE_DEEP  = "#42145f"
PURPLE_MID   = "#6b3fa0"
PURPLE_LIGHT = "#c4b0e0"
AMBER        = "#f59e0b"
GREEN        = "#10b981"
RED          = "#ef4444"
BLUE         = "#3b82f6"
GREY         = "#9ca3af"

PLOTLY_LAYOUT = dict(
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    font          = dict(family="DM Sans, sans-serif", size=12, color="#3d3d4a"),
    margin        = dict(l=16, r=16, t=40, b=16),
    legend        = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis         = dict(showgrid=False, linecolor="#ebebed"),
    yaxis         = dict(gridcolor="#f5f5f7", linecolor="#ebebed"),
)


def render_chart(chart: dict):
    """
    Auto-select and render the appropriate chart for the given query result.

    Args:
        chart: Dict containing query_type, data, columns from the pipeline
    """
    if not chart or not chart.get("data"):
        return

    query_type = chart.get("query_type", "unknown")
    data       = chart.get("data", [])
    columns    = chart.get("columns", [])

    if not data or not columns:
        return

    df = pd.DataFrame(data)

    try:
        if query_type == "change_analysis":
            _render_change_chart(df)
        elif query_type == "compare":
            _render_compare_chart(df)
        elif query_type == "breakdown":
            _render_breakdown_chart(df)
        elif query_type == "summarize":
            _render_summary_chart(df)
        else:
            _render_default_chart(df)
    except Exception as e:
        # Graceful fallback — never crash the UI because of chart error
        _render_default_chart(df)


def _render_change_chart(df: pd.DataFrame):
    """Line chart for change analysis — highlights drops/spikes."""
    if df.empty:
        return

    # Find date column and value column
    date_col  = _find_col(df, ["month", "week", "date", "period"])
    value_col = _find_numeric_col(df, ["revenue", "amount", "value", "count", "total"])

    if not date_col or not value_col:
        _render_default_chart(df)
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x    = df[date_col],
        y    = df[value_col],
        mode = "lines+markers",
        name = value_col.replace("_", " ").title(),
        line = dict(color=PURPLE_DEEP, width=2.5),
        marker = dict(size=6, color=PURPLE_DEEP),
        fill = "tozeroy",
        fillcolor = "rgba(66,20,95,0.06)",
    ))

    layout = dict(**PLOTLY_LAYOUT)
    layout["title"] = dict(
        text    = f"{value_col.replace('_', ' ').title()} over time",
        font    = dict(size=13, color="#3d3d4a", family="DM Sans"),
        x       = 0,
        xanchor = "left",
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_compare_chart(df: pd.DataFrame):
    """Grouped bar chart for comparisons."""
    if df.empty:
        return

    # Try to find category and value columns
    cat_col   = _find_col(df, ["region", "product", "segment", "channel", "department", "category"])
    value_cols = [c for c in df.select_dtypes(include=["float64", "int64"]).columns]

    if not cat_col or not value_cols:
        _render_default_chart(df)
        return

    colors = [PURPLE_DEEP, AMBER, GREEN, BLUE, RED, GREY]
    fig    = go.Figure()

    for i, val_col in enumerate(value_cols[:4]):
        fig.add_trace(go.Bar(
            name = val_col.replace("_", " ").title(),
            x    = df[cat_col],
            y    = df[val_col],
            marker_color = colors[i % len(colors)],
        ))

    layout = dict(**PLOTLY_LAYOUT)
    layout["barmode"] = "group"
    layout["title"]   = dict(
        text    = "Comparison",
        font    = dict(size=13, color="#3d3d4a", family="DM Sans"),
        x       = 0,
        xanchor = "left",
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_breakdown_chart(df: pd.DataFrame):
    """Horizontal bar chart showing % contribution for breakdowns."""
    if df.empty:
        return

    cat_col   = _find_col(df, ["department", "region", "product", "segment", "channel", "category", "cost_category"])
    value_col = _find_numeric_col(df, ["total", "cost", "revenue", "amount", "value", "sum"])

    if not cat_col or not value_col:
        _render_default_chart(df)
        return

    df_sorted = df.sort_values(value_col, ascending=True).tail(10)

    # Calculate percentage
    total = df_sorted[value_col].sum()
    df_sorted = df_sorted.copy()
    df_sorted["pct"] = (df_sorted[value_col] / total * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y           = df_sorted[cat_col],
        x           = df_sorted[value_col],
        orientation = "h",
        marker      = dict(
            color     = [PURPLE_DEEP] * (len(df_sorted) - 1) + [AMBER],
            line      = dict(color="white", width=1),
        ),
        text        = [f"{p}%" for p in df_sorted["pct"]],
        textposition = "outside",
        textfont    = dict(size=11, color="#3d3d4a"),
    ))

    layout = dict(**PLOTLY_LAYOUT)
    layout["title"] = dict(
        text    = f"{value_col.replace('_',' ').title()} breakdown",
        font    = dict(size=13, color="#3d3d4a", family="DM Sans"),
        x       = 0,
        xanchor = "left",
    )
    layout["xaxis"] = dict(showgrid=True, gridcolor="#f5f5f7")
    layout["yaxis"] = dict(showgrid=False)
    layout["height"] = max(200, len(df_sorted) * 36 + 60)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_summary_chart(df: pd.DataFrame):
    """Multi-line chart for KPI summaries."""
    if df.empty:
        return

    date_col     = _find_col(df, ["week", "month", "date", "period"])
    numeric_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)

    if not date_col or not numeric_cols:
        _render_default_chart(df)
        return

    colors = [PURPLE_DEEP, AMBER, GREEN, BLUE, RED]
    fig    = go.Figure()

    for i, col in enumerate(numeric_cols[:3]):
        fig.add_trace(go.Scatter(
            x    = df[date_col],
            y    = df[col],
            mode = "lines+markers",
            name = col.replace("_", " ").title(),
            line = dict(color=colors[i % len(colors)], width=2),
            marker = dict(size=5),
        ))

    layout = dict(**PLOTLY_LAYOUT)
    layout["title"] = dict(
        text    = "KPI Summary",
        font    = dict(size=13, color="#3d3d4a", family="DM Sans"),
        x       = 0,
        xanchor = "left",
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_default_chart(df: pd.DataFrame):
    """Default bar chart fallback for any query type."""
    if df.empty:
        return

    cat_col   = None
    value_col = None

    # Find first text column as category
    for col in df.columns:
        if df[col].dtype == "object":
            cat_col = col
            break

    # Find first numeric column as value
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        value_col = col
        break

    if not cat_col or not value_col:
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        return

    fig = px.bar(
        df.head(15),
        x     = cat_col,
        y     = value_col,
        color_discrete_sequence = [PURPLE_DEEP],
    )
    layout = dict(**PLOTLY_LAYOUT)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column from a list of candidates."""
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    # Fallback: first text column
    for col in df.columns:
        if df[col].dtype == "object":
            return col
    return None


def _find_numeric_col(df: pd.DataFrame, preferences: list[str]) -> str | None:
    """Find the best numeric column from a list of preferences."""
    cols_lower = {c.lower(): c for c in df.columns}
    for pref in preferences:
        for col_lower, col in cols_lower.items():
            if pref.lower() in col_lower and df[col].dtype in ["float64", "int64"]:
                return col
    # Fallback: first numeric column
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        return col
    return None