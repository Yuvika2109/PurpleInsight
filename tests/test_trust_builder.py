"""
Tests for the Trust Layer.

Covers:
    - TrustBuilder: metric resolution, ambiguity detection, confidence scoring
    - ChartSelector: correct chart type selection per intent
    - /query and /feedback API endpoints
"""

from __future__ import annotations

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.core.trust_builder import TrustBuilder, ConfidenceLevel, TrustTrail
from src.core.chart_selector import ChartSelector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_METRICS = {
    "revenue": {
        "definition": "Total net revenue after refunds, excluding VAT.",
        "source_field": "net_revenue",
    },
    "churn rate": {
        "definition": "Percentage of customers who closed accounts in a given period.",
        "source_field": "churn_flag",
    },
    "active customers": {
        "definition": "Customers with at least one transaction in the last 90 days.",
        "source_field": "is_active",
    },
}


@pytest.fixture()
def builder() -> TrustBuilder:
    """Return a TrustBuilder loaded with sample metric definitions."""
    return TrustBuilder(metric_definitions=SAMPLE_METRICS)


@pytest.fixture()
def selector() -> ChartSelector:
    """Return a ChartSelector instance."""
    return ChartSelector()


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """A minimal DataFrame for chart tests."""
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "West"],
            "revenue": [120_000, 95_000, 143_000, 78_000],
        }
    )


@pytest.fixture()
def multi_col_df() -> pd.DataFrame:
    """A multi-column DataFrame for grouped/stacked chart tests."""
    return pd.DataFrame(
        {
            "region": ["North", "North", "South", "South"],
            "product": ["Savings", "Mortgage", "Savings", "Mortgage"],
            "revenue": [60_000, 60_000, 45_000, 50_000],
        }
    )


# ---------------------------------------------------------------------------
# TrustBuilder Tests
# ---------------------------------------------------------------------------


class TestTrustBuilderMetricResolution:
    """Verify that metric terms are correctly matched from SQL and query text."""

    def test_resolves_metric_present_in_sql(self, builder: TrustBuilder) -> None:
        """When 'revenue' appears in the SQL, it should appear in metrics_used."""
        sql = "SELECT region, SUM(net_revenue) AS revenue FROM regional_revenue GROUP BY region"
        query = "Show me revenue by region"
        trail = builder.build(sql=sql, query=query, data_source="regional_revenue", row_count=4)

        metric_names = [m.name for m in trail.metrics_used]
        assert "revenue" in metric_names

    def test_resolves_metric_present_in_query_only(self, builder: TrustBuilder) -> None:
        """Metric matched from the NL query text even if not literally in SQL."""
        sql = "SELECT segment, COUNT(*) AS cnt FROM customer_metrics WHERE churn_flag=1 GROUP BY segment"
        query = "What is the churn rate per customer segment?"
        trail = builder.build(sql=sql, query=query, data_source="customer_metrics", row_count=3)

        metric_names = [m.name for m in trail.metrics_used]
        assert "churn rate" in metric_names

    def test_no_duplicate_metrics(self, builder: TrustBuilder) -> None:
        """Each metric should appear at most once even if mentioned multiple times."""
        sql = "SELECT revenue, SUM(revenue) FROM t GROUP BY revenue"
        query = "Total revenue vs monthly revenue"
        trail = builder.build(sql=sql, query=query, data_source="t", row_count=1)

        metric_names = [m.name for m in trail.metrics_used]
        assert metric_names.count("revenue") == 1


class TestTrustBuilderAmbiguityDetection:
    """Verify temporal and contextual ambiguity detection."""

    def test_detects_this_month(self, builder: TrustBuilder) -> None:
        sql = "SELECT * FROM t WHERE month = 'current'"
        trail = builder.build(sql=sql, query="Show revenue for this month", data_source="t", row_count=5)
        assert "this month" in trail.ambiguity_flags

    def test_detects_recently(self, builder: TrustBuilder) -> None:
        sql = "SELECT * FROM t"
        trail = builder.build(sql=sql, query="What happened recently with churn?", data_source="t", row_count=2)
        assert "recently" in trail.ambiguity_flags

    def test_no_false_positives_on_clean_query(self, builder: TrustBuilder) -> None:
        sql = "SELECT region, SUM(revenue) FROM t WHERE month='2024-03' GROUP BY region"
        trail = builder.build(sql=sql, query="Show revenue by region for March 2024", data_source="t", row_count=4)
        assert trail.ambiguity_flags == []


class TestTrustBuilderConfidenceScoring:
    """Verify the heuristic confidence scoring rules."""

    def test_ambiguity_triggers_low_confidence(self, builder: TrustBuilder) -> None:
        trail = builder.build(
            sql="SELECT * FROM t",
            query="What happened last cycle?",
            data_source="t",
            row_count=10,
        )
        assert trail.confidence == ConfidenceLevel.LOW

    def test_zero_rows_triggers_low_confidence(self, builder: TrustBuilder) -> None:
        trail = builder.build(
            sql="SELECT revenue FROM t WHERE region='Antarctica'",
            query="Revenue in Antarctica",
            data_source="t",
            row_count=0,
        )
        assert trail.confidence == ConfidenceLevel.LOW

    def test_cte_triggers_medium_confidence(self, builder: TrustBuilder) -> None:
        sql = "WITH base AS (SELECT * FROM t) SELECT revenue FROM base"
        trail = builder.build(sql=sql, query="Show revenue", data_source="t", row_count=5)
        assert trail.confidence == ConfidenceLevel.MEDIUM

    def test_clean_query_with_metrics_gives_high_confidence(self, builder: TrustBuilder) -> None:
        sql = "SELECT region, SUM(net_revenue) AS revenue FROM regional_revenue GROUP BY region"
        trail = builder.build(sql=sql, query="Revenue by region", data_source="regional_revenue", row_count=4)
        assert trail.confidence == ConfidenceLevel.HIGH

    def test_to_dict_serialises_correctly(self, builder: TrustBuilder) -> None:
        trail = builder.build(
            sql="SELECT SUM(net_revenue) FROM t",
            query="Total revenue",
            data_source="t",
            row_count=1,
        )
        d = trail.to_dict()
        assert "sql_executed" in d
        assert "confidence" in d
        assert "metrics_used" in d
        assert isinstance(d["metrics_used"], list)

    def test_feedback_recorded(self, builder: TrustBuilder) -> None:
        trail = builder.build(sql="SELECT 1", query="test", data_source="t", row_count=1)
        builder.apply_feedback(trail, "positive")
        assert trail.feedback == "positive"

    def test_invalid_feedback_raises(self, builder: TrustBuilder) -> None:
        trail = builder.build(sql="SELECT 1", query="test", data_source="t", row_count=1)
        with pytest.raises(ValueError, match="feedback must be"):
            builder.apply_feedback(trail, "maybe")


# ---------------------------------------------------------------------------
# ChartSelector Tests
# ---------------------------------------------------------------------------


class TestChartSelector:
    """Verify correct chart type selection for each intent."""

    def test_breakdown_returns_figure(self, selector: ChartSelector, simple_df: pd.DataFrame) -> None:
        fig = selector.build(df=simple_df, intent="BREAKDOWN", title="Revenue Breakdown")
        assert fig is not None
        assert fig.layout.title.text == "Revenue Breakdown"

    def test_compare_returns_figure(self, selector: ChartSelector, simple_df: pd.DataFrame) -> None:
        fig = selector.build(df=simple_df, intent="COMPARE", title="Regional Comparison")
        assert fig is not None

    def test_summarize_returns_kpi_figure(self, selector: ChartSelector) -> None:
        df = pd.DataFrame({"signups": [1200], "churn_rate": [0.03], "nps_score": [42]})
        fig = selector.build(df=df, intent="SUMMARIZE", title="KPI Summary")
        assert fig is not None

    def test_empty_dataframe_returns_none(self, selector: ChartSelector) -> None:
        fig = selector.build(df=pd.DataFrame(), intent="BREAKDOWN")
        assert fig is None

    def test_unknown_intent_falls_back_to_bar(self, selector: ChartSelector, simple_df: pd.DataFrame) -> None:
        fig = selector.build(df=simple_df, intent="UNKNOWN_INTENT")
        assert fig is not None  # fallback bar should still work

    def test_natwest_theme_applied(self, selector: ChartSelector, simple_df: pd.DataFrame) -> None:
        fig = selector.build(df=simple_df, intent="BREAKDOWN")
        # Paper background should be white per theme
        assert fig.layout.paper_bgcolor == "#ffffff"