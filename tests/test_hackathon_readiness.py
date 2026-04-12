from __future__ import annotations

from src.core.intent_router import IntentResult, IntentRouter, UseCase
from src.core.nl_to_sql import NLToSQL


def test_intent_router_can_rank_registered_custom_dataset(monkeypatch) -> None:
    custom_registry = [
        {
            "dataset_id": "branch_service_quality",
            "display_name": "Branch Service Quality",
            "description": "Weekly branch service metrics including complaints and satisfaction",
            "category": "Operations",
            "primary_use_cases": ["compare", "summarize"],
            "exists": True,
        }
    ]
    monkeypatch.setattr("src.core.intent_router.list_registered_datasets", lambda: custom_registry)

    router = IntentRouter(metrics_yaml_path="src/semantic/metrics.yaml")
    result = IntentResult(
        use_case=UseCase.COMPARE,
        confidence=0.9,
        method="keyword",
        matched_keywords=["compare"],
        needs_clarification=False,
    )

    ranked = router.get_dataset_hint(result, "Compare branch complaints by week")
    assert ranked[0] == "branch_service_quality"


def test_nl_to_sql_builds_schema_context_for_registered_custom_dataset(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.core.nl_to_sql.get_dataset_profile",
        lambda dataset_id: {
            "description": "Weekly branch service metrics",
            "columns": [
                {"name": "week", "dtype": "object", "sample": "2025-W01"},
                {"name": "branch", "dtype": "object", "sample": "London"},
                {"name": "complaints", "dtype": "int64", "sample": "12"},
            ],
            "date_like_columns": ["week"],
            "categorical_columns": ["branch"],
            "numeric_columns": ["complaints"],
        },
    )
    monkeypatch.setattr(
        "src.core.nl_to_sql.list_registered_datasets",
        lambda: [{"dataset_id": "branch_service_quality"}],
    )

    generator = NLToSQL(metrics_yaml_path="src/semantic/metrics.yaml")
    schema_context = generator._build_schema_context(["branch_service_quality"])
    assert "Table: branch_service_quality" in schema_context
    assert "complaints" in schema_context


def test_nl_to_sql_generic_fallback_supports_registered_custom_dataset(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.core.nl_to_sql.get_dataset_profile",
        lambda dataset_id: {
            "description": "Weekly branch service metrics",
            "columns": [
                {"name": "week", "dtype": "object", "sample": "2025-W01"},
                {"name": "branch", "dtype": "object", "sample": "London"},
                {"name": "complaints", "dtype": "int64", "sample": "12"},
            ],
            "date_like_columns": ["week"],
            "categorical_columns": ["branch"],
            "numeric_columns": ["complaints"],
        },
    )
    monkeypatch.setattr(
        "src.core.nl_to_sql.list_registered_datasets",
        lambda: [{"dataset_id": "branch_service_quality"}],
    )

    generator = NLToSQL(metrics_yaml_path="src/semantic/metrics.yaml")
    result = generator._fallback_generate(
        query="Compare branch complaints by week",
        use_case=UseCase.COMPARE,
        datasets=["branch_service_quality"],
    )
    assert result["valid"] is True
    assert "FROM branch_service_quality" in result["sql"]
