"""
FastAPI routes for DataTalk.
"""

from __future__ import annotations

import logging
import uuid
import yaml
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config.settings import settings
from src.core.trust_builder import TrustBuilder, TrustTrail
from src.core.chart_selector import ChartSelector
from src.core.intent_router import IntentRouter
from src.core.nl_to_sql import NLToSQL
from src.core.narrative import Narrative
from src.core.query_engine import QueryEngine

logger = logging.getLogger(__name__)
router = APIRouter()

def _load_metric_definitions() -> dict:
    try:
        with open(settings.METRICS_YAML_PATH, "r") as f:
            raw = yaml.safe_load(f) or {}
        metrics = raw.get("metrics", {})
        return {
            name: {
                "definition": meta.get("definition", "").strip(),
                "source_field": meta.get("column", "unknown"),
            }
            for name, meta in metrics.items()
        }
    except FileNotFoundError:
        logger.warning("metrics.yaml not found at %s", settings.METRICS_YAML_PATH)
        return {}


_metric_definitions: dict = _load_metric_definitions()
_trust_builder = TrustBuilder(metric_definitions=_metric_definitions)
_chart_selector = ChartSelector()
_intent_router  = IntentRouter(metrics_yaml_path=settings.METRICS_YAML_PATH)
_nl_to_sql      = NLToSQL(metrics_yaml_path=settings.METRICS_YAML_PATH)
_query_engine   = QueryEngine(data_dir=settings.DATA_DIR)
_narrative      = Narrative(metrics_yaml_path=settings.METRICS_YAML_PATH)
_trail_store: dict[str, TrustTrail] = {}


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    query_id: str | None = Field(default=None)

class FeedbackRequest(BaseModel):
    query_id: str
    feedback: str = Field(..., pattern="^(positive|negative)$")

class QueryResponse(BaseModel):
    query_id: str
    intent: str
    narrative: str
    trust_trail: dict
    chart_type: str | None = None
    sql: str

class MetricItem(BaseModel):
    name: str
    definition: str
    source_field: str


@router.get("/health", tags=["system"])
def health_check() -> dict:
    return {"status": "ok", "service": "DataTalk"}


@router.get("/metrics", response_model=list[MetricItem], tags=["semantic"])
def list_metrics() -> list[MetricItem]:
    return [
        MetricItem(name=name, definition=meta.get("definition",""), source_field=meta.get("source_field",""))
        for name, meta in _metric_definitions.items()
    ]


@router.post("/query", response_model=QueryResponse, tags=["query"])
def run_query(request: QueryRequest) -> QueryResponse:
    query_id = request.query_id or str(uuid.uuid4())

    # 1. Classify intent
    intent_result = _intent_router.classify(request.query)

    # 2. Get datasets
    datasets = _intent_router.get_dataset_hint(intent_result, request.query)

    # 3. Generate SQL — use_case and datasets as NLToSQL.generate() expects
    try:
        sql_result = _nl_to_sql.generate(
            query=request.query,
            use_case=intent_result.use_case,
            datasets=datasets,
        )
        sql = sql_result.get("sql") if isinstance(sql_result, dict) else str(sql_result)
        if not sql:
            err = sql_result.get("error", "Unknown error") if isinstance(sql_result, dict) else "SQL generation returned None"
            raise HTTPException(status_code=500, detail=f"SQL generation failed: {err}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {exc}") from exc

    # 4. Execute SQL
    try:
        exec_result = _query_engine.execute(sql)
        exec_result["datasets_used"] = datasets
        result_df = pd.DataFrame(exec_result.get("data", []))
        data_source = datasets[0] if datasets else "unknown"
        if not exec_result.get("success", True):
            raise HTTPException(status_code=500, detail=exec_result.get("error", "Query failed"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {exc}") from exc

    # 5. Trust trail
    trail = _trust_builder.build(
        sql=sql, query=request.query, data_source=data_source, row_count=len(result_df)
    )
    _trail_store[query_id] = trail

    # 6. Intent string
    intent_str = intent_result.use_case.value if hasattr(intent_result.use_case, "value") else str(intent_result.use_case)

    return QueryResponse(
        query_id=query_id,
        intent=intent_str,
        narrative=_generate_narrative(exec_result, intent_result.use_case, request.query),
        trust_trail=trail.to_dict(),
        chart_type=_resolve_chart_type(intent_str),
        sql=sql,
    )


@router.post("/feedback", tags=["feedback"])
def record_feedback(request: FeedbackRequest) -> dict:
    trail = _trail_store.get(request.query_id)
    if trail is None:
        raise HTTPException(status_code=404, detail=f"No query found with id: {request.query_id}")
    _trust_builder.apply_feedback(trail, request.feedback)
    return {"status": "recorded", "query_id": request.query_id, "feedback": request.feedback}


def _resolve_chart_type(intent: str) -> str | None:
    return {"BREAKDOWN":"stacked_bar","COMPARE":"grouped_bar_or_line","CHANGE_ANALYSIS":"waterfall","SUMMARIZE":"kpi_cards"}.get(intent.upper())


def _generate_narrative(query_result: dict, use_case, query: str) -> str:
    if not query_result.get("data"):
        return "No data found. Try adjusting filters or time range."
    try:
        narrative_result = _narrative.generate(
            query=query,
            use_case=use_case,
            query_result=query_result,
            metric_definitions=[],
        )
        return narrative_result.get("narrative", "Analysis complete.")
    except Exception as exc:
        logger.warning("Narrative fallback: %s", exc)
        preview = pd.DataFrame(query_result.get("data", [])).head(5).to_dict(orient="records")
        return f"Query returned {len(query_result.get('data', []))} rows. Top results: {preview}"
