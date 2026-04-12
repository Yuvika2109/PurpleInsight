"""
Trust Builder — generates a full audit trail for every answer.

Every response from DataTalk includes:
- The exact SQL that was executed
- Which metric definitions were applied
- The data source referenced
- A confidence score with justification
- A feedback slot for thumbs up/down

This directly addresses NatWest's compliance and explainability requirements.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence tier for a generated answer."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class MetricUsage:
    """Records which metric definition was applied during query resolution."""
    name: str
    definition: str
    source_field: str


@dataclass
class TrustTrail:
    """
    Complete audit record attached to every DataTalk answer.

    Attributes:
        sql_executed:       The exact SQL run against DuckDB.
        metrics_used:       List of metric definitions resolved for this query.
        data_source:        Which dataset(s) were queried.
        confidence:         HIGH / MEDIUM / LOW with a reason string.
        confidence_reason:  Human-readable explanation of the confidence tier.
        row_count:          Number of rows the SQL returned.
        ambiguity_flags:    Any ambiguous terms detected in the original query.
        feedback:           Optional thumbs-up / thumbs-down from the user.
    """
    sql_executed: str
    metrics_used: list[MetricUsage]
    data_source: str
    confidence: ConfidenceLevel
    confidence_reason: str
    row_count: int = 0
    ambiguity_flags: list[str] = field(default_factory=list)
    feedback: Optional[str] = None  # "positive" | "negative" | None

    def to_dict(self) -> dict:
        """Serialise the trail for API responses and UI rendering."""
        return {
            "sql_executed": self.sql_executed,
            "metrics_used": [
                {
                    "name": m.name,
                    "definition": m.definition,
                    "source_field": m.source_field,
                }
                for m in self.metrics_used
            ],
            "data_source": self.data_source,
            "confidence": self.confidence.value,
            "confidence_reason": self.confidence_reason,
            "row_count": self.row_count,
            "ambiguity_flags": self.ambiguity_flags,
            "feedback": self.feedback,
        }


class TrustBuilder:
    """
    Constructs a TrustTrail for every query-answer pair.

    Usage:
        builder = TrustBuilder(metric_definitions)
        trail = builder.build(
            sql=sql_string,
            query=user_query,
            data_source="regional_revenue",
            row_count=42,
        )
    """

    # Terms that commonly cause ambiguity in banking queries
    _AMBIGUOUS_PATTERNS: list[str] = [
        r"\bthis month\b",
        r"\blast month\b",
        r"\brecently\b",
        r"\blast cycle\b",
        r"\bcurrent period\b",
        r"\blatest\b",
        r"\byesterday\b",
        r"\bthis quarter\b",
    ]

    def __init__(self, metric_definitions: dict[str, dict]) -> None:
        """
        Args:
            metric_definitions: Loaded from metrics.yaml — maps term → {definition, source_field}.
        """
        self.metric_definitions = metric_definitions
        logger.debug("TrustBuilder initialised with %d metric definitions.", len(metric_definitions))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        sql: str,
        query: str,
        data_source: str,
        row_count: int = 0,
    ) -> TrustTrail:
        """
        Build a complete TrustTrail for a given SQL + user query pair.

        Args:
            sql:         The SQL string that was executed against DuckDB.
            query:       The original natural-language question from the user.
            data_source: Name of the dataset/table that was queried.
            row_count:   Number of rows returned by the SQL.

        Returns:
            A fully populated TrustTrail instance.
        """
        metrics_used = self._resolve_metrics(sql, query)
        ambiguity_flags = self._detect_ambiguity(query)
        confidence, reason = self._score_confidence(
            sql=sql,
            metrics_used=metrics_used,
            ambiguity_flags=ambiguity_flags,
            row_count=row_count,
        )

        trail = TrustTrail(
            sql_executed=sql,
            metrics_used=metrics_used,
            data_source=data_source,
            confidence=confidence,
            confidence_reason=reason,
            row_count=row_count,
            ambiguity_flags=ambiguity_flags,
        )
        logger.info(
            "TrustTrail built | source=%s | confidence=%s | metrics=%d | rows=%d",
            data_source,
            confidence.value,
            len(metrics_used),
            row_count,
        )
        return trail

    def apply_feedback(self, trail: TrustTrail, feedback: str) -> TrustTrail:
        """
        Attach user feedback (thumbs up/down) to an existing TrustTrail.

        Args:
            trail:    The trail to update.
            feedback: 'positive' or 'negative'.

        Returns:
            The updated TrustTrail.
        """
        if feedback not in ("positive", "negative"):
            raise ValueError(f"feedback must be 'positive' or 'negative', got: {feedback!r}")
        trail.feedback = feedback
        logger.info("Feedback recorded: %s", feedback)
        return trail

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_metrics(self, sql: str, query: str) -> list[MetricUsage]:
        """
        Match known metric terms against the SQL and the original query.

        Returns a deduplicated list of MetricUsage objects for every term found.
        """
        combined_text = (sql + " " + query).lower()
        used: list[MetricUsage] = []
        seen: set[str] = set()

        for term, meta in self.metric_definitions.items():
            if term.lower() in combined_text and term not in seen:
                used.append(
                    MetricUsage(
                        name=term,
                        definition=meta.get("definition", "No definition available."),
                        source_field=meta.get("source_field", "unknown"),
                    )
                )
                seen.add(term)

        return used

    def _detect_ambiguity(self, query: str) -> list[str]:
        """
        Scan the user query for vague temporal or contextual references.

        Returns a list of matched ambiguous phrases.
        """
        flags: list[str] = []
        for pattern in self._AMBIGUOUS_PATTERNS:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                flags.append(match.group(0).strip())
        return flags

    def _score_confidence(
        self,
        sql: str,
        metrics_used: list[MetricUsage],
        ambiguity_flags: list[str],
        row_count: int,
    ) -> tuple[ConfidenceLevel, str]:
        """
        Heuristic confidence scoring based on query characteristics.

        Rules (applied in priority order):
        - LOW  → ambiguous terms present OR zero rows returned
        - LOW  → SQL contains LIKE wildcard (fuzzy match)
        - MEDIUM → SQL has subqueries or CTEs (complex, higher error surface)
        - MEDIUM → no metric definitions were resolved
        - HIGH → everything looks clean

        Returns:
            Tuple of (ConfidenceLevel, human-readable reason string).
        """
        if ambiguity_flags:
            return (
                ConfidenceLevel.LOW,
                f"Ambiguous terms detected: {', '.join(ambiguity_flags)}. "
                "Results may not reflect intended time period or scope.",
            )

        if row_count == 0:
            return (
                ConfidenceLevel.LOW,
                "Query returned zero rows. The filter conditions may be too narrow "
                "or the data may not cover this period.",
            )

        sql_lower = sql.lower()

        if "like '%" in sql_lower or "like \"%" in sql_lower:
            return (
                ConfidenceLevel.LOW,
                "SQL uses a wildcard (LIKE) match which may produce imprecise results.",
            )

        if "with " in sql_lower and " as (" in sql_lower:
            return (
                ConfidenceLevel.MEDIUM,
                "Query uses a CTE/subquery. Results are correct but the logic is "
                "more complex — review the SQL for accuracy.",
            )

        if not metrics_used:
            return (
                ConfidenceLevel.MEDIUM,
                "No registered metric definitions were matched. The answer is based "
                "on raw column names which may have inconsistent meaning.",
            )

        return (
            ConfidenceLevel.HIGH,
            "All metric terms resolved via the semantic dictionary. "
            "SQL is straightforward with no ambiguity detected.",
        )