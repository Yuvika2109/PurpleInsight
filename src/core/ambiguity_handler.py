"""
ambiguity_handler.py
--------------------
Detects and resolves ambiguous or vague terms in user queries.

Handles:
    1. Vague time references: "this month", "last week", "recently", "last cycle"
    2. Ambiguous metric names: "performance", "numbers", "results"
    3. Ambiguous comparisons: "better", "worse" without a baseline

For each ambiguity detected, the handler either:
    - Auto-resolves it (for time references with a clear mapping)
    - Asks a single clarifying question (for genuine ambiguity)

The problem doc explicitly calls out handling "this month" and "last cycle"
as requirements — this module delivers that.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from loguru import logger
import yaml
import re


# ── Result types ──────────────────────────────────────────────────────────────
@dataclass
class AmbiguityResult:
    """Result of ambiguity analysis for a query."""
    has_ambiguity:        bool          # whether ambiguity was detected
    needs_clarification:  bool          # whether user input is needed
    clarification_question: str         # question to ask user (if needed)
    resolved_query:       str           # query with ambiguities resolved
    resolutions:          dict          # what was auto-resolved and how
    detected_terms:       list[str]     # ambiguous terms found


# ── Current reference date ────────────────────────────────────────────────────
def _now() -> datetime:
    """Return current datetime. Centralised for easy testing."""
    return datetime.now()


# ── Time period resolver ──────────────────────────────────────────────────────
def _resolve_time_period(term: str) -> tuple[str, str]:
    """
    Resolve a vague time reference to an exact date range or period string.

    Args:
        term: Vague time reference (e.g. "last month", "this week")

    Returns:
        tuple: (resolved_label, SQL_hint)
            resolved_label — human-readable resolved period
            SQL_hint       — concrete date string to inject into SQL
    """
    now      = _now()
    term_low = term.lower().strip()

    # ── Monthly references ────────────────────────────────────────────────────
    if term_low in ("last month", "previous month"):
        first_of_this = now.replace(day=1)
        last_month    = first_of_this - relativedelta(months=1)
        return (
            f"last month ({last_month.strftime('%B %Y')})",
            last_month.strftime("%Y-%m"),
        )

    if term_low in ("this month", "current month"):
        return (
            f"this month ({now.strftime('%B %Y')})",
            now.strftime("%Y-%m"),
        )

    if term_low in ("2 months ago", "two months ago"):
        two_ago = now - relativedelta(months=2)
        return (
            f"2 months ago ({two_ago.strftime('%B %Y')})",
            two_ago.strftime("%Y-%m"),
        )

    # ── Weekly references ─────────────────────────────────────────────────────
    if term_low in ("last week", "previous week"):
        last_week_start = now - timedelta(days=now.weekday() + 7)
        week_str        = last_week_start.strftime("%Y-W%W")
        return (
            f"last week (week starting {last_week_start.strftime('%d %b %Y')})",
            week_str,
        )

    if term_low in ("this week", "current week"):
        this_week_start = now - timedelta(days=now.weekday())
        week_str        = this_week_start.strftime("%Y-W%W")
        return (
            f"this week (week starting {this_week_start.strftime('%d %b %Y')})",
            week_str,
        )

    # ── Quarterly references ──────────────────────────────────────────────────
    if term_low in ("last quarter", "previous quarter", "last cycle"):
        current_q = (now.month - 1) // 3 + 1
        if current_q == 1:
            prev_q    = 4
            prev_year = now.year - 1
        else:
            prev_q    = current_q - 1
            prev_year = now.year
        return (
            f"last quarter (Q{prev_q}-{prev_year})",
            f"Q{prev_q}-{prev_year}",
        )

    if term_low in ("this quarter", "current quarter"):
        current_q = (now.month - 1) // 3 + 1
        return (
            f"this quarter (Q{current_q}-{now.year})",
            f"Q{current_q}-{now.year}",
        )

    # ── Yearly references ─────────────────────────────────────────────────────
    if term_low in ("last year", "previous year"):
        return (
            f"last year ({now.year - 1})",
            str(now.year - 1),
        )

    if term_low in ("this year", "current year", "ytd", "year to date"):
        return (
            f"this year ({now.year}, year to date)",
            str(now.year),
        )

    # ── Relative references ───────────────────────────────────────────────────
    if term_low in ("recently", "recent", "lately"):
        four_weeks_ago = now - timedelta(weeks=4)
        return (
            f"recently (last 4 weeks, since {four_weeks_ago.strftime('%d %b %Y')})",
            four_weeks_ago.strftime("%Y-%m"),
        )

    # Unknown — cannot auto-resolve
    return None, None


# ── Ambiguous metric terms ────────────────────────────────────────────────────
AMBIGUOUS_METRICS = {
    "performance":  ["revenue", "NPS score", "transaction volume", "churn rate"],
    "numbers":      ["revenue", "signups", "complaints"],
    "results":      ["revenue", "NPS score", "churn rate"],
    "metrics":      ["revenue", "signups", "NPS", "churn", "complaints"],
    "figures":      ["revenue", "cost", "signups"],
    "data":         ["revenue", "customer metrics", "product performance"],
    "stats":        ["revenue", "signups", "NPS", "churn"],
}

# ── Vague time terms to detect ────────────────────────────────────────────────
VAGUE_TIME_TERMS = [
    "last month", "this month", "previous month", "current month",
    "last week", "this week", "previous week", "current week",
    "last quarter", "this quarter", "previous quarter", "current quarter",
    "last cycle", "last year", "this year", "previous year", "current year",
    "recently", "recent", "lately", "2 months ago", "two months ago",
    "ytd", "year to date", "qtd", "quarter to date",
]


class AmbiguityHandler:
    """
    Detects and resolves ambiguous terms in natural language banking queries.

    Prioritises auto-resolution over asking questions to keep the
    "minimal steps" requirement from the NatWest problem doc.
    Only asks clarification when auto-resolution is genuinely impossible.
    """

    def __init__(self, metrics_yaml_path: str = "src/semantic/metrics.yaml"):
        """
        Initialise the ambiguity handler.

        Args:
            metrics_yaml_path: Path to metrics.yaml for time period mappings.
        """
        self.metrics_yaml_path = metrics_yaml_path
        self._yaml_time_periods = self._load_yaml_time_periods()
        logger.info("AmbiguityHandler initialised")

    def _load_yaml_time_periods(self) -> dict:
        """
        Load time period definitions from metrics.yaml.

        Returns:
            dict: Term → resolution mapping from YAML
        """
        try:
            with open(self.metrics_yaml_path, "r") as f:
                data = yaml.safe_load(f)
            return data.get("time_periods", {})
        except FileNotFoundError:
            return {}

    def analyse(self, query: str) -> AmbiguityResult:
        """
        Analyse a query for ambiguities and resolve where possible.

        Args:
            query: Natural language query from the user

        Returns:
            AmbiguityResult: Analysis result with resolutions and any
                             clarification question needed
        """
        detected_terms = []
        resolutions    = {}
        resolved_query = query
        needs_clarification   = False
        clarification_question = ""

        # ── Step 1: Detect and resolve vague time references ──────────────────
        for term in VAGUE_TIME_TERMS:
            if term.lower() in query.lower():
                detected_terms.append(term)
                resolved_label, sql_hint = _resolve_time_period(term)

                if resolved_label:
                    # Auto-resolve: replace vague term with exact label in context
                    resolutions[term] = {
                        "resolved_to": resolved_label,
                        "sql_hint":    sql_hint,
                        "auto":        True,
                    }
                    logger.debug(f"Auto-resolved '{term}' → '{resolved_label}'")
                else:
                    # Cannot auto-resolve — ask for clarification
                    needs_clarification    = True
                    clarification_question = (
                        f"Could you clarify what time period you mean by '{term}'? "
                        f"For example: a specific month (e.g. March 2024), "
                        f"a quarter (e.g. Q1 2024), or a date range?"
                    )

        # ── Step 2: Detect ambiguous metric terms ─────────────────────────────
        query_lower = query.lower()
        for vague_term, options in AMBIGUOUS_METRICS.items():
            # Only flag if the vague term appears without a more specific metric
            if vague_term in query_lower:
                specific_found = any(
                    specific in query_lower
                    for specific in ["revenue", "churn", "nps", "cost", "signup",
                                     "complaint", "transaction", "satisfaction"]
                )
                if not specific_found:
                    detected_terms.append(vague_term)
                    if not needs_clarification:
                        needs_clarification    = True
                        options_str = ", ".join(options[:4])
                        clarification_question = (
                            f"When you say '{vague_term}', which metrics are you "
                            f"most interested in? For example: {options_str}?"
                        )
                    break  # ask one question at a time

        # ── Step 3: Detect region/product ambiguity ───────────────────────────
        comparison_words = ["vs", "versus", "compare", "compared to", "against"]
        has_comparison   = any(w in query_lower for w in comparison_words)

        if has_comparison:
            # Check if both sides of the comparison are specified
            regions   = ["north", "south", "east", "west", "scotland", "wales"]
            products  = ["current account", "savings", "mortgage", "credit card", "business loan"]
            segments  = ["retail", "business", "premier", "student"]

            found_regions   = [r for r in regions if r in query_lower]
            found_products  = [p for p in products if p in query_lower]
            found_segments  = [s for s in segments if s in query_lower]

            # If comparison intent but nothing specific mentioned — ask
            if not found_regions and not found_products and not found_segments:
                if not needs_clarification:
                    needs_clarification    = True
                    clarification_question = (
                        "What would you like to compare? For example: "
                        "two regions (North vs South), two products "
                        "(Mortgage vs Credit Card), or two time periods "
                        "(this month vs last month)?"
                    )

        has_ambiguity = len(detected_terms) > 0

        return AmbiguityResult(
            has_ambiguity         = has_ambiguity,
            needs_clarification   = needs_clarification,
            clarification_question = clarification_question,
            resolved_query        = resolved_query,
            resolutions           = resolutions,
            detected_terms        = detected_terms,
        )

    def get_sql_time_hints(self, result: AmbiguityResult) -> dict:
        """
        Extract SQL-ready time hints from resolution results.
        These are injected into the NL-to-SQL prompt.

        Args:
            result: AmbiguityResult from analyse()

        Returns:
            dict: term → sql_hint for all auto-resolved time periods
        """
        hints = {}
        for term, resolution in result.resolutions.items():
            if resolution.get("auto") and resolution.get("sql_hint"):
                hints[term] = resolution["sql_hint"]
        return hints

    def format_resolution_summary(self, result: AmbiguityResult) -> str:
        """
        Format a human-readable summary of what was auto-resolved.
        Shown in the trust trail so users know how their query was interpreted.

        Args:
            result: AmbiguityResult from analyse()

        Returns:
            str: Human-readable resolution summary
        """
        if not result.resolutions:
            return ""

        lines = ["Time periods interpreted as:"]
        for term, resolution in result.resolutions.items():
            if resolution.get("auto"):
                lines.append(f"  • '{term}' → {resolution['resolved_to']}")

        return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    handler = AmbiguityHandler(metrics_yaml_path="src/semantic/metrics.yaml")

    test_queries = [
        "Why did revenue drop last month?",
        "Compare performance this week vs last week",
        "Show me the results recently",
        "Give me a summary of this quarter",
        "Compare North vs South",
        "What are the numbers for last cycle?",
        "How did we do this year so far?",
        "Show me the breakdown",
    ]

    print("\nPurpleInsight — Ambiguity Handler Test")
    print("=" * 60)

    for query in test_queries:
        result = handler.analyse(query)
        print(f"\nQuery     : {query}")
        print(f"Ambiguous : {result.has_ambiguity}")
        print(f"Needs Q   : {result.needs_clarification}")
        if result.needs_clarification:
            print(f"Question  : {result.clarification_question}")
        if result.resolutions:
            print(f"Resolved  : {handler.format_resolution_summary(result)}")
