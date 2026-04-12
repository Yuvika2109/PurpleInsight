"""
nl_to_sql.py
------------
Converts natural language queries into valid DuckDB SQL statements
using Google Gemini 1.5 Flash (free tier).

Key features:
    - Injects metric definitions from metrics.yaml into every prompt
      so the LLM always uses canonical NatWest definitions
    - Use-case-specific prompt templates per the 4 NatWest use cases
    - SQL validation before returning — rejects unsafe or invalid SQL
    - Never exposes raw PII or sensitive row-level data in queries

Security:
    - Only SELECT statements are allowed (no INSERT/UPDATE/DELETE/DROP)
    - All queries run against pre-loaded DuckDB views (not raw files)
    - Row-level data is never surfaced — only aggregations
"""

import os
import re
from urllib import response
import yaml
from loguru import logger
from dotenv import load_dotenv
from google import genai

from src.core.intent_router import UseCase

load_dotenv()

# ── Gemini setup ──────────────────────────────────────────────────────────────
_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Dataset schemas — what DuckDB can query ───────────────────────────────────
DATASET_SCHEMAS = {
    "regional_revenue": """
        Table: regional_revenue
        Columns:
            month           TEXT        -- format: YYYY-MM (e.g. '2024-02')
            region          TEXT        -- one of: North, South, East, West, Scotland, Wales
            product         TEXT        -- one of: Personal Current Account, Savings Account, Mortgage, Credit Card, Business Loan
            channel         TEXT        -- one of: Branch, Mobile App, Online, Telephone
            revenue         FLOAT       -- revenue in GBP
            transactions    INTEGER     -- number of transactions
            avg_ticket      FLOAT       -- average transaction value in GBP
            ad_spend        FLOAT       -- advertising spend in GBP
    """,
    "customer_metrics": """
        Table: customer_metrics
        Columns:
            month               TEXT    -- format: YYYY-MM
            segment             TEXT    -- one of: Retail, Business, Premier, Student
            region              TEXT    -- one of: North, South, East, West, Scotland, Wales
            new_signups         INTEGER -- new customers in period
            churned             INTEGER -- customers who left
            churn_rate_pct      FLOAT   -- churn as percentage (e.g. 1.8 = 1.8%)
            complaints          INTEGER -- number of formal complaints
            nps_score           INTEGER -- Net Promoter Score (0-100)
            avg_handle_time     INTEGER -- average call handle time in seconds
            active_customers    INTEGER -- total active customers
    """,
    "product_performance": """
        Table: product_performance
        Columns:
            week                TEXT    -- format: YYYY-WNN (e.g. '2024-W05')
            week_start_date     TEXT    -- format: YYYY-MM-DD
            product             TEXT    -- one of: Personal Current Account, Savings Account, Mortgage, Credit Card, Business Loan
            region              TEXT    -- one of: North, South, East, West, Scotland, Wales
            transaction_volume  INTEGER -- number of transactions
            revenue             FLOAT   -- revenue in GBP
            new_customers       INTEGER -- new customers for this product
            return_customers    INTEGER -- returning customers
            return_rate_pct     FLOAT   -- return customer rate as percentage
            conversion_rate_pct FLOAT   -- conversion rate as percentage
            satisfaction_score  FLOAT   -- satisfaction score (1.0-5.0)
    """,
    "cost_breakdown": """
        Table: cost_breakdown
        Columns:
            quarter         TEXT    -- format: QN-YYYY (e.g. 'Q1-2024')
            department      TEXT    -- one of: Technology, Operations, Risk & Compliance, Marketing, Customer Service, Finance
            cost_category   TEXT    -- one of: Headcount, Infrastructure, Software Licences, Marketing Spend, Training, Outsourcing
            cost_gbp        FLOAT   -- actual cost in GBP
            headcount       INTEGER -- number of FTE employees
            budget_gbp      FLOAT   -- approved budget in GBP
            variance_pct    FLOAT   -- budget variance percentage
    """,
    "weekly_kpis": """
        Table: weekly_kpis
        Columns:
            week                        TEXT    -- format: YYYY-WNN
            week_start_date             TEXT    -- format: YYYY-MM-DD
            new_signups                 INTEGER -- new customers this week
            wow_signup_change_pct       FLOAT   -- week-over-week signup change %
            churn_rate_pct              FLOAT   -- weekly churn rate %
            total_active_customers      INTEGER -- total active customers
            complaints                  INTEGER -- total complaints
            complaint_resolution_rate   FLOAT   -- % complaints resolved
            avg_handle_time_secs        INTEGER -- average handle time in seconds
            nps_score                   INTEGER -- NPS score (0-100)
            digital_adoption_pct        FLOAT   -- % interactions via digital channels
            net_revenue_gbp             FLOAT   -- net revenue in GBP
    """,
}

# ── Use-case-specific prompt templates ────────────────────────────────────────
PROMPT_TEMPLATES = {
    UseCase.CHANGE_ANALYSIS: """
You are a NatWest banking data analyst. Generate a DuckDB SQL query to explain
what changed and why. Focus on identifying the biggest contributors to the change.

Use case: CHANGE ANALYSIS — identify drivers behind increases or decreases.

Rules:
- ONLY return a single valid SQL SELECT statement. No explanation, no markdown.
- Always aggregate — never return individual rows.
- Include percentage change calculations where relevant.
- Limit results to top 10 rows maximum using LIMIT 10.
- Reference only the tables and columns listed in the schema below.
- For month comparisons, use string comparison on the 'month' column (YYYY-MM format).
""",

    UseCase.COMPARE: """
You are a NatWest banking data analyst. Generate a DuckDB SQL query to compare
two or more entities (time periods, regions, products, or segments).

Use case: COMPARE — side-by-side comparison with clear differences highlighted.

Rules:
- ONLY return a single valid SQL SELECT statement. No explanation, no markdown.
- Always aggregate — never return individual rows.
- Include both values being compared AND the difference/percentage difference.
- Use CASE WHEN or CTEs for side-by-side comparisons.
- Limit results to top 10 rows maximum using LIMIT 10.
- Reference only the tables and columns listed in the schema below.
""",

    UseCase.BREAKDOWN: """
You are a NatWest banking data analyst. Generate a DuckDB SQL query to decompose
a metric into its components and show the contribution of each.

Use case: BREAKDOWN — decompose totals, surface patterns and biggest contributors.

Rules:
- ONLY return a single valid SQL SELECT statement. No explanation, no markdown.
- Always aggregate — never return individual rows.
- Include both absolute values AND percentage of total (contribution share).
- Order by the main metric descending so biggest contributors appear first.
- Limit results to top 10 rows maximum using LIMIT 10.
- Reference only the tables and columns listed in the schema below.
""",

    UseCase.SUMMARIZE: """
You are a NatWest banking data analyst. Generate a DuckDB SQL query to produce
a concise summary of key metrics for a time period.

Use case: SUMMARIZE — scan for trends, anomalies, and important shifts.

Rules:
- ONLY return a single valid SQL SELECT statement. No explanation, no markdown.
- Always aggregate — never return individual rows.
- Include week-over-week or period-over-period comparisons where possible.
- Focus on metrics that matter most: signups, churn, NPS, revenue, complaints.
- Limit results to most recent 8 periods maximum using LIMIT 8.
- Reference only the tables and columns listed in the schema below.
""",

    UseCase.UNKNOWN: """
You are a NatWest banking data analyst. Generate a DuckDB SQL query to answer
the user's question using the available data.

Rules:
- ONLY return a single valid SQL SELECT statement. No explanation, no markdown.
- Always aggregate — never return individual rows.
- Limit results to top 10 rows maximum using LIMIT 10.
- Reference only the tables and columns listed in the schema below.
""",
}


class NLToSQL:
    """
    Converts natural language banking queries into valid DuckDB SQL
    using Google Gemini 1.5 Flash with metric-aware prompting.
    """

    def __init__(self, metrics_yaml_path: str = "src/semantic/metrics.yaml"):
        """
        Initialise the NL-to-SQL converter.

        Args:
            metrics_yaml_path: Path to the semantic metric dictionary.
        """
        self.metrics_yaml_path = metrics_yaml_path
        self._metric_context   = self._build_metric_context()
        logger.info("NLToSQL initialised")

    def _build_metric_context(self) -> str:
        """
        Build a concise metric definition context string from metrics.yaml.
        This is injected into every LLM prompt to ensure consistent definitions.

        Returns:
            str: Formatted metric definitions for prompt injection
        """
        try:
            with open(self.metrics_yaml_path, "r") as f:
                metrics_data = yaml.safe_load(f)

            lines = ["METRIC DEFINITIONS (always use these exact definitions):"]
            for name, defn in metrics_data.get("metrics", {}).items():
                aliases = ", ".join(defn.get("aliases", [])[:3])
                lines.append(
                    f"  - {defn['display_name']} ({aliases}): "
                    f"column='{defn['column']}' in {defn['datasets']}, "
                    f"aggregation={defn['aggregation']}"
                )

            return "\n".join(lines)

        except FileNotFoundError:
            logger.warning("metrics.yaml not found — using minimal context")
            return "Use standard SQL aggregations for all metrics."

    def _build_schema_context(self, datasets: list[str]) -> str:
        """
        Build schema context for the relevant datasets only.

        Args:
            datasets: List of dataset names relevant to the query

        Returns:
            str: Formatted schema information for prompt injection
        """
        schemas = []
        for ds in datasets:
            if ds in DATASET_SCHEMAS:
                schemas.append(DATASET_SCHEMAS[ds])
        return "\n".join(schemas)

    def _build_prompt(
        self,
        query: str,
        use_case: UseCase,
        datasets: list[str],
        resolved_time: str = None,
    ) -> str:
        """
        Build the full prompt for Gemini including use-case template,
        metric definitions, schema, and the user query.

        Args:
            query: Original natural language query
            use_case: Classified use case from IntentRouter
            datasets: Relevant dataset names
            resolved_time: Optional resolved time period (from ambiguity handler)

        Returns:
            str: Complete prompt for Gemini
        """
        template      = PROMPT_TEMPLATES.get(use_case, PROMPT_TEMPLATES[UseCase.UNKNOWN])
        schema_context = self._build_schema_context(datasets)

        time_note = ""
        if resolved_time:
            time_note = f"\nTIME RESOLUTION: '{resolved_time}' has been resolved. Use this exact date/period in your query.\n"

        prompt = f"""{template}

{self._metric_context}

AVAILABLE TABLE SCHEMAS:
{schema_context}
{time_note}
IMPORTANT RULES:
- Return ONLY a SQL SELECT statement — no markdown, no explanation, no ```sql fences
- Do NOT use INSERT, UPDATE, DELETE, DROP, CREATE, or ALTER
- Always use aggregate functions (SUM, AVG, COUNT) — never return raw rows
- Use ROUND(value, 2) for all decimal values
- Column names must exactly match the schema above

USER QUESTION: {query}

SQL:"""

        return prompt

    def _validate_sql(self, sql: str) -> tuple[bool, str]:
        """
        Validate that the generated SQL is safe and structurally correct.

        Args:
            sql: Generated SQL string

        Returns:
            tuple: (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL generated"

        sql_upper = sql.upper().strip()

        # Must be a SELECT statement
        if not sql_upper.startswith("SELECT"):
            return False, "Only SELECT statements are allowed"

        # Block dangerous keywords
        dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
                     "TRUNCATE", "EXEC", "EXECUTE", "GRANT", "REVOKE"]
        for keyword in dangerous:
            if re.search(rf"\b{keyword}\b", sql_upper):
                return False, f"Dangerous keyword detected: {keyword}"

        # Must have at least one aggregate function (no raw row dumps)
        aggregates = ["SUM(", "AVG(", "COUNT(", "MIN(", "MAX(", "GROUP BY"]
        has_aggregate = any(agg in sql_upper for agg in aggregates)
        if not has_aggregate:
            return False, "Query must use aggregate functions — raw row queries not allowed"

        # Must reference a known table
        known_tables = list(DATASET_SCHEMAS.keys())
        references_known_table = any(t in sql.lower() for t in known_tables)
        if not references_known_table:
            return False, f"Query must reference a known table: {known_tables}"

        return True, ""

    def _clean_sql(self, raw_output: str) -> str:
        """
        Clean the raw LLM output to extract just the SQL statement.

        Args:
            raw_output: Raw text from Gemini

        Returns:
            str: Cleaned SQL string
        """
        # Remove markdown code fences if present
        sql = re.sub(r"```sql\s*", "", raw_output, flags=re.IGNORECASE)
        sql = re.sub(r"```\s*", "", sql)

        # Remove leading/trailing whitespace
        sql = sql.strip()

        # Take only up to the first semicolon
        if ";" in sql:
            sql = sql[:sql.index(";") + 1]

        return sql.strip()

    def generate(
        self,
        query: str,
        use_case: UseCase,
        datasets: list[str],
        resolved_time: str = None,
        max_retries: int = 2,
    ) -> dict:
        """
        Generate a validated DuckDB SQL query from a natural language question.

        Args:
            query: Natural language query from the user
            use_case: Classified use case from IntentRouter
            datasets: Relevant datasets from IntentRouter.get_dataset_hint()
            resolved_time: Optional time resolution from AmbiguityHandler
            max_retries: Number of retry attempts if validation fails

        Returns:
            dict: {
                "sql": str,              — the validated SQL query
                "valid": bool,           — whether SQL passed validation
                "error": str | None,     — validation error if any
                "datasets_used": list,   — datasets referenced in the query
                "metric_definitions": list — metric definitions applied
            }
        """
        logger.info(f"Generating SQL for use case: {use_case.value}")

        prompt = self._build_prompt(query, use_case, datasets, resolved_time)

        for attempt in range(max_retries + 1):
            try:
                
                response = _CLIENT.models.generate_content(model="gemini-2.0-flash-lite", contents=prompt)
                raw_sql = response.text

                is_valid, error = self._validate_sql(clean_sql)

                if is_valid:
                    logger.info(f"SQL generated and validated (attempt {attempt + 1})")
                    return {
                        "sql":                 clean_sql,
                        "valid":               True,
                        "error":               None,
                        "datasets_used":       datasets,
                        "metric_definitions":  self._get_applied_metrics(clean_sql),
                    }
                else:
                    logger.warning(f"SQL validation failed (attempt {attempt + 1}): {error}")
                    if attempt < max_retries:
                        # Add error feedback to prompt for retry
                        prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {error}. Please fix and try again."

            except Exception as e:
                logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    return {
                        "sql":                None,
                        "valid":              False,
                        "error":              str(e),
                        "datasets_used":      datasets,
                        "metric_definitions": [],
                    }

        return {
            "sql":                None,
            "valid":              False,
            "error":              f"SQL validation failed after {max_retries + 1} attempts",
            "datasets_used":      datasets,
            "metric_definitions": [],
        }

    def _get_applied_metrics(self, sql: str) -> list[str]:
        """
        Identify which metric definitions were applied in the generated SQL.
        Used for the trust trail panel.

        Args:
            sql: Generated SQL string

        Returns:
            list[str]: Names of metrics referenced in the SQL
        """
        try:
            with open(self.metrics_yaml_path, "r") as f:
                metrics_data = yaml.safe_load(f)

            applied = []
            sql_lower = sql.lower()
            for name, defn in metrics_data.get("metrics", {}).items():
                col = defn.get("column", "")
                if col and col.lower() in sql_lower:
                    applied.append(defn["display_name"])

            return applied
        except Exception:
            return []


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.core.intent_router import IntentRouter

    router    = IntentRouter(metrics_yaml_path="src/semantic/metrics.yaml")
    nl_to_sql = NLToSQL(metrics_yaml_path="src/semantic/metrics.yaml")

    test_queries = [
        ("Why did revenue drop in the South region in February 2024?", UseCase.CHANGE_ANALYSIS),
        ("Compare North vs South region revenue for 2024", UseCase.COMPARE),
        ("Show the breakdown of costs by department", UseCase.BREAKDOWN),
        ("Give me a weekly summary for the last 4 weeks", UseCase.SUMMARIZE),
    ]

    print("\n PurpleInsight — NL to SQL Test")
    print("=" * 60)

    for query, use_case in test_queries:
        result   = router.classify(query)
        datasets = router.get_dataset_hint(result, query)
        sql_result = nl_to_sql.generate(query, use_case, datasets)

        print(f"\nQuery   : {query}")
        print(f"UseCase : {use_case.value}")
        print(f"Valid   : {sql_result['valid']}")
        print(f"SQL     :\n{sql_result['sql']}")
        print(f"Metrics : {sql_result['metric_definitions']}")