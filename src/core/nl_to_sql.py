"""
nl_to_sql.py
------------
Converts natural language queries into valid DuckDB SQL statements
using Groq + Llama 3.1 70B (free tier, fastest inference available).

Key features:
    - Injects metric definitions from metrics.yaml into every prompt
    - Use-case-specific prompt templates per the 4 NatWest use cases
    - SQL validation before returning — rejects unsafe or invalid SQL
    - Never exposes raw PII or sensitive row-level data in queries

Security:
    - Only SELECT statements are allowed
    - Row-level data is never surfaced — only aggregations
"""

import os
import re
import yaml
from loguru import logger
from dotenv import load_dotenv
from groq import Groq

from config.dataset_registry import build_schema_prompt_block, get_dataset_profile, list_registered_datasets
from src.core.intent_router import UseCase

load_dotenv()

# ── Groq client ───────────────────────────────────────────────────────────────
_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL  = "llama-3.1-70b-versatile"

# ── Dataset schemas ───────────────────────────────────────────────────────────
DATASET_SCHEMAS = {
    "regional_revenue": """
        Table: regional_revenue
        Columns:
            month           TEXT    -- format: YYYY-MM (e.g. '2024-02')
            region          TEXT    -- one of: North, South, East, West, Scotland, Wales
            product         TEXT    -- one of: Personal Current Account, Savings Account, Mortgage, Credit Card, Business Loan
            channel         TEXT    -- one of: Branch, Mobile App, Online, Telephone
            revenue         FLOAT   -- revenue in GBP
            transactions    INTEGER -- number of transactions
            avg_ticket      FLOAT   -- average transaction value in GBP
            ad_spend        FLOAT   -- advertising spend in GBP
    """,
    "customer_metrics": """
        Table: customer_metrics
        Columns:
            month               TEXT    -- format: YYYY-MM
            segment             TEXT    -- one of: Retail, Business, Premier, Student
            region              TEXT    -- one of: North, South, East, West, Scotland, Wales
            new_signups         INTEGER
            churned             INTEGER
            churn_rate_pct      FLOAT   -- e.g. 1.8 means 1.8%
            complaints          INTEGER
            nps_score           INTEGER -- 0 to 100
            avg_handle_time     INTEGER -- seconds
            active_customers    INTEGER
    """,
    "product_performance": """
        Table: product_performance
        Columns:
            week                TEXT    -- format: YYYY-WNN
            week_start_date     TEXT    -- format: YYYY-MM-DD
            product             TEXT    -- one of: Personal Current Account, Savings Account, Mortgage, Credit Card, Business Loan
            region              TEXT    -- one of: North, South, East, West, Scotland, Wales
            transaction_volume  INTEGER
            revenue             FLOAT
            new_customers       INTEGER
            return_customers    INTEGER
            return_rate_pct     FLOAT
            conversion_rate_pct FLOAT
            satisfaction_score  FLOAT   -- 1.0 to 5.0
    """,
    "cost_breakdown": """
        Table: cost_breakdown
        Columns:
            quarter         TEXT    -- format: QN-YYYY (e.g. 'Q1-2024')
            department      TEXT    -- one of: Technology, Operations, Risk & Compliance, Marketing, Customer Service, Finance
            cost_category   TEXT    -- one of: Headcount, Infrastructure, Software Licences, Marketing Spend, Training, Outsourcing
            cost_gbp        FLOAT
            headcount       INTEGER
            budget_gbp      FLOAT
            variance_pct    FLOAT
    """,
    "weekly_kpis": """
        Table: weekly_kpis
        Columns:
            week                        TEXT    -- format: YYYY-WNN
            week_start_date             TEXT    -- format: YYYY-MM-DD
            new_signups                 INTEGER
            wow_signup_change_pct       FLOAT
            churn_rate_pct              FLOAT
            total_active_customers      INTEGER
            complaints                  INTEGER
            complaint_resolution_rate   FLOAT
            avg_handle_time_secs        INTEGER
            nps_score                   INTEGER
            digital_adoption_pct        FLOAT
            net_revenue_gbp             FLOAT
    """,
}

PROMPT_TEMPLATES = {
    UseCase.CHANGE_ANALYSIS: "Generate DuckDB SQL to identify drivers behind a metric change. Focus on biggest contributors. Use aggregates only.",
    UseCase.COMPARE:         "Generate DuckDB SQL for a side-by-side comparison. Include both values and the percentage difference.",
    UseCase.BREAKDOWN:       "Generate DuckDB SQL to decompose a metric into components. Include percentage of total. Order by value DESC.",
    UseCase.SUMMARIZE:       "Generate DuckDB SQL for a concise KPI summary. Include trends. LIMIT 8.",
    UseCase.UNKNOWN:         "Generate DuckDB SQL to answer the question. Use aggregates only. LIMIT 10.",
}


class NLToSQL:
    """Converts natural language banking queries into valid DuckDB SQL using Groq."""

    def __init__(self, metrics_yaml_path: str = "src/semantic/metrics.yaml"):
        self.metrics_yaml_path = metrics_yaml_path
        self._metric_context   = self._build_metric_context()
        logger.info("NLToSQL initialised")

    def _build_metric_context(self) -> str:
        try:
            with open(self.metrics_yaml_path, "r") as f:
                metrics_data = yaml.safe_load(f)
            lines = ["METRIC DEFINITIONS:"]
            for name, defn in metrics_data.get("metrics", {}).items():
                lines.append(
                    f"  - {defn['display_name']}: column='{defn['column']}' "
                    f"in {defn['datasets']}, aggregation={defn['aggregation']}"
                )
            return "\n".join(lines)
        except FileNotFoundError:
            return "Use standard SQL aggregations for all metrics."

    def _registered_dataset_ids(self) -> list[str]:
        """Return dataset ids currently registered in the project."""
        return [item["dataset_id"] for item in list_registered_datasets()]

    def _build_schema_context(self, datasets: list) -> str:
        blocks = []
        for dataset_id in datasets:
            if dataset_id in DATASET_SCHEMAS:
                blocks.append(DATASET_SCHEMAS[dataset_id])
                continue

            profile = get_dataset_profile(dataset_id)
            if profile:
                blocks.append(build_schema_prompt_block(dataset_id, profile))

        return "\n\n".join(blocks)

    def _build_prompt(self, query: str, use_case: UseCase, datasets: list, resolved_time: str = None) -> str:
        template       = PROMPT_TEMPLATES.get(use_case, PROMPT_TEMPLATES[UseCase.UNKNOWN])
        schema_context = self._build_schema_context(datasets)
        time_note      = f"\nTIME: '{resolved_time}' — use this exact value in WHERE clause.\n" if resolved_time else ""

        return f"""You are a NatWest banking SQL expert. {template}

{self._metric_context}

SCHEMAS:
{schema_context}
{time_note}
RULES — MUST FOLLOW EXACTLY:
1. Return ONLY a raw SQL SELECT statement — nothing else
2. NO markdown, NO backticks, NO explanation, NO preamble
3. Start your entire response with the word SELECT
4. Only SELECT allowed — no INSERT/UPDATE/DELETE/DROP/CREATE
5. Always use GROUP BY with aggregate functions
6. ROUND all floats to 2 decimal places
7. Column names must exactly match the schemas above

QUESTION: {query}

SELECT"""

    def _validate_sql(self, sql: str) -> tuple:
        if not sql or not sql.strip():
            return False, "Empty SQL"
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith("SELECT"):
            return False, "Must start with SELECT"
        for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]:
            if re.search(rf"\b{kw}\b", sql_upper):
                return False, f"Dangerous keyword: {kw}"
        if not any(agg in sql_upper for agg in ["SUM(", "AVG(", "COUNT(", "MIN(", "MAX(", "GROUP BY"]):
            return False, "Must use aggregate functions"
        known_tables = self._registered_dataset_ids() or list(DATASET_SCHEMAS.keys())
        if not any(t in sql.lower() for t in known_tables):
            return False, "Must reference a known table"
        return True, ""

    def _clean_sql(self, raw: str) -> str:
        sql = re.sub(r"```sql\s*", "", raw, flags=re.IGNORECASE)
        sql = re.sub(r"```\s*", "", sql).strip()
        # Ensure starts with SELECT
        match = re.search(r"\bSELECT\b", sql, re.IGNORECASE)
        if match:
            sql = sql[match.start():]
        if ";" in sql:
            sql = sql[:sql.index(";") + 1]
        return sql.strip()

    def _extract_time_filter(self, dataset: str, resolved_time: str = None, query: str = "") -> str:
        """Build a simple WHERE clause when a clear time hint is available."""
        profile = get_dataset_profile(dataset)
        date_cols = profile.get("date_like_columns", []) if profile else []
        if resolved_time:
            if dataset in {"regional_revenue", "customer_metrics"}:
                return f"WHERE month = '{resolved_time}'"
            if dataset == "product_performance":
                return f"WHERE week = '{resolved_time}' OR week_start_date = '{resolved_time}'"
            if dataset == "weekly_kpis":
                return f"WHERE week = '{resolved_time}' OR week_start_date = '{resolved_time}'"
            if dataset == "cost_breakdown":
                return f"WHERE quarter = '{resolved_time}'"
            if date_cols:
                return f"WHERE {date_cols[0]} = '{resolved_time}'"

        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            year = year_match.group(1)
            if dataset in {"regional_revenue", "customer_metrics"}:
                return f"WHERE month LIKE '{year}-%'"
            if dataset in {"product_performance", "weekly_kpis"}:
                return f"WHERE week LIKE '{year}-%' OR week_start_date LIKE '{year}-%'"
            if dataset == "cost_breakdown":
                return f"WHERE quarter LIKE '%-{year}'"
            if date_cols:
                return f"WHERE CAST({date_cols[0]} AS VARCHAR) LIKE '{year}%'"

        return ""

    def _fallback_generate_generic(self, dataset: str, query: str, resolved_time: str = None) -> str | None:
        """Build generic aggregate SQL for any registered dataset profile."""
        profile = get_dataset_profile(dataset)
        if not profile:
            return None

        query_lower = query.lower()
        date_cols = profile.get("date_like_columns", [])
        numeric_cols = profile.get("numeric_columns", [])
        categorical_cols = profile.get("categorical_columns", [])

        group_col = None
        for column in categorical_cols + date_cols:
            if column.lower() in query_lower:
                group_col = column
                break
        if group_col is None:
            if any(token in query_lower for token in ["trend", "over time", "week", "month", "quarter", "year"]) and date_cols:
                group_col = date_cols[0]
            elif categorical_cols:
                group_col = categorical_cols[0]
            elif date_cols:
                group_col = date_cols[0]

        metric_col = None
        for column in numeric_cols:
            if column.lower() in query_lower:
                metric_col = column
                break
        if metric_col is None and numeric_cols:
            metric_col = numeric_cols[0]

        if not metric_col:
            if group_col:
                return f"""
                SELECT
                    {group_col},
                    COUNT(*) AS row_count
                FROM {dataset}
                GROUP BY {group_col}
                ORDER BY row_count DESC
                LIMIT 10;
                """
            return f"SELECT COUNT(*) AS row_count FROM {dataset};"

        time_filter = self._extract_time_filter(dataset, resolved_time, query)
        if group_col:
            return f"""
            SELECT
                {group_col},
                ROUND(SUM({metric_col}), 2) AS total_{metric_col},
                ROUND(AVG({metric_col}), 2) AS avg_{metric_col},
                COUNT(*) AS row_count
            FROM {dataset}
            {time_filter}
            GROUP BY {group_col}
            ORDER BY total_{metric_col} DESC
            LIMIT 10;
            """

        return f"""
        SELECT
            ROUND(SUM({metric_col}), 2) AS total_{metric_col},
            ROUND(AVG({metric_col}), 2) AS avg_{metric_col},
            COUNT(*) AS row_count
        FROM {dataset}
        {time_filter};
        """

    def _fallback_generate(self, query: str, use_case: UseCase, datasets: list, resolved_time: str = None) -> dict:
        """Fallback SQL generator used when Groq is unavailable."""
        query_lower = query.lower()
        dataset = datasets[0] if datasets else "weekly_kpis"
        time_filter = self._extract_time_filter(dataset, resolved_time, query)

        regions = [r for r in ["North", "South", "East", "West", "Scotland", "Wales"] if r.lower() in query_lower]
        products = [
            p for p in [
                "Personal Current Account", "Savings Account", "Mortgage", "Credit Card", "Business Loan"
            ] if p.lower() in query_lower
        ]

        if dataset == "cost_breakdown":
            sql = f"""
            SELECT
                department,
                ROUND(SUM(cost_gbp), 2) AS total_cost_gbp,
                ROUND(100.0 * SUM(cost_gbp) / SUM(SUM(cost_gbp)) OVER (), 2) AS pct_of_total,
                ROUND(SUM(budget_gbp), 2) AS total_budget_gbp,
                ROUND(AVG(variance_pct), 2) AS avg_variance_pct
            FROM cost_breakdown
            {time_filter}
            GROUP BY department
            ORDER BY total_cost_gbp DESC
            LIMIT 10;
            """
        elif dataset == "regional_revenue":
            filters = [time_filter.replace("WHERE ", "", 1)] if time_filter else []
            if regions:
                region_list = ", ".join(f"'{region}'" for region in regions)
                filters.append(f"region IN ({region_list})")
            if products:
                product_list = ", ".join(f"'{product}'" for product in products)
                filters.append(f"product IN ({product_list})")
            where_clause = f"WHERE {' AND '.join(f for f in filters if f)}" if filters else ""
            group_col = "region" if "region" in query_lower or regions else "product"
            sql = f"""
            SELECT
                {group_col},
                ROUND(SUM(revenue), 2) AS total_revenue,
                SUM(transactions) AS total_transactions,
                ROUND(AVG(avg_ticket), 2) AS avg_ticket_gbp,
                ROUND(SUM(ad_spend), 2) AS total_ad_spend
            FROM regional_revenue
            {where_clause}
            GROUP BY {group_col}
            ORDER BY total_revenue DESC
            LIMIT 10;
            """
        elif dataset == "customer_metrics":
            metric_cols = []
            if "churn" in query_lower:
                metric_cols.extend([
                    "ROUND(AVG(churn_rate_pct), 2) AS churn_rate_pct",
                    "SUM(churned) AS churned_customers",
                ])
            if "complaint" in query_lower:
                metric_cols.append("SUM(complaints) AS complaints")
            if "nps" in query_lower:
                metric_cols.append("ROUND(AVG(nps_score), 2) AS nps_score")
            if not metric_cols:
                metric_cols = [
                    "SUM(new_signups) AS new_signups",
                    "ROUND(AVG(churn_rate_pct), 2) AS churn_rate_pct",
                    "SUM(complaints) AS complaints",
                    "ROUND(AVG(nps_score), 2) AS nps_score",
                ]
            sql = f"""
            SELECT
                month,
                {", ".join(metric_cols)}
            FROM customer_metrics
            {time_filter}
            GROUP BY month
            ORDER BY month DESC
            LIMIT 8;
            """
        elif dataset == "product_performance":
            filters = [time_filter.replace("WHERE ", "", 1)] if time_filter else []
            if products:
                product_list = ", ".join(f"'{product}'" for product in products)
                filters.append(f"product IN ({product_list})")
            where_clause = f"WHERE {' AND '.join(f for f in filters if f)}" if filters else ""
            sql = f"""
            SELECT
                product,
                ROUND(SUM(revenue), 2) AS total_revenue,
                SUM(transaction_volume) AS transaction_volume,
                ROUND(AVG(return_rate_pct), 2) AS return_rate_pct,
                ROUND(AVG(conversion_rate_pct), 2) AS conversion_rate_pct,
                ROUND(AVG(satisfaction_score), 2) AS satisfaction_score
            FROM product_performance
            {where_clause}
            GROUP BY product
            ORDER BY total_revenue DESC
            LIMIT 10;
            """
        else:
            sql = self._fallback_generate_generic(dataset, query, resolved_time)
            if not sql:
                sql = f"""
                SELECT
                    week,
                    new_signups,
                    ROUND(churn_rate_pct, 2) AS churn_rate_pct,
                    complaints,
                    nps_score,
                    ROUND(net_revenue_gbp, 2) AS net_revenue_gbp,
                    avg_handle_time_secs
                FROM weekly_kpis
                {time_filter}
                ORDER BY week DESC
                LIMIT 8;
                """

        clean_sql = self._clean_sql(sql)
        is_valid, error = self._validate_sql(clean_sql)

        if is_valid:
            logger.warning("Using local fallback SQL generator because Groq is unavailable")
            return {
                "sql": clean_sql,
                "valid": True,
                "error": None,
                "datasets_used": datasets,
                "metric_definitions": self._get_applied_metrics(clean_sql),
            }

        return {
            "sql": None,
            "valid": False,
            "error": error or "Fallback SQL generation failed",
            "datasets_used": datasets,
            "metric_definitions": [],
        }

    def _is_connection_error(self, error: Exception) -> bool:
        """Identify provider/network connectivity failures."""
        message = str(error).lower()
        return any(token in message for token in ["connection error", "timed out", "api connection", "network"])

    def generate(self, query: str, use_case: UseCase, datasets: list,
                 resolved_time: str = None, max_retries: int = 2) -> dict:
        """Generate and validate a DuckDB SQL query from natural language."""
        logger.info(f"Generating SQL | use_case={use_case.value}")
        # Because prompt ends with SELECT, prepend it back to response
        prompt = self._build_prompt(query, use_case, datasets, resolved_time)

        for attempt in range(max_retries + 1):
            try:
                response = _CLIENT.chat.completions.create(
                    model       = _MODEL,
                    messages    = [{"role": "user", "content": prompt}],
                    max_tokens  = 500,
                    temperature = 0,
                )
                raw_content = response.choices[0].message.content
                # Prompt ends with SELECT so prepend it if not present
                if not raw_content.strip().upper().startswith("SELECT"):
                    raw_content = "SELECT " + raw_content
                clean_sql   = self._clean_sql(raw_content)
                is_valid, error = self._validate_sql(clean_sql)

                if is_valid:
                    logger.info(f"SQL validated on attempt {attempt + 1}")
                    return {
                        "sql":                clean_sql,
                        "valid":              True,
                        "error":              None,
                        "datasets_used":      datasets,
                        "metric_definitions": self._get_applied_metrics(clean_sql),
                    }
                else:
                    logger.warning(f"Attempt {attempt + 1} failed: {error}")
                    if attempt < max_retries:
                        prompt += f"\n\nERROR: {error}. Fix it. Return ONLY corrected SQL starting with SELECT."

            except Exception as e:
                if self._is_connection_error(e):
                    logger.warning(f"Groq unavailable, switching to local SQL fallback: {e}")
                    fallback_result = self._fallback_generate(
                        query=query,
                        use_case=use_case,
                        datasets=datasets,
                        resolved_time=resolved_time,
                    )
                    if fallback_result["valid"]:
                        return fallback_result
                else:
                    logger.error(f"Groq error attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    fallback_result = self._fallback_generate(
                        query=query,
                        use_case=use_case,
                        datasets=datasets,
                        resolved_time=resolved_time,
                    )
                    if fallback_result["valid"]:
                        return fallback_result
                    return {"sql": None, "valid": False, "error": str(e),
                            "datasets_used": datasets, "metric_definitions": []}

        return {"sql": None, "valid": False,
                "error": f"Failed after {max_retries + 1} attempts",
                "datasets_used": datasets, "metric_definitions": []}

    def _get_applied_metrics(self, sql: str) -> list:
        try:
            with open(self.metrics_yaml_path, "r") as f:
                data = yaml.safe_load(f)
            sql_lower = sql.lower()
            return [
                defn["display_name"]
                for defn in data.get("metrics", {}).values()
                if defn.get("column", "").lower() in sql_lower
            ]
        except Exception:
            return []
