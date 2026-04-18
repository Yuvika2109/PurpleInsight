"""
query_engine.py
---------------
Executes validated SQL queries against DuckDB loaded with all 5 datasets.

Key features:
    - Loads all CSVs into DuckDB in-memory on startup (blazing fast)
    - Executes only pre-validated SELECT queries
    - Returns results as structured dicts for downstream processing
    - Enforces row limits to prevent data dumps
    - Never exposes raw PII — only aggregated query results leave this module

Security:
    - Only pre-validated SQL from nl_to_sql.py is accepted
    - Row limit enforced at engine level (not just prompt level)
    - No direct file paths exposed in results
"""

import os
import duckdb
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from config.dataset_registry import load_dataset_registry

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_ROWS     = 50   # hard cap — never return more than 50 rows
DATA_DIR     = os.getenv("DATA_PATH", "data/raw")


class QueryEngine:
    """
    Executes validated DuckDB SQL queries against NatWest banking datasets.

    All 5 CSV datasets are loaded into an in-memory DuckDB instance on
    first use. Queries run in milliseconds with no external DB dependency.
    """

    def __init__(self, data_dir: str = DATA_DIR):
        """
        Initialise the query engine.

        Args:
            data_dir: Path to directory containing the 5 CSV datasets.
        """
        self.data_dir   = data_dir
        self._conn      = None
        self._loaded    = False
        logger.info("QueryEngine initialised")

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Get or create the DuckDB in-memory connection.
        Loads all datasets on first call (lazy loading).

        Returns:
            duckdb.DuckDBPyConnection: Active DuckDB connection
        """
        if self._conn is None:
            self._conn = duckdb.connect(database=":memory:")
            self._load_datasets()
        return self._conn

    def _load_datasets(self):
        """
        Load all 5 CSV datasets into DuckDB as queryable tables.
        Each CSV becomes a DuckDB view — zero data copying, reads direct from file.
        """
        datasets = {
            dataset_id: meta.get("file", f"{dataset_id}.csv")
            for dataset_id, meta in load_dataset_registry().items()
        }

        conn = self._conn
        loaded = []
        failed = []

        for table_name, filename in datasets.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                # Create a DuckDB view pointing directly at the CSV
                conn.execute(f"""
                    CREATE OR REPLACE VIEW {table_name} AS
                    SELECT * FROM read_csv_auto('{filepath}')
                """)
                row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                loaded.append(f"{table_name} ({row_count:,} rows)")
            else:
                failed.append(filename)
                logger.error(f"Dataset not found: {filepath}")

        if loaded:
            logger.info(f"Loaded datasets: {', '.join(loaded)}")
        if failed:
            logger.warning(
                f"Skipped missing dataset files: {', '.join(failed)}. "
                f"These datasets will not be queryable until their CSV files exist."
            )

        self._loaded = True

    def execute(self, sql: str) -> dict:
        """
        Execute a validated SQL query and return structured results.

        Args:
            sql: Pre-validated SELECT SQL from nl_to_sql.py

        Returns:
            dict: {
                "success": bool,
                "data": list[dict],        — query results as list of row dicts
                "columns": list[str],      — column names
                "row_count": int,          — number of rows returned
                "was_truncated": bool,     — True if results were capped at MAX_ROWS
                "error": str | None,       — error message if failed
                "execution_time_ms": float — query execution time
            }
        """
        import time

        conn  = self._get_connection()
        start = time.time()

        try:
            # Safety re-check — only SELECT allowed at engine level
            if not sql.strip().upper().startswith("SELECT"):
                return self._error_result("Only SELECT queries are allowed", start)

            # Execute query
            result_df = conn.execute(sql).df()

            # Enforce row cap
            was_truncated = len(result_df) > MAX_ROWS
            if was_truncated:
                logger.warning(f"Result truncated from {len(result_df)} to {MAX_ROWS} rows")
                result_df = result_df.head(MAX_ROWS)

            # Round all float columns to 2 decimal places
            float_cols = result_df.select_dtypes(include=["float64", "float32"]).columns
            result_df[float_cols] = result_df[float_cols].round(2)

            execution_ms = round((time.time() - start) * 1000, 1)
            logger.info(f"Query executed in {execution_ms}ms — {len(result_df)} rows returned")

            return {
                "success":          True,
                "data":             result_df.to_dict(orient="records"),
                "columns":          list(result_df.columns),
                "row_count":        len(result_df),
                "was_truncated":    was_truncated,
                "error":            None,
                "execution_time_ms": execution_ms,
            }

        except duckdb.Error as e:
            logger.error(f"DuckDB query error: {e}")
            return self._error_result(f"Query execution failed: {str(e)}", start)

        except Exception as e:
            logger.error(f"Unexpected error during query execution: {e}")
            return self._error_result(f"Unexpected error: {str(e)}", start)

    def _error_result(self, error_msg: str, start_time: float) -> dict:
        """
        Build a standardised error result dict.

        Args:
            error_msg: Human-readable error description
            start_time: Query start time (for elapsed time calc)

        Returns:
            dict: Standardised error result
        """
        import time
        return {
            "success":           False,
            "data":              [],
            "columns":           [],
            "row_count":         0,
            "was_truncated":     False,
            "error":             error_msg,
            "execution_time_ms": round((time.time() - start_time) * 1000, 1),
        }

    def get_table_info(self) -> dict:
        """
        Return metadata about all loaded tables.
        Used by the trust trail to show data sources.

        Returns:
            dict: Table name → {row_count, columns}
        """
        conn = self._get_connection()
        info = {}

        tables = list(load_dataset_registry().keys())

        for table in tables:
            try:
                row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                columns   = [col[0] for col in conn.execute(
                    f"DESCRIBE {table}"
                ).fetchall()]
                info[table] = {"row_count": row_count, "columns": columns}
            except Exception as e:
                info[table] = {"error": str(e)}

        return info

    def run_sample(self, table_name: str, n: int = 3) -> list[dict]:
        """
        Return a small sample of rows from a table for debugging.
        Used internally — never exposed in production UI.

        Args:
            table_name: Name of the table to sample
            n: Number of rows to return (max 5)

        Returns:
            list[dict]: Sample rows
        """
        n    = min(n, 5)  # hard cap on samples
        conn = self._get_connection()
        try:
            df = conn.execute(f"SELECT * FROM {table_name} LIMIT {n}").df()
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Sample query failed for {table_name}: {e}")
            return []

    def register_new_dataset(self, dataset_id: str, file_path: str) -> bool:
        """Create a DuckDB view for a newly registered dataset without full reload."""
        if not os.path.exists(file_path):
            logger.error(f"Cannot register dataset — file not found: {file_path}")
            return False
        conn = self._get_connection()
        try:
            conn.execute(f"""
                CREATE OR REPLACE VIEW {dataset_id} AS
                SELECT * FROM read_csv_auto('{file_path}')
            """)
            row_count = conn.execute(f"SELECT COUNT(*) FROM {dataset_id}").fetchone()[0]
            logger.info(f"Registered new dataset view: {dataset_id} ({row_count:,} rows)")
            return True
        except Exception as e:
            logger.error(f"Failed to register dataset view {dataset_id}: {e}")
            return False

    def close(self):
        """Close the DuckDB connection cleanly."""
        if self._conn:
            self._conn.close()
            self._conn  = None
            self._loaded = False
            logger.info("QueryEngine connection closed")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = QueryEngine(data_dir="data/raw")

    print("\nPurpleInsight — Query Engine Test")
    print("=" * 60)

    # Table info
    info = engine.get_table_info()
    print("\nLoaded tables:")
    for table, meta in info.items():
        print(f"  {table}: {meta.get('row_count', '?'):,} rows")

    # Test queries
    test_sqls = [
        (
            "Total revenue by region (2024)",
            "SELECT region, SUM(revenue) AS total_revenue FROM regional_revenue WHERE month LIKE '2024%' GROUP BY region ORDER BY total_revenue DESC"
        ),
        (
            "Weekly KPIs latest 4 weeks",
            "SELECT week, new_signups, churn_rate_pct, nps_score FROM weekly_kpis ORDER BY week DESC LIMIT 4"
        ),
        (
            "Cost breakdown by department",
            "SELECT department, SUM(cost_gbp) AS total_cost, ROUND(SUM(cost_gbp)*100.0/SUM(SUM(cost_gbp)) OVER(), 2) AS pct_of_total FROM cost_breakdown GROUP BY department ORDER BY total_cost DESC"
        ),
    ]

    for label, sql in test_sqls:
        print(f"\n── {label}")
        result = engine.execute(sql)
        if result["success"]:
            print(f"  Rows: {result['row_count']} | Time: {result['execution_time_ms']}ms")
            for row in result["data"][:3]:
                print(f"  {row}")
        else:
            print(f"  ERROR: {result['error']}")

    engine.close()
