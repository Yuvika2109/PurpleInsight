# 👾PurpleInsight - Trusted Conversational Intelligence for NatWest
<p align="center">
    <picture>
        <img src="https://github.com/user-attachments/assets/544c9e57-a67c-4ec3-8f95-1f6478e4fb62" 
             alt="Project Image" 
             width="700">
    </picture>
</p>

> *"Every answer must be explainable, auditable, and traceable — because in banking, trust isn't optional."*

---

## Overview

**PurpleInsight** is a self-service analytics platform built for NatWest analysts. A user types a business question in plain English — *"Why did revenue drop in the South last month?"* — and receives a clear answer, an appropriate chart, and a full audit trail showing exactly how that answer was produced.

The system is designed around NatWest's core requirements: **trust, auditability, and compliance**. Every response surfaces the SQL executed, the metric definitions applied, a confidence rating, and the data source — so no answer is ever a black box.

Intended users are non-technical NatWest business analysts who need fast, reliable answers from banking data without writing SQL or waiting for the data team.

---

## Features

All features listed below are implemented and working.

- **Natural Language to SQL** — type a plain-English question and receive a validated DuckDB SQL query and result
- **4 Use Case Intent Router** — queries are automatically classified into `change_analysis`, `compare`, `breakdown`, or `summarize`, each with a tailored prompt and output template
- **Semantic Metric Dictionary** — 20 NatWest banking metrics (revenue, churn rate, NPS, complaints, etc.) defined in `src/semantic/metrics.yaml`; every query resolves terms through this dictionary before hitting the LLM
- **Trust Trail Panel** — every answer shows: SQL executed, metric definitions applied, data source, confidence level (HIGH / MEDIUM / LOW with reason), and a thumbs up/down feedback button
- **Ambiguity Handler** — vague time references like "last month", "this quarter", "recently" are resolved into concrete date filters before SQL generation
- **Auto Chart Selection** — breakdown → stacked bar, compare → grouped bar or line, change analysis → waterfall, summarize → KPI cards; all styled in NatWest purple
- **Streamlit UI** — clean product interface with NatWest branding, quick-start query prompts, data explorer, and dataset registry page
- **FastAPI Backend** — programmatic access via `/query`, `/metrics`, `/health`, and `/feedback` endpoints
- **Dataset Registry** — new CSVs can be uploaded and registered through the UI; no code changes required
- **Zero Raw Data Exposure** — only aggregated query results are ever returned; row-level data never leaves the engine
- **LLM Fallback** — if the primary LLM is unavailable, deterministic local SQL generation keeps the demo running
- **23 Tests Passing** — covering intent routing, ambiguity handling, SQL generation, query engine, trust builder, and dataset registry

---

## Install and Run

### Prerequisites

- Python 3.10 or higher
- A Groq API key (free tier): https://console.groq.com

### Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd PurpleInsight
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

All other variables have working defaults and do not need to be changed.

### Step 5 — Run the Streamlit UI

```bash
PYTHONPATH=. ./venv/bin/streamlit run src/ui/app.py
```

The app opens at **http://localhost:8501**

### Step 6 — (Optional) Run the FastAPI backend

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs available at **http://localhost:8000/docs**

### Step 7 — (Optional) Run the tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| Analytics engine | DuckDB (in-memory SQL) |
| LLM — SQL generation | Groq (llama3-70b-8192) with local fallback |
| LLM — narrative | Groq with local fallback |
| Intent classification | Keyword matching + Sentence Transformers (all-MiniLM-L6-v2) |
| Charts | Plotly |
| Data handling | Pandas |
| Semantic layer | PyYAML |
| Testing | pytest |
| Config | python-dotenv |

---

## Usage Examples

### Streamlit UI

Start the app and try these queries on the **Analyse** page:

```
Why did revenue drop in the South region last month?
Compare complaints by region this quarter vs last quarter
Show the breakdown of costs by department for Q1-2024
Give me a weekly KPI summary for the last 4 weeks
Which product has the highest churn rate?
```

Each query returns a plain-English answer, an auto-selected chart, and a Trust Trail panel.

### API — Health check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "service": "DataTalk"}
```

### API — List all metric definitions

```bash
curl http://localhost:8000/metrics
```

Returns all 20 NatWest metric definitions from the semantic layer.

### API — Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me revenue by region for 2024"}'
```

Example response:

```json
{
  "query_id": "abc-123",
  "intent": "BREAKDOWN",
  "narrative": "The North region led revenue at £4.2M, followed by Scotland at £3.8M...",
  "sql": "SELECT region, ROUND(SUM(revenue), 2) AS total_revenue FROM regional_revenue WHERE month LIKE '2024%' GROUP BY region ORDER BY total_revenue DESC LIMIT 10",
  "chart_type": "stacked_bar",
  "trust_trail": {
    "sql_executed": "SELECT region, ROUND(SUM(revenue)...",
    "metrics_used": [{"name": "revenue", "definition": "Sum of all revenue...", "source_field": "revenue"}],
    "data_source": "regional_revenue",
    "confidence": "HIGH",
    "confidence_reason": "All metric terms resolved via semantic dictionary. SQL is straightforward with no ambiguity detected.",
    "row_count": 6,
    "ambiguity_flags": []
  }
}
```

### API — Record feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"query_id": "abc-123", "feedback": "positive"}'
```

---

## Architecture


<img width="403" height="893" alt="image" src="https://github.com/user-attachments/assets/a960deb2-18bb-488b-91bb-84d6fae5bdbf" />



---

## Semantic Layer

The metric dictionary at `src/semantic/metrics.yaml` is one of the most important files in the project. It defines 20 NatWest banking metrics with:

- Canonical definition
- Source column name
- Applicable datasets
- Aggregation method (SUM, AVG, COUNT)
- Aliases (so "income", "earnings", and "sales" all resolve to `revenue`)
- Valid dimension values for region, product, channel, segment

This means "revenue" always means the same thing regardless of who asks, which dataset is queried, or how the question is phrased — a direct requirement from the NatWest problem statement.

---

## Datasets

Five synthetic NatWest-aligned datasets are bundled in `data/raw/`:

| Dataset | Description | Primary Use Cases |
|---|---|---|
| `regional_revenue.csv` | Monthly revenue by region, product, channel | Change analysis, Compare |
| `customer_metrics.csv` | Monthly churn, signups, NPS, complaints by segment | Change analysis, Summarize |
| `product_performance.csv` | Weekly transaction volume, satisfaction, conversion | Compare, Breakdown |
| `cost_breakdown.csv` | Quarterly costs by department and category | Breakdown |
| `weekly_kpis.csv` | Weekly headline KPIs across all business metrics | Summarize |

New datasets can be registered through the **Dataset Registry** page in the app UI.

---

## Technical Depth

**Why DuckDB?** DuckDB runs SQL directly against CSV files in memory with no database server, no setup, and sub-second query times. This means the demo works immediately after `pip install` with no infrastructure.

**Why a Semantic Layer?** LLMs hallucinate column names and metric definitions. By injecting the canonical definitions from `metrics.yaml` into every prompt, we force the model to use the correct column (`churn_rate_pct`, not `churn`) and the correct aggregation (`AVG`, not `SUM`). This is the difference between a toy demo and something production-ready.

**Why Sentence Transformers for intent routing?** Keyword matching handles 85% of queries fast and offline. For ambiguous queries like "how are we doing?", a local `all-MiniLM-L6-v2` model computes semantic similarity against use-case anchors. This runs entirely on-device — no extra API calls.

**Why a Trust Trail?** NatWest is a regulated bank. Analysts need to know *why* an answer was produced, not just what it says. The trust trail is not a UI decoration — it is an audit log that shows the SQL, the metric definitions applied, and the confidence tier with a human-readable justification.

---

## Limitations

- Ambiguity resolution handles date-related phrases; product and region ambiguity detection is not yet implemented
- The LLM fallback SQL covers the most common query patterns; highly complex multi-join queries may not always validate correctly
- Feedback recorded via `/feedback` is stored in-memory only; it is not persisted to a database between restarts
- The dataset registry stores file metadata in `config/datasets.yaml`; schema inference is automatic but may need manual correction for non-standard column names
- The sentence-transformer model (`all-MiniLM-L6-v2`) is downloaded on first run — this requires an internet connection on first launch

---

## Future Improvements

- Persist user feedback to a database and use it to fine-tune prompt templates over time
- Add product and region disambiguation to the ambiguity handler
- Support multi-turn conversations so users can ask follow-up questions in context
- Add a scheduled job to refresh datasets from source systems automatically
- Extend the trust trail with a lineage graph showing which source rows contributed to the answer
- Add role-based access control so different analyst tiers see only permitted datasets

---

## Repository Structure

```
PurpleInsight/
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI entrypoint
│   │   └── routes.py             # API endpoints
│   ├── core/
│   │   ├── ambiguity_handler.py  # Time reference resolution
│   │   ├── intent_router.py      # Query classification
│   │   ├── nl_to_sql.py          # LLM SQL generation
│   │   ├── query_engine.py       # DuckDB execution
│   │   ├── narrative.py          # Plain-English answers
│   │   ├── chart_selector.py     # Plotly chart selection
│   │   └── trust_builder.py      # Trust trail construction
│   ├── semantic/
│   │   └── metrics.yaml          # Canonical metric definitions
│   └── ui/
│       ├── app.py                # Streamlit main app
│       └── components/           # UI components
├── config/
│   ├── settings.py               # Central configuration
│   ├── datasets.yaml             # Dataset registry
│   └── dataset_registry.py      # Registry loader
├── data/
│   ├── raw/                      # Five bundled CSV datasets
│   └── schema/
│       └── data_dictionary.yaml
├── tests/                        # 23 passing tests
├── scripts/
│   ├── generate_synthetic_data.py
│   └── validate_datasets.py
├── assets/
│   └── screenshots/
├── docs/
├── .env.example
├── LICENSE
├── requirements.txt
└── README.md
```

---

## License

Apache License 2.0 — see `LICENSE` for details.

All commits are signed off in accordance with the Developer Certificate of Origin (DCO).
