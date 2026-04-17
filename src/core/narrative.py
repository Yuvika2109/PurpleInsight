"""
narrative.py
------------
Converts DuckDB query results into clear, plain-English narratives
using Groq

Key features:
    - Use-case-specific narrative templates matching NatWest problem doc
    - Jargon-free language for non-technical users and leadership
    - Structured output: headline + narrative + key facts
    - Source references included in every narrative
    - Anomaly flagging when values fall outside expected ranges
"""

import os
import yaml
from loguru import logger
from dotenv import load_dotenv
from groq import Groq

from src.core.intent_router import UseCase

load_dotenv()

# ── Groq client ───────────────────────────────────────────────────────────────
_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
# ── Narrative templates per use case ─────────────────────────────────────────
NARRATIVE_TEMPLATES = {
    UseCase.CHANGE_ANALYSIS: """You are a NatWest banking analyst explaining data to a non-technical audience.

Data:
{data}

Question: "{query}"

Write a clear explanation in this EXACT format (no deviations):

HEADLINE: [One sentence — what changed and by how much]

EXPLANATION: [2-3 sentences explaining the main drivers. Use simple language. Include specific numbers. Example: "Revenue decreased by 11% in Feb. The biggest contributor was a 22% drop in the South region due to reduced ad spend."]

KEY FACTS:
• [Specific data point with number]
• [Specific data point with number]
• [Specific data point with number]

DATA SOURCE: [Dataset name(s) used]

Rules: No jargon. Specific numbers only. HEADLINE under 20 words. Keep it concise.""",

    UseCase.COMPARE: """You are a NatWest banking analyst explaining a comparison to a non-technical audience.

Data:
{data}

Question: "{query}"

Write a clear comparison in this EXACT format:

HEADLINE: [One sentence — which performed better and by how much]

COMPARISON: [2-3 sentences comparing the two. Example: "Product A grew by 8% WoW, outperforming Product B (+2%). Primary reason: higher return customer rate."]

KEY DIFFERENCES:
• [Metric 1: value A vs value B — difference]
• [Metric 2: value A vs value B — difference]
• [Metric 3: value A vs value B — difference]

DATA SOURCE: [Dataset name(s) used]

Rules: Be direct. State direction (up/down, higher/lower). HEADLINE under 20 words.""",

    UseCase.BREAKDOWN: """You are a NatWest banking analyst explaining a breakdown to a non-technical audience.

Data:
{data}

Question: "{query}"

Write a clear breakdown in this EXACT format:

HEADLINE: [One sentence about the biggest contributor]

BREAKDOWN: [2-3 sentences explaining the composition. Example: "North region accounts for 40% of total sales, with Retail contributing most of that share."]

TOP CONTRIBUTORS:
• [#1: value and % of total]
• [#2: value and % of total]
• [#3: value and % of total]

NOTABLE PATTERN: [One sentence about any outlier or surprise]

DATA SOURCE: [Dataset name(s) used]

Rules: Always mention percentages. Highlight dominant categories. HEADLINE under 20 words.""",

    UseCase.SUMMARIZE: """You are a NatWest banking analyst writing a weekly digest for leadership.

Data:
{data}

Question: "{query}"

Write a concise summary in this EXACT format:

HEADLINE: [One sentence — most important thing that happened]

THIS PERIOD:
• Signups: [value and change vs previous]
• Churn: [value — stable/improving/worsening]
• NPS: [value and trend]
• Revenue: [value and change]
• Handle time: [value and change]

KEY TAKEAWAY: [One sentence for leadership — the single most important thing]

DATA SOURCE: [Dataset name(s) used]

Rules: Executive briefing tone. Plain numbers with units. Example: "Signups grew by 5%, churn remained stable, handle time improved by 12 seconds." """,

    UseCase.UNKNOWN: """You are a NatWest banking analyst answering a data question clearly.

Data:
{data}

Question: "{query}"

Answer clearly:

ANSWER: [Direct answer in 2-3 sentences with specific numbers]

KEY FACTS:
• [Fact with number]
• [Fact with number]

DATA SOURCE: [Dataset name(s) used]

Rules: No jargon. Include specific numbers. Answer the question first.""",
}


class Narrative:
    """Converts query results into plain-English banking narratives using Groq."""

    def __init__(self, metrics_yaml_path: str = "src/semantic/metrics.yaml"):
        self.metrics_yaml_path = metrics_yaml_path
        logger.info("Narrative generator initialised")

    def _format_data(self, data: list, columns: list) -> str:
        """Format query results into a clean table string for the prompt."""
        if not data:
            return "No data returned."
        lines = [" | ".join(str(c) for c in columns)]
        lines.append("-" * len(lines[0]))
        for row in data[:15]:
            lines.append(" | ".join(
                f"£{v:,.2f}" if isinstance(v, float) and any(
                    w in str(k).lower() for w in ["revenue", "cost", "gbp", "spend"]
                )
                else f"{v:,.0f}" if isinstance(v, (int, float)) and v > 1000
                else str(round(v, 2)) if isinstance(v, float)
                else str(v)
                for k, v in row.items()
            ))
        return "\n".join(lines)

    def _format_value(self, key: str, value) -> str:
        """Render values for fallback narratives."""
        if isinstance(value, float):
            if any(token in key.lower() for token in ["revenue", "cost", "gbp", "spend", "budget"]):
                return f"GBP {value:,.2f}"
            if "pct" in key.lower() or "rate" in key.lower() or "variance" in key.lower():
                return f"{value:.2f}%"
            return f"{value:.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    def _is_connection_error(self, error: Exception) -> bool:
        """Identify provider/network connectivity failures."""
        message = str(error).lower()
        return any(token in message for token in ["connection error", "timed out", "api connection", "network"])

    def _generate_fallback_narrative(self, query: str, use_case: UseCase, query_result: dict) -> dict:
        """Generate a simple local narrative when Groq is unavailable."""
        data = query_result.get("data", [])
        datasets_used = ", ".join(query_result.get("datasets_used", []))
        anomalies = self._detect_anomalies(data)

        if not data:
            return {
                "headline": "No data found for this query.",
                "narrative": "The query returned no results. Try rephrasing or broadening your question.",
                "key_facts": [],
                "data_source": datasets_used or "N/A",
                "anomalies": anomalies,
                "success": False,
            }

        top_row = data[0]
        dims = [key for key, value in top_row.items() if not isinstance(value, (int, float, float))]
        metrics = [key for key, value in top_row.items() if isinstance(value, (int, float))]

        lead_dim = dims[0] if dims else None
        lead_metric = metrics[0] if metrics else None
        lead_label = top_row.get(lead_dim, "latest result") if lead_dim else "latest result"
        lead_value = self._format_value(lead_metric, top_row.get(lead_metric)) if lead_metric else "available"

        if use_case == UseCase.BREAKDOWN and lead_dim and lead_metric:
            headline = f"{lead_label} is the largest contributor."
            narrative = (
                f"HEADLINE: {headline}\n\n"
                f"BREAKDOWN: Based on the available data, {lead_label} leads this breakdown with "
                f"{lead_value}. The results below are generated locally because the external LLM "
                f"service is currently unavailable.\n\n"
                "TOP CONTRIBUTORS:\n"
            )
        elif use_case == UseCase.COMPARE and lead_dim and lead_metric:
            headline = f"{lead_label} leads on {lead_metric.replace('_', ' ')}."
            narrative = (
                f"HEADLINE: {headline}\n\n"
                f"COMPARISON: The top result is {lead_label} with {lead_value}. "
                f"This comparison was summarised locally because the external LLM service "
                f"is currently unavailable.\n\n"
                "KEY DIFFERENCES:\n"
            )
        elif use_case == UseCase.SUMMARIZE:
            headline = "Latest KPI snapshot is ready."
            narrative = (
                f"HEADLINE: {headline}\n\n"
                "THIS PERIOD:\n"
            )
        else:
            headline = f"{lead_label} stands out in the current results." if lead_dim else "Analysis complete."
            narrative = (
                f"HEADLINE: {headline}\n\n"
                f"ANSWER: The current result set highlights {lead_label}"
                f"{' at ' + lead_value if lead_metric else ''}. "
                "This explanation was generated locally because the external LLM service is currently unavailable.\n\n"
                "KEY FACTS:\n"
            )

        key_facts = []
        for row in data[:3]:
            parts = []
            for key, value in row.items():
                parts.append(f"{key.replace('_', ' ')}: {self._format_value(key, value)}")
            fact = " | ".join(parts)
            key_facts.append(fact)
            narrative += f"• {fact}\n"

        narrative += f"\nDATA SOURCE: {datasets_used or 'N/A'}"

        return {
            "headline": headline,
            "narrative": narrative,
            "key_facts": key_facts,
            "data_source": datasets_used or "N/A",
            "anomalies": anomalies,
            "success": True,
        }

    def generate(self, query: str, use_case: UseCase, query_result: dict,
                 metric_definitions: list = None) -> dict:
        """
        Generate a plain-English narrative from query results.

        Args:
            query: Original user question
            use_case: Classified use case
            query_result: Result dict from QueryEngine.execute()
            metric_definitions: Metric definitions applied (for trust trail)

        Returns:
            dict with headline, narrative, key_facts, data_source, anomalies, success
        """
        if not query_result.get("success") or not query_result.get("data"):
            return {
                "headline":   "No data found for this query.",
                "narrative":  "The query returned no results. Try rephrasing or broadening your question.",
                "key_facts":  [],
                "data_source": "N/A",
                "anomalies":  [],
                "success":    False,
            }

        template = NARRATIVE_TEMPLATES.get(use_case, NARRATIVE_TEMPLATES[UseCase.UNKNOWN])
        data_str = self._format_data(query_result["data"], query_result["columns"])
        prompt   = template.format(data=data_str, query=query)

        try:
            response = _CLIENT.chat.completions.create(
                model       = _MODEL,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 800,
                temperature = 0.3,
            )
            raw_text = response.choices[0].message.content.strip()

            headline  = self._extract_section(raw_text, "HEADLINE")
            data_src  = self._extract_section(raw_text, "DATA SOURCE")
            key_facts = self._extract_bullets(raw_text)
            anomalies = self._detect_anomalies(query_result["data"])

            logger.info(f"Narrative generated | use_case={use_case.value}")

            return {
                "headline":    headline or "Analysis complete.",
                "narrative":   raw_text,
                "key_facts":   key_facts,
                "data_source": data_src or ", ".join(query_result.get("datasets_used", [])),
                "anomalies":   anomalies,
                "success":     True,
            }

        except Exception as e:
            if self._is_connection_error(e):
                logger.warning(f"Groq unavailable, switching to local narrative fallback: {e}")
            else:
                logger.error(f"Narrative generation failed: {e}")
                logger.warning("Using local fallback narrative because Groq is unavailable")
            return self._generate_fallback_narrative(query, use_case, query_result)

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from the narrative text."""
        for i, line in enumerate(text.split("\n")):
            if line.strip().startswith(f"{section_name}:"):
                content = line.split(":", 1)[-1].strip()
                return content if content else ""
        return ""

    def _extract_bullets(self, text: str) -> list:
        """Extract bullet points from the narrative text."""
        bullets = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("•") or stripped.startswith("-"):
                bullet = stripped.lstrip("•-").strip()
                if bullet and len(bullet) > 5:
                    bullets.append(bullet)
        return bullets[:5]

    def _detect_anomalies(self, data: list) -> list:
        """Flag statistical anomalies in query results."""
        anomalies = []
        for row in data:
            for col, val in row.items():
                if not isinstance(val, (int, float)):
                    continue
                col_lower = col.lower()
                if "churn" in col_lower and "pct" in col_lower and val > 5.0:
                    anomalies.append(f"Warning: High churn at {val:.1f}% (typical range: 1–3%)")
                if "nps" in col_lower and val < 20:
                    anomalies.append(f"Warning: Low NPS at {val} (target: 40+)")
                if "variance" in col_lower and abs(val) > 15:
                    direction = "overspend" if val > 0 else "underspend"
                    anomalies.append(f"Warning: Budget {direction} at {val:+.1f}%")
                if "change" in col_lower and "pct" in col_lower and val < -20:
                    anomalies.append(f"Warning: Significant drop of {val:.1f}%")
        return list(set(anomalies))[:3]
