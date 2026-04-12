"""
intent_router.py
----------------
Classifies incoming natural language queries into one of four use cases:
    1. CHANGE_ANALYSIS  — "Why did revenue drop last month?"
    2. COMPARE          — "This week vs last week / Region A vs Region B"
    3. BREAKDOWN        — "What makes up total sales? / Cost by department"
    4. SUMMARIZE        — "Give me a weekly summary of customer metrics"

Routing strategy:
    Step 1 — Keyword pre-filter using metrics.yaml use_case_keywords
    Step 2 — Semantic similarity using sentence-transformers (local, no API cost)
    Step 3 — Confidence scoring — if low confidence, flags for clarification

This two-step approach ensures speed (keyword fast path) with accuracy
(semantic fallback) without burning LLM API tokens on classification.
"""

import os
import yaml
from enum import Enum
from dataclasses import dataclass
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config.dataset_registry import list_registered_datasets


# ── Use case enum ─────────────────────────────────────────────────────────────
class UseCase(str, Enum):
    CHANGE_ANALYSIS = "change_analysis"
    COMPARE         = "compare"
    BREAKDOWN       = "breakdown"
    SUMMARIZE       = "summarize"
    UNKNOWN         = "unknown"


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class IntentResult:
    """Result of intent classification for a single query."""
    use_case: UseCase
    confidence: float          # 0.0 – 1.0
    method: str                # "keyword" or "semantic"
    matched_keywords: list     # keywords that triggered keyword match
    needs_clarification: bool  # True if confidence < threshold


# ── Confidence threshold ──────────────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.45

# ── Semantic anchor queries per use case ─────────────────────────────────────
# These are representative example queries the model compares against.
# Chosen to cover the exact examples from the NatWest problem doc.
SEMANTIC_ANCHORS = {
    UseCase.CHANGE_ANALYSIS: [
        "Why did revenue drop last month?",
        "What caused customer complaints to rise?",
        "What drove the increase in churn?",
        "Why did sales decrease in the South region?",
        "What is behind the fall in NPS score?",
        "Explain the spike in costs last quarter",
        "What changed in product performance this week?",
    ],
    UseCase.COMPARE: [
        "This week vs last week performance",
        "Compare Region A vs Region B",
        "How does Product X compare to Product Y?",
        "Show me the difference between North and South revenue",
        "Personal Current Account vs Savings Account",
        "How did Q1 2024 perform versus Q1 2023?",
        "Which region is doing better this month?",
    ],
    UseCase.BREAKDOWN: [
        "What makes up total sales?",
        "Show the breakdown of costs by department",
        "Decompose revenue by product and channel",
        "What are the biggest contributors to churn?",
        "Break down customer complaints by segment",
        "Show cost composition for Technology department",
        "What proportion of revenue comes from each region?",
    ],
    UseCase.SUMMARIZE: [
        "Give me a weekly summary for customer metrics",
        "What happened this week?",
        "Monthly overview of key metrics",
        "Summarise performance for this quarter",
        "How are we doing this week?",
        "Give me a digest of today's numbers",
        "Weekly report for leadership",
    ],
}


class IntentRouter:
    """
    Routes natural language queries to the appropriate analytical use case.

    Uses a two-step approach:
        1. Fast keyword match from metrics.yaml
        2. Semantic similarity fallback using sentence-transformers
    """

    def __init__(self, metrics_yaml_path: str = "src/semantic/metrics.yaml"):
        """
        Initialise the intent router.

        Args:
            metrics_yaml_path: Path to the metrics.yaml semantic layer file.
        """
        self.metrics_yaml_path = metrics_yaml_path
        self._keywords = self._load_keywords()
        self._model = None          # lazy loaded — heavy, load only when needed
        self._anchor_embeddings = {}

        logger.info("IntentRouter initialised")

    def _load_keywords(self) -> dict:
        """
        Load use case keyword hints from metrics.yaml.

        Returns:
            dict: Mapping of use_case → list of trigger keywords
        """
        try:
            with open(self.metrics_yaml_path, "r") as f:
                metrics = yaml.safe_load(f)
            keywords = metrics.get("use_case_keywords", {})
            logger.debug(f"Loaded keywords for {len(keywords)} use cases")
            return keywords
        except FileNotFoundError:
            logger.warning(f"metrics.yaml not found at {self.metrics_yaml_path}, using defaults")
            return {}

    def _load_semantic_model(self):
        """
        Lazy-load the sentence transformer model.
        Only called if keyword matching fails — avoids startup delay.
        """
        if self._model is None:
            logger.info("Loading sentence transformer model (first time only)...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

            # Pre-compute anchor embeddings for all use cases
            for use_case, anchors in SEMANTIC_ANCHORS.items():
                self._anchor_embeddings[use_case] = self._model.encode(anchors)

            logger.info("Sentence transformer model loaded and anchors encoded")

    def _keyword_match(self, query: str) -> tuple[UseCase, float, list]:
        """
        Fast keyword-based intent classification.

        Args:
            query: Lowercased user query string

        Returns:
            tuple: (UseCase, confidence_score, matched_keywords)
                   Returns (UNKNOWN, 0.0, []) if no match found
        """
        query_lower = query.lower()
        scores = {}

        for use_case_str, keywords in self._keywords.items():
            matched = [kw for kw in keywords if kw.lower() in query_lower]
            if matched:
                # Score = matched keywords / total keywords (normalised)
                scores[use_case_str] = (len(matched), matched)

        if not scores:
            return UseCase.UNKNOWN, 0.0, []

        # Pick use case with most keyword matches
        best = max(scores.items(), key=lambda x: x[1][0])
        use_case_str, (count, matched_kws) = best

        # Confidence: 0.6 base + 0.1 per additional keyword match (max 0.95)
        confidence = min(0.6 + (count - 1) * 0.1, 0.95)

        return UseCase(use_case_str), confidence, matched_kws

    def _semantic_match(self, query: str) -> tuple[UseCase, float]:
        """
        Semantic similarity-based intent classification using sentence transformers.

        Args:
            query: User query string

        Returns:
            tuple: (UseCase, confidence_score)
        """
        self._load_semantic_model()

        query_embedding = self._model.encode([query])
        best_use_case   = UseCase.UNKNOWN
        best_score      = 0.0

        for use_case, anchor_embeddings in self._anchor_embeddings.items():
            # Compare query against all anchors, take the best match
            similarities = cosine_similarity(query_embedding, anchor_embeddings)[0]
            max_sim      = float(np.max(similarities))

            if max_sim > best_score:
                best_score    = max_sim
                best_use_case = use_case

        return best_use_case, best_score

    def classify(self, query: str) -> IntentResult:
        """
        Classify a natural language query into one of 4 use cases.

        Strategy:
            1. Try keyword match first (fast, cheap)
            2. If no keyword match, fall back to semantic similarity
            3. Flag low-confidence results for clarification

        Args:
            query: Natural language query from the user

        Returns:
            IntentResult: Classification result with use case, confidence, and metadata
        """
        if not query or not query.strip():
            return IntentResult(
                use_case=UseCase.UNKNOWN,
                confidence=0.0,
                method="none",
                matched_keywords=[],
                needs_clarification=True,
            )

        logger.debug(f"Classifying query: '{query[:80]}...'")

        # Step 1 — keyword match
        kw_use_case, kw_confidence, matched_kws = self._keyword_match(query)

        if kw_use_case != UseCase.UNKNOWN and kw_confidence >= LOW_CONFIDENCE_THRESHOLD:
            logger.debug(f"Keyword match → {kw_use_case} (confidence={kw_confidence:.2f})")
            return IntentResult(
                use_case=kw_use_case,
                confidence=kw_confidence,
                method="keyword",
                matched_keywords=matched_kws,
                needs_clarification=kw_confidence < LOW_CONFIDENCE_THRESHOLD,
            )

        # Step 2 — semantic fallback
        logger.debug("Keyword match insufficient, falling back to semantic similarity")
        sem_use_case, sem_confidence = self._semantic_match(query)

        logger.debug(f"Semantic match → {sem_use_case} (confidence={sem_confidence:.2f})")

        return IntentResult(
            use_case=sem_use_case,
            confidence=sem_confidence,
            method="semantic",
            matched_keywords=matched_kws,
            needs_clarification=sem_confidence < LOW_CONFIDENCE_THRESHOLD,
        )

    def get_dataset_hint(self, result: IntentResult, query: str) -> list[str]:
        """
        Suggest which datasets are most relevant for a given intent and query.

        Args:
            result: IntentResult from classify()
            query: Original user query

        Returns:
            list[str]: Ordered list of dataset names most relevant to the query
        """
        registered = list_registered_datasets()
        if not registered:
            return ["weekly_kpis"]

        query_lower = query.lower()
        use_case_name = result.use_case.value
        scored: list[tuple[int, str]] = []

        for dataset in registered:
            score = 0
            dataset_id = dataset["dataset_id"]
            haystacks = [
                dataset_id.replace("_", " "),
                dataset.get("display_name", ""),
                dataset.get("description", ""),
                dataset.get("category", ""),
                " ".join(dataset.get("primary_use_cases", [])),
            ]

            if use_case_name in dataset.get("primary_use_cases", []):
                score += 5
            if dataset.get("category", "").lower() in query_lower:
                score += 3

            for haystack in haystacks:
                for token in set(haystack.lower().replace("_", " ").split()):
                    if len(token) > 3 and token in query_lower:
                        score += 1

            if any(w in query_lower for w in ["cost", "department", "budget", "spend", "headcount"]) and "cost" in dataset_id:
                score += 4
            if any(w in query_lower for w in ["product", "mortgage", "credit card", "savings"]) and "product" in dataset_id:
                score += 4
            if any(w in query_lower for w in ["region", "north", "south", "east", "west", "scotland", "wales"]) and "region" in dataset_id:
                score += 4
            if any(w in query_lower for w in ["customer", "churn", "signup", "complaint", "nps"]) and "customer" in dataset_id:
                score += 4
            if any(w in query_lower for w in ["weekly", "week", "kpi", "summary", "overview"]) and "kpi" in dataset_id:
                score += 4

            scored.append((score, dataset_id))

        ranked = [dataset_id for score, dataset_id in sorted(scored, key=lambda item: (-item[0], item[1])) if score > 0]
        if ranked:
            return ranked

        defaults = [
            dataset["dataset_id"]
            for dataset in registered
            if use_case_name in dataset.get("primary_use_cases", [])
        ]
        return defaults or [registered[0]["dataset_id"]]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    router = IntentRouter(metrics_yaml_path="src/semantic/metrics.yaml")

    test_queries = [
        "Why did revenue drop last month?",
        "Compare North vs South region performance",
        "Show the breakdown of costs by department",
        "Give me a weekly summary for customer metrics",
        "What caused customer complaints to rise in Q3?",
        "Product A vs Product B this week",
        "What makes up total sales in 2024?",
        "How are we doing this week?",
    ]

    print("\nPurpleInsight — Intent Router Test")
    print("=" * 60)
    for q in test_queries:
        result = router.classify(q)
        datasets = router.get_dataset_hint(result, q)
        print(f"\nQuery   : {q}")
        print(f"UseCase : {result.use_case.value}")
        print(f"Confidence: {result.confidence:.2f} ({result.method})")
        print(f"Datasets: {datasets[:2]}")
        if result.matched_keywords:
            print(f"Keywords: {result.matched_keywords}")
