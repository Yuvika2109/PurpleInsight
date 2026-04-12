"""
Configuration and environment settings for DataTalk.
Loads all environment variables and provides a central settings object.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration for the DataTalk application."""

    # LLM
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))

    # DuckDB / Data
    DATA_DIR: str = os.getenv("DATA_DIR", "data/raw")
    DB_PATH: str = os.getenv("DB_PATH", ":memory:")

    # Semantic layer
    METRICS_YAML_PATH: str = os.getenv("METRICS_YAML_PATH", "data/schema/data_dictionary.yaml")

    # Trust layer
    CONFIDENCE_HIGH_THRESHOLD: float = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.85"))
    CONFIDENCE_LOW_THRESHOLD: float = float(os.getenv("CONFIDENCE_LOW_THRESHOLD", "0.50"))

    # App
    APP_TITLE: str = "DataTalk — Trusted Intelligence for NatWest"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()