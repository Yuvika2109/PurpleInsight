"""Helpers for loading and maintaining the dataset registry."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = ROOT / "config" / "datasets.yaml"
DEFAULT_DATA_DIR = ROOT / "data" / "raw"


def load_dataset_registry(registry_path: str | os.PathLike | None = None) -> dict[str, dict]:
    """Load dataset metadata from the central registry."""
    path = Path(registry_path or DEFAULT_REGISTRY_PATH)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload.get("datasets", {})


def list_registered_datasets(
    data_dir: str | os.PathLike | None = None,
    registry_path: str | os.PathLike | None = None,
) -> list[dict]:
    """Return registry items with resolved file paths and availability flags."""
    registry = load_dataset_registry(registry_path=registry_path)
    base_dir = Path(data_dir or DEFAULT_DATA_DIR)
    items: list[dict] = []

    for dataset_id, meta in registry.items():
        file_name = meta.get("file", f"{dataset_id}.csv")
        file_path = base_dir / file_name
        items.append(
            {
                "dataset_id": dataset_id,
                "file": file_name,
                "file_path": str(file_path),
                "display_name": meta.get("display_name", dataset_id.replace("_", " ").title()),
                "description": meta.get("description", ""),
                "category": meta.get("category", "General"),
                "primary_use_cases": meta.get("primary_use_cases", []),
                "exists": file_path.exists(),
            }
        )

    return items


def slugify_dataset_id(raw_name: str) -> str:
    """Convert a display name into a safe dataset id."""
    slug = re.sub(r"[^a-z0-9]+", "_", raw_name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "dataset"


def register_dataset(
    *,
    dataset_id: str,
    display_name: str,
    description: str,
    category: str,
    file_name: str,
    primary_use_cases: list[str],
    registry_path: str | os.PathLike | None = None,
) -> str:
    """Create or update a dataset entry in the registry."""
    path = Path(registry_path or DEFAULT_REGISTRY_PATH)
    payload = {"datasets": load_dataset_registry(registry_path=path)}
    payload["datasets"][dataset_id] = {
        "file": file_name,
        "display_name": display_name,
        "description": description,
        "category": category,
        "primary_use_cases": primary_use_cases,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    return str(path)


def get_dataset_profile(
    dataset_id: str,
    data_dir: str | os.PathLike | None = None,
    registry_path: str | os.PathLike | None = None,
    sample_rows: int = 50,
) -> dict | None:
    """Inspect a registered dataset and return a lightweight schema profile."""
    items = list_registered_datasets(data_dir=data_dir, registry_path=registry_path)
    match = next((item for item in items if item["dataset_id"] == dataset_id), None)
    if not match or not match["exists"]:
        return None

    df = pd.read_csv(match["file_path"], nrows=sample_rows)
    columns = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    date_like_columns: list[str] = []

    for column in df.columns:
        series = df[column]
        dtype = str(series.dtype)
        sample = "" if series.empty else str(series.iloc[0])
        columns.append(
            {
                "name": column,
                "dtype": dtype,
                "sample": sample,
                "unique_values": int(series.nunique(dropna=True)),
            }
        )

        lower_name = column.lower()
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        elif any(token in lower_name for token in ["date", "day", "week", "month", "year", "quarter", "time"]):
            date_like_columns.append(column)
        else:
            categorical_columns.append(column)

        if lower_name not in [c.lower() for c in date_like_columns] and any(
            token in lower_name for token in ["date", "day", "week", "month", "year", "quarter", "time"]
        ):
            date_like_columns.append(column)

    return {
        "dataset_id": dataset_id,
        "display_name": match["display_name"],
        "description": match["description"],
        "category": match["category"],
        "primary_use_cases": match["primary_use_cases"],
        "file_path": match["file_path"],
        "columns": columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_like_columns": date_like_columns,
    }


def build_schema_prompt_block(dataset_id: str, profile: dict) -> str:
    """Convert a dataset profile into prompt-friendly schema text."""
    lines = [
        f"Table: {dataset_id}",
        f"Description: {profile.get('description', '')}",
        "Columns:",
    ]
    for column in profile.get("columns", []):
        sample = f" sample='{column['sample']}'" if column.get("sample") else ""
        lines.append(f"    {column['name']} {column['dtype']}{sample}")
    if profile.get("date_like_columns"):
        lines.append(f"Date-like columns: {', '.join(profile['date_like_columns'])}")
    if profile.get("categorical_columns"):
        lines.append(f"Categorical columns: {', '.join(profile['categorical_columns'][:8])}")
    if profile.get("numeric_columns"):
        lines.append(f"Numeric columns: {', '.join(profile['numeric_columns'][:8])}")
    return "\n".join(lines)
