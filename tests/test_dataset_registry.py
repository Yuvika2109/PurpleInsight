from __future__ import annotations

import pandas as pd

from config.dataset_registry import (
    build_schema_prompt_block,
    get_dataset_profile,
    list_registered_datasets,
    register_dataset,
)


def test_register_and_profile_dataset(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    registry_path = tmp_path / "datasets.yaml"
    csv_path = data_dir / "branch_service_quality.csv"

    pd.DataFrame(
        {
            "week": ["2025-W01", "2025-W02"],
            "branch": ["London", "Manchester"],
            "satisfaction_score": [4.3, 4.1],
            "complaints": [12, 18],
        }
    ).to_csv(csv_path, index=False)

    register_dataset(
        dataset_id="branch_service_quality",
        display_name="Branch Service Quality",
        description="Weekly branch service metrics",
        category="Operations",
        file_name="branch_service_quality.csv",
        primary_use_cases=["compare", "summarize"],
        registry_path=registry_path,
    )

    items = list_registered_datasets(data_dir=data_dir, registry_path=registry_path)
    assert len(items) == 1
    assert items[0]["dataset_id"] == "branch_service_quality"
    assert items[0]["exists"] is True

    profile = get_dataset_profile(
        "branch_service_quality",
        data_dir=data_dir,
        registry_path=registry_path,
    )
    assert profile is not None
    assert "complaints" in profile["numeric_columns"]
    assert "branch" in profile["categorical_columns"]
    assert "week" in profile["date_like_columns"]

    prompt_block = build_schema_prompt_block("branch_service_quality", profile)
    assert "Table: branch_service_quality" in prompt_block
    assert "satisfaction_score" in prompt_block
