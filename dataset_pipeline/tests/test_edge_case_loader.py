"""
Unit tests for ai.dataset_pipeline.edge_case_loader
"""

import pandas as pd

from ai.dataset_pipeline import edge_case_loader


def test_load_edge_case_csv_success(tmp_path):
    # Create a small CSV file for edge cases
    csv_path = tmp_path / "edge_cases.csv"
    df = pd.DataFrame({"case": ["empty", "long"], "value": ["", "x" * 10000]})
    df.to_csv(csv_path, index=False)
    loaded = edge_case_loader.load_edge_case_csv(str(csv_path))
    assert loaded is not None
    assert loaded.equals(df)


def test_load_edge_case_csv_not_found(tmp_path):
    # Try to load a non-existent file
    missing_path = tmp_path / "missing_edge_cases.csv"
    loaded = edge_case_loader.load_edge_case_csv(str(missing_path))
    assert loaded is None
