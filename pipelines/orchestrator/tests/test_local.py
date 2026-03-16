"""
Unit tests for ai.pipelines.orchestrator.local_loader
"""

import pandas as pd

from ai.pipelines.orchestrator import local_loader


def test_load_local_csv_success(tmp_path):
    # Create a small CSV file
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df.to_csv(csv_path, index=False)
    progress_messages = []
    loaded = local_loader.load_local_csv(str(csv_path), progress_callback=progress_messages.append)
    assert loaded is not None
    assert loaded.equals(df)
    assert any("Starting local CSV load" in msg for msg in progress_messages)
    assert any("Loaded local dataset" in msg for msg in progress_messages)


def test_load_local_csv_not_found(tmp_path):
    # Try to load a non-existent file
    missing_path = tmp_path / "missing.csv"
    error_messages = []
    loaded = local_loader.load_local_csv(str(missing_path), error_callback=error_messages.append)
    assert loaded is None
    assert any("Local dataset not found" in msg for msg in error_messages)
