"""
Test suite for ai/dataset_pipeline/load.py

Covers:
- Successful dataset loading (file/dir, all supported formats)
- Error handling (missing/empty/invalid files, unsupported types)
- Config override logic
- Logging and edge cases (partial loads, warnings)
- No secrets or env vars hardcoded

TDD anchors: Red phase (tests must fail until implemented)
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Union
from unittest import mock

from ai.dataset_pipeline import load

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test datasets."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)

def make_csv(path: Path, data: List[Dict[str, Any]]):
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

def make_json(path: Path, data: List[Dict[str, Any]]):
    df = pd.DataFrame(data)
    df.to_json(path, orient="records")

def make_jsonl(path: Path, data: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(pd.Series(rec).to_json(force_ascii=False) + "\n")

def make_parquet(path: Path, data: List[Dict[str, Any]]):
    df = pd.DataFrame(data)
    df.to_parquet(path)

def test_load_single_csv_file(temp_data_dir):
    file = temp_data_dir / "test.csv"
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    make_csv(file, data)
    config = {"test": str(file)}
    result = load.load_datasets(config)
    assert "test" in result
    assert isinstance(result["test"], pd.DataFrame)
    assert result["test"].shape == (2, 2)
    assert set(result["test"].columns) == {"a", "b"}

def test_load_multiple_formats_in_dir(temp_data_dir):
    # Create files of all supported types
    make_csv(temp_data_dir / "a.csv", [{"x": 1}])
    make_json(temp_data_dir / "b.json", [{"y": 2}])
    make_jsonl(temp_data_dir / "c.jsonl", [{"z": 3}])
    make_parquet(temp_data_dir / "d.parquet", [{"w": 4}])
    config = {"mixed": str(temp_data_dir)}
    result = load.load_datasets(config)
    assert "mixed" in result
    dfs = result["mixed"]
    assert isinstance(dfs, list)
    assert len(dfs) == 4
    # Each should be a DataFrame
    for df in dfs:
        assert isinstance(df, pd.DataFrame)

def test_missing_dataset_path_raises(temp_data_dir):
    config = {"missing": str(temp_data_dir / "nope")}
    with pytest.raises(load.DatasetLoaderError):
        load.load_datasets(config)

def test_empty_directory_raises(temp_data_dir):
    config = {"empty": str(temp_data_dir)}
    with pytest.raises(load.DatasetLoaderError):
        load.load_datasets(config)

def test_empty_file_raises(temp_data_dir):
    file = temp_data_dir / "empty.csv"
    file.touch()
    config = {"emptyfile": str(file)}
    with pytest.raises(load.DatasetLoaderError):
        load.load_datasets(config)

def test_unsupported_file_type_raises(temp_data_dir):
    file = temp_data_dir / "foo.txt"
    file.write_text("not a dataset")
    config = {"bad": str(file)}
    with pytest.raises(load.DatasetLoaderError):
        load.load_datasets(config)

def test_partial_load_warns_and_skips(temp_data_dir):
    # One good, one bad file
    make_csv(temp_data_dir / "good.csv", [{"a": 1}])
    bad = temp_data_dir / "bad.txt"
    bad.write_text("bad")
    config = {"mixed": str(temp_data_dir)}
    # Patch logger to capture warnings
    with mock.patch.object(load.logger, "warning") as warn_mock:
        result = load.load_datasets(config)
        assert "mixed" in result
        assert any("Skipping file" in str(call) for call in warn_mock.call_args_list)

def test_logger_override_and_custom_config(temp_data_dir):
    make_csv(temp_data_dir / "foo.csv", [{"a": 1}])
    custom_logger = mock.Mock()
    config = {"foo": str(temp_data_dir / "foo.csv")}
    result = load.load_datasets(config, logger_override=custom_logger)
    # Should use the custom logger for info
    assert custom_logger.info.called

def test_default_config_loads(monkeypatch):
    # Patch _default_dataset_config to point to a temp dir with a valid file
    with tempfile.TemporaryDirectory() as d:
        file = Path(d) / "d.csv"
        make_csv(file, [{"a": 1}])
        monkeypatch.setattr(load, "_default_dataset_config", lambda: {"d": str(file)})
        result = load.load_datasets()
        assert "d" in result

# TDD anchor: Add more edge case tests for symlinks, permission errors, and logging if needed.