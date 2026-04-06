"""
Tests for dataset Asana sync service.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ai.pipelines.orchestrator.orchestration.dataset_asana_sync_service import (
    DatasetAsanaSyncService,
)
from ai.pipelines.orchestrator.systems.dataset_inventory import get_dataset_metadata


class MockConfig:
    enable_dataset_asana_sync = True
    asana_project_gid = "1234567890"
    asana_dataset_section_gid = "987654321"
    asana_dataset_task_gid_output_path = "/tmp/test_task_gid.txt"
    asana_dataset_mapping_output_path = "/tmp/test_task_mapping.json"
    dataset_scan_directory = "/tmp/test_datasets"
    dataset_file_patterns = [".csv", ".json"]
    dataset_task_prefix = "DATASET"


class MockStats:
    def __init__(self) -> None:
        self.warnings: list[str] = []


@pytest.fixture
def mock_asana_client():
    return Mock()


@pytest.fixture
def service(mock_asana_client):
    return DatasetAsanaSyncService(
        config=MockConfig(),
        stats=MockStats(),
        asana_client=mock_asana_client,
    )


def test_is_valid_gid(service):
    """Test GID validation."""
    assert service._is_valid_gid("12345") is True
    assert service._is_valid_gid("0") is True
    assert service._is_valid_gid(None) is False
    assert service._is_valid_gid("") is False
    assert service._is_valid_gid("abc") is False


def test_generate_dataset_task_key(service):
    """Test dataset task key generation."""
    dataset_meta = {
        "name": "test_dataset.csv",
        "path": "/data/test_dataset.csv",
        "size_bytes": 1024,
        "last_modified": 1234567890,
        "columns": ["col1", "col2"],
    }
    task_key = service._generate_dataset_task_key(dataset_meta)
    assert task_key == "DATASET-TEST-DATASET"


def test_build_dataset_task_payload(service):
    """Test building Asana task payload from dataset metadata."""
    dataset_meta = {
        "name": "test.csv",
        "path": "/data/test.csv",
        "size_bytes": 2048,
        "last_modified": 1234567890,
        "columns": ["id", "name", "value"],
        "preview_rows": [{"id": 1, "name": "test", "value": 100}],
    }
    payload = service._build_dataset_task_payload(dataset_meta, "DATASET-TEST")

    assert "DATASET test.csv" in payload["name"]  # Note: prefix is DATASET not DATASET-TEST
    assert "test.csv" in payload["notes"]
    assert "/data/test.csv" in payload["notes"]
    size_mb = dataset_meta["size_bytes"] / (1024 * 1024)
    assert f"{size_mb:.2f} MB" in payload["notes"]
    assert "Columns: 3" in payload["notes"]


@patch("ai.pipelines.orchestrator.orchestration.dataset_asana_sync_service.scan_datasets")
def test_scan_and_filter_datasets(mock_scan, service):
    """Test dataset scanning and filtering."""
    # Set the scan directory to match what we'll return
    service.config.dataset_scan_directory = "/tmp/test_datasets"
    # Mock the scan_datasets function to return our test data
    # Pass empty pattern to get all files (our implementation uses pattern="" to get all)
    mock_scan.return_value = [
        {
            "path": "/tmp/test_datasets/dataset1.csv",
            "name": "dataset1.csv",
            "size_bytes": 1024,
            "last_modified": 1234567890,
            "columns": ["a", "b"],
        },
        {
            "path": "/tmp/test_datasets/dataset2.json",
            "name": "dataset2.json",
            "size_bytes": 2048,
            "last_modified": 1234567891,
            "columns": ["x", "y", "z"],
        },
        {
            "path": "/tmp/test_datasets/readme.txt",
            "name": "readme.txt",
            "size_bytes": 512,
            "last_modified": 1234567892,
            "columns": [],
        },
    ]

    datasets = service._scan_and_filter_datasets()
    assert len(datasets) == 2  # Only .csv and .json files
    assert datasets[0]["name"] == "dataset1.csv"
    assert datasets[1]["name"] == "dataset2.json"


def test_load_dataset_task_mapping_file_not_found(service, tmp_path):
    """Test loading task mapping when file doesn't exist."""
    service.config.asana_dataset_mapping_output_path = str(tmp_path / "nonexistent.json")
    mapping = service._load_dataset_task_mapping("12345")
    assert mapping == {}


def test_load_dataset_task_mapping_invalid_json(service, tmp_path):
    """Test loading task mapping with invalid JSON."""
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text("invalid json")
    service.config.asana_dataset_mapping_output_path = str(mapping_file)
    mapping = service._load_dataset_task_mapping("12345")
    assert mapping == {}


def test_load_dataset_task_mapping_valid(service, tmp_path):
    """Test loading valid task mapping."""
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"TEST-123": "98765"}))
    service.config.asana_dataset_mapping_output_path = str(mapping_file)
    mapping = service._load_dataset_task_mapping("12345")
    assert mapping == {"TEST-123": "98765"}


def test_save_dataset_task_mapping(service, tmp_path):
    """Test saving task mapping."""
    mapping_file = tmp_path / "mapping.json"
    service.config.asana_dataset_mapping_output_path = str(mapping_file)
    test_mapping = {"DATASET-ABC": "11111", "DATASET-XYZ": "22222"}
    service._save_dataset_task_mapping(test_mapping, "12345")
    assert mapping_file.exists()
    content = json.loads(mapping_file.read_text())
    assert content == {"DATASET-ABC": "11111", "DATASET-XYZ": "22222"}


def test_sync_dataset_inventory_disabled(service):
    """Test sync is skipped when disabled."""
    service.config.enable_dataset_asana_sync = False
    service.asana_client.request.return_value = {"gid": "123"}
    # Should not call any Asana methods
    service.sync_dataset_inventory()
    service.asana_client.request.assert_not_called()


def test_sync_dataset_inventory_invalid_project_gid(service):
    """Test sync is skipped when project GID is invalid."""
    service.config.asana_project_gid = "invalid"
    service.sync_dataset_inventory()
    assert len(service.stats.warnings) > 0
    assert "ASANA_PROJECT_GID missing or invalid" in service.stats.warnings[0]


@patch("ai.pipelines.orchestrator.systems.dataset_inventory.scan_datasets")
def test_sync_dataset_inventory_no_datasets(mock_scan, service):
    """Test sync when no datasets found."""
    mock_scan.return_value = []
    service.sync_dataset_inventory()
    assert len(service.stats.warnings) > 0
    assert "No datasets found to sync to Asana" in service.stats.warnings[0]


@patch("ai.pipelines.orchestrator.systems.dataset_inventory.scan_datasets")
def test_sync_dataset_inventory_success(mock_scan, service, mock_asana_client):
    """Test successful dataset sync."""
    # Set scan directory to match what we'll return
    service.config.dataset_scan_directory = "/tmp/test_datasets"
    # Mock dataset scan
    mock_scan.return_value = [
        {
            "path": "/tmp/test_datasets/test.csv",
            "name": "test.csv",
            "size_bytes": 1024,
            "last_modified": 1234567890,
            "columns": ["col1", "col2"],
            "preview_rows": [],
        }
    ]

    # Mock Asana responses
    mock_asana_client.request.side_effect = [
        # First call: get tasks for mapping (returns empty list)
        [],
        # Second call: create task
        {"gid": "task123"},
        # Third call: add story
        {},
    ]

    service.sync_dataset_inventory()

    # Verify Asana calls were made
    assert mock_asana_client.request.call_count >= 2
    # Check that task was created with correct parameters
    create_call = mock_asana_client.request.call_args_list[1]
    assert create_call[0][0] == "POST"  # method
    assert create_call[0][1] == "/tasks"  # path
    assert "test.csv" in create_call[1]["name"]  # payload contains dataset name

    # Verify task mapping was updated
    assert service.stats.warnings == []  # No warnings on success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
