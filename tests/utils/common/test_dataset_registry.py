from pathlib import Path
from unittest.mock import mock_open, patch

from utils.common.dataset_registry import (
    DatasetRef,
    _iter_registry_groups,
    get_default_registry_path,
    iter_dataset_refs,
    load_registry,
    resolve_fallback_path,
)


def test_get_default_registry_path():
    path = get_default_registry_path()
    assert isinstance(path, Path)
    assert path.name == "dataset_registry.json"
    assert path.parent.name == "data"


@patch("builtins.open", new_callable=mock_open, read_data='{"datasets": {}}')
def test_load_registry(mock_file):
    registry = load_registry()
    assert registry == {"datasets": {}}
    mock_file.assert_called_once_with(get_default_registry_path(), encoding="utf-8")


@patch("builtins.open", new_callable=mock_open, read_data='{"datasets": {"section": {}}}')
def test_load_registry_custom_path(mock_file):
    custom_path = Path("/custom/path.json")
    registry = load_registry(custom_path)
    assert registry == {"datasets": {"section": {}}}
    mock_file.assert_called_once_with(custom_path, encoding="utf-8")


def test_iter_registry_groups():
    registry = {
        "datasets": {
            "section1": {"ds1": {}},
        },
        "edge_case_sources": {"ds2": {}},
        "voice_persona": {"ds3": {}},
        "other": {"ignored": {}},
    }

    groups = list(_iter_registry_groups(registry))
    expected_groups_len = 3
    assert len(groups) == expected_groups_len

    names = [name for name, _ in groups]
    assert "section1" in names
    assert "edge_case_sources" in names
    assert "voice_persona" in names


def test_iter_registry_groups_missing():
    registry = {}
    groups = list(_iter_registry_groups(registry))
    assert len(groups) == 0


def test_iter_dataset_refs():
    registry = {
        "datasets": {
            "section1": {
                "valid1": {
                    "path": "s3://bucket/ds1",
                    "stage": "prod",
                    "quality_profile": "high",
                    "type": "audio",
                    "focus": "speech",
                    "fallback_paths": {"local": "/local/ds1", "gdrive": "gdrive://ds1"},
                    "legacy_paths": ["/legacy/ds1"],
                },
                "non_s3": {
                    "path": "/local/only",
                },
                "not_a_dict": "ignore_me",
            }
        },
        "edge_case_sources": {
            "valid2": {
                "path": "s3://bucket/ds2",
                "fallback_paths": "not_a_dict",  # Should be handled
                "legacy_paths": "not_a_list",  # Should be handled
            }
        },
    }

    refs = list(iter_dataset_refs(registry))
    expected_refs_len = 2
    assert len(refs) == expected_refs_len

    # Check valid1
    ref1 = next(r for r in refs if r.key == "section1.valid1")
    assert ref1.s3_path == "s3://bucket/ds1"
    assert ref1.stage == "prod"
    assert ref1.quality_profile == "high"
    assert ref1.type == "audio"
    assert ref1.focus == "speech"
    assert ref1.fallback_paths == {"local": "/local/ds1", "gdrive": "gdrive://ds1"}
    assert ref1.legacy_paths == ["/legacy/ds1"]

    # Check valid2
    ref2 = next(r for r in refs if r.key == "edge_case_sources.valid2")
    assert ref2.s3_path == "s3://bucket/ds2"
    assert ref2.stage is None
    assert ref2.fallback_paths == {}
    assert ref2.legacy_paths == []


def test_resolve_fallback_path():
    dataset = DatasetRef(
        key="test",
        s3_path="s3://test",
        stage=None,
        quality_profile=None,
        type=None,
        focus=None,
        fallback_paths={"local": "/path/local", "gdrive": "/path/gdrive", "custom": "/path/custom"},
        legacy_paths=[],
    )

    # Default preference (local, then gdrive)
    assert resolve_fallback_path(dataset) == "/path/local"

    # Custom preference
    assert resolve_fallback_path(dataset, prefer=["gdrive", "local"]) == "/path/gdrive"
    assert resolve_fallback_path(dataset, prefer=["custom"]) == "/path/custom"

    # Preference not found, fallback to alphabetical
    assert resolve_fallback_path(dataset, prefer=["missing"]) == "/path/custom"


def test_resolve_fallback_path_no_fallbacks():
    dataset = DatasetRef(
        key="test",
        s3_path="s3://test",
        stage=None,
        quality_profile=None,
        type=None,
        focus=None,
        fallback_paths={},
        legacy_paths=[],
    )
    assert resolve_fallback_path(dataset) is None
