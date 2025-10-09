"""
Unit tests for utility functions in ai.dataset_pipeline.utils
"""

from ai.dataset_pipeline import utils


def test_ensure_dir(tmp_path):
    test_dir = tmp_path / "test_dir"
    utils.ensure_dir(str(test_dir))
    assert test_dir.exists()
    assert test_dir.is_dir()


def test_write_and_read_json(tmp_path):
    data = {"a": 1, "b": [2, 3]}
    file_path = tmp_path / "test.json"
    utils.write_json(str(file_path), data)
    loaded = utils.read_json(str(file_path))
    assert loaded == data
