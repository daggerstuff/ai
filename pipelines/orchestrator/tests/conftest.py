"""
Pytest configuration and shared fixtures for dataset pipeline tests.
"""

import pytest


@pytest.fixture(scope="session")
def sample_json(tmp_path_factory):
    """Fixture for creating a sample JSON file."""
    file_path = tmp_path_factory.mktemp("data") / "sample.json"
    file_path.write_text('{"foo": "bar", "num": 42}', encoding="utf-8")
    return str(file_path)
