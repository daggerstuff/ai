"""Expose the fixtures declared in this package to pytest.

pytest only discovers fixtures from ``conftest.py``, so the ones declared in
``__init__.py`` were invisible to every module in this directory and each test
that requested one errored at setup with "fixture not found".
"""

from cli.tests import (
    cli_runner,
    mock_auth_manager,
    mock_config,
    mock_pipeline_manager,
    mock_progress_tracker,
    temp_config_dir,
)

__all__ = [
    "cli_runner",
    "mock_auth_manager",
    "mock_config",
    "mock_pipeline_manager",
    "mock_progress_tracker",
    "temp_config_dir",
]

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_cli_config_dir(tmp_path_factory):
    """Keep every CLI test off the real ``~/.pixelated-ai`` directory."""
    config_dir = tmp_path_factory.mktemp("cli-config")
    patcher = pytest.MonkeyPatch()
    patcher.setenv("PIXELATED_CONFIG_DIR", str(config_dir))
    yield config_dir
    patcher.undo()
