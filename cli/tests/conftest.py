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
