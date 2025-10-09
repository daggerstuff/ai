"""
Test package for Pixelated AI CLI.

This package contains comprehensive tests for all CLI components including:
- Configuration management
- Authentication
- Command execution
- Progress tracking
- Error handling
- Integration tests
"""

import pytest
from pathlib import Path
import tempfile
import os
from typing import Dict, Any


# Test configuration
TEST_CONFIG = {
    "api_base_url": "http://localhost:8000",
    "timeout": 30,
    "max_retries": 3,
    "log_level": "DEBUG",
    "profiles": {
        "test": {
            "api_base_url": "http://localhost:8000",
            "auth_endpoint": "/auth/token",
            "pipeline_endpoint": "/api/v1/pipelines",
            "timeout": 30,
            "max_retries": 3
        }
    }
}


@pytest.fixture
def temp_config_dir():
    """Create a temporary configuration directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / ".pixelated"
        config_dir.mkdir(exist_ok=True)
        yield config_dir


@pytest.fixture
def mock_config(temp_config_dir):
    """Create a mock configuration for testing"""
    from cli.config import CLIConfig
    
    config_file = temp_config_dir / "config.yaml"
    config_file.write_text("""
api_base_url: http://localhost:8000
timeout: 30
max_retries: 3
log_level: DEBUG
profiles:
  test:
    api_base_url: http://localhost:8000
    auth_endpoint: /auth/token
    pipeline_endpoint: /api/v1/pipelines
    timeout: 30
    max_retries: 3
""")
    
    return CLIConfig(config_file=config_file, profile="test")


@pytest.fixture
def mock_auth_manager(mock_config):
    """Create a mock authentication manager for testing"""
    from cli.auth import AuthManager
    
    auth_manager = AuthManager(mock_config)
    # Mock successful authentication
    auth_manager._access_token = "test_access_token"
    auth_manager._refresh_token = "test_refresh_token"
    auth_manager._token_expiry = 9999999999  # Far future
    
    return auth_manager


@pytest.fixture
def mock_pipeline_manager(mock_config, mock_auth_manager):
    """Create a mock pipeline manager for testing"""
    from cli.pipeline import PipelineManager
    
    return PipelineManager(mock_config, mock_auth_manager)


@pytest.fixture
def mock_progress_tracker(mock_config):
    """Create a mock progress tracker for testing"""
    from cli.progress import ProgressTracker
    
    return ProgressTracker(mock_config)


@pytest.fixture
def cli_runner():
    """Create a CLI test runner"""
    from click.testing import CliRunner
    
    return CliRunner()


class MockResponse:
    """Mock HTTP response for testing"""
    
    def __init__(self, json_data: Dict[str, Any] = None, status_code: int = 200, text: str = ""):
        self.json_data = json_data or {}
        self.status_code = status_code
        self.text = text or ""
        self.ok = 200 <= status_code < 300
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if not self.ok:
            raise Exception(f"HTTP {self.status_code}: {self.text}")


def assert_command_success(result, expected_output: str = None):
    """Assert that a CLI command executed successfully"""
    assert result.exit_code == 0, f"Command failed with exit code {result.exit_code}: {result.output}"
    if expected_output:
        assert expected_output in result.output


def assert_command_failure(result, expected_error: str = None):
    """Assert that a CLI command failed as expected"""
    assert result.exit_code != 0, "Command should have failed but succeeded"
    if expected_error:
        assert expected_error in result.output


def create_test_config_file(path: Path, content: str = None):
    """Create a test configuration file"""
    if content is None:
        content = """
api_base_url: http://localhost:8000
timeout: 30
max_retries: 3
log_level: DEBUG
profiles:
  test:
    api_base_url: http://localhost:8000
    auth_endpoint: /auth/token
    pipeline_endpoint: /api/v1/pipelines
    timeout: 30
    max_retries: 3
"""
    path.write_text(content)
    return path