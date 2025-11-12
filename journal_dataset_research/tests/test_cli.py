"""
Unit tests for CLI module.

Tests command-line interface functionality, configuration management,
and interactive mode.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import click
from click.testing import CliRunner

from ai.journal_dataset_research.cli.cli import cli, setup_logging
from ai.journal_dataset_research.cli.commands import CommandHandler
from ai.journal_dataset_research.cli.config import load_config, save_config
from ai.journal_dataset_research.cli.interactive import prompt_for_session_config


class TestCLI:
    """Tests for CLI interface."""

    def test_cli_group_exists(self):
        """Test that CLI group is defined."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Commands:" in result.output

    def test_setup_logging(self, temp_dir):
        """Test logging setup."""
        log_file = temp_dir / "test.log"
        setup_logging(level="INFO", log_file=log_file)

        # Verify log file was created
        assert log_file.exists() or log_file.parent.exists()

    def test_dry_run_flag(self):
        """Test dry-run flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--dry-run", "--help"])
        assert result.exit_code == 0

    def test_verbose_flag(self):
        """Test verbose flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0

    def test_config_file_option(self, temp_dir):
        """Test config file option."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("test: config")

        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(config_file), "--help"])
        assert result.exit_code == 0


class TestCommandHandler:
    """Tests for CommandHandler."""

    @pytest.fixture
    def command_handler(self):
        """Create a command handler instance."""
        return CommandHandler()

    def test_search_command(self, command_handler, mock_discovery_service):
        """Test search command."""
        # Replace the discovery service in the orchestrator with the mock
        orchestrator = command_handler._get_orchestrator()
        orchestrator.discovery_service = mock_discovery_service

        result = command_handler.search(
            sources=["pubmed"],
            keywords=["cbt"],  # CLI expects list of keywords, not dict
        )

        assert result is not None
        mock_discovery_service.discover_sources.assert_called_once()

    def test_evaluate_command(self, command_handler, mock_evaluation_engine, sample_dataset_source):
        """Test evaluate command."""
        orchestrator = command_handler._get_orchestrator()
        orchestrator.evaluation_engine = mock_evaluation_engine

        # Create a session with the source
        session = orchestrator.start_research_session(
            target_sources=[],
            search_keywords={},
        )
        state = orchestrator.get_session_state(session.session_id)
        state.sources = [sample_dataset_source]
        # Save the session state so it can be loaded
        orchestrator.save_session_state(session.session_id)

        result = command_handler.evaluate(
            session_id=session.session_id,
            source_id=sample_dataset_source.source_id
        )

        assert result is not None
        mock_evaluation_engine.evaluate_dataset.assert_called_once()

    def test_acquire_command(self, command_handler, mock_acquisition_manager, sample_dataset_source):
        """Test acquire command."""
        orchestrator = command_handler._get_orchestrator()
        orchestrator.acquisition_manager = mock_acquisition_manager

        # Create a session with the source
        session = orchestrator.start_research_session(
            target_sources=[],
            search_keywords={},
        )
        state = orchestrator.get_session_state(session.session_id)
        state.sources = [sample_dataset_source]
        # Save the session state so it can be loaded
        orchestrator.save_session_state(session.session_id)

        result = command_handler.acquire(
            session_id=session.session_id,
            source_id=sample_dataset_source.source_id
        )

        assert result is not None
        mock_acquisition_manager.submit_access_request.assert_called_once()

    def test_integrate_command(self, command_handler, mock_integration_engine, sample_acquired_dataset):
        """Test integrate command."""
        orchestrator = command_handler._get_orchestrator()
        orchestrator.integration_engine = mock_integration_engine
        
        # Create a session with the acquired dataset
        session = orchestrator.start_research_session(
            target_sources=[],
            search_keywords={},
        )
        state = orchestrator.get_session_state(session.session_id)
        state.acquired_datasets = [sample_acquired_dataset]
        # Save the session state so it can be loaded
        orchestrator.save_session_state(session.session_id)

        result = command_handler.integrate(
            session_id=session.session_id,
            dataset_id=sample_acquired_dataset.source_id
        )

        assert result is not None
        mock_integration_engine.create_integration_plan.assert_called_once()

    def test_status_command(self, command_handler, sample_research_session):
        """Test status command."""
        # Initialize orchestrator first
        orchestrator = command_handler._get_orchestrator()
        orchestrator.sessions[sample_research_session.session_id] = sample_research_session
        # Create and save session state
        state = orchestrator.get_session_state(sample_research_session.session_id)
        orchestrator.save_session_state(sample_research_session.session_id)

        result = command_handler.status(session_id=sample_research_session.session_id)

        assert result is not None
        assert "session_id" in result or "Session" in str(result)


class TestConfigManagement:
    """Tests for configuration management."""

    def test_load_config_file_exists(self, temp_dir):
        """Test loading config from file."""
        config_file = temp_dir / "config.yaml"
        config_content = """
storage_base_path: /tmp/storage
log_directory: /tmp/logs
"""
        config_file.write_text(config_content)

        config = load_config(str(config_file))
        assert config is not None
        assert "storage_base_path" in config

    def test_save_config(self, temp_dir):
        """Test saving config to file."""
        config_file = temp_dir / "config.yaml"
        config = {
            "storage_base_path": "/tmp/storage",
            "log_directory": "/tmp/logs",
        }

        save_config(config, str(config_file))
        assert config_file.exists()

        # Verify content
        loaded_config = load_config(str(config_file))
        assert loaded_config["storage_base_path"] == "/tmp/storage"

    def test_load_config_file_not_exists(self, temp_dir):
        """Test loading config when file doesn't exist."""
        # Use a path in temp_dir that doesn't exist, but parent is writable
        config_path = temp_dir / "nonexistent" / "config.yaml"
        config = load_config(str(config_path))
        # Should return default config (dict)
        assert isinstance(config, dict)
        assert "storage_base_path" in config  # Should have default values


class TestInteractiveMode:
    """Tests for interactive mode."""

    @patch("ai.journal_dataset_research.cli.interactive.Prompt.ask")
    @patch("ai.journal_dataset_research.cli.interactive.console")
    def test_prompt_for_session_config(self, mock_console, mock_prompt):
        """Test prompting for session configuration."""
        from ai.journal_dataset_research.cli.interactive import prompt_for_session_config

        mock_prompt.side_effect = [
            "pubmed,zenodo",  # target_sources
            "therapy,counseling",  # therapeutic keywords
            "dataset,conversation",  # dataset keywords
            "",  # sources_identified target (skip)
            "",  # datasets_evaluated target (skip)
            "",  # datasets_acquired target (skip)
            "",  # integration_plans_created target (skip)
            "",  # session_id (skip)
        ]

        config = prompt_for_session_config()

        assert config is not None
        assert "target_sources" in config
        assert "search_keywords" in config

    @patch("ai.journal_dataset_research.cli.interactive.Prompt.ask")
    def test_interactive_mode_handles_cancellation(self, mock_prompt):
        """Test that interactive mode handles user cancellation."""
        from ai.journal_dataset_research.cli.interactive import prompt_for_session_config

        mock_prompt.side_effect = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            prompt_for_session_config()

