"""
CLI interface for journal dataset research system.
"""

__all__ = ["cli", "load_config", "save_config"]

from ai.pipelines.data_processing.journal.cli.cli import cli
from ai.pipelines.data_processing.journal.cli.config import load_config, save_config
