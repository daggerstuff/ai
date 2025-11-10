"""
CLI interface for journal dataset research system.
"""

__all__ = ["cli", "load_config", "save_config"]

from ai.journal_dataset_research.cli.cli import cli
from ai.journal_dataset_research.cli.config import load_config, save_config

