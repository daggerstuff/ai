"""
CLI interface for journal dataset research system.
"""

__all__ = ["cli", "load_config", "save_config"]

from ai.sourcing.journal.cli.cli import cli
from ai.sourcing.journal.cli.config import load_config, save_config

