#!/usr/bin/env python3
"""
Entry point for the journal dataset research CLI.

This can be used as a console script entry point in setup.py or pyproject.toml.
"""

import sys

from ai.journal_dataset_research.cli.cli import cli

if __name__ == "__main__":
    sys.exit(cli())

