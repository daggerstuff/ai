"""Compatibility wrapper for the pipelines data-processing journal CLI group."""

from __future__ import annotations

from ai.pipelines.data_processing.journal.cli.cli import cli, setup_logging

__all__ = ["cli", "setup_logging"]
