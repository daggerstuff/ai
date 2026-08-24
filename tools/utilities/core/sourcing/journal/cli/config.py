"""Compatibility wrapper for legacy ``ai.tools.utilities.core.sourcing.journal.cli.config`` imports."""

from __future__ import annotations

from typing import Any

from ai.pipelines.data_processing.journal.cli.config import (
    ConfigManager,
    get_config_value,
    load_config,
    save_config,
)


class Config(ConfigManager):
    """Compatibility shim for legacy callers that imported ``Config``."""

    def __init__(self, config_path=None):
        super().__init__(config_path=config_path)

    # Keep an explicit ``load`` method to satisfy expected interface
    def load(self) -> dict[str, Any]:
        return load_config(self.config_path)


__all__ = ["Config", "ConfigManager", "get_config_value", "load_config", "save_config"]
