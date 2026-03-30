"""
Integration configuration helpers for tracker-backed runtime defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

DEFAULT_INTERNAL_CONFIG_PATH = Path(".agent/internal/config.json")


class IntegrationConfigResolver:
    """Resolve tracker-related defaults from the internal integration config."""

    def __init__(self, *, config_path: Path | None = None) -> None:
        self.config_path = config_path

    def resolve_training_asana_project_gid(self) -> str | None:
        payload = self._load_internal_config()
        integration = payload.get("integration")
        if not isinstance(integration, Mapping):
            return None

        asana = integration.get("asana")
        if not isinstance(asana, Mapping):
            return None

        all_projects = asana.get("all_projects")
        if isinstance(all_projects, Mapping):
            for project_key in (
                "master_training_gap_closure",
                "active_sprint",
                "master_training_epic",
            ):
                project_gid = all_projects.get(project_key)
                if isinstance(project_gid, str) and project_gid.strip():
                    return project_gid.strip()

        project_gid = asana.get("project_id")
        if isinstance(project_gid, str) and project_gid.strip():
            return project_gid.strip()
        return None

    def _load_internal_config(self) -> dict[str, Any]:
        configured_path = os.getenv("PIXELATED_INTERNAL_CONFIG_PATH", "").strip()
        path = (
            Path(configured_path)
            if configured_path
            else self.config_path or DEFAULT_INTERNAL_CONFIG_PATH
        )
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}
