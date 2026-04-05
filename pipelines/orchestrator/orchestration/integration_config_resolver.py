"""
Integration configuration helpers for tracker-backed runtime defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ai.pipelines.orchestrator.utils.logger import get_logger

DEFAULT_INTERNAL_CONFIG_PATH = Path(".agent/internal/config.json")
logger = get_logger("dataset_pipeline.integration_config_resolver")


class IntegrationConfigResolver:
    """Resolve tracker-related defaults from the internal integration config."""

    def __init__(self, *, config_path: Path | None = None) -> None:
        self.config_path = config_path
        self._cached_payload: dict[str, Any] = {}
        self._cache_loaded = False

    def resolve_training_asana_project_gid(self) -> str | None:
        payload = self._load_internal_config()
        integration = payload.get("integration")
        if not isinstance(integration, Mapping):
            return None

        asana = integration.get("asana")
        if not isinstance(asana, Mapping):
            return None

        for mapping_key in ("task_sync_projects", "all_projects"):
            project_mapping = asana.get(mapping_key)
            project_gid = self._resolve_project_gid_from_mapping(project_mapping)
            if project_gid is not None:
                return project_gid

        project_gid = asana.get("project_id")
        if isinstance(project_gid, str) and project_gid.strip():
            return project_gid.strip()
        return None

    @staticmethod
    def _resolve_project_gid_from_mapping(project_mapping: Any) -> str | None:
        if not isinstance(project_mapping, Mapping):
            return None

        for project_key in (
            "master_training_gap_closure",
            "active_sprint",
            "master_training_epic",
        ):
            project_gid = project_mapping.get(project_key)
            if isinstance(project_gid, str) and project_gid.strip():
                return project_gid.strip()
        return None

    def _load_internal_config(self) -> dict[str, Any]:
        if self._cache_loaded:
            return self._cached_payload

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
        except OSError as exc:
            logger.warning("Failed to read internal config %s: %s", path, exc)
            return {}
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse internal config %s: %s", path, exc)
            return {}

        self._cached_payload = payload if isinstance(payload, dict) else {}
        self._cache_loaded = True
        return self._cached_payload
