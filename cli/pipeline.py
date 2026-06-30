"""
Pipeline management for Pixelated AI CLI.

Handles pipeline creation, execution, monitoring, and status tracking
for training and inference workflows.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

import requests

logger = logging.getLogger(__name__)


class PipelineStatus(StrEnum):
    """Pipeline execution status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineManager:
    """Manages training and inference pipeline operations."""

    def __init__(self, config, auth_manager) -> None:
        self._config = config
        self._auth = auth_manager
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._pipeline_cache: dict[str, dict[str, Any]] = {}

    @property
    def _base_url(self) -> str:
        return f"{self._config.api_base_url}/pipelines"

    def get_status(self, pipeline_id: str | None = None) -> str:
        """Get pipeline status. Returns overall status if no ID given."""
        if pipeline_id is None:
            running = [
                pid
                for pid, p in self._pipeline_cache.items()
                if p.get("status") in (PipelineStatus.RUNNING, PipelineStatus.QUEUED)
            ]
            if not running:
                return "No active pipelines"
            pipeline_id = running[0]

        cached = self._pipeline_cache.get(pipeline_id)
        if cached:
            return cached.get("status", PipelineStatus.RUNNING).value

        try:
            response = requests.get(
                f"{self._base_url}/{pipeline_id}/status",
                headers=self._auth.get_auth_headers(),
                timeout=self._config.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("status", PipelineStatus.RUNNING.value)
            return f"Error: HTTP {response.status_code}"
        except requests.RequestException:
            return "Running"

    def check_api_health(self) -> str:
        """Check pipeline API health and return status."""
        try:
            response = requests.get(
                f"{self._base_url.rsplit('/pipelines', 1)[0]}/health",
                headers=self._auth.get_auth_headers(),
                timeout=self._config.timeout,
            )
            if response.status_code == 200:
                return "Healthy"
            return f"Unhealthy: HTTP {response.status_code}"
        except requests.RequestException:
            return "Unreachable"

    def list_pipelines(self, status: PipelineStatus | None = None) -> list[dict[str, Any]]:
        """List pipelines, optionally filtered by status."""
        try:
            params = {}
            if status:
                params["status"] = status.value
            response = requests.get(
                self._base_url,
                headers=self._auth.get_auth_headers(),
                params=params,
                timeout=self._config.timeout,
            )
            if response.status_code == 200:
                pipelines = response.json()
                for p in pipelines:
                    self._pipeline_cache[p["id"]] = p
                return pipelines
            return []
        except requests.RequestException as e:
            logger.warning(f"Failed to list pipelines: {e}")
            return []

    def create_pipeline(
        self,
        name: str,
        config: dict[str, Any],
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create and start a new pipeline."""
        payload = {
            "name": name,
            "config": config,
            "description": description or "",
        }
        try:
            response = requests.post(
                self._base_url,
                headers=self._auth.get_auth_headers(),
                json=payload,
                timeout=self._config.timeout,
            )
            response.raise_for_status()
            pipeline = response.json()
            self._pipeline_cache[pipeline["id"]] = pipeline
            return pipeline
        except requests.RequestException as e:
            raise PipelineError(f"Failed to create pipeline: {e}") from e

    def get_pipeline(self, pipeline_id: str) -> dict[str, Any] | None:
        """Get pipeline details by ID."""
        if pipeline_id in self._pipeline_cache:
            return self._pipeline_cache[pipeline_id]

        try:
            response = requests.get(
                f"{self._base_url}/{pipeline_id}",
                headers=self._auth.get_auth_headers(),
                timeout=self._config.timeout,
            )
            if response.status_code == 200:
                pipeline = response.json()
                self._pipeline_cache[pipeline["id"]] = pipeline
                return pipeline
            return None
        except requests.RequestException:
            return None

    def stop_pipeline(self, pipeline_id: str) -> bool:
        """Stop a running pipeline."""
        try:
            response = requests.post(
                f"{self._base_url}/{pipeline_id}/cancel",
                headers=self._auth.get_auth_headers(),
                timeout=self._config.timeout,
            )
            if response.ok:
                if pipeline_id in self._pipeline_cache:
                    self._pipeline_cache[pipeline_id]["status"] = PipelineStatus.CANCELLED.value
                return True
            return False
        except requests.RequestException:
            return False

    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline."""
        try:
            response = requests.delete(
                f"{self._base_url}/{pipeline_id}",
                headers=self._auth.get_auth_headers(),
                timeout=self._config.timeout,
            )
            if response.ok:
                self._pipeline_cache.pop(pipeline_id, None)
                return True
            return False
        except requests.RequestException:
            return False

    def stream_logs(self, pipeline_id: str) -> str:
        """Stream pipeline logs. Returns log content."""
        try:
            response = requests.get(
                f"{self._base_url}/{pipeline_id}/logs",
                headers=self._auth.get_auth_headers(),
                timeout=self._config.timeout,
                stream=True,
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise PipelineError(f"Failed to get logs: {e}") from e


class PipelineError(Exception):
    """Raised when a pipeline operation fails."""


__all__ = ["PipelineError", "PipelineManager", "PipelineStatus"]
