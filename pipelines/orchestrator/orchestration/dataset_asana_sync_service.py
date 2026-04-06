"""
Dataset inventory synchronization to Asana for improved dataset tracking and management.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ai.pipelines.orchestrator.orchestration.asana_tracker_client import (
    AsanaTrackerClient,
)
from ai.pipelines.orchestrator.systems.dataset_inventory import (
    get_dataset_metadata,
    scan_datasets,
)
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.dataset_asana_sync_service")


class DatasetAsanaConfigProtocol(Protocol):
    enable_dataset_asana_sync: bool
    asana_project_gid: str | None
    asana_dataset_section_gid: str | None
    asana_dataset_task_gid_output_path: str
    asana_dataset_mapping_output_path: str
    dataset_scan_directory: str
    dataset_file_patterns: list[str]
    dataset_task_prefix: str


class DatasetAsanaStatsProtocol(Protocol):
    warnings: list[str]


class DatasetAsanaSyncService:
    """Sync dataset inventory and metadata to Asana for improved tracking."""

    def __init__(
        self,
        *,
        config: DatasetAsanaConfigProtocol,
        stats: DatasetAsanaStatsProtocol,
        asana_client: AsanaTrackerClient,
    ) -> None:
        self.config = config
        self.stats = stats
        self.asana_client = asana_client

    @staticmethod
    def _is_valid_gid(value: str | None) -> bool:
        """Validate Asana GID (should be numeric string)."""
        return isinstance(value, str) and value.isdigit() and len(value) > 0

    def sync_dataset_inventory(self) -> None:
        """Scan dataset directory and sync inventory to Asana tasks."""
        if not self.config.enable_dataset_asana_sync:
            return

        project_gid = self.config.asana_project_gid
        if not self._is_valid_gid(project_gid):
            self._warn("Dataset Asana sync skipped: ASANA_PROJECT_GID missing or invalid")
            return

        section_gid = self._validated_optional_gid(
            self.config.asana_dataset_section_gid,
            "Dataset sync skipped section assignment: ASANA_DATASET_SECTION_GID is invalid",
        )

        # Scan for datasets
        datasets = self._scan_and_filter_datasets()
        print(f"DEBUG sync_dataset_inventory: Found {len(datasets)} datasets")
        if not datasets:
            self._warn("No datasets found to sync to Asana")
            return

        # Load existing task mappings
        task_mapping = self._load_dataset_task_mapping(project_gid)

        # Prepare work items
        work_items = []
        for dataset_meta in datasets:
            task_key = self._generate_dataset_task_key(dataset_meta)
            work_items.append((dataset_meta, task_key))

        # Process datasets in parallel
        max_workers = min(8, max(1, len(work_items)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    self._sync_single_dataset,
                    dataset_meta,
                    task_key,
                    project_gid,
                    section_gid,
                    task_mapping,
                ): (dataset_meta["path"], task_key)
                for dataset_meta, task_key in work_items
            }
            for future in as_completed(future_map):
                dataset_path, task_key, update_result = future.result()
                if update_result:
                    self._update_task_mapping(
                        task_key, update_result["gid"], project_gid
                    )
                if update_result.get("warning"):
                    self._warn(update_result["warning"])

        # Save updated task mapping
        self._save_dataset_task_mapping(task_mapping, project_gid)

    def _scan_and_filter_datasets(self) -> list[dict]:
        """Scan configured directory for datasets matching patterns."""
        try:
            all_datasets = scan_datasets(
                self.config.dataset_scan_directory,
                pattern="",  # Get all files, we'll filter by extension ourselves
            )

            # Filter by configured file patterns
            filtered_datasets = []
            for dataset in all_datasets:
                if any(
                    dataset["path"].endswith(pattern)
                    for pattern in self.config.dataset_file_patterns
                ):
                    filtered_datasets.append(dataset)

            return filtered_datasets
        except Exception as exc:
            self._warn(f"Failed to scan datasets: {exc}")
            return []

    def _generate_dataset_task_key(self, dataset_meta: dict) -> str:
        """Generate a consistent task key for a dataset."""
        # Use filename without extension as base
        name = Path(dataset_meta["name"]).stem
        # Clean for task key format
        task_key = "".join(c if c.isalnum() else "-" for c in name.upper())
        # Remove leading/trailing hyphens and replace multiple hyphens
        task_key = "-".join(filter(None, task_key.split("-")))
        # Ensure it starts with letter
        if not task_key or not task_key[0].isalpha():
            task_key = "DS" + (task_key or "DATASET")
        return f"DATASET-{task_key}"

    def _sync_single_dataset(
        self,
        dataset_meta: dict,
        task_key: str,
        project_gid: str,
        section_gid: str | None,
        existing_mapping: dict[str, str],
    ) -> tuple[str, dict[str, Any] | None, str | None]:
        """Sync a single dataset to Asana."""
        try:
            gid = existing_mapping.get(task_key)

            # Prepare task payload
            task_payload = self._build_dataset_task_payload(dataset_meta, task_key)

            # Create or update task
            if gid and self._is_valid_gid(gid):
                # Update existing task
                task_data = self._update_existing_task(gid, task_payload)
                action = "updated"
            else:
                # Create new task
                task_data = self._create_new_task(
                    task_payload, project_gid, section_gid
                )
                action = "created"
                gid = str(task_data.get("gid", "")).strip()

            if not self._is_valid_gid(gid):
                raise RuntimeError(f"Invalid GID returned for {task_key}: {gid}")

            # Add dataset metadata as story/comment
            self._add_dataset_story(gid, dataset_meta)

            # Prepare result
            result = {
                "action": action,
                "gid": gid,
                "task_key": task_key,
                "dataset_name": dataset_meta["name"],
                "size_mb": round(dataset_meta["size_bytes"] / (1024 * 1024), 2),
                "columns": len(dataset_meta.get("columns", [])),
            }

            return dataset_meta["path"], result, None

        except Exception as exc:
            return (
                dataset_meta["path"],
                None,
                f"Failed to sync dataset {dataset_meta.get('name', 'unknown')}: {exc}",
            )

    def _build_dataset_task_payload(self, dataset_meta: dict, task_key: str) -> dict[str, Any]:
        """Build Asana task payload from dataset metadata."""
        size_mb = dataset_meta["size_bytes"] / (1024 * 1024)
        modified_time = datetime.fromtimestamp(
            dataset_meta["last_modified"], tz=timezone.utc
        ).isoformat()

        task_name = f"{self.config.dataset_task_prefix} {dataset_meta['name']}"

        notes_lines = [
            f"Dataset: {dataset_meta['name']}",
            f"Path: {dataset_meta['path']}",
            f"Size: {size_mb:.2f} MB",
            f"Last Modified: {modified_time}",
            f"Columns: {len(dataset_meta.get('columns', []))}",
            "",
            "Preview (first 5 rows):",
        ]

        # Add preview data
        preview_rows = dataset_meta.get("preview_rows", [])
        for i, row in enumerate(preview_rows[:3]):  # Limit preview
            notes_lines.append(f"Row {i+1}: {json.dumps(row)}")

        if len(preview_rows) > 3:
            notes_lines.append(f"... and {len(preview_rows) - 3} more rows")

        task_payload: dict[str, Any] = {
            "name": task_name,
            "notes": "\n".join(notes_lines),
            "projects": [dataset_meta.get("project_gid", "")],
        }

        # Add custom fields if supported (would need Asana API v2+ for custom fields)
        # For now, we'll put key metadata in notes

        return task_payload

    def _create_new_task(
        self, task_payload: dict[str, Any], project_gid: str, section_gid: str | None
    ) -> dict[str, Any]:
        """Create a new Asana task for a dataset."""
        if section_gid:
            task_data = self.asana_client.request(
                "POST",
                "/tasks",
                {
                    **task_payload,
                    "projects": [project_gid],
                    "memberships": [{"project": project_gid, "section": section_gid}],
                },
            )
        else:
            task_data = self.asana_client.request(
                "POST",
                "/tasks",
                {**task_payload, "projects": [project_gid]},
            )

        if not isinstance(task_data, dict):
            raise RuntimeError("Asana task creation returned invalid payload")
        return task_data

    def _update_existing_task(self, gid: str, task_payload: dict[str, Any]) -> dict[str, Any]:
        """Update an existing Asana task."""
        task_data = self.asana_client.request(
            "PUT",
            f"/tasks/{gid}",
            task_payload,
        )
        if not isinstance(task_data, dict):
            raise RuntimeError("Asana task update returned invalid payload")
        return task_data

    def _add_dataset_story(self, gid: str, dataset_meta: dict) -> None:
        """Add dataset information as a story/comment to the task."""
        story_text = (
            f"Dataset sync update: {dataset_meta['name']}\n"
            f"Size: {dataset_meta['size_bytes'] / (1024 * 1024):.2f} MB\n"
            f"Synced at: {datetime.now(timezone.utc).isoformat()}"
        )

        self.asana_client.request(
            "POST",
            f"/tasks/{gid}/stories",
            {"text": story_text},
        )

    def _validated_optional_gid(self, gid: str | None, warning: str) -> str | None:
        if gid and not self._is_valid_gid(gid):
            self._warn(warning)
            return None
        return gid

    def _load_dataset_task_mapping(self, project_gid: str) -> dict[str, str]:
        """Load existing dataset-to-task GID mapping from file."""
        mapping: dict[str, str] = {}
        mapping_path = Path(self.config.asana_dataset_mapping_output_path)

        if mapping_path.exists():
            try:
                existing = json.loads(mapping_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    for task_key, gid in existing.items():
                        gid_str = str(gid)
                        if (
                            isinstance(task_key, str)
                            and self._is_valid_gid(gid_str)
                            and len(gid_str) > 0
                        ):
                            mapping[task_key] = gid_str
            except Exception as exc:
                self._warn(f"Failed to read dataset task mapping file: {exc}")

        # Also try to resolve from Asana (similar to progress service)
        try:
            tasks = self.asana_client.request(
                "GET",
                f"/projects/{project_gid}/tasks",
                payload=None,
                query_params={"limit": 100, "opt_fields": "gid,name"},
            )
            if isinstance(tasks, list):
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    # Extract task key from name (assuming format PREFIX-NAME)
                    task_name = task.get("name", "")
                    if task_name.startswith("DATASET-"):
                        task_key = task_name  # Use full name as key for now
                        gid = str(task.get("gid", "")).strip()
                        if task_key and self._is_valid_gid(gid):
                            mapping[task_key] = gid
        except Exception as exc:
            self._warn(f"Failed to resolve dataset task keys from Asana project: {exc}")

        return mapping

    def _save_dataset_task_mapping(
        self, mapping: dict[str, str], project_gid: str
    ) -> None:
        """Save dataset-to-task GID mapping to file."""
        try:
            mapping_path = Path(self.config.asana_dataset_mapping_output_path)
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            # Sort by task key for consistent output
            sorted_mapping = dict(sorted(mapping.items()))
            mapping_path.write_text(
                json.dumps(sorted_mapping, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            self._warn(f"Failed to save dataset task mapping file: {exc}")

    def _update_task_mapping(
        self, task_key: str, gid: str, project_gid: str
    ) -> None:
        """Update in-memory mapping (will be saved later)."""
        # This is handled by the caller after saving the full mapping
        pass

    def _warn(self, message: str) -> None:
        logger.warning(message)
        self.stats.warnings.append(message)


__all__ = ["DatasetAsanaSyncService"]
