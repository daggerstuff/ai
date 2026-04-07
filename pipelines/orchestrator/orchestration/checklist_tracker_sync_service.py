"""Checklist emission and tracker synchronization for dataset assembly runs."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from ai.pipelines.orchestrator.orchestration.report_validators import (
    collect_stage_drift_failures,
)
from ai.pipelines.orchestrator.orchestration.tracker_sync import (
    TrackerSyncCoordinator,
    TrackerSyncEvent,
)
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.checklist_tracker_sync")


class TrackerSyncConfigProtocol(Protocol):
    enable_tracker_sync: bool
    tracker_sync_output_path: str
    tracker_sync_state_output_path: str
    enable_beads_sync: bool
    enable_jira_sync: bool
    enable_linear_sync: bool
    stage_health_report_output_path: str


class ChecklistStatsProtocol(Protocol):
    warnings: list[str]
    errors: list[str]


AsanaSync = Callable[[dict[str, Any], Path], None]
OpsFreshnessCollector = Callable[[], dict[str, Any]]
TrackerSyncFactory = Callable[[], TrackerSyncCoordinator]


@dataclass
class ChecklistTrackerSyncService:
    """Persist training checklists and fan them out to tracker systems."""

    config: TrackerSyncConfigProtocol
    stats: ChecklistStatsProtocol
    stage_drift_tolerance: float
    collect_ops_freshness: OpsFreshnessCollector
    asana_sync: AsanaSync
    tracker_sync_factory: TrackerSyncFactory | None = None

    def sync_run_checklist(self, report: dict[str, Any]) -> None:
        """Persist checklist payload and optionally emit to tracker webhook."""
        if not self.config.enable_tracker_sync:
            return

        stage_balance = report.get("stage_balance", {})
        drift_failures = collect_stage_drift_failures(
            stage_balance if isinstance(stage_balance, dict) else {},
            tolerance=self.stage_drift_tolerance,
        )

        checklist = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": report.get("total_samples", 0),
            "stage_drift_within_tolerance": not drift_failures,
            "stage_drift_failures": drift_failures,
            "split_counts": report.get("split_counts", {}),
            "ops_freshness": self.collect_ops_freshness(),
            "stage_health_report_path": self.config.stage_health_report_output_path,
            "warnings": report.get("warnings", []),
            "errors": report.get("errors", []),
            "report": report,
        }

        output_path = Path(self.config.tracker_sync_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(checklist, handle, indent=2)

        webhook_url = os.getenv("TRAINING_CHECKLIST_WEBHOOK_URL", "").strip()
        if webhook_url:
            self._send_webhook(webhook_url, checklist)

        self._sync_tracker_state(checklist, output_path)

    def _send_webhook(self, webhook_url: str, checklist: dict[str, Any]) -> None:
        shared_secret = os.getenv("TRAINING_CHECKLIST_WEBHOOK_SECRET", "").strip()
        if not shared_secret:
            self.stats.warnings.append(
                "Checklist webhook skipped: TRAINING_CHECKLIST_WEBHOOK_SECRET is not configured"
            )
            return

        payload = json.dumps(checklist).encode("utf-8")
        signature = hmac.new(
            shared_secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()
        try:
            request = urllib.request.Request(
                webhook_url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Pixelated-Signature": f"sha256={signature}",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10):
                return
        except (urllib.error.URLError, TimeoutError) as exc:
            self.stats.warnings.append(f"Checklist webhook sync failed: {exc}")

    def _sync_tracker_state(
        self, checklist: dict[str, Any], checklist_path: Path
    ) -> None:
        coordinator = (
            self.tracker_sync_factory()
            if self.tracker_sync_factory is not None
            else TrackerSyncCoordinator(
                state_path=Path(self.config.tracker_sync_state_output_path),
                asana_sync=self.asana_sync,
                enable_beads_sync=self.config.enable_beads_sync,
                enable_jira_sync=self.config.enable_jira_sync,
                enable_linear_sync=self.config.enable_linear_sync,
            )
        )

        generated_at = str(checklist.get("generated_at", datetime.now(timezone.utc).isoformat()))
        event = TrackerSyncEvent(
            source="dataset_assembly_workflow",
            source_id=generated_at,
            title=f"Training Checklist {generated_at}",
            body=json.dumps(checklist, indent=2),
            status="done" if checklist.get("stage_drift_within_tolerance", False) else "open",
            checklist_path=checklist_path,
            payload=checklist,
        )
        summary = coordinator.sync(event)
        self.stats.warnings.extend(summary.warnings)
        self.stats.warnings.extend(summary.errors)


__all__ = ["ChecklistTrackerSyncService"]
