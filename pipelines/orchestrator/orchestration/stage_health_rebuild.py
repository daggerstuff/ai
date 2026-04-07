"""Rebuild stage health artifacts from persisted dataset assembly outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai.pipelines.orchestrator.orchestration.run_artifact_service import (
    RunArtifactPaths,
    RunArtifactService,
)


@dataclass
class PersistedRunStats:
    """Minimal stats view backed by a persisted dataset assembly report."""

    total_samples: int
    samples_by_source: dict[str, int] = field(default_factory=dict)
    samples_by_stage: dict[str, int] = field(default_factory=dict)
    stage_balance: dict[str, dict[str, float | int]] = field(default_factory=dict)
    split_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    integration_time: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    bias_detection_results: dict[str, Any] = field(default_factory=dict)
    stage_policy_enforcement: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageHealthRebuildResult:
    """Paths and payloads emitted during stage health rebuild."""

    checklist_path: Path
    stage_health_report_path: Path
    closure_pack_path: Path
    stage_health_report: dict[str, Any]
    closure_pack: dict[str, Any]


def load_checklist_payload(checklist_path: Path) -> dict[str, Any]:
    """Read a persisted training checklist and validate its report payload."""
    payload = json.loads(checklist_path.read_text(encoding="utf-8"))
    report = payload.get("report")
    if not isinstance(report, dict):
        raise ValueError(
            f"Checklist at {checklist_path} does not contain a valid report payload"
        )
    return payload


def _stats_from_report(report: dict[str, Any]) -> PersistedRunStats:
    stage_balance = report.get("stage_balance", {})
    return PersistedRunStats(
        total_samples=int(report.get("total_samples", 0) or 0),
        samples_by_source=dict(report.get("samples_by_source", {})),
        samples_by_stage={
            stage: int((entry or {}).get("actual", 0) or 0)
            for stage, entry in stage_balance.items()
            if isinstance(entry, dict)
        },
        stage_balance=dict(stage_balance),
        split_counts=dict(report.get("split_counts", {})),
        integration_time=float(report.get("integration_time_seconds", 0.0) or 0.0),
        warnings=list(report.get("warnings", [])),
        errors=list(report.get("errors", [])),
        bias_detection_results=dict(report.get("bias_detection", {})),
        stage_policy_enforcement=dict(report.get("stage_policy_enforcement", {})),
    )


def rebuild_stage_health_artifacts(
    *,
    checklist_path: Path,
    manifest_path: Path,
    stage_health_report_output_path: Path,
    closure_pack_output_path: Path,
    asana_task_key_mapping_output_path: Path,
    asana_task_transition_output_path: Path,
    enable_asana_sync: bool | None = None,
) -> StageHealthRebuildResult:
    """Rebuild stage health report and closure pack from a persisted checklist."""
    checklist_payload = load_checklist_payload(checklist_path)
    report = checklist_payload["report"]
    stats = _stats_from_report(report)
    should_sync_asana = (
        enable_asana_sync
        if enable_asana_sync is not None
        else (
            asana_task_key_mapping_output_path.exists()
            or asana_task_transition_output_path.exists()
        )
    )

    service = RunArtifactService(
        paths=RunArtifactPaths(
            tracker_sync_output_path=str(checklist_path),
            asana_task_key_mapping_output_path=str(asana_task_key_mapping_output_path),
            asana_task_transition_output_path=str(asana_task_transition_output_path),
            stage_health_report_output_path=str(stage_health_report_output_path),
            closure_pack_output_path=str(closure_pack_output_path),
        ),
        stats=stats,
        stage_distribution=dict(report.get("stage_distribution_targets", {})),
        fail_on_missing_stage_artifacts=bool(
            report.get("fail_on_missing_stage_artifacts", False)
        ),
        stage_drift_tolerance=0.02,
        stage_drift_waivers=dict(report.get("stage_drift_waivers", {})),
        manifest_path=manifest_path,
        enable_asana_sync=should_sync_asana,
    )

    stage_health_report = service.build_stage_health_report(report)
    service.write_stage_health_report(stage_health_report)
    closure_pack = service.build_mtgc_closure_pack(report, stage_health_report)
    service.write_mtgc_closure_pack(closure_pack)

    return StageHealthRebuildResult(
        checklist_path=checklist_path,
        stage_health_report_path=stage_health_report_output_path,
        closure_pack_path=closure_pack_output_path,
        stage_health_report=stage_health_report,
        closure_pack=closure_pack,
    )


__all__ = [
    "PersistedRunStats",
    "StageHealthRebuildResult",
    "load_checklist_payload",
    "rebuild_stage_health_artifacts",
]
