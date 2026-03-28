"""
Run artifact generation and persistence for the integrated training pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ai.pipelines.orchestrator.orchestration.report_validators import (
    build_closure_success_criteria,
    build_stage_health_blockers,
    build_validator_status_by_stage,
    collect_stage_drift_failures,
)


@dataclass(frozen=True)
class RunArtifactPaths:
    """Filesystem paths for generated run artifacts."""

    tracker_sync_output_path: str
    asana_task_key_mapping_output_path: str
    asana_task_transition_output_path: str
    stage_health_report_output_path: str
    closure_pack_output_path: str


class IntegrationStatsProtocol(Protocol):
    total_samples: int
    samples_by_source: dict[str, int]
    samples_by_stage: dict[str, int]
    stage_balance: dict[str, dict[str, float | int]]
    split_counts: dict[str, dict[str, int]]
    integration_time: float
    warnings: list[str]
    errors: list[str]
    bias_detection_results: dict[str, Any]
    stage_policy_enforcement: dict[str, Any]


class RunArtifactService:
    """Own report, health-report, and closure-pack generation for a run."""

    def __init__(
        self,
        *,
        paths: RunArtifactPaths,
        stats: IntegrationStatsProtocol,
        stage_distribution: dict[str, float],
        fail_on_missing_stage_artifacts: bool,
        stage_drift_tolerance: float,
        stage_drift_waivers: dict[str, Any],
        manifest_path: Path,
        enable_asana_sync: bool,
    ) -> None:
        self.paths = paths
        self.stats = stats
        self.stage_distribution = stage_distribution
        self.fail_on_missing_stage_artifacts = fail_on_missing_stage_artifacts
        self.stage_drift_tolerance = stage_drift_tolerance
        self.stage_drift_waivers = stage_drift_waivers
        self.manifest_path = manifest_path
        self.enable_asana_sync = enable_asana_sync

    def generate_report(self) -> dict[str, Any]:
        """Generate the integration report from current run stats."""
        stage_sample_total = sum(self.stats.samples_by_stage.values())
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_samples": self.stats.total_samples,
            "samples_by_source": self.stats.samples_by_source,
            "stage_distribution_targets": self.stage_distribution,
            "fail_on_missing_stage_artifacts": self.fail_on_missing_stage_artifacts,
            "stage_balance": self.stats.stage_balance,
            "actual_stage_percentages": {
                stage: count / stage_sample_total if stage_sample_total > 0 else 0
                for stage, count in self.stats.samples_by_stage.items()
            },
            "split_counts": self.stats.split_counts,
            "integration_time_seconds": self.stats.integration_time,
            "warnings": self.stats.warnings,
            "errors": self.stats.errors,
            "bias_detection": self.stats.bias_detection_results,
            "stage_policy_enforcement": self.stats.stage_policy_enforcement,
            "stage_drift_waivers": self.stage_drift_waivers,
        }

    def build_stage_health_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """Build the MTGC integrated stage health report payload."""
        stage_balance = report.get("stage_balance", {})
        split_counts = report.get("split_counts", {})
        enforcement = report.get("stage_policy_enforcement", {})

        removed_by_stage = {}
        failure_reasons_by_stage = {}
        if isinstance(enforcement, dict):
            removed_by_stage = enforcement.get("removed_by_stage", {})
            failure_reasons_by_stage = enforcement.get("failure_reasons_by_stage", {})

        drift_failures = collect_stage_drift_failures(
            stage_balance if isinstance(stage_balance, dict) else {},
            tolerance=self.stage_drift_tolerance,
        )
        validator_status_by_stage = build_validator_status_by_stage(
            stage_distribution=self.stage_distribution,
            removed_by_stage=removed_by_stage if isinstance(removed_by_stage, dict) else {},
            failure_reasons_by_stage=(
                failure_reasons_by_stage if isinstance(failure_reasons_by_stage, dict) else {}
            ),
            drift_failures=drift_failures,
        )
        blockers = build_stage_health_blockers(
            errors=report.get("errors", []),
            split_counts=split_counts if isinstance(split_counts, dict) else {},
            drift_failures=drift_failures,
            validator_status_by_stage=validator_status_by_stage,
        )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": report.get("total_samples", 0),
            "integration_time_seconds": report.get("integration_time_seconds", 0.0),
            "stage_distribution_targets": report.get("stage_distribution_targets", {}),
            "stage_balance": stage_balance,
            "split_counts": split_counts,
            "validator_status_by_stage": validator_status_by_stage,
            "blockers": sorted(set(blockers)),
            "warnings": report.get("warnings", []),
            "errors": report.get("errors", []),
            "pass": len(blockers) == 0,
        }

    def write_stage_health_report(self, stage_health_report: dict[str, Any]) -> None:
        """Persist the integrated stage health report."""
        output_path = Path(self.paths.stage_health_report_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(stage_health_report, handle, indent=2)

    def build_mtgc_closure_pack(
        self,
        report: dict[str, Any],
        stage_health_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the MTGC closure pack summarizing status and evidence artifacts."""
        checklist_path = Path(self.paths.tracker_sync_output_path)
        asana_mapping_path = Path(self.paths.asana_task_key_mapping_output_path)
        asana_transition_path = Path(self.paths.asana_task_transition_output_path)
        stage_health_path = Path(self.paths.stage_health_report_output_path)

        ops_freshness_all_fresh = False
        if checklist_path.exists():
            try:
                with checklist_path.open(encoding="utf-8") as handle:
                    checklist_payload = json.load(handle)
                ops_payload = checklist_payload.get("ops_freshness", {})
                if isinstance(ops_payload, dict):
                    ops_freshness_all_fresh = bool(ops_payload.get("all_fresh", False))
            except Exception as exc:
                self.stats.warnings.append(
                    f"Failed to read checklist for closure pack: {exc}"
                )

        stage_health_pass = bool(stage_health_report.get("pass", False))
        stage_blockers = stage_health_report.get("blockers", [])
        if not isinstance(stage_blockers, list):
            stage_blockers = []
        drift_ok = not any(
            str(blocker).startswith("stage_drift_exceeds_tolerance:")
            for blocker in stage_blockers
        )
        split_counts = report.get("split_counts", {})
        aggregate_split = split_counts.get("aggregate") if isinstance(split_counts, dict) else None
        split_artifacts_present = isinstance(aggregate_split, dict)

        success_criteria = build_closure_success_criteria(
            stage_health_pass=stage_health_pass,
            drift_ok=drift_ok,
            manifest_exists=self.manifest_path.exists(),
            stage_health_exists=stage_health_path.exists(),
            report_warnings=report.get("warnings", []),
            fail_on_missing_stage_artifacts=self.fail_on_missing_stage_artifacts,
            split_artifacts_present=split_artifacts_present,
            asana_mapping_exists=asana_mapping_path.exists(),
            enable_asana_sync=self.enable_asana_sync,
            ops_freshness_all_fresh=ops_freshness_all_fresh,
            asana_transition_exists=asana_transition_path.exists(),
        )

        completion_pass = all(bool(entry.get("passed", False)) for entry in success_criteria.values())

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "task_key": "MTGC-13",
            "overall_pass": completion_pass,
            "success_criteria": success_criteria,
            "artifact_paths": {
                "checklist": str(checklist_path),
                "stage_health_report": str(stage_health_path),
                "stage_manifest": str(self.manifest_path),
                "asana_task_key_mapping": str(asana_mapping_path),
                "asana_task_transition_results": str(asana_transition_path),
            },
            "warnings": report.get("warnings", []),
            "errors": report.get("errors", []),
        }

    def write_mtgc_closure_pack(self, closure_pack: dict[str, Any]) -> None:
        """Persist the MTGC closure pack."""
        output_path = Path(self.paths.closure_pack_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(closure_pack, handle, indent=2)


__all__ = ["RunArtifactPaths", "RunArtifactService"]
