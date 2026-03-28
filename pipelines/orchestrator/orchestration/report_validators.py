"""
Shared validation helpers for orchestrator run reporting artifacts.
"""

from __future__ import annotations

from typing import Any


def collect_stage_drift_failures(
    stage_balance: dict[str, Any],
    *,
    tolerance: float,
) -> list[str]:
    """Return the stages whose drift exceeds the configured tolerance."""
    failures: list[str] = []
    for stage, metrics in stage_balance.items():
        if not isinstance(metrics, dict):
            continue
        drift = metrics.get("drift_vs_target")
        if isinstance(drift, (int, float)) and abs(drift) > tolerance:
            failures.append(stage)
    return failures


def build_validator_status_by_stage(
    *,
    stage_distribution: dict[str, float],
    removed_by_stage: dict[str, Any],
    failure_reasons_by_stage: dict[str, Any],
    drift_failures: list[str],
) -> dict[str, Any]:
    """Build validator status payloads for each configured stage."""
    validator_status_by_stage: dict[str, Any] = {}
    for stage in stage_distribution:
        removed_count = 0
        value = removed_by_stage.get(stage, 0)
        if isinstance(value, int):
            removed_count = value

        reasons: dict[str, Any] = {}
        stage_reasons = failure_reasons_by_stage.get(stage, {})
        if isinstance(stage_reasons, dict):
            reasons = stage_reasons

        validator_status_by_stage[stage] = {
            "passed": stage not in drift_failures and removed_count == 0,
            "removed_count": removed_count,
            "failure_reasons": reasons,
            "drift_within_tolerance": stage not in drift_failures,
        }
    return validator_status_by_stage


def build_stage_health_blockers(
    *,
    errors: list[Any],
    split_counts: dict[str, Any],
    drift_failures: list[str],
    validator_status_by_stage: dict[str, Any],
) -> list[str]:
    """Build the blocker list for the stage health report."""
    blockers: list[str] = []
    if errors:
        blockers.append("pipeline_errors_present")

    aggregate = split_counts.get("aggregate") if isinstance(split_counts, dict) else None
    if not isinstance(aggregate, dict):
        blockers.append("aggregate_splits_missing")
    else:
        train = aggregate.get("train", 0)
        val = aggregate.get("val", 0)
        test = aggregate.get("test", 0)
        if all(isinstance(v, int) for v in (train, val, test)):
            if train + val + test == 0:
                blockers.append("aggregate_splits_empty")
        else:
            blockers.append("aggregate_split_counts_invalid")

    blockers.extend(f"stage_drift_exceeds_tolerance:{stage}" for stage in drift_failures)

    for stage, status in validator_status_by_stage.items():
        if isinstance(status, dict) and not status.get("passed", False):
            if status.get("removed_count", 0):
                blockers.append(f"validator_failures:{stage}")

    return sorted(set(blockers))


def build_closure_success_criteria(
    *,
    stage_health_pass: bool,
    drift_ok: bool,
    manifest_exists: bool,
    stage_health_exists: bool,
    report_warnings: list[Any],
    fail_on_missing_stage_artifacts: bool,
    split_artifacts_present: bool,
    asana_mapping_exists: bool,
    enable_asana_sync: bool,
    ops_freshness_all_fresh: bool,
    asana_transition_exists: bool,
) -> dict[str, dict[str, Any]]:
    """Build closure-pack success criteria payload."""
    return {
        "stage_drift_within_tolerance": {
            "passed": bool(stage_health_pass) and bool(drift_ok),
            "evidence": "stage_health_report.blockers + report.stage_balance",
        },
        "manifest_and_report_generated": {
            "passed": manifest_exists and stage_health_exists,
            "evidence": "MASTER_STAGE_MANIFEST.json + integrated_stage_health_report.json",
        },
        "stage_3_4_inputs_checked": {
            "passed": any(
                "missing required artifacts for stage3_edge_stress_test" in str(item).lower()
                or "missing required artifacts for stage4_voice_persona" in str(item).lower()
                for item in report_warnings
            )
            or bool(fail_on_missing_stage_artifacts),
            "evidence": "pipeline warnings + strict artifact validation gate",
        },
        "aggregate_and_stage_split_artifacts_emitted": {
            "passed": split_artifacts_present,
            "evidence": "report.split_counts.aggregate + split directories",
        },
        "asana_task_graph_evidence_available": {
            "passed": asana_mapping_exists or not enable_asana_sync,
            "evidence": "asana_task_key_mapping.json",
        },
        "ops_freshness_reflected": {
            "passed": ops_freshness_all_fresh,
            "evidence": "training_run_checklist.json:ops_freshness",
        },
        "asana_transition_results_recorded": {
            "passed": asana_transition_exists or not enable_asana_sync,
            "evidence": "asana_task_transition_results.json",
        },
    }


__all__ = [
    "build_closure_success_criteria",
    "build_stage_health_blockers",
    "build_validator_status_by_stage",
    "collect_stage_drift_failures",
]
