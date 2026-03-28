"""
Dataset assembly coordinator for the integrated training pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class RunArtifactServiceProtocol(Protocol):
    def generate_report(self) -> dict[str, Any]: ...
    def build_stage_health_report(self, report: dict[str, Any]) -> dict[str, Any]: ...
    def write_stage_health_report(self, stage_health_report: dict[str, Any]) -> None: ...
    def build_mtgc_closure_pack(
        self, report: dict[str, Any], stage_health_report: dict[str, Any]
    ) -> dict[str, Any]: ...
    def write_mtgc_closure_pack(self, closure_pack: dict[str, Any]) -> None: ...


class ChecklistTrackerSyncServiceProtocol(Protocol):
    def sync_run_checklist(self, report: dict[str, Any]) -> None: ...


class DatasetAssembler:
    """Coordinate balancing, validation, and artifact emission for training data."""

    def __init__(
        self,
        *,
        enable_bias_detection: bool,
        enable_quality_validation: bool,
        balance_dataset: Callable[[list[dict[str, Any]]], tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]],
        run_bias_detection: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        run_quality_validation: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        validate_final_stage_balance: Callable[[list[dict[str, Any]]], None],
        finalize_stats: Callable[[list[dict[str, Any]]], None],
        save_dataset: Callable[[list[dict[str, Any]]], str],
        write_stage_outputs: Callable[[dict[str, list[dict[str, Any]]]], None],
        write_split_outputs: Callable[[list[dict[str, Any]]], None],
        run_artifact_service: RunArtifactServiceProtocol,
        checklist_tracker_sync_service: ChecklistTrackerSyncServiceProtocol,
    ) -> None:
        self.enable_bias_detection = enable_bias_detection
        self.enable_quality_validation = enable_quality_validation
        self.balance_dataset = balance_dataset
        self.run_bias_detection = run_bias_detection
        self.run_quality_validation = run_quality_validation
        self.validate_final_stage_balance = validate_final_stage_balance
        self.finalize_stats = finalize_stats
        self.save_dataset = save_dataset
        self.write_stage_outputs = write_stage_outputs
        self.write_split_outputs = write_split_outputs
        self.run_artifact_service = run_artifact_service
        self.checklist_tracker_sync_service = checklist_tracker_sync_service

    def assemble(
        self, all_training_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Balance, validate, and persist the integrated dataset."""
        balanced_data, stage_segments = self.balance_dataset(all_training_data)

        if self.enable_bias_detection:
            balanced_data = self.run_bias_detection(balanced_data)

        if self.enable_quality_validation:
            balanced_data = self.run_quality_validation(balanced_data)

        self.validate_final_stage_balance(balanced_data)
        self.finalize_stats(balanced_data)
        output_path = self.save_dataset(balanced_data)
        self.write_stage_outputs(stage_segments)
        self.write_split_outputs(balanced_data)

        report = self.run_artifact_service.generate_report()
        stage_health_report = self.run_artifact_service.build_stage_health_report(report)
        self.run_artifact_service.write_stage_health_report(stage_health_report)
        self.checklist_tracker_sync_service.sync_run_checklist(report)
        closure_pack = self.run_artifact_service.build_mtgc_closure_pack(
            report, stage_health_report
        )
        self.run_artifact_service.write_mtgc_closure_pack(closure_pack)

        return {
            "training_data": balanced_data,
            "output_path": output_path,
            "report": report,
            "stage_health_report": stage_health_report,
            "closure_pack": closure_pack,
        }


__all__ = ["DatasetAssembler"]
