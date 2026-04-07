"""
Factory helpers for wiring integrated pipeline services.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from ai.pipelines.orchestrator.orchestration.asana_progress_service import (
    AsanaProgressSyncService,
)
from ai.pipelines.orchestrator.orchestration.asana_tracker_client import (
    AsanaTrackerClient,
)
from ai.pipelines.orchestrator.orchestration.checklist_tracker_sync_service import (
    ChecklistTrackerSyncService,
)
from ai.pipelines.orchestrator.orchestration.curriculum_enforcement_service import (
    CurriculumEnforcementService,
)
from ai.pipelines.orchestrator.orchestration.data_ingestion_coordinator import (
    DataIngestionCoordinator,
)
from ai.pipelines.orchestrator.orchestration.dataset_asana_sync_service import (
    DatasetAsanaSyncService,
)
from ai.pipelines.orchestrator.orchestration.dataset_assembler import (
    DatasetAssembler,
)
from ai.pipelines.orchestrator.orchestration.dataset_output_service import (
    DatasetOutputService,
)
from ai.pipelines.orchestrator.orchestration.dataset_quality_service import (
    DatasetQualityService,
)
from ai.pipelines.orchestrator.orchestration.run_artifact_service import (
    RunArtifactPaths,
    RunArtifactService,
)
from ai.pipelines.orchestrator.orchestration.source_loader_service import (
    SourceLoaderService,
)
from ai.pipelines.orchestrator.orchestration.standard_therapeutic_loader_service import (
    StandardTherapeuticLoaderService,
)
from ai.pipelines.orchestrator.orchestration.storage_resolver import StorageResolver


class RuntimePolicyServiceProtocol(Protocol):
    def collect_ops_freshness(self) -> dict[str, Any]: ...


@dataclass
class PipelineServiceBundle:
    asana_client: AsanaTrackerClient
    curriculum_enforcement_service: CurriculumEnforcementService
    asana_progress_service: AsanaProgressSyncService
    dataset_asana_sync_service: DatasetAsanaSyncService
    standard_therapeutic_loader_service: StandardTherapeuticLoaderService
    source_loader_service: SourceLoaderService
    dataset_quality_service: DatasetQualityService
    dataset_output_service: DatasetOutputService
    run_artifact_service: RunArtifactService
    checklist_tracker_sync_service: ChecklistTrackerSyncService
    dataset_assembler: DatasetAssembler
    data_ingestion: DataIngestionCoordinator


def build_pipeline_services(
    *,
    config: Any,
    stats: Any,
    runtime_policy_service: RuntimePolicyServiceProtocol,
    stage_drift_tolerance: float,
    stage_drift_waivers: dict[str, Any],
    stage_quality_profiles: dict[str, Any],
    storage_resolver: StorageResolver,
    cache_data: Callable[[str | None], Path | None],
    apply_intake_routing: Callable[[list[dict], str], list[dict]],
) -> PipelineServiceBundle:
    """Build the service graph for the integrated training pipeline."""
    asana_client = AsanaTrackerClient()
    curriculum_enforcement_service = CurriculumEnforcementService(
        config=config,
        stats=stats,
        runtime_policy_service=runtime_policy_service,
        stage_drift_waivers=stage_drift_waivers,
        stage_quality_profiles=stage_quality_profiles,
    )
    asana_progress_service = AsanaProgressSyncService(
        config=config,
        stats=stats,
        asana_client=asana_client,
    )
    dataset_asana_sync_service = DatasetAsanaSyncService(
        config=config,
        stats=stats,
        asana_client=asana_client,
    )
    standard_therapeutic_loader_service = StandardTherapeuticLoaderService(
        config=config.standard_therapeutic,
        stats=stats,
        cache_data=cache_data,
    )
    source_loader_service = SourceLoaderService(stats=stats, config=config)
    dataset_quality_service = DatasetQualityService(
        stats=stats,
        curriculum_enforcement_service=curriculum_enforcement_service,
        stage_quality_profiles=stage_quality_profiles,
    )
    dataset_output_service = DatasetOutputService(
        config=config,
        stats=stats,
    )
    run_artifact_service = RunArtifactService(
        paths=RunArtifactPaths(
            tracker_sync_output_path=config.tracker_sync_output_path,
            asana_task_key_mapping_output_path=config.asana_task_key_mapping_output_path,
            asana_task_transition_output_path=config.asana_task_transition_output_path,
            stage_health_report_output_path=config.stage_health_report_output_path,
            closure_pack_output_path=config.closure_pack_output_path,
        ),
        stats=stats,
        stage_distribution=config.stage_distribution,
        fail_on_missing_stage_artifacts=config.fail_on_missing_stage_artifacts,
        stage_drift_tolerance=stage_drift_tolerance,
        stage_drift_waivers=stage_drift_waivers,
        manifest_path=Path("ai/training_data_consolidated/final/MASTER_STAGE_MANIFEST.json"),
        enable_asana_sync=config.enable_asana_sync,
    )
    checklist_tracker_sync_service = ChecklistTrackerSyncService(
        config=config,
        stats=stats,
        stage_drift_tolerance=stage_drift_tolerance,
        collect_ops_freshness=runtime_policy_service.collect_ops_freshness,
        asana_sync=asana_progress_service.sync_checklist_task,
    )
    dataset_assembler = DatasetAssembler(
        enable_bias_detection=config.enable_bias_detection,
        enable_quality_validation=config.enable_quality_validation,
        balance_dataset=curriculum_enforcement_service.balance_dataset,
        run_bias_detection=dataset_quality_service.run_bias_detection,
        run_quality_validation=dataset_quality_service.run_quality_validation,
        validate_final_stage_balance=curriculum_enforcement_service.validate_final_stage_balance,
        finalize_stats=dataset_quality_service.finalize_stats,
        save_dataset=dataset_output_service.save_dataset,
        write_stage_outputs=dataset_output_service.write_stage_outputs,
        write_split_outputs=dataset_output_service.write_split_outputs,
        run_artifact_service=run_artifact_service,
        checklist_tracker_sync_service=checklist_tracker_sync_service,
    )
    data_ingestion = DataIngestionCoordinator(
        config=config,
        storage_resolver=storage_resolver,
        load_edge_cases=source_loader_service.load_edge_cases,
        load_pixel_voice=source_loader_service.load_pixel_voice,
        load_psychology_knowledge=source_loader_service.load_psychology_knowledge,
        load_dual_persona=source_loader_service.load_dual_persona,
        load_standard_therapeutic=standard_therapeutic_loader_service.load,
        apply_intake_routing=apply_intake_routing,
        samples_by_source=stats.samples_by_source,
    )
    return PipelineServiceBundle(
        asana_client=asana_client,
        curriculum_enforcement_service=curriculum_enforcement_service,
        asana_progress_service=asana_progress_service,
        dataset_asana_sync_service=dataset_asana_sync_service,
        standard_therapeutic_loader_service=standard_therapeutic_loader_service,
        source_loader_service=source_loader_service,
        dataset_quality_service=dataset_quality_service,
        dataset_output_service=dataset_output_service,
        run_artifact_service=run_artifact_service,
        checklist_tracker_sync_service=checklist_tracker_sync_service,
        dataset_assembler=dataset_assembler,
        data_ingestion=data_ingestion,
    )


__all__ = ["PipelineServiceBundle", "build_pipeline_services"]
