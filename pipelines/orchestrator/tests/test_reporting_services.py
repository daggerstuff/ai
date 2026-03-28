from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai.pipelines.orchestrator.orchestration.checklist_tracker_sync_service import (
    ChecklistTrackerSyncService,
)
from ai.pipelines.orchestrator.orchestration.run_artifact_service import (
    RunArtifactPaths,
    RunArtifactService,
)
from ai.pipelines.orchestrator.orchestration.tracker_sync import (
    TrackerSyncEvent,
    TrackerSyncSummary,
)


@dataclass
class _StatsStub:
    total_samples: int = 12
    samples_by_source: dict[str, int] = field(
        default_factory=lambda: {"psych8k": 7, "youtube_transcript": 5}
    )
    samples_by_stage: dict[str, int] = field(
        default_factory=lambda: {
            "stage1_foundation": 7,
            "stage4_voice_persona": 5,
        }
    )
    stage_balance: dict[str, dict[str, float | int]] = field(
        default_factory=lambda: {
            "stage1_foundation": {"drift_vs_target": 0.01},
            "stage4_voice_persona": {"drift_vs_target": 0.0},
        }
    )
    split_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"aggregate": {"train": 8, "val": 2, "test": 2}}
    )
    integration_time: float = 3.2
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    bias_detection_results: dict[str, Any] = field(default_factory=dict)
    stage_policy_enforcement: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ConfigStub:
    enable_tracker_sync: bool = True
    tracker_sync_output_path: str = ""
    tracker_sync_state_output_path: str = ""
    enable_beads_sync: bool = False
    enable_jira_sync: bool = False
    enable_linear_sync: bool = False
    stage_health_report_output_path: str = ""


class _TrackerCoordinatorStub:
    def __init__(self) -> None:
        self.events: list[TrackerSyncEvent] = []

    def sync(self, event: TrackerSyncEvent) -> TrackerSyncSummary:
        self.events.append(event)
        return TrackerSyncSummary(
            sync_key="training-checklist",
            source=event.source,
            source_id=event.source_id,
            success=True,
            asana_synced=True,
        )


def test_run_artifact_service_generates_and_writes_artifacts(tmp_path: Path):
    stats = _StatsStub()
    manifest_path = tmp_path / "MASTER_STAGE_MANIFEST.json"
    manifest_path.write_text("{}", encoding="utf-8")

    checklist_path = tmp_path / "training_run_checklist.json"
    checklist_path.write_text(
        json.dumps({"ops_freshness": {"all_fresh": True}}), encoding="utf-8"
    )
    asana_mapping_path = tmp_path / "asana_task_key_mapping.json"
    asana_mapping_path.write_text("{}", encoding="utf-8")
    asana_transition_path = tmp_path / "asana_task_transition_results.json"
    asana_transition_path.write_text("{}", encoding="utf-8")

    service = RunArtifactService(
        paths=RunArtifactPaths(
            tracker_sync_output_path=str(checklist_path),
            asana_task_key_mapping_output_path=str(asana_mapping_path),
            asana_task_transition_output_path=str(asana_transition_path),
            stage_health_report_output_path=str(tmp_path / "stage_health_report.json"),
            closure_pack_output_path=str(tmp_path / "closure_pack.json"),
        ),
        stats=stats,
        stage_distribution={
            "stage1_foundation": 0.4,
            "stage4_voice_persona": 0.15,
        },
        fail_on_missing_stage_artifacts=True,
        stage_drift_tolerance=0.02,
        stage_drift_waivers={"stage4_voice_persona": "approved"},
        manifest_path=manifest_path,
        enable_asana_sync=True,
    )

    report = service.generate_report()
    assert report["total_samples"] == 12
    assert report["actual_stage_percentages"]["stage1_foundation"] == 7 / 12

    stage_health_report = service.build_stage_health_report(report)
    assert stage_health_report["pass"] is True
    service.write_stage_health_report(stage_health_report)
    assert Path(service.paths.stage_health_report_output_path).exists()

    closure_pack = service.build_mtgc_closure_pack(report, stage_health_report)
    assert closure_pack["overall_pass"] is True
    service.write_mtgc_closure_pack(closure_pack)
    assert Path(service.paths.closure_pack_output_path).exists()


def test_checklist_tracker_sync_service_writes_checklist_and_syncs(tmp_path: Path):
    stats = _StatsStub()
    config = _ConfigStub(
        tracker_sync_output_path=str(tmp_path / "training_run_checklist.json"),
        tracker_sync_state_output_path=str(tmp_path / "tracker_sync_state.json"),
        stage_health_report_output_path=str(tmp_path / "stage_health_report.json"),
    )
    tracker = _TrackerCoordinatorStub()

    service = ChecklistTrackerSyncService(
        config=config,
        stats=stats,
        stage_drift_tolerance=0.02,
        collect_ops_freshness=lambda: {"all_fresh": True},
        asana_sync=lambda payload, path: None,
        tracker_sync_factory=lambda: tracker,
    )

    report = {
        "total_samples": 12,
        "stage_balance": {"stage1_foundation": {"drift_vs_target": 0.01}},
        "split_counts": {"aggregate": {"train": 8, "val": 2, "test": 2}},
        "warnings": [],
        "errors": [],
    }

    service.sync_run_checklist(report)

    checklist_path = Path(config.tracker_sync_output_path)
    assert checklist_path.exists()
    payload = json.loads(checklist_path.read_text(encoding="utf-8"))
    assert payload["ops_freshness"]["all_fresh"] is True
    assert payload["stage_drift_within_tolerance"] is True
    assert tracker.events
