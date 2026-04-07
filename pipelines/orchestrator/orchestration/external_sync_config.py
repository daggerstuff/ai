"""External tracker and artifact sync configuration for dataset assembly runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExternalSyncConfig:
    """Tracker and artifact emission settings for one integrated run."""

    enable_progress_tracking: bool = True
    progress_tracker_path: str = "ai/lightning/therapeutic_progress_tracker.py"
    enable_tracker_sync: bool = True
    tracker_sync_output_path: str = "ai/lightning/training_run_checklist.json"
    tracker_sync_state_output_path: str = "ai/lightning/tracker_sync_state.json"
    enable_asana_sync: bool = True
    enable_beads_sync: bool = True
    enable_jira_sync: bool = True
    enable_linear_sync: bool = True
    enable_dataset_asana_sync: bool = True
    asana_project_gid: str | None = None
    asana_section_gid: str | None = None
    asana_dataset_section_gid: str | None = None
    asana_parent_task_gid: str | None = None
    asana_task_gid_output_path: str = "ai/lightning/training_run_asana_task_gid.txt"
    asana_task_key_mapping_output_path: str = "ai/lightning/asana_task_key_mapping.json"
    asana_task_transition_output_path: str = (
        "ai/lightning/asana_task_transition_results.json"
    )
    asana_dataset_mapping_output_path: str = (
        "ai/lightning/asana_dataset_task_mapping.json"
    )
    dataset_scan_directory: str = "ai/datasets"
    dataset_file_patterns: list[str] = field(
        default_factory=lambda: [".jsonl", ".json", ".csv", ".parquet"]
    )
    dataset_task_prefix: str = "DATASET"
    stage_health_report_output_path: str = (
        "ai/lightning/integrated_stage_health_report.json"
    )
    closure_pack_output_path: str = "ai/lightning/mtgc_closure_pack.json"
