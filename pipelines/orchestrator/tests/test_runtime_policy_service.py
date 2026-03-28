from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from ai.pipelines.orchestrator.orchestration.runtime_policy_service import (
    RuntimePolicyService,
)


@dataclass
class _ConfigStub:
    asana_project_gid: str | None = None
    asana_section_gid: str | None = None
    asana_parent_task_gid: str | None = None
    enable_asana_sync: bool = True
    enable_beads_sync: bool = True
    enable_jira_sync: bool = True
    enable_linear_sync: bool = True
    tracker_sync_state_output_path: str = "tracker-state.json"
    stage_distribution: dict[str, float] = field(
        default_factory=lambda: {
            "stage1_foundation": 0.4,
            "stage2_therapeutic_expertise": 0.25,
            "stage3_edge_stress_test": 0.2,
            "stage4_voice_persona": 0.15,
        }
    )


@dataclass
class _WarningSink:
    warnings: list[str] = field(default_factory=list)


def test_runtime_policy_service_hydrates_tracker_env(monkeypatch):
    config = _ConfigStub()
    warnings = _WarningSink()
    service = RuntimePolicyService(
        manifest_path=Path("missing.json"),
        warning_sink=warnings,
        default_stage_distribution=config.stage_distribution,
        default_stage_drift_tolerance=0.02,
    )

    monkeypatch.setenv("ASANA_PROJECT_GID", "123")
    monkeypatch.setenv("ASANA_SECTION_GID", "456")
    monkeypatch.setenv("ENABLE_BEADS_SYNC", "false")
    monkeypatch.setenv("TRACKER_SYNC_STATE_PATH", "/tmp/custom-state.json")

    service.hydrate_tracker_config(config)

    assert config.asana_project_gid == "123"
    assert config.asana_section_gid == "456"
    assert config.enable_beads_sync is False
    assert config.tracker_sync_state_output_path == "/tmp/custom-state.json"


def test_runtime_policy_service_loads_manifest_policy(tmp_path: Path):
    manifest_path = tmp_path / "training_policy_manifest.json"
    manifest_path.write_text(
        """
        {
          "stages": {
            "stage1_foundation": {
              "target_percentage": 0.5,
              "quality_profile": {"min_quality": 0.8}
            },
            "stage2_therapeutic_expertise": {
              "target_percentage": 0.2,
              "quality_profile": {"min_quality": 0.7}
            },
            "stage3_edge_stress_test": {"target_percentage": 0.2},
            "stage4_voice_persona": {"target_percentage": 0.1}
          },
          "stage_drift_waivers": {
            "stage4_voice_persona": {
              "max_drift": 0.05,
              "reason": "manual override",
              "approved_by": "ops",
              "expires_at": "2999-01-01T00:00:00Z"
            }
          }
        }
        """,
        encoding="utf-8",
    )
    warnings = _WarningSink()
    service = RuntimePolicyService(
        manifest_path=manifest_path,
        warning_sink=warnings,
        default_stage_distribution={
            "stage1_foundation": 0.4,
            "stage2_therapeutic_expertise": 0.25,
            "stage3_edge_stress_test": 0.2,
            "stage4_voice_persona": 0.15,
        },
        default_stage_drift_tolerance=0.02,
    )

    bundle = service.load_stage_policy()

    assert bundle.stage_distribution["stage1_foundation"] == 0.5
    assert bundle.quality_profiles["stage1_foundation"]["min_quality"] == 0.8
    assert bundle.drift_waivers["stage4_voice_persona"]["max_drift"] == 0.05
    tolerance, waiver_applied = service.resolve_stage_drift_tolerance(
        "stage4_voice_persona",
        drift_waivers=bundle.drift_waivers,
    )
    assert tolerance == 0.05
    assert waiver_applied is True


def test_runtime_policy_service_collects_ops_freshness(tmp_path: Path, monkeypatch):
    inventory = tmp_path / "manifest.json"
    inventory.write_text("{}", encoding="utf-8")
    prompt_mirror = tmp_path / "prompt_corpus"
    prompt_mirror.mkdir()
    voice_export = tmp_path / "transcripts"
    voice_export.mkdir()

    monkeypatch.setenv("TRAINING_OPS_INVENTORY_PATH", str(inventory))
    monkeypatch.setenv("TRAINING_OPS_PROMPT_MIRROR_PATH", str(prompt_mirror))
    monkeypatch.setenv("TRAINING_OPS_VOICE_EXPORT_PATH", str(voice_export))
    monkeypatch.setenv("TRAINING_OPS_FRESHNESS_HOURS", "1000")

    warnings = _WarningSink()
    service = RuntimePolicyService(
        manifest_path=Path("missing.json"),
        warning_sink=warnings,
        default_stage_distribution={"stage1_foundation": 1.0},
        default_stage_drift_tolerance=0.02,
    )

    freshness = service.collect_ops_freshness()

    assert freshness["all_fresh"] is True
    assert freshness["checks"]["inventory"]["exists"] is True
    assert freshness["checks"]["prompt_mirror"]["fresh"] is True
