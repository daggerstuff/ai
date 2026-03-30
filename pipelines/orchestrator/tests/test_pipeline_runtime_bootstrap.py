from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ai.pipelines.orchestrator.orchestration.pipeline_runtime_bootstrap import (
    PipelineRuntimeBootstrap,
)
from ai.pipelines.orchestrator.storage_config import StorageBackend, StorageConfig


@dataclass
class _ConfigStub:
    stage_distribution: dict[str, float] = field(
        default_factory=lambda: {
            "stage1_foundation": 0.4,
            "stage2_therapeutic_expertise": 0.25,
            "stage3_edge_stress_test": 0.2,
            "stage4_voice_persona": 0.15,
        }
    )
    fail_on_stage_drift: bool = True
    fail_on_missing_stage_artifacts: bool = True
    edge_cases: object | None = None
    pixel_voice: object | None = None
    psychology_knowledge: object | None = None
    dual_persona: object | None = None
    standard_therapeutic: object | None = None


def test_pipeline_runtime_bootstrap_applies_strict_mode_override(monkeypatch):
    config = _ConfigStub()
    bootstrap = PipelineRuntimeBootstrap(
        manifest_path=Path("missing.json"),
        stage_drift_tolerance=0.02,
    )

    monkeypatch.setenv("TRAINING_STRICT_MODE", "false")

    bootstrap.apply_strict_mode_overrides(config)

    assert config.fail_on_stage_drift is False
    assert config.fail_on_missing_stage_artifacts is False


def test_pipeline_runtime_bootstrap_prefers_s3_when_curated_sources_use_s3():
    config = _ConfigStub()
    bootstrap = PipelineRuntimeBootstrap(
        manifest_path=Path("missing.json"),
        stage_drift_tolerance=0.02,
        storage_config=StorageConfig(
            backend=StorageBackend.LOCAL,
            local_base_path=Path("/tmp/pixelated-storage"),
            s3_bucket="pixel-data",
            s3_region="sfo3",
            s3_access_key_id="key",
            s3_secret_access_key="secret",
            s3_endpoint_url="https://sfo3.digitaloceanspaces.com",
        ),
    )

    storage = bootstrap.initialize_storage_manager(
        [
            "s3://pixel-data/curated_sources/consolidated/final_datasets/ULTIMATE_FINAL_DATASET.jsonl",
            "ai/pipelines/dual_persona",
        ]
    )

    assert storage is not None
    assert storage.config.backend == StorageBackend.S3
