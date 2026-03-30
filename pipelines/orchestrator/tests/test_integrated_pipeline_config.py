from __future__ import annotations

from importlib import reload

from ai.pipelines.orchestrator.orchestration import integrated_training_pipeline as itp


def test_integrated_pipeline_config_defaults_to_curated_s3_prefix(monkeypatch):
    monkeypatch.setenv("PIXELATED_CURATED_S3_PREFIX", "s3://pixel-data/curated_sources")
    module = reload(itp)

    config = module.IntegratedPipelineConfig()

    assert (
        config.edge_cases.source_path
        == "s3://pixel-data/curated_sources/consolidated/edge_cases/existing_edge_cases.jsonl"
    )
    assert (
        config.psychology_knowledge.source_path
        == "s3://pixel-data/curated_sources/consolidated/psychology_knowledge/enhanced_psychology_knowledge_base.json"
    )
    assert (
        config.standard_therapeutic.source_path
        == "s3://pixel-data/curated_sources/consolidated/final_datasets/ULTIMATE_FINAL_DATASET.jsonl"
    )


def test_integrated_pipeline_config_supports_source_path_env_overrides(monkeypatch):
    monkeypatch.setenv("PIXELATED_CURATED_S3_PREFIX", "s3://pixel-data/curated_sources")
    monkeypatch.setenv("PIXELATED_PIXEL_VOICE_SOURCE_PATH", "s3://pixel-data/voice_exports")
    monkeypatch.setenv(
        "PIXELATED_STANDARD_THERAPEUTIC_SOURCE_PATH",
        "s3://pixel-data/custom/standard.jsonl",
    )
    module = reload(itp)

    config = module.IntegratedPipelineConfig()

    assert config.pixel_voice.source_path == "s3://pixel-data/voice_exports"
    assert config.standard_therapeutic.source_path == "s3://pixel-data/custom/standard.jsonl"


def test_integrated_training_pipeline_keeps_service_bundle_off_public_surface():
    pipeline = itp.IntegratedTrainingPipeline()

    assert hasattr(pipeline, "services")
    assert hasattr(pipeline, "context")
    assert not hasattr(pipeline, "asana_client")


def test_integrated_pipeline_config_exposes_nested_policy_domains():
    config = itp.IntegratedPipelineConfig()

    assert config.quality.enable_bias_detection is True
    assert config.sync.enable_asana_sync is True
    assert config.outputs.output_filename == "training_dataset.json"
    assert config.enable_asana_sync is True
    assert config.output_filename == "training_dataset.json"
