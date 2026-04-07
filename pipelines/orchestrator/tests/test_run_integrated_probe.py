from __future__ import annotations

from argparse import Namespace

from ai.pipelines.orchestrator.scripts.run_integrated_probe import (
    STANDARD_THERAPEUTIC_MONOLITH,
    STANDARD_THERAPEUTIC_SPECIALISTS,
    _build_config,
    _sufficiency,
)


def test_build_config_applies_probe_overrides():
    args = Namespace(
        preset=None,
        target_total=1000,
        standard_max_samples=2000,
        edge_max_samples=300,
        specialists_first=True,
        disable_source=["dual_persona", "psychology_knowledge"],
        output=None,
    )

    config = _build_config(args)

    assert config.target_total_samples == 1000
    assert config.enable_bias_detection is False
    assert config.enable_quality_validation is False
    assert config.standard_therapeutic.max_samples == 2000
    assert config.edge_cases.max_samples == 300
    assert config.dual_persona.enabled is False
    assert config.psychology_knowledge.enabled is False
    assert config.sync.enable_tracker_sync is False
    assert config.sync.enable_asana_sync is False
    assert config.standard_therapeutic.source_path is None
    assert config.standard_therapeutic.source_paths == (
        *STANDARD_THERAPEUTIC_SPECIALISTS,
        STANDARD_THERAPEUTIC_MONOLITH,
    )


def test_build_config_applies_full_capped_preset():
    args = Namespace(
        preset="full-capped-1k",
        target_total=999,
        standard_max_samples=None,
        edge_max_samples=None,
        specialists_first=False,
        disable_source=[],
        output=None,
    )

    config = _build_config(args)

    assert config.target_total_samples == 1000
    assert config.standard_therapeutic.max_samples == 2000
    assert config.edge_cases.max_samples == 2000
    assert config.standard_therapeutic.source_paths == (
        *STANDARD_THERAPEUTIC_SPECIALISTS,
        STANDARD_THERAPEUTIC_MONOLITH,
    )


def test_sufficiency_reports_deficits():
    result = _sufficiency(
        {
            "stage1_foundation": 400,
            "stage2_therapeutic_expertise": 250,
        },
        {
            "stage1_foundation": 450,
            "stage2_therapeutic_expertise": 200,
        },
    )

    assert result["stage1_foundation"]["sufficient"] is True
    assert result["stage1_foundation"]["deficit"] == 0
    assert result["stage2_therapeutic_expertise"]["sufficient"] is False
    assert result["stage2_therapeutic_expertise"]["deficit"] == 50
