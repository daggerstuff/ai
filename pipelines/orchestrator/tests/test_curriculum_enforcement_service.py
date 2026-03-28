from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ai.pipelines.orchestrator.orchestration.curriculum_enforcement_service import (
    CurriculumEnforcementService,
)


@dataclass
class _ConfigStub:
    stage_distribution: dict[str, float] = field(
        default_factory=lambda: {
            "stage1_foundation": 0.5,
            "stage4_voice_persona": 0.5,
        }
    )
    target_total_samples: int = 4
    fail_on_missing_stage_artifacts: bool = False
    fail_on_stage_drift: bool = True


@dataclass
class _StatsStub:
    warnings: list[str] = field(default_factory=list)
    samples_by_stage: dict[str, int] = field(default_factory=dict)
    stage_balance: dict[str, dict[str, Any]] = field(default_factory=dict)
    stage_policy_enforcement: dict[str, Any] = field(default_factory=dict)


class _RuntimePolicyStub:
    def __init__(self, tolerance: float = 0.2, waiver_applied: bool = False) -> None:
        self.tolerance = tolerance
        self.waiver_applied = waiver_applied

    def resolve_stage_drift_tolerance(
        self,
        stage: str,
        *,
        drift_waivers: dict[str, dict[str, Any]],
    ) -> tuple[float, bool]:
        return self.tolerance, self.waiver_applied


def test_balance_dataset_preserves_passthrough_lanes():
    config = _ConfigStub()
    stats = _StatsStub()
    service = CurriculumEnforcementService(
        config=config,
        stats=stats,
        runtime_policy_service=_RuntimePolicyStub(),
        stage_drift_waivers={},
        stage_quality_profiles={},
    )

    data = [
        {"metadata": {"stage": "stage1_foundation"}, "id": "s1a"},
        {"metadata": {"stage": "stage1_foundation"}, "id": "s1b"},
        {"metadata": {"stage": "stage4_voice_persona"}, "id": "s4a"},
        {"metadata": {"stage": "stage4_voice_persona"}, "id": "s4b"},
        {"metadata": {"stage": "continuity_holdout"}, "id": "holdout"},
    ]

    balanced, segments = service.balance_dataset(data)

    assert len(balanced) == 5
    assert len(segments["stage1_foundation"]) == 2
    assert len(segments["stage4_voice_persona"]) == 2
    assert len(segments["continuity_holdout"]) == 1
    assert stats.stage_balance["continuity_holdout"]["passthrough"] is True


def test_validate_final_stage_balance_raises_on_excessive_drift():
    config = _ConfigStub()
    stats = _StatsStub()
    service = CurriculumEnforcementService(
        config=config,
        stats=stats,
        runtime_policy_service=_RuntimePolicyStub(tolerance=0.05),
        stage_drift_waivers={},
        stage_quality_profiles={},
    )

    data = [
        {"metadata": {"stage": "stage1_foundation"}},
        {"metadata": {"stage": "stage1_foundation"}},
        {"metadata": {"stage": "stage1_foundation"}},
        {"metadata": {"stage": "stage1_foundation"}},
    ]

    with pytest.raises(RuntimeError, match="Stage balance drift exceeds tolerance"):
        service.validate_final_stage_balance(data)


def test_apply_stage_quality_profiles_filters_by_profile_rules():
    config = _ConfigStub()
    stats = _StatsStub()
    service = CurriculumEnforcementService(
        config=config,
        stats=stats,
        runtime_policy_service=_RuntimePolicyStub(),
        stage_drift_waivers={},
        stage_quality_profiles={
            "stage4_voice_persona": {
                "min_empathy": 0.7,
                "requires_voice_signature": True,
            }
        },
    )

    kept, removed = service.apply_stage_quality_profiles(
        [
            {
                "metadata": {
                    "stage": "stage4_voice_persona",
                    "quality_scoring_v1": {"signals": {"empathy": 0.8, "harm": 0.0}},
                    "voice_signature": "tf",
                    "persona_id": "tim-fletcher",
                }
            },
            {
                "metadata": {
                    "stage": "stage4_voice_persona",
                    "quality_scoring_v1": {"signals": {"empathy": 0.2, "harm": 0.0}},
                }
            },
        ]
    )

    assert len(kept) == 1
    assert removed == 1
    assert stats.stage_policy_enforcement["removed_total"] == 1
