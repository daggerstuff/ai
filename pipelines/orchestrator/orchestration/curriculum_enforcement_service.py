"""Curriculum and stage-quality enforcement for assembled training datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Protocol

from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.curriculum_enforcement")


class CurriculumConfigProtocol(Protocol):
    stage_distribution: dict[str, float]
    target_total_samples: int
    fail_on_missing_stage_artifacts: bool
    fail_on_stage_drift: bool


class CurriculumStatsProtocol(Protocol):
    warnings: list[str]
    samples_by_stage: dict[str, int]
    stage_balance: dict[str, dict[str, Any]]
    stage_policy_enforcement: dict[str, Any]


class DriftPolicyServiceProtocol(Protocol):
    def resolve_stage_drift_tolerance(
        self,
        stage: str,
        *,
        drift_waivers: dict[str, dict[str, Any]],
    ) -> tuple[float, bool]: ...


class CurriculumEnforcementService:
    """Own curriculum-stage balancing and stage-quality enforcement policy."""

    def __init__(
        self,
        *,
        config: CurriculumConfigProtocol,
        stats: CurriculumStatsProtocol,
        runtime_policy_service: DriftPolicyServiceProtocol,
        stage_drift_waivers: dict[str, dict[str, Any]],
        stage_quality_profiles: dict[str, dict[str, Any]],
    ) -> None:
        self.config = config
        self.stats = stats
        self.runtime_policy_service = runtime_policy_service
        self.stage_drift_waivers = stage_drift_waivers
        self.stage_quality_profiles = stage_quality_profiles

    def validate_required_stage_artifacts(self) -> None:
        """Preflight required Stage 3/4 artifacts before dataset assembly."""
        required_artifacts = {
            "stage3_edge_stress_test": [
                Path("ai/pipelines/edge_case/output/edge_cases_training_format.jsonl"),
                Path("ai/pipelines/orchestrator/prompt_corpus"),
            ],
            "stage4_voice_persona": [
                Path("ai/data/tim_fletcher_voice"),
                Path("ai/training_data_consolidated/transcripts"),
            ],
        }

        missing: dict[str, list[str]] = {}
        for stage, paths in required_artifacts.items():
            if self.config.stage_distribution.get(stage, 0.0) <= 0:
                continue
            stage_missing = [str(path) for path in paths if not path.exists()]
            if stage_missing:
                missing[stage] = stage_missing

        if not missing:
            logger.info("✅ All required stage artifacts present")
            return

        for stage, paths in missing.items():
            self._warn(f"Missing required artifacts for {stage}: {', '.join(paths)}")

        if self.config.fail_on_missing_stage_artifacts:
            missing_str = " | ".join(
                f"{stage}: {', '.join(paths)}" for stage, paths in missing.items()
            )
            raise RuntimeError(
                "STRICT MODE: Required stage artifacts missing. "
                "To override, set TRAINING_ALLOW_MISSING_ARTIFACTS=true "
                "(development only). Missing: " + missing_str
            )

        logger.warning(
            "⚠️  NON-STRICT MODE: Continuing despite missing artifacts. "
            "Dataset quality may be reduced."
        )

    def balance_dataset(
        self, data: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """Balance the dataset according to the configured stage distribution."""
        logger.info("⚖️  Balancing dataset by stage...")

        stage_buckets: dict[str, list[dict[str, Any]]] = {}
        for item in data:
            stage = item.get("metadata", {}).get("stage", "stage1_foundation")
            stage_buckets.setdefault(stage, []).append(item)

        balanced: list[dict[str, Any]] = []
        stage_segments: dict[str, list[dict[str, Any]]] = {}

        for stage, percentage in self.config.stage_distribution.items():
            target_count = int(self.config.target_total_samples * percentage)
            bucket = stage_buckets.get(stage, [])

            if not bucket:
                self._warn(f"No data found for stage '{stage}' (target: {target_count}).")
                self.stats.stage_balance[stage] = {
                    "target": target_count,
                    "available": 0,
                    "actual": 0,
                }
                continue

            if len(bucket) <= target_count:
                stage_sample = bucket
                if len(bucket) < target_count:
                    self._warn(
                        f"Stage '{stage}' has only {len(bucket)} samples (target: {target_count})."
                    )
            else:
                stage_sample = random.sample(bucket, target_count)

            balanced.extend(stage_sample)
            stage_segments[stage] = stage_sample
            actual = len(stage_sample)
            self.stats.samples_by_stage[stage] = actual
            self.stats.stage_balance[stage] = {
                "target": target_count,
                "available": len(bucket),
                "actual": actual,
            }

        passthrough_stages = sorted(
            stage for stage in stage_buckets if stage not in self.config.stage_distribution
        )
        for stage in passthrough_stages:
            records = list(stage_buckets[stage])
            balanced.extend(records)
            stage_segments[stage] = records
            self.stats.samples_by_stage[stage] = len(records)
            self.stats.stage_balance[stage] = {
                "target": None,
                "available": len(records),
                "actual": len(records),
                "passthrough": True,
            }
            logger.info(
                "   Preserved %s passthrough records for non-curriculum lane '%s'",
                len(records),
                stage,
            )

        logger.info("   Stage-balanced to %s samples", len(balanced))
        return balanced, stage_segments

    def validate_final_stage_balance(self, data: list[dict[str, Any]]) -> None:
        """Record final stage percentages and fail or warn on drift violations."""
        if not data:
            return

        counts: dict[str, int] = {}
        for item in data:
            stage = item.get("metadata", {}).get("stage", "stage1_foundation")
            counts[stage] = counts.get(stage, 0) + 1

        total = len(data)
        self.stats.samples_by_stage = counts
        drift_violations: list[str] = []

        for stage, target in self.config.stage_distribution.items():
            actual_count = counts.get(stage, 0)
            actual_share = actual_count / total if total else 0.0
            drift = actual_share - target
            tolerance, waiver_applied = self.runtime_policy_service.resolve_stage_drift_tolerance(
                stage,
                drift_waivers=self.stage_drift_waivers,
            )

            balance = self.stats.stage_balance.get(stage, {})
            balance.update(
                {
                    "final_actual": actual_count,
                    "final_share": round(actual_share, 6),
                    "drift_vs_target": round(drift, 6),
                    "drift_tolerance": round(tolerance, 6),
                    "drift_waiver_applied": waiver_applied,
                }
            )
            self.stats.stage_balance[stage] = balance

            if abs(drift) > tolerance:
                warning = (
                    f"Stage '{stage}' drift {drift:+.3f} exceeds tolerance "
                    f"{tolerance:.2f} (target={target:.2%}, actual={actual_share:.2%}, "
                    f"waiver_applied={waiver_applied})"
                )
                self._warn(warning)
                drift_violations.append(stage)

        if self.config.fail_on_stage_drift and drift_violations:
            raise RuntimeError(
                "Stage balance drift exceeds tolerance: " + ", ".join(drift_violations)
            )

    def apply_stage_quality_profiles(
        self, data: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        """Apply stage-specific quality profile constraints."""
        kept: list[dict[str, Any]] = []
        removed_count = 0
        removed_by_stage: dict[str, int] = {}
        crisis_override_by_stage: dict[str, int] = {}
        fail_reasons_by_stage: dict[str, dict[str, int]] = {}

        for item in data:
            metadata = item.get("metadata", {})
            stage = metadata.get("stage", "stage1_foundation")
            profile = self.stage_quality_profiles.get(stage)
            if not profile:
                kept.append(item)
                continue

            score = metadata.get("quality_scoring_v1", {})
            signals = score.get("signals", {}) if isinstance(score, dict) else {}
            empathy = float(signals.get("empathy", 0.0))
            safety = 1.0 - float(signals.get("harm", 0.0))
            bias_score = metadata.get("bias_score")
            bias = float(bias_score) if isinstance(bias_score, (int, float)) else 0.0

            min_empathy = profile.get("min_empathy")
            min_safety = profile.get("min_safety")
            bias_max = profile.get("bias_max")
            requires_reasoning_metadata = bool(
                profile.get("requires_reasoning_metadata", False)
            )
            requires_voice_signature = bool(
                profile.get("requires_voice_signature", False)
            )
            allow_crisis_override = bool(profile.get("allow_crisis_override", False))

            if allow_crisis_override and metadata.get("crisis_intensity") == "very_high":
                crisis_override_by_stage[stage] = crisis_override_by_stage.get(stage, 0) + 1
                kept.append(item)
                continue

            fail_reasons: list[str] = []
            if isinstance(min_empathy, (int, float)) and empathy < float(min_empathy):
                fail_reasons.append(
                    f"empathy {empathy:.3f} < min_empathy {float(min_empathy):.3f}"
                )
            if isinstance(min_safety, (int, float)) and safety < float(min_safety):
                fail_reasons.append(
                    f"safety {safety:.3f} < min_safety {float(min_safety):.3f}"
                )
            if isinstance(bias_max, (int, float)) and bias > float(bias_max):
                fail_reasons.append(f"bias {bias:.3f} > bias_max {float(bias_max):.3f}")

            if requires_reasoning_metadata:
                reasoning_keys = ("chain_of_thought", "summary", "technique")
                if not any(key in metadata and metadata.get(key) for key in reasoning_keys):
                    fail_reasons.append("missing required reasoning metadata")

            if requires_voice_signature:
                if not metadata.get("voice_signature"):
                    fail_reasons.append("missing required voice_signature")
                if not metadata.get("persona_id"):
                    fail_reasons.append("missing required persona_id")

            if fail_reasons:
                removed_count += 1
                removed_by_stage[stage] = removed_by_stage.get(stage, 0) + 1
                stage_fail_reasons = fail_reasons_by_stage.setdefault(stage, {})
                for reason in fail_reasons:
                    stage_fail_reasons[reason] = stage_fail_reasons.get(reason, 0) + 1
                self.stats.warnings.append(
                    f"Dropped sample from {stage} due to stage quality profile: "
                    + "; ".join(fail_reasons)
                )
                continue

            kept.append(item)

        self.stats.stage_policy_enforcement = {
            "removed_total": removed_count,
            "removed_by_stage": removed_by_stage,
            "crisis_overrides_by_stage": crisis_override_by_stage,
            "failure_reasons_by_stage": fail_reasons_by_stage,
        }
        return kept, removed_count

    def _warn(self, message: str) -> None:
        logger.warning(message)
        self.stats.warnings.append(message)


__all__ = ["CurriculumEnforcementService"]
