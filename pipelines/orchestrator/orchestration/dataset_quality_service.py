"""Quality filtering and aggregate stats helpers for dataset assembly workflows."""

from __future__ import annotations

from typing import Any, Protocol

from ai.pipelines.orchestrator.quality.evidence_based_practice_validator import (
    validate_bias,
)
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.dataset_quality_service")


class QualityStatsProtocol(Protocol):
    total_samples: int
    samples_by_stage: dict[str, int]
    samples_by_category: dict[str, int]
    bias_detection_results: dict[str, Any]


class CurriculumEnforcementProtocol(Protocol):
    def apply_stage_quality_profiles(
        self, data: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]: ...


class DatasetQualityService:
    """Own bias filtering, quality scoring, and final stat hydration."""

    def __init__(
        self,
        *,
        stats: QualityStatsProtocol,
        curriculum_enforcement_service: CurriculumEnforcementProtocol,
        stage_quality_profiles: dict[str, Any],
    ) -> None:
        self.stats = stats
        self.curriculum_enforcement_service = curriculum_enforcement_service
        self.stage_quality_profiles = stage_quality_profiles

    def run_bias_detection(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run bias detection on training data."""
        logger.info("🔍 Running bias detection...")
        try:
            flagged_count = 0
            filtered_data: list[dict[str, Any]] = []

            for item in data:
                text = item.get("text", "")
                if validate_bias(text):
                    filtered_data.append(item)
                else:
                    flagged_count += 1

            self.stats.bias_detection_results = {
                "total_checked": len(data),
                "flagged": flagged_count,
                "passed": len(filtered_data),
            }
            logger.info("   Flagged %s items for bias", flagged_count)
            return filtered_data
        except Exception as exc:
            logger.warning("Bias detection failed: %s", exc)
            return data

    def run_quality_validation(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run quality validation on training data using Quality Scoring v1."""
        logger.info("✓ Running quality validation...")
        try:
            from ai.pipelines.orchestrator.quality.quality_filter_v1 import (
                QualityFilterV1,
            )

            quality_filter = QualityFilterV1(
                min_decision="curate",
                min_composite=0.6,
                enabled=True,
            )
            filtered, scoring_results = quality_filter.filter_batch(data)

            for item, result in zip(data, scoring_results):
                metadata = item.setdefault("metadata", {})
                metadata["quality_scoring_v1"] = result

            if self.stage_quality_profiles:
                filtered, stage_filtered_count = (
                    self.curriculum_enforcement_service.apply_stage_quality_profiles(
                        filtered
                    )
                )
                if stage_filtered_count > 0:
                    logger.info(
                        "   Stage policy filters removed %s additional samples",
                        stage_filtered_count,
                    )

            logger.info(
                "   Validated %s samples, filtered to %s high-quality samples",
                len(data),
                len(filtered),
            )
            return filtered
        except ImportError:
            logger.warning(
                "Quality Scoring v1 not available, skipping quality validation"
            )
            logger.info("   Validated %s samples (no filtering)", len(data))
            return data

    def finalize_stats(self, data: list[dict[str, Any]]) -> None:
        """Hydrate aggregate stats after final dataset assembly."""
        self.stats.total_samples = len(data)
        category_counts: dict[str, int] = {}
        for item in data:
            metadata = item.get("metadata", {})
            source_family = metadata.get("source_family") or metadata.get("source")
            if isinstance(source_family, str) and source_family:
                category_counts[source_family] = category_counts.get(source_family, 0) + 1
        self.stats.samples_by_category = category_counts


__all__ = ["DatasetQualityService"]
