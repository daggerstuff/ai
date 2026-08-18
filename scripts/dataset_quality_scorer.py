#!/usr/bin/env python3
"""
Dataset quality scoring script that computes quality metrics
and assigns quality tiers based on configurable thresholds.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from s3_client_helper import get_s3_client


class DatasetQualityScorer:
    """Computes quality metrics and assigns quality tiers to datasets."""

    # Quality tier thresholds
    QUALITY_THRESHOLDS = {
        "excellent": 90,
        "good": 75,
        "acceptable": 60,
        "needs_review": 0,
    }

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.s3_client = get_s3_client()

    def load_registry(self) -> dict[str, Any]:
        """Load the dataset registry."""
        with open(self.registry_path) as f:
            return json.load(f)

    def save_registry(self, registry: dict[str, Any]) -> None:
        """Save the updated registry."""
        registry["last_updated"] = datetime.now(UTC).isoformat() + "Z"
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    def calculate_completeness_score(self, dataset_entry: dict[str, Any]) -> float:
        """
        Calculate completeness score based on metadata presence.

        Args:
            dataset_entry: Dataset entry from registry

        Returns:
            Completeness score (0-100)
        """
        required_fields = ["path", "type", "focus", "stage", "quality_profile"]

        optional_fields = ["size_mb", "fallback_paths", "notes"]

        enhanced_fields = [
            "validation",
            "usage_analytics",
            "quality_metrics",
            "lineage",
            "sync_status",
            "version_control",
        ]

        score = 0.0

        # Check required fields (50 points)
        required_present = sum(1 for field in required_fields if field in dataset_entry)
        score += (required_present / len(required_fields)) * 50

        # Check optional fields (25 points)
        optional_present = sum(1 for field in optional_fields if field in dataset_entry)
        score += (optional_present / len(optional_fields)) * 25

        # Check enhanced fields (25 points)
        enhanced_present = sum(1 for field in enhanced_fields if field in dataset_entry)
        score += (enhanced_present / len(enhanced_fields)) * 25

        return round(score, 2)

    def calculate_consistency_score(self, dataset_entry: dict[str, Any]) -> float:
        """
        Calculate consistency score based on validation and sync status.

        Args:
            dataset_entry: Dataset entry from registry

        Returns:
            Consistency score (0-100)
        """
        score = 100.0

        # Check validation status
        validation = dataset_entry.get("validation", {})
        if validation:
            if validation.get("validation_status") == "failed":
                score -= 30
            elif validation.get("validation_status") == "pending":
                score -= 10

            if validation.get("validation_errors"):
                score -= min(len(validation.get("validation_errors", [])) * 5, 20)

        # Check sync status
        sync_status = dataset_entry.get("sync_status", {})
        if sync_status:
            if not sync_status.get("s3_synced", True):
                score -= 20

            if sync_status.get("sync_discrepancies"):
                score -= min(len(sync_status.get("sync_discrepancies", [])) * 5, 15)

        return max(0.0, round(score, 2))

    def calculate_annotation_quality(self, dataset_entry: dict[str, Any]) -> float | None:
        """
        Calculate annotation quality based on dataset type and profile.

        Args:
            dataset_entry: Dataset entry from registry

        Returns:
            Annotation quality score (0-100) or None if not applicable
        """
        dataset_type = dataset_entry.get("type", "")
        quality_profile = dataset_entry.get("quality_profile", "")

        # Base scores by quality profile
        profile_scores = {
            "foundation": 85.0,
            "cot_reasoning": 90.0,
            "emotion_recognition": 80.0,
            "advanced_reasoning": 85.0,
            "edge_crisis": 70.0,
            "edge_raw": 65.0,
            "rlhf": 90.0,
            "voice": 75.0,
        }

        base_score = profile_scores.get(quality_profile, 70.0)

        # Adjustments by dataset type
        type_adjustments = {
            "curated_gold": 10.0,
            "professional": 5.0,
            "chain_of_thought": 5.0,
            "synthetic_edge": -5.0,
            "raw_forum": -10.0,
        }

        adjustment = type_adjustments.get(dataset_type, 0.0)

        return max(0.0, min(100.0, round(base_score + adjustment, 2)))

    def determine_quality_tier(self, quality_score: float) -> str:
        """
        Determine quality tier based on score.

        Args:
            quality_score: Quality score (0-100)

        Returns:
            Quality tier name
        """
        if quality_score >= self.QUALITY_THRESHOLDS["excellent"]:
            return "excellent"
        if quality_score >= self.QUALITY_THRESHOLDS["good"]:
            return "good"
        if quality_score >= self.QUALITY_THRESHOLDS["acceptable"]:
            return "acceptable"
        return "needs_review"

    def detect_anomalies(self, dataset_entry: dict[str, Any]) -> list[str]:
        """
        Detect anomalies in dataset.

        Args:
            dataset_entry: Dataset entry from registry

        Returns:
            List of anomaly flags
        """
        anomalies = []

        # Check validation errors
        validation = dataset_entry.get("validation", {})
        if validation.get("validation_status") == "failed":
            anomalies.append("validation_failed")

        if validation.get("validation_errors"):
            anomalies.append("has_validation_errors")

        # Check sync discrepancies
        sync_status = dataset_entry.get("sync_status", {})
        if sync_status.get("sync_discrepancies"):
            anomalies.append("sync_discrepancies")

        # Check data freshness
        quality_metrics = dataset_entry.get("quality_metrics", {})
        freshness_days = quality_metrics.get("data_freshness_days")
        if freshness_days and freshness_days > 365:
            anomalies.append("stale_data")

        # Check size anomalies
        size_mb = dataset_entry.get("size_mb")
        if size_mb == 0:
            anomalies.append("empty_dataset")
        elif size_mb and size_mb > 1000:
            anomalies.append("large_dataset")

        # Check deprecation
        version_control = dataset_entry.get("version_control", {})
        if version_control.get("deprecated"):
            anomalies.append("deprecated")

        return anomalies

    def score_dataset(self, dataset_name: str, dataset_entry: dict[str, Any]) -> dict[str, Any]:
        """
        Compute quality metrics for a dataset.

        Args:
            dataset_name: Name of the dataset
            dataset_entry: Dataset entry from registry

        Returns:
            Quality metrics dictionary
        """

        # Calculate individual scores
        completeness_score = self.calculate_completeness_score(dataset_entry)
        consistency_score = self.calculate_consistency_score(dataset_entry)
        annotation_quality = self.calculate_annotation_quality(dataset_entry)

        # Calculate overall quality score (weighted average)
        weights = {"completeness": 0.3, "consistency": 0.4, "annotation": 0.3}

        quality_score = completeness_score * weights["completeness"] + consistency_score * weights["consistency"]

        if annotation_quality is not None:
            quality_score += annotation_quality * weights["annotation"]
        else:
            # Redistribute weight if annotation quality not applicable
            quality_score = completeness_score * 0.5 + consistency_score * 0.5

        quality_score = round(quality_score, 2)
        quality_tier = self.determine_quality_tier(quality_score)

        # Detect anomalies
        anomaly_flags = self.detect_anomalies(dataset_entry)

        return {
            "quality_score": quality_score,
            "quality_tier": quality_tier,
            "completeness_score": completeness_score,
            "consistency_score": consistency_score,
            "annotation_quality": annotation_quality,
            "data_freshness_days": dataset_entry.get("quality_metrics", {}).get("data_freshness_days"),
            "anomaly_flags": anomaly_flags,
        }

    def score_all_datasets(self, limit: int | None = None) -> dict[str, Any]:
        """
        Score all datasets in the registry.

        Args:
            limit: Maximum number of datasets to score

        Returns:
            Statistics about scoring
        """
        registry = self.load_registry()
        stats = {
            "total_scored": 0,
            "by_tier": {"excellent": 0, "good": 0, "acceptable": 0, "needs_review": 0},
            "anomalies_detected": 0,
        }

        # Collect all datasets
        datasets_to_score = []

        if "datasets" in registry:
            for category_name, category_data in registry["datasets"].items():
                if isinstance(category_data, dict):
                    for dataset_name, dataset_entry in category_data.items():
                        if isinstance(dataset_entry, dict) and "path" in dataset_entry:
                            datasets_to_score.append(
                                (
                                    f"datasets.{category_name}.{dataset_name}",
                                    dataset_entry,
                                )
                            )

        other_sections = [
            "rlhf_alignment",
            "emotion_recognition",
            "advanced_reasoning",
            "embeddings",
            "edge_case_sources",
            "voice_persona",
            "supplementary",
        ]

        for section_name in other_sections:
            if section_name in registry:
                section_data = registry[section_name]
                if isinstance(section_data, dict):
                    for dataset_name, dataset_entry in section_data.items():
                        if isinstance(dataset_entry, dict) and "path" in dataset_entry:
                            datasets_to_score.append((f"{section_name}.{dataset_name}", dataset_entry))

        if limit:
            datasets_to_score = datasets_to_score[:limit]

        # Score each dataset
        for dataset_path_key, dataset_entry in datasets_to_score:
            try:
                quality_metrics = self.score_dataset(dataset_path_key, dataset_entry)

                # Update registry
                parts = dataset_path_key.split(".")
                if len(parts) == 3:
                    registry["datasets"][parts[1]][parts[2]]["quality_metrics"] = quality_metrics
                elif len(parts) == 2:
                    registry[parts[0]][parts[1]]["quality_metrics"] = quality_metrics

                stats["total_scored"] += 1
                stats["by_tier"][quality_metrics["quality_tier"]] += 1

                if quality_metrics["anomaly_flags"]:
                    stats["anomalies_detected"] += 1

            except Exception:
                pass

        # Update registry statistics
        if "registry_statistics" in registry:
            registry["registry_statistics"]["datasets_by_quality"] = stats["by_tier"]

        # Save updated registry
        self.save_registry(registry)

        return stats


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Score dataset quality")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/home/vivi/pixelated/ai/config/dataset_registry.json"),
        help="Path to dataset registry",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of datasets to score")

    args = parser.parse_args()

    scorer = DatasetQualityScorer(args.registry)
    stats = scorer.score_all_datasets(limit=args.limit)

    for _tier, _count in stats["by_tier"].items():
        pass


if __name__ == "__main__":
    main()
