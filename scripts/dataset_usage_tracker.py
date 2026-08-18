#!/usr/bin/env python3
"""
Dataset usage analytics tracking script that monitors access patterns,
training job correlations, and data freshness.
"""
import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from s3_client_helper import get_s3_client


@dataclass
class AccessRecord:
    """Record of dataset access."""

    timestamp: str
    access_type: str  # 'training', 'validation', 'exploration'
    job_id: str | None = None
    user: str | None = None
    duration_seconds: float | None = None


@dataclass
class TrainingJobRecord:
    """Record of training job using a dataset."""

    job_id: str
    timestamp: str
    stage: str
    model: str
    epochs: int
    status: str


class DatasetUsageTracker:
    """Tracks dataset usage analytics."""

    def __init__(self, registry_path: Path, analytics_dir: Path | None = None):
        self.registry_path = registry_path
        self.analytics_dir = (
            analytics_dir or Path.home() / ".cache" / "dataset_analytics"
        )
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
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

    def load_analytics(self, dataset_name: str) -> dict[str, Any]:
        """Load analytics data for a dataset."""
        analytics_file = self.analytics_dir / f"{dataset_name}_analytics.json"

        if analytics_file.exists():
            with open(analytics_file) as f:
                return json.load(f)

        return {
            "dataset_name": dataset_name,
            "access_history": [],
            "training_jobs": [],
            "created_at": datetime.now(UTC).isoformat() + "Z",
        }

    def save_analytics(self, dataset_name: str, analytics: dict[str, Any]) -> None:
        """Save analytics data for a dataset."""
        analytics_file = self.analytics_dir / f"{dataset_name}_analytics.json"
        analytics["last_updated"] = datetime.now(UTC).isoformat() + "Z"

        with open(analytics_file, "w") as f:
            json.dump(analytics, f, indent=2)

    def record_access(
        self,
        dataset_name: str,
        access_type: str,
        job_id: str | None = None,
        user: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """
        Record a dataset access event.

        Args:
            dataset_name: Name of the accessed dataset
            access_type: Type of access (training, validation, exploration)
            job_id: ID of the job accessing the dataset
            user: User who accessed the dataset
            duration_seconds: Duration of access
        """
        analytics = self.load_analytics(dataset_name)

        access_record = asdict(
            AccessRecord(
                timestamp=datetime.now(UTC).isoformat() + "Z",
                access_type=access_type,
                job_id=job_id,
                user=user,
                duration_seconds=duration_seconds,
            )
        )

        analytics["access_history"].append(access_record)

        # Keep only last 1000 access records
        if len(analytics["access_history"]) > 1000:
            analytics["access_history"] = analytics["access_history"][-1000:]

        self.save_analytics(dataset_name, analytics)

    def record_training_job(
        self,
        dataset_name: str,
        job_id: str,
        stage: str,
        model: str,
        epochs: int,
        status: str,
    ) -> None:
        """
        Record a training job that used the dataset.

        Args:
            dataset_name: Name of the dataset
            job_id: ID of the training job
            stage: Training stage
            model: Model name
            epochs: Number of epochs
            status: Job status (running, completed, failed)
        """
        analytics = self.load_analytics(dataset_name)

        job_record = asdict(
            TrainingJobRecord(
                job_id=job_id,
                timestamp=datetime.now(UTC).isoformat() + "Z",
                stage=stage,
                model=model,
                epochs=epochs,
                status=status,
            )
        )

        analytics["training_jobs"].append(job_record)

        # Keep only last 100 job records
        if len(analytics["training_jobs"]) > 100:
            analytics["training_jobs"] = analytics["training_jobs"][-100:]

        self.save_analytics(dataset_name, analytics)

    def calculate_data_freshness(self, dataset_path: str) -> int | None:
        """
        Calculate data freshness in days.

        Args:
            dataset_path: S3 path to dataset

        Returns:
            Number of days since last modification, or None if unavailable
        """
        try:
            if not dataset_path.startswith("s3://"):
                return None

            parts = dataset_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            last_modified = response.get("LastModified")

            if last_modified:
                delta = datetime.now(UTC) - last_modified.replace(tzinfo=None)
                return delta.days

            return None
        except ClientError:
            return None

    def get_usage_summary(self, dataset_name: str) -> dict[str, Any]:
        """
        Get usage summary for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Usage summary dictionary
        """
        analytics = self.load_analytics(dataset_name)

        access_history = analytics.get("access_history", [])
        training_jobs = analytics.get("training_jobs", [])

        # Calculate statistics
        total_accesses = len(access_history)

        # Get last access
        last_accessed = None
        if access_history:
            last_accessed = access_history[-1]["timestamp"]

        # Count accesses by type
        access_by_type = {}
        for access in access_history:
            access_type = access.get("access_type", "unknown")
            access_by_type[access_type] = access_by_type.get(access_type, 0) + 1

        # Get recent training jobs
        recent_jobs = [
            job
            for job in training_jobs
            if datetime.fromisoformat(job["timestamp"].rstrip("Z"))
            > datetime.now(UTC) - timedelta(days=30)
        ]

        return {
            "total_accesses": total_accesses,
            "last_accessed": last_accessed,
            "access_by_type": access_by_type,
            "total_training_jobs": len(training_jobs),
            "recent_training_jobs": len(recent_jobs),
            "last_training_job": training_jobs[-1] if training_jobs else None,
        }

    def update_registry_usage_metrics(
        self, limit: int | None = None
    ) -> dict[str, Any]:
        """
        Update usage metrics in registry for all datasets.

        Args:
            limit: Maximum number of datasets to update

        Returns:
            Statistics about the update
        """
        registry = self.load_registry()
        stats = {"total_updated": 0, "total_skipped": 0}

        # Collect all datasets
        datasets_to_update = []

        if "datasets" in registry:
            for category_name, category_data in registry["datasets"].items():
                if isinstance(category_data, dict):
                    for dataset_name, dataset_entry in category_data.items():
                        if isinstance(dataset_entry, dict) and "path" in dataset_entry:
                            datasets_to_update.append(
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
                            datasets_to_update.append(
                                (f"{section_name}.{dataset_name}", dataset_entry)
                            )

        if limit:
            datasets_to_update = datasets_to_update[:limit]

        # Update each dataset
        for dataset_path_key, dataset_entry in datasets_to_update:
            try:
                # Extract dataset name
                parts = dataset_path_key.split(".")
                dataset_name = parts[-1]

                # Get usage summary
                usage_summary = self.get_usage_summary(dataset_name)

                # Calculate data freshness
                dataset_path = dataset_entry.get("path", "")
                freshness_days = self.calculate_data_freshness(dataset_path)

                # Update usage_analytics in dataset entry
                usage_analytics = {
                    "last_accessed": usage_summary["last_accessed"],
                    "access_count": usage_summary["total_accesses"],
                    "last_training_job": usage_summary["last_training_job"],
                    "training_jobs_used_in": usage_summary["total_training_jobs"],
                }

                # Update quality_metrics data_freshness
                quality_metrics = dataset_entry.get("quality_metrics", {})
                quality_metrics["data_freshness_days"] = freshness_days

                # Update registry
                if len(parts) == 3:
                    if (
                        "usage_analytics"
                        not in registry["datasets"][parts[1]][parts[2]]
                    ):
                        registry["datasets"][parts[1]][parts[2]]["usage_analytics"] = {}
                    registry["datasets"][parts[1]][parts[2]]["usage_analytics"] = (
                        usage_analytics
                    )
                    registry["datasets"][parts[1]][parts[2]]["quality_metrics"] = (
                        quality_metrics
                    )
                elif len(parts) == 2:
                    if "usage_analytics" not in registry[parts[0]][parts[1]]:
                        registry[parts[0]][parts[1]]["usage_analytics"] = {}
                    registry[parts[0]][parts[1]]["usage_analytics"] = usage_analytics
                    registry[parts[0]][parts[1]]["quality_metrics"] = quality_metrics

                stats["total_updated"] += 1

            except Exception:
                stats["total_skipped"] += 1

        # Save updated registry
        self.save_registry(registry)

        return stats


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Track dataset usage analytics")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/home/vivi/pixelated/ai/config/dataset_registry.json"),
        help="Path to dataset registry",
    )
    parser.add_argument(
        "--action",
        choices=["update", "access", "job"],
        default="update",
        help="Action to perform",
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset name (for access/job actions)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of datasets to update"
    )

    args = parser.parse_args()

    tracker = DatasetUsageTracker(args.registry)

    if args.action == "update":

        tracker.update_registry_usage_metrics(limit=args.limit)


    elif args.action == "access":
        if not args.dataset:
            return

        tracker.record_access(args.dataset, access_type="exploration")

    elif args.action == "job":
        if not args.dataset:
            return

        tracker.record_training_job(
            args.dataset,
            job_id="test-job-001",
            stage="stage1",
            model="test-model",
            epochs=1,
            status="completed",
        )


if __name__ == "__main__":
    main()
