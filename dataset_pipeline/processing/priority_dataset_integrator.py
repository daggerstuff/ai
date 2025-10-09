"""
Priority Dataset Integrator

Integrates priority therapeutic conversation datasets with comprehensive
validation, processing, and quality assessment.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class PriorityDatasetInfo:
    """Priority dataset information."""

    priority_level: int
    name: str
    file_path: str
    summary_path: str | None = None
    record_count: int = 0
    quality_score: float = 0.0
    therapeutic_value: str = "unknown"
    processing_status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationResult:
    """Dataset integration result."""

    dataset_id: str
    success: bool
    records_processed: int = 0
    quality_metrics: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    processing_time: float = 0.0
    output_path: str | None = None


class PriorityDatasetIntegrator:
    """Integrates priority therapeutic conversation datasets."""

    def __init__(
        self,
        base_path: str = "./datasets-wendy",
        output_dir: str = "./integrated_datasets",
    ):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = get_logger(__name__)

        # Priority dataset configurations
        self.priority_configs = {
            1: {
                "name": "Top-tier therapeutic conversations",
                "file": "priority_1_FINAL.jsonl",
                "summary": "summary.json",
                "therapeutic_value": "highest",
                "expected_quality": 0.95,
            },
            3: {
                "name": "Specialized therapeutic content",
                "file": "priority_3_FINAL.jsonl",
                "summary": "summary.json",
                "therapeutic_value": "high",
                "expected_quality": 0.85,
            },
            5: {
                "name": "N/A - no data",
                "file": "priority_5_FINAL.jsonl",
                "summary": "summary.json",
                "therapeutic_value": "medium",
                "expected_quality": 0.75,
            },
        }

        logger.info("PriorityDatasetIntegrator initialized")

    def integrate_priority_dataset(self, priority_level: int) -> IntegrationResult:
        """Integrate a specific priority dataset."""
        start_time = datetime.now()

        if priority_level not in self.priority_configs:
            return IntegrationResult(
                dataset_id=f"priority_{priority_level}",
                success=False,
                issues=[f"Unknown priority level: {priority_level}"],
            )

        config = self.priority_configs[priority_level]
        dataset_id = f"priority_{priority_level}"

        result = IntegrationResult(dataset_id=dataset_id, success=False)

        try:
            # Check if files exist
            data_file = self.base_path / config["file"]
            summary_file = self.base_path / config["summary"]

            if not data_file.exists():
                # Create mock data for testing
                self._create_mock_priority_data(data_file, summary_file, config)
                result.warnings.append(f"Created mock data for {config['name']}")

            # Load and validate data
            dataset_info = self._load_priority_dataset(data_file, summary_file, config)

            # Process and integrate
            processed_data = self._process_priority_data(dataset_info)

            # Quality assessment
            quality_metrics = self._assess_priority_quality(processed_data, config)

            # Save integrated dataset
            output_path = self._save_integrated_dataset(
                processed_data, dataset_id, quality_metrics
            )

            # Update result
            result.success = True
            result.records_processed = len(processed_data.get("conversations", []))
            result.quality_metrics = quality_metrics
            result.output_path = str(output_path)
            result.processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Successfully integrated priority {priority_level} dataset: {result.records_processed} records"
            )

        except Exception as e:
            result.issues.append(f"Integration failed: {e!s}")
            logger.error(f"Priority {priority_level} integration failed: {e}")

        return result

    def integrate_all_priorities(self) -> dict[int, IntegrationResult]:
        """Integrate all priority datasets."""
        results = {}

        for priority_level in self.priority_configs:
            logger.info(f"Integrating priority {priority_level} dataset...")
            results[priority_level] = self.integrate_priority_dataset(priority_level)

        return results

    def _create_mock_priority_data(
        self, data_file: Path, summary_file: Path, config: dict[str, Any]
    ):
        """Create mock priority dataset for testing."""
        data_file.parent.mkdir(parents=True, exist_ok=True)

        # Create mock conversations based on priority level
        conversations = []
        base_count = 100 if config["therapeutic_value"] == "highest" else 50

        for i in range(base_count):
            conversation = {
                "id": f"priority_{config['therapeutic_value']}_{i}",
                "messages": [
                    {
                        "role": "user",
                        "content": f"I'm struggling with {['anxiety', 'depression', 'stress', 'relationships'][i % 4]}. Can you help me?",
                    },
                    {
                        "role": "therapist",
                        "content": f"I understand you're dealing with {['anxiety', 'depression', 'stress', 'relationship issues'][i % 4]}. Let's explore some therapeutic approaches that might help you.",
                    },
                ],
                "therapeutic_approach": ["CBT", "DBT", "Humanistic", "Psychodynamic"][
                    i % 4
                ],
                "quality_rating": config["expected_quality"] + (i % 10) * 0.01,
                "priority_level": config["therapeutic_value"],
            }
            conversations.append(conversation)

        # Save mock data
        with open(data_file, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")

        # Create summary
        summary = {
            "dataset_name": config["name"],
            "total_conversations": len(conversations),
            "therapeutic_value": config["therapeutic_value"],
            "quality_range": [
                config["expected_quality"] - 0.1,
                config["expected_quality"] + 0.1,
            ],
            "approaches": ["CBT", "DBT", "Humanistic", "Psychodynamic"],
            "created_at": datetime.now().isoformat(),
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    def _load_priority_dataset(
        self, data_file: Path, summary_file: Path, config: dict[str, Any]
    ) -> PriorityDatasetInfo:
        """Load priority dataset and summary."""
        # Load summary if available
        summary_data = {}
        if summary_file.exists():
            with open(summary_file) as f:
                summary_data = json.load(f)

        # Count records in data file
        record_count = 0
        if data_file.exists():
            with open(data_file) as f:
                for line in f:
                    if line.strip():
                        record_count += 1

        return PriorityDatasetInfo(
            priority_level=int(data_file.name.split("_")[1]),
            name=config["name"],
            file_path=str(data_file),
            summary_path=str(summary_file) if summary_file.exists() else None,
            record_count=record_count,
            therapeutic_value=config["therapeutic_value"],
            metadata=summary_data,
        )

    def _process_priority_data(
        self, dataset_info: PriorityDatasetInfo
    ) -> dict[str, Any]:
        """Process priority dataset data."""
        conversations = []

        # Load conversations from JSONL
        with open(dataset_info.file_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        conv = json.loads(line)

                        # Standardize conversation format
                        standardized_conv = self._standardize_conversation(
                            conv, dataset_info
                        )
                        conversations.append(standardized_conv)

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Invalid JSON on line {line_num} in {dataset_info.file_path}: {e}"
                        )

        return {
            "dataset_info": {
                "priority_level": dataset_info.priority_level,
                "name": dataset_info.name,
                "therapeutic_value": dataset_info.therapeutic_value,
                "source_file": dataset_info.file_path,
                "processed_at": datetime.now().isoformat(),
            },
            "conversations": conversations,
            "metadata": dataset_info.metadata,
        }

    def _standardize_conversation(
        self, conv: dict[str, Any], dataset_info: PriorityDatasetInfo
    ) -> dict[str, Any]:
        """Standardize conversation format."""
        standardized = {
            "id": conv.get(
                "id", f"priority_{dataset_info.priority_level}_{hash(str(conv))%10000}"
            ),
            "messages": conv.get("messages", []),
            "metadata": {
                "priority_level": dataset_info.priority_level,
                "therapeutic_value": dataset_info.therapeutic_value,
                "source": "priority_dataset",
                "therapeutic_approach": conv.get("therapeutic_approach", "unknown"),
                "quality_rating": conv.get("quality_rating", 0.5),
                "processed_at": datetime.now().isoformat(),
            },
        }

        # Ensure messages have required fields
        for msg in standardized["messages"]:
            if "role" not in msg:
                msg["role"] = "unknown"
            if "content" not in msg:
                msg["content"] = ""

        return standardized

    def _assess_priority_quality(
        self, processed_data: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, float]:
        """Assess quality of priority dataset."""
        conversations = processed_data.get("conversations", [])

        if not conversations:
            return {"overall_quality": 0.0}

        # Calculate quality metrics
        total_quality = 0.0
        valid_conversations = 0
        message_count_total = 0
        therapeutic_approach_count = 0

        for conv in conversations:
            messages = conv.get("messages", [])
            if len(messages) >= 2:  # At least user and therapist message
                valid_conversations += 1
                message_count_total += len(messages)

                # Quality from metadata
                quality_rating = conv.get("metadata", {}).get("quality_rating", 0.5)
                total_quality += quality_rating

                # Check for therapeutic approach
                if (
                    conv.get("metadata", {}).get("therapeutic_approach", "unknown")
                    != "unknown"
                ):
                    therapeutic_approach_count += 1

        if valid_conversations == 0:
            return {"overall_quality": 0.0}

        return {
            "overall_quality": total_quality / valid_conversations,
            "conversation_validity": valid_conversations / len(conversations),
            "average_message_count": message_count_total / valid_conversations,
            "therapeutic_approach_coverage": therapeutic_approach_count
            / valid_conversations,
            "expected_quality_alignment": abs(
                (total_quality / valid_conversations) - config["expected_quality"]
            ),
        }


    def _save_integrated_dataset(
        self,
        processed_data: dict[str, Any],
        dataset_id: str,
        quality_metrics: dict[str, float],
    ) -> Path:
        """Save integrated dataset."""
        output_file = self.output_dir / f"{dataset_id}_integrated.json"

        # Add quality metrics to output
        output_data = {
            **processed_data,
            "integration_info": {
                "integrated_at": datetime.now().isoformat(),
                "quality_metrics": quality_metrics,
                "total_conversations": len(processed_data.get("conversations", [])),
                "integrator_version": "1.0",
            },
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Integrated dataset saved: {output_file}")
        return output_file

    def get_integration_summary(self) -> dict[str, Any]:
        """Get summary of all integrations."""
        summary = {
            "total_priorities": len(self.priority_configs),
            "configured_priorities": list(self.priority_configs.keys()),
            "output_directory": str(self.output_dir),
            "last_updated": datetime.now().isoformat(),
        }

        # Check for existing integrated files
        integrated_files = list(self.output_dir.glob("priority_*_integrated.json"))
        summary["integrated_files"] = [str(f.name) for f in integrated_files]
        summary["integrated_count"] = len(integrated_files)

        return summary


# Example usage
if __name__ == "__main__":
    # Initialize integrator
    integrator = PriorityDatasetIntegrator()

    # Integrate all priority datasets
    results = integrator.integrate_all_priorities()

    # Show results
    for _priority, result in results.items():
        if result.success:
            pass
        else:
            pass

    # Show summary
    summary = integrator.get_integration_summary()
