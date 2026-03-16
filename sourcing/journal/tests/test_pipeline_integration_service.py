"""
Unit tests for pipeline integration service.

Tests format conversion, schema validation, dataset merging, and quality checks.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ai.sourcing.journal.integration.pipeline_integration_service import (
    PipelineIntegrationService,
)
from ai.sourcing.journal.integration.pipeline_integrator import (
    DatasetMerger,
    PipelineFormatConverter,
    PipelineSchemaValidator,
    QualityChecker,
)
from ai.sourcing.journal.models.dataset_models import (
    AcquiredDataset,
    IntegrationPlan,
)


class TestPipelineFormatConverter:
    """Tests for PipelineFormatConverter."""

    @pytest.fixture
    def converter(self):
        """Create a format converter instance."""
        return PipelineFormatConverter()

    def test_convert_csv_to_chatml(self, converter, sample_csv_dataset, sample_integration_plan, temp_dir):
        """Test converting CSV to ChatML format."""
        dataset = AcquiredDataset(
            source_id="test-001",
            storage_path=str(sample_csv_dataset),
            file_format="csv",
        )

        output_path = temp_dir / "output.jsonl"
        result = converter.convert_dataset(
            dataset=dataset,
            integration_plan=sample_integration_plan,
            output_path=str(output_path),
            target_format="chatml",
        )

        assert result.success
        assert output_path.exists()
        assert result.records_converted > 0

    def test_convert_jsonl_to_chatml(self, converter, sample_jsonl_dataset, sample_integration_plan, temp_dir):
        """Test converting JSONL to ChatML format."""
        dataset = AcquiredDataset(
            source_id="test-001",
            storage_path=str(sample_jsonl_dataset),
            file_format="jsonl",
        )

        output_path = temp_dir / "output.jsonl"
        result = converter.convert_dataset(
            dataset=dataset,
            integration_plan=sample_integration_plan,
            output_path=str(output_path),
            target_format="chatml",
        )

        assert result.success
        assert output_path.exists()

    def test_convert_to_conversation_record(self, converter, sample_csv_dataset, sample_integration_plan, temp_dir):
        """Test converting to ConversationRecord format."""
        dataset = AcquiredDataset(
            source_id="test-001",
            storage_path=str(sample_csv_dataset),
            file_format="csv",
        )

        output_path = temp_dir / "output.jsonl"
        result = converter.convert_dataset(
            dataset=dataset,
            integration_plan=sample_integration_plan,
            output_path=str(output_path),
            target_format="conversation_record",
        )

        assert result.success
        assert output_path.exists()


class TestPipelineSchemaValidator:
    """Tests for PipelineSchemaValidator."""

    @pytest.fixture
    def validator(self):
        """Create a schema validator instance."""
        return PipelineSchemaValidator()

    def test_validate_chatml_format(self, validator, temp_dir):
        """Test validating ChatML format."""
        # Create a valid ChatML file
        chatml_file = temp_dir / "test_chatml.jsonl"
        with open(chatml_file, "w") as f:
            record = {
                "id": "test-001",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
            }
            f.write(json.dumps(record) + "\n")

        result = validator.validate_dataset(str(chatml_file), format="chatml")

        assert result.valid
        assert result.errors == []

    def test_validate_invalid_format(self, validator, temp_dir):
        """Test validating invalid format."""
        invalid_file = temp_dir / "invalid.jsonl"
        invalid_file.write_text("invalid json content")

        result = validator.validate_dataset(str(invalid_file), format="chatml")

        assert not result.valid
        assert len(result.errors) > 0

    def test_validate_missing_required_fields(self, validator, temp_dir):
        """Test validating dataset with missing required fields."""
        incomplete_file = temp_dir / "incomplete.jsonl"
        with open(incomplete_file, "w") as f:
            record = {"id": "test-001"}  # Missing messages
            f.write(json.dumps(record) + "\n")

        result = validator.validate_dataset(str(incomplete_file), format="chatml")

        assert not result.valid
        assert len(result.errors) > 0


class TestDatasetMerger:
    """Tests for DatasetMerger."""

    @pytest.fixture
    def merger(self):
        """Create a dataset merger instance."""
        return DatasetMerger(similarity_threshold=0.85)

    def test_merge_datasets(self, merger, temp_dir):
        """Test merging two datasets."""
        # Create first dataset
        dataset1 = temp_dir / "dataset1.jsonl"
        with open(dataset1, "w") as f:
            record = {
                "id": "test-001",
                "messages": [{"role": "user", "content": "Hello"}],
            }
            f.write(json.dumps(record) + "\n")

        # Create second dataset
        dataset2 = temp_dir / "dataset2.jsonl"
        with open(dataset2, "w") as f:
            record = {
                "id": "test-002",
                "messages": [{"role": "user", "content": "Hi"}],
            }
            f.write(json.dumps(record) + "\n")

        output_path = temp_dir / "merged.jsonl"
        result = merger.merge_datasets(
            dataset1_path=str(dataset1),
            dataset2_path=str(dataset2),
            output_path=str(output_path),
        )

        assert result.success
        assert output_path.exists()
        assert result.total_records >= 2

    def test_deduplicate_records(self, merger, temp_dir):
        """Test deduplicating records."""
        # Create dataset with duplicate records
        dataset = temp_dir / "duplicates.jsonl"
        with open(dataset, "w") as f:
            record = {
                "id": "test-001",
                "messages": [{"role": "user", "content": "Hello"}],
            }
            # Write same record twice
            f.write(json.dumps(record) + "\n")
            f.write(json.dumps(record) + "\n")

        output_path = temp_dir / "deduplicated.jsonl"
        result = merger.deduplicate_dataset(
            dataset_path=str(dataset),
            output_path=str(output_path),
        )

        assert result.success
        assert output_path.exists()
        # Should have only one record after deduplication
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 1


class TestQualityChecker:
    """Tests for QualityChecker."""

    @pytest.fixture
    def quality_checker(self):
        """Create a quality checker instance."""
        return QualityChecker()

    def test_check_therapeutic_content(self, quality_checker, temp_dir):
        """Test checking therapeutic content."""
        dataset = temp_dir / "therapeutic.jsonl"
        with open(dataset, "w") as f:
            record = {
                "id": "test-001",
                "messages": [
                    {"role": "user", "content": "I'm feeling depressed"},
                    {"role": "assistant", "content": "I understand. Let's talk about that."},
                ],
            }
            f.write(json.dumps(record) + "\n")

        result = quality_checker.check_therapeutic_content(str(dataset))

        assert result.passed
        assert result.therapeutic_content_score >= 0

    def test_check_pii_detection(self, quality_checker, temp_dir):
        """Test PII detection."""
        dataset = temp_dir / "pii_test.jsonl"
        with open(dataset, "w") as f:
            record = {
                "id": "test-001",
                "messages": [
                    {"role": "user", "content": "My email is test@example.com"},
                ],
            }
            f.write(json.dumps(record) + "\n")

        result = quality_checker.check_pii(str(dataset))

        assert result.pii_detected
        assert len(result.pii_types) > 0

    def test_check_structure_validation(self, quality_checker, temp_dir):
        """Test structure validation."""
        dataset = temp_dir / "structured.jsonl"
        with open(dataset, "w") as f:
            record = {
                "id": "test-001",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            }
            f.write(json.dumps(record) + "\n")

        result = quality_checker.check_structure(str(dataset))

        assert result.passed
        assert result.structure_valid


class TestPipelineIntegrationService:
    """Tests for PipelineIntegrationService."""

    @pytest.fixture
    def service(self):
        """Create a pipeline integration service instance."""
        return PipelineIntegrationService()

    def test_integrate_dataset_full_workflow(
        self, service, sample_acquired_dataset, sample_integration_plan, sample_csv_dataset, temp_dir
    ):
        """Test complete integration workflow."""
        # Update dataset to point to actual file
        sample_acquired_dataset.storage_path = str(sample_csv_dataset)
        sample_acquired_dataset.file_format = "csv"

        output_path = temp_dir / "integrated.jsonl"
        result = service.integrate_dataset(
            dataset=sample_acquired_dataset,
            integration_plan=sample_integration_plan,
            output_path=str(output_path),
            target_format="chatml",
            validate=True,
            merge=False,
            quality_check=True,
        )

        assert result["success"]
        assert "conversion" in result
        assert "validation" in result
        assert "quality_check" in result

    def test_integrate_dataset_with_merging(
        self, service, sample_acquired_dataset, sample_integration_plan, sample_csv_dataset, temp_dir
    ):
        """Test integration with dataset merging."""
        sample_acquired_dataset.storage_path = str(sample_csv_dataset)
        sample_acquired_dataset.file_format = "csv"

        # Create existing dataset
        existing_dataset = temp_dir / "existing.jsonl"
        with open(existing_dataset, "w") as f:
            record = {
                "id": "existing-001",
                "messages": [{"role": "user", "content": "Existing message"}],
            }
            f.write(json.dumps(record) + "\n")

        output_path = temp_dir / "merged_integrated.jsonl"
        result = service.integrate_dataset(
            dataset=sample_acquired_dataset,
            integration_plan=sample_integration_plan,
            existing_dataset_path=str(existing_dataset),
            output_path=str(output_path),
            target_format="chatml",
            validate=True,
            merge=True,
            quality_check=True,
        )

        assert result["success"]
        assert "merge" in result
        assert result["merge"]["success"]

    def test_integrate_dataset_validation_failure(self, service, sample_acquired_dataset, sample_integration_plan, temp_dir):
        """Test integration with validation failure."""
        # Create invalid dataset file
        invalid_file = temp_dir / "invalid.csv"
        invalid_file.write_text("invalid,data\n")

        sample_acquired_dataset.storage_path = str(invalid_file)
        sample_acquired_dataset.file_format = "csv"

        output_path = temp_dir / "output.jsonl"
        result = service.integrate_dataset(
            dataset=sample_acquired_dataset,
            integration_plan=sample_integration_plan,
            output_path=str(output_path),
            validate=True,
            merge=False,
            quality_check=False,
        )

        # Should handle validation failure gracefully
        assert "conversion" in result
        assert "validation" in result

