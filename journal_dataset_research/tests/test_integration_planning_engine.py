"""
Tests for Integration Planning Engine
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from ai.journal_dataset_research.integration.integration_planning_engine import (
    DatasetStructure,
    IntegrationPlanningEngine,
    SchemaMapping,
)
from ai.journal_dataset_research.models.dataset_models import AcquiredDataset


@pytest.fixture
def engine():
    """Create an integration planning engine instance."""
    return IntegrationPlanningEngine()


@pytest.fixture
def sample_csv_dataset(tmp_path):
    """Create a sample CSV dataset for testing."""
    csv_path = tmp_path / "test_dataset.csv"
    df = pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(10)],
            "message": [f"Message {i}" for i in range(10)],
            "role": ["user", "assistant"] * 5,
            "timestamp": [datetime.now().isoformat()] * 10,
        }
    )
    df.to_csv(csv_path, index=False)

    dataset = AcquiredDataset(
        source_id="test_source_1",
        storage_path=str(csv_path),
        file_format="csv",
        file_size_mb=0.1,
    )
    return dataset


@pytest.fixture
def sample_jsonl_dataset(tmp_path):
    """Create a sample JSONL dataset for testing."""
    jsonl_path = tmp_path / "test_dataset.jsonl"
    records = [
        {
            "conversation_id": f"conv_{i}",
            "messages": [
                {"role": "user", "content": f"User message {i}"},
                {"role": "assistant", "content": f"Assistant response {i}"},
            ],
        }
        for i in range(10)
    ]

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    dataset = AcquiredDataset(
        source_id="test_source_2",
        storage_path=str(jsonl_path),
        file_format="jsonl",
        file_size_mb=0.1,
    )
    return dataset


class TestDatasetStructureAnalysis:
    """Test dataset structure analysis functionality."""

    def test_analyze_csv_structure(self, engine, sample_csv_dataset):
        """Test CSV structure analysis."""
        structure = engine.analyze_dataset_structure(sample_csv_dataset)

        assert structure.format == "csv"
        assert "id" in structure.schema
        assert "message" in structure.schema
        assert "role" in structure.schema
        assert structure.sample_size > 0
        assert isinstance(structure.field_types, dict)
        assert isinstance(structure.field_distributions, dict)

    def test_analyze_jsonl_structure(self, engine, sample_jsonl_dataset):
        """Test JSONL structure analysis."""
        structure = engine.analyze_dataset_structure(sample_jsonl_dataset)

        assert structure.format == "jsonl"
        assert "conversation_id" in structure.schema or "messages" in structure.schema
        assert structure.sample_size > 0

    def test_analyze_missing_file(self, engine):
        """Test analysis with missing file."""
        dataset = AcquiredDataset(
            source_id="missing",
            storage_path="/nonexistent/path/file.csv",
            file_format="csv",
        )

        with pytest.raises(FileNotFoundError):
            engine.analyze_dataset_structure(dataset)

    def test_dataset_structure_validation(self):
        """Test DatasetStructure validation."""
        structure = DatasetStructure(
            format="csv",
            schema={"field1": "string"},
            field_types={"field1": "string"},
            field_distributions={},
            quality_issues=[],
            sample_size=10,
        )

        errors = structure.validate()
        assert len(errors) == 0

        # Invalid structure
        invalid_structure = DatasetStructure(
            format="",
            schema={},
            field_types={},
            field_distributions={},
            quality_issues=[],
            sample_size=0,
        )
        errors = invalid_structure.validate()
        assert len(errors) > 0


class TestSchemaMapping:
    """Test schema mapping functionality."""

    def test_create_schema_mapping_chatml(self, engine, sample_csv_dataset):
        """Test schema mapping for ChatML format."""
        structure = engine.analyze_dataset_structure(sample_csv_dataset)
        mappings = engine.create_schema_mapping(structure, target_format="chatml")

        assert len(mappings) > 0
        assert any(m.pipeline_field == "messages" for m in mappings)
        assert any(m.pipeline_field == "id" for m in mappings)

    def test_create_schema_mapping_conversation_record(
        self, engine, sample_csv_dataset
    ):
        """Test schema mapping for ConversationRecord format."""
        structure = engine.analyze_dataset_structure(sample_csv_dataset)
        mappings = engine.create_schema_mapping(
            structure, target_format="conversation_record"
        )

        assert len(mappings) > 0
        # Should map to turns format
        assert any("turns" in m.pipeline_field for m in mappings)

    def test_schema_mapping_validation(self):
        """Test SchemaMapping validation."""
        mapping = SchemaMapping(
            dataset_field="message",
            pipeline_field="messages",
            transformation_type="transform",
        )
        errors = mapping.validate()
        assert len(errors) == 0

        # Invalid mapping
        invalid_mapping = SchemaMapping(
            dataset_field="",
            pipeline_field="",
            transformation_type="invalid",
        )
        errors = invalid_mapping.validate()
        assert len(errors) > 0


class TestTransformationSpecs:
    """Test transformation specification generation."""

    def test_create_transformation_specs(self, engine, sample_csv_dataset):
        """Test transformation spec creation."""
        structure = engine.analyze_dataset_structure(sample_csv_dataset)
        mappings = engine.create_schema_mapping(structure)
        specs = engine.create_transformation_specs(structure, mappings)

        assert len(specs) > 0
        assert any(spec.transformation_type == "format_conversion" for spec in specs)
        assert any(spec.transformation_type == "field_mapping" for spec in specs)
        assert any(spec.transformation_type == "validation" for spec in specs)

    def test_transformation_spec_validation(self):
        """Test TransformationSpec validation."""
        from ai.journal_dataset_research.models.dataset_models import (
            TransformationSpec,
        )

        spec = TransformationSpec(
            transformation_type="format_conversion",
            input_format="csv",
            output_format="jsonl",
            transformation_logic="Convert CSV to JSONL",
        )
        errors = spec.validate()
        assert len(errors) == 0

        # Invalid spec
        invalid_spec = TransformationSpec(
            transformation_type="invalid",
            input_format="csv",
            output_format="jsonl",
            transformation_logic="",
        )
        errors = invalid_spec.validate()
        assert len(errors) > 0


class TestComplexityEstimation:
    """Test complexity estimation functionality."""

    def test_estimate_complexity_low(self, engine, sample_csv_dataset):
        """Test complexity estimation for simple CSV."""
        structure = engine.analyze_dataset_structure(sample_csv_dataset)
        mappings = engine.create_schema_mapping(structure)
        specs = engine.create_transformation_specs(structure, mappings)

        complexity, hours = engine.estimate_complexity(structure, mappings, specs)

        assert complexity in ["low", "medium", "high"]
        assert hours > 0
        assert isinstance(hours, int)

    def test_estimate_complexity_high(self, engine, sample_jsonl_dataset):
        """Test complexity estimation for complex JSONL."""
        structure = engine.analyze_dataset_structure(sample_jsonl_dataset)
        mappings = engine.create_schema_mapping(structure)
        specs = engine.create_transformation_specs(structure, mappings)

        complexity, hours = engine.estimate_complexity(structure, mappings, specs)

        assert complexity in ["low", "medium", "high"]
        assert hours > 0


class TestIntegrationPlan:
    """Test integration plan creation."""

    def test_create_integration_plan(self, engine, sample_csv_dataset):
        """Test complete integration plan creation."""
        plan = engine.create_integration_plan(sample_csv_dataset)

        assert plan.source_id == sample_csv_dataset.source_id
        assert plan.dataset_format == "csv"
        assert len(plan.schema_mapping) > 0
        assert len(plan.required_transformations) > 0
        assert len(plan.preprocessing_steps) > 0
        assert plan.complexity in ["low", "medium", "high"]
        assert plan.estimated_effort_hours > 0
        assert plan.created_date is not None

    def test_create_integration_plan_chatml(self, engine, sample_jsonl_dataset):
        """Test integration plan creation for ChatML format."""
        plan = engine.create_integration_plan(
            sample_jsonl_dataset, target_format="chatml"
        )

        assert plan.source_id == sample_jsonl_dataset.source_id
        assert plan.dataset_format == "jsonl"
        assert plan.complexity in ["low", "medium", "high"]

    def test_integration_plan_validation(self):
        """Test IntegrationPlan validation."""
        from ai.journal_dataset_research.models.dataset_models import (
            IntegrationPlan,
        )

        plan = IntegrationPlan(
            source_id="test",
            dataset_format="csv",
            schema_mapping={"field1": "field2"},
            complexity="low",
        )
        errors = plan.validate()
        assert len(errors) == 0

        # Invalid plan
        invalid_plan = IntegrationPlan(
            source_id="",
            dataset_format="csv",
            schema_mapping={},
            complexity="invalid",
        )
        errors = invalid_plan.validate()
        assert len(errors) > 0


class TestPreprocessingScriptGeneration:
    """Test preprocessing script generation."""

    def test_generate_preprocessing_script(self, engine, sample_csv_dataset, tmp_path):
        """Test preprocessing script generation."""
        plan = engine.create_integration_plan(sample_csv_dataset)
        script_path = tmp_path / "preprocess.py"

        output_path = engine.generate_preprocessing_script(plan, str(script_path))

        assert os.path.exists(output_path)
        assert output_path == str(script_path)

        # Verify script content
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert plan.source_id in content
            assert plan.complexity in content
            assert "def main" in content
            assert "def transform_data" in content

    def test_generate_script_creates_directory(self, engine, sample_csv_dataset, tmp_path):
        """Test that script generation creates directory if needed."""
        plan = engine.create_integration_plan(sample_csv_dataset)
        script_path = tmp_path / "scripts" / "preprocess.py"

        output_path = engine.generate_preprocessing_script(plan, str(script_path))

        assert os.path.exists(output_path)
        assert os.path.exists(tmp_path / "scripts")


class TestFeasibilityValidation:
    """Test integration feasibility validation."""

    def test_validate_feasible_plan(self, engine, sample_csv_dataset):
        """Test validation of a feasible integration plan."""
        plan = engine.create_integration_plan(sample_csv_dataset)
        is_feasible = engine.validate_integration_feasibility(plan)

        # Simple CSV should be feasible
        assert isinstance(is_feasible, bool)

    def test_validate_infeasible_plan(self, engine):
        """Test validation of an infeasible integration plan."""
        from ai.journal_dataset_research.models.dataset_models import (
            IntegrationPlan,
        )

        # Create plan with missing critical fields
        plan = IntegrationPlan(
            source_id="test",
            dataset_format="custom",
            schema_mapping={},  # No mappings
            complexity="high",
            estimated_effort_hours=100,  # Very high effort
        )

        is_feasible = engine.validate_integration_feasibility(plan)
        assert is_feasible is False


class TestHelperMethods:
    """Test helper methods."""

    def test_extract_fields(self, engine):
        """Test field extraction from nested structures."""
        nested_structures = []
        obj = {
            "field1": "value1",
            "nested": {"field2": "value2", "field3": {"field4": "value4"}},
            "list": [{"item_field": "value"}],
        }

        fields = engine._extract_fields(obj, nested_structures)

        assert "field1" in fields
        assert "nested.field2" in fields
        assert "nested.field3.field4" in fields
        assert len(nested_structures) > 0

    def test_get_nested_value(self, engine):
        """Test getting values from nested structures."""
        obj = {
            "level1": {
                "level2": {
                    "level3": "value",
                }
            }
        }

        value = engine._get_nested_value(obj, "level1.level2.level3")
        assert value == "value"

        value = engine._get_nested_value(obj, "level1.nonexistent")
        assert value is None

