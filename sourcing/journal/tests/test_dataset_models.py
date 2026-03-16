"""
Unit tests for data models.

Tests validation methods and data integrity for all data model classes.
"""

import pytest
from datetime import datetime

from ai.sourcing.journal.models.dataset_models import (
    AccessRequest,
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
    ResearchLog,
    ResearchSession,
    TransformationSpec,
)


class TestDatasetSource:
    """Tests for DatasetSource model."""

    def test_valid_source(self):
        """Test that a valid source passes validation."""
        source = DatasetSource(
            source_id="test-001",
            title="Test Dataset",
            authors=["Author 1"],
            publication_date=datetime(2024, 1, 1),
            source_type="journal",
            url="https://example.com/dataset",
        )
        errors = source.validate()
        assert len(errors) == 0

    def test_missing_source_id(self):
        """Test that missing source_id fails validation."""
        source = DatasetSource(
            source_id="",
            title="Test Dataset",
            authors=[],
            publication_date=datetime(2024, 1, 1),
            source_type="journal",
            url="https://example.com/dataset",
        )
        errors = source.validate()
        assert len(errors) > 0
        assert "source_id is required" in errors

    def test_missing_title(self):
        """Test that missing title fails validation."""
        source = DatasetSource(
            source_id="test-001",
            title="",
            authors=[],
            publication_date=datetime(2024, 1, 1),
            source_type="journal",
            url="https://example.com/dataset",
        )
        errors = source.validate()
        assert len(errors) > 0
        assert "title is required" in errors

    def test_missing_url(self):
        """Test that missing url fails validation."""
        source = DatasetSource(
            source_id="test-001",
            title="Test Dataset",
            authors=[],
            publication_date=datetime(2024, 1, 1),
            source_type="journal",
            url="",
        )
        errors = source.validate()
        assert len(errors) > 0
        assert "url is required" in errors

    def test_invalid_source_type(self):
        """Test that invalid source_type fails validation."""
        source = DatasetSource(
            source_id="test-001",
            title="Test Dataset",
            authors=[],
            publication_date=datetime(2024, 1, 1),
            source_type="invalid_type",
            url="https://example.com/dataset",
        )
        errors = source.validate()
        assert len(errors) > 0
        assert "source_type must be one of" in errors[0]

    def test_invalid_data_availability(self):
        """Test that invalid data_availability fails validation."""
        source = DatasetSource(
            source_id="test-001",
            title="Test Dataset",
            authors=[],
            publication_date=datetime(2024, 1, 1),
            source_type="journal",
            url="https://example.com/dataset",
            data_availability="invalid",
        )
        errors = source.validate()
        assert len(errors) > 0
        assert "data_availability must be one of" in errors[0]

    def test_valid_source_types(self):
        """Test that all valid source types pass validation."""
        valid_types = ["journal", "repository", "clinical_trial", "training_material"]
        for source_type in valid_types:
            source = DatasetSource(
                source_id="test-001",
                title="Test Dataset",
                authors=[],
                publication_date=datetime(2024, 1, 1),
                source_type=source_type,
                url="https://example.com/dataset",
            )
            errors = source.validate()
            assert len(errors) == 0, f"Source type {source_type} should be valid"

    def test_valid_data_availability_values(self):
        """Test that all valid data_availability values pass validation."""
        valid_values = ["available", "upon_request", "restricted", "unknown"]
        for availability in valid_values:
            source = DatasetSource(
                source_id="test-001",
                title="Test Dataset",
                authors=[],
                publication_date=datetime(2024, 1, 1),
                source_type="journal",
                url="https://example.com/dataset",
                data_availability=availability,
            )
            errors = source.validate()
            assert len(errors) == 0, f"Data availability {availability} should be valid"


class TestDatasetEvaluation:
    """Tests for DatasetEvaluation model."""

    def test_valid_evaluation(self):
        """Test that a valid evaluation passes validation."""
        evaluation = DatasetEvaluation(
            source_id="test-001",
            therapeutic_relevance=8,
            data_structure_quality=7,
            training_integration=8,
            ethical_accessibility=9,
        )
        errors = evaluation.validate()
        assert len(errors) == 0

    def test_missing_source_id(self):
        """Test that missing source_id fails validation."""
        evaluation = DatasetEvaluation(
            source_id="",
            therapeutic_relevance=8,
            data_structure_quality=7,
            training_integration=8,
            ethical_accessibility=9,
        )
        errors = evaluation.validate()
        assert len(errors) > 0
        assert "source_id is required" in errors

    def test_invalid_therapeutic_relevance_too_low(self):
        """Test that therapeutic_relevance below 1 fails validation."""
        evaluation = DatasetEvaluation(
            source_id="test-001",
            therapeutic_relevance=0,
            data_structure_quality=7,
            training_integration=8,
            ethical_accessibility=9,
        )
        errors = evaluation.validate()
        assert len(errors) > 0
        assert "therapeutic_relevance must be between 1 and 10" in errors

    def test_invalid_therapeutic_relevance_too_high(self):
        """Test that therapeutic_relevance above 10 fails validation."""
        evaluation = DatasetEvaluation(
            source_id="test-001",
            therapeutic_relevance=11,
            data_structure_quality=7,
            training_integration=8,
            ethical_accessibility=9,
        )
        errors = evaluation.validate()
        assert len(errors) > 0
        assert "therapeutic_relevance must be between 1 and 10" in errors

    def test_valid_score_range(self):
        """Test that scores in valid range (1-10) pass validation."""
        for score in range(1, 11):
            evaluation = DatasetEvaluation(
                source_id="test-001",
                therapeutic_relevance=score,
                data_structure_quality=score,
                training_integration=score,
                ethical_accessibility=score,
            )
            errors = evaluation.validate()
            assert len(errors) == 0, f"Score {score} should be valid"

    def test_invalid_priority_tier(self):
        """Test that invalid priority_tier fails validation."""
        evaluation = DatasetEvaluation(
            source_id="test-001",
            therapeutic_relevance=8,
            data_structure_quality=7,
            training_integration=8,
            ethical_accessibility=9,
            priority_tier="invalid",
        )
        errors = evaluation.validate()
        assert len(errors) > 0
        assert "priority_tier must be one of" in errors[0]

    def test_valid_priority_tiers(self):
        """Test that all valid priority tiers pass validation."""
        valid_tiers = ["high", "medium", "low"]
        for tier in valid_tiers:
            evaluation = DatasetEvaluation(
                source_id="test-001",
                therapeutic_relevance=8,
                data_structure_quality=7,
                training_integration=8,
                ethical_accessibility=9,
                priority_tier=tier,
            )
            errors = evaluation.validate()
            assert len(errors) == 0, f"Priority tier {tier} should be valid"


class TestAccessRequest:
    """Tests for AccessRequest model."""

    def test_valid_access_request(self):
        """Test that a valid access request passes validation."""
        request = AccessRequest(
            source_id="test-001",
            access_method="direct",
        )
        errors = request.validate()
        assert len(errors) == 0

    def test_missing_source_id(self):
        """Test that missing source_id fails validation."""
        request = AccessRequest(
            source_id="",
            access_method="direct",
        )
        errors = request.validate()
        assert len(errors) > 0
        assert "source_id is required" in errors

    def test_invalid_access_method(self):
        """Test that invalid access_method fails validation."""
        request = AccessRequest(
            source_id="test-001",
            access_method="invalid_method",
        )
        errors = request.validate()
        assert len(errors) > 0
        assert "access_method must be one of" in errors[0]

    def test_valid_access_methods(self):
        """Test that all valid access methods pass validation."""
        valid_methods = ["direct", "api", "request_form", "collaboration", "registration"]
        for method in valid_methods:
            request = AccessRequest(
                source_id="test-001",
                access_method=method,
            )
            errors = request.validate()
            assert len(errors) == 0, f"Access method {method} should be valid"

    def test_invalid_status(self):
        """Test that invalid status fails validation."""
        request = AccessRequest(
            source_id="test-001",
            access_method="direct",
            status="invalid_status",
        )
        errors = request.validate()
        assert len(errors) > 0
        assert "status must be one of" in errors[0]

    def test_valid_statuses(self):
        """Test that all valid statuses pass validation."""
        valid_statuses = ["pending", "approved", "denied", "downloaded", "error"]
        for status in valid_statuses:
            request = AccessRequest(
                source_id="test-001",
                access_method="direct",
                status=status,
            )
            errors = request.validate()
            assert len(errors) == 0, f"Status {status} should be valid"


class TestAcquiredDataset:
    """Tests for AcquiredDataset model."""

    def test_valid_acquired_dataset(self):
        """Test that a valid acquired dataset passes validation."""
        dataset = AcquiredDataset(
            source_id="test-001",
            storage_path="/path/to/dataset.zip",
        )
        errors = dataset.validate()
        assert len(errors) == 0

    def test_missing_source_id(self):
        """Test that missing source_id fails validation."""
        dataset = AcquiredDataset(
            source_id="",
            storage_path="/path/to/dataset.zip",
        )
        errors = dataset.validate()
        assert len(errors) > 0
        assert "source_id is required" in errors

    def test_missing_storage_path(self):
        """Test that missing storage_path fails validation."""
        dataset = AcquiredDataset(
            source_id="test-001",
            storage_path="",
        )
        errors = dataset.validate()
        assert len(errors) > 0
        assert "storage_path is required" in errors


class TestTransformationSpec:
    """Tests for TransformationSpec model."""

    def test_valid_transformation_spec(self):
        """Test that a valid transformation spec passes validation."""
        spec = TransformationSpec(
            transformation_type="format_conversion",
            input_format="csv",
            output_format="jsonl",
            transformation_logic="Convert CSV to JSONL",
        )
        errors = spec.validate()
        assert len(errors) == 0

    def test_invalid_transformation_type(self):
        """Test that invalid transformation_type fails validation."""
        spec = TransformationSpec(
            transformation_type="invalid_type",
            input_format="csv",
            output_format="jsonl",
            transformation_logic="Convert",
        )
        errors = spec.validate()
        assert len(errors) > 0
        assert "transformation_type must be one of" in errors[0]

    def test_valid_transformation_types(self):
        """Test that all valid transformation types pass validation."""
        valid_types = ["format_conversion", "field_mapping", "cleaning", "validation"]
        for trans_type in valid_types:
            spec = TransformationSpec(
                transformation_type=trans_type,
                input_format="csv",
                output_format="jsonl",
                transformation_logic="Transform",
            )
            errors = spec.validate()
            assert len(errors) == 0, f"Transformation type {trans_type} should be valid"


class TestIntegrationPlan:
    """Tests for IntegrationPlan model."""

    def test_valid_integration_plan(self):
        """Test that a valid integration plan passes validation."""
        plan = IntegrationPlan(
            source_id="test-001",
            dataset_format="csv",
            complexity="medium",
        )
        errors = plan.validate()
        assert len(errors) == 0

    def test_missing_source_id(self):
        """Test that missing source_id fails validation."""
        plan = IntegrationPlan(
            source_id="",
            dataset_format="csv",
            complexity="medium",
        )
        errors = plan.validate()
        assert len(errors) > 0
        assert "source_id is required" in errors

    def test_invalid_complexity(self):
        """Test that invalid complexity fails validation."""
        plan = IntegrationPlan(
            source_id="test-001",
            dataset_format="csv",
            complexity="invalid",
        )
        errors = plan.validate()
        assert len(errors) > 0
        assert "complexity must be one of" in errors[0]

    def test_valid_complexities(self):
        """Test that all valid complexities pass validation."""
        valid_complexities = ["low", "medium", "high"]
        for complexity in valid_complexities:
            plan = IntegrationPlan(
                source_id="test-001",
                dataset_format="csv",
                complexity=complexity,
            )
            errors = plan.validate()
            assert len(errors) == 0, f"Complexity {complexity} should be valid"


class TestResearchSession:
    """Tests for ResearchSession model."""

    def test_valid_research_session(self):
        """Test that a valid research session passes validation."""
        session = ResearchSession(
            session_id="test-session-001",
            current_phase="discovery",
        )
        errors = session.validate()
        assert len(errors) == 0

    def test_missing_session_id(self):
        """Test that missing session_id fails validation."""
        session = ResearchSession(
            session_id="",
            current_phase="discovery",
        )
        errors = session.validate()
        assert len(errors) > 0
        assert "session_id is required" in errors

    def test_invalid_current_phase(self):
        """Test that invalid current_phase fails validation."""
        session = ResearchSession(
            session_id="test-session-001",
            current_phase="invalid_phase",
        )
        errors = session.validate()
        assert len(errors) > 0
        assert "current_phase must be one of" in errors[0]

    def test_valid_phases(self):
        """Test that all valid phases pass validation."""
        valid_phases = ["discovery", "evaluation", "acquisition", "integration"]
        for phase in valid_phases:
            session = ResearchSession(
                session_id="test-session-001",
                current_phase=phase,
            )
            errors = session.validate()
            assert len(errors) == 0, f"Phase {phase} should be valid"


class TestResearchLog:
    """Tests for ResearchLog model."""

    def test_valid_research_log(self):
        """Test that a valid research log passes validation."""
        log = ResearchLog(
            activity_type="search",
            description="Searching for datasets",
            outcome="success",
        )
        errors = log.validate()
        assert len(errors) == 0

    def test_invalid_activity_type(self):
        """Test that invalid activity_type fails validation."""
        log = ResearchLog(
            activity_type="invalid_activity",
            description="Test",
            outcome="success",
        )
        errors = log.validate()
        assert len(errors) > 0
        assert "activity_type must be one of" in errors[0]

    def test_valid_activity_types(self):
        """Test that all valid activity types pass validation."""
        valid_types = ResearchLog.ALLOWED_ACTIVITY_TYPES
        for activity_type in valid_types:
            log = ResearchLog(
                activity_type=activity_type,
                description="Test activity",
                outcome="success",
            )
            errors = log.validate()
            assert len(errors) == 0, f"Activity type {activity_type} should be valid"

