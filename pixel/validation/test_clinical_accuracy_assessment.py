"""
Unit tests for Clinical Accuracy Assessment Framework
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from clinical_accuracy_assessment import (
    ClinicalAccuracyAssessmentFramework,
    ClinicalDomain,
    AccuracyLevel,
    ValidationStatus,
    ClinicalCriteria,
    AssessmentResult,
    ClinicalAssessment,
)


class TestClinicalCriteria:
    """Test clinical criteria configuration."""

    def test_valid_criteria_creation(self):
        """Test creating valid clinical criteria."""
        criteria = ClinicalCriteria(
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            criterion_id="test_criterion",
            name="Test Criterion",
            description="Test description",
            weight=0.8,
            required_accuracy=0.75,
        )

        assert criteria.domain == ClinicalDomain.DSM5_DIAGNOSTIC
        assert criteria.criterion_id == "test_criterion"
        assert criteria.weight == 0.8
        assert criteria.required_accuracy == 0.75

    def test_invalid_weight_raises_error(self):
        """Test that invalid weight raises ValueError."""
        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            ClinicalCriteria(
                domain=ClinicalDomain.DSM5_DIAGNOSTIC,
                criterion_id="test",
                name="Test",
                description="Test",
                weight=1.5,
            )

    def test_invalid_accuracy_raises_error(self):
        """Test that invalid required accuracy raises ValueError."""
        with pytest.raises(ValueError, match="Required accuracy must be between 0 and 1"):
            ClinicalCriteria(
                domain=ClinicalDomain.DSM5_DIAGNOSTIC,
                criterion_id="test",
                name="Test",
                description="Test",
                required_accuracy=1.2,
            )


class TestAssessmentResult:
    """Test assessment result validation."""

    def test_valid_assessment_result(self):
        """Test creating valid assessment result."""
        result = AssessmentResult(
            criterion_id="test_criterion",
            score=0.85,
            accuracy_level=AccuracyLevel.GOOD,
            feedback="Good performance",
            assessor_id="expert_001",
        )

        assert result.criterion_id == "test_criterion"
        assert result.score == 0.85
        assert result.accuracy_level == AccuracyLevel.GOOD
        assert result.assessor_id == "expert_001"

    def test_invalid_score_raises_error(self):
        """Test that invalid score raises ValueError."""
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            AssessmentResult(
                criterion_id="test", score=1.5, accuracy_level=AccuracyLevel.GOOD, feedback="Test"
            )


class TestClinicalAccuracyAssessmentFramework:
    """Test the main assessment framework."""

    @pytest.fixture
    def framework(self):
        """Create framework instance for testing."""
        return ClinicalAccuracyAssessmentFramework()

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "assessment_timeout_hours": 24,
            "minimum_assessors_per_domain": 1,
            "consensus_threshold": 0.7,
            "auto_approval_threshold": 0.85,
        }

    def test_framework_initialization(self, framework):
        """Test framework initialization."""
        assert isinstance(framework.criteria_registry, dict)
        assert isinstance(framework.assessments, dict)
        assert len(framework.criteria_registry) > 0  # Should have default criteria

    def test_config_loading_with_file(self, sample_config):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_config, f)
            config_path = f.name

        try:
            framework = ClinicalAccuracyAssessmentFramework(config_path)
            assert framework.config["assessment_timeout_hours"] == 24
            assert framework.config["minimum_assessors_per_domain"] == 1
        finally:
            Path(config_path).unlink()

    def test_config_loading_with_invalid_file(self):
        """Test loading configuration with invalid file."""
        framework = ClinicalAccuracyAssessmentFramework("nonexistent.json")
        # Should use default config
        assert framework.config["assessment_timeout_hours"] == 48

    def test_register_criterion(self, framework):
        """Test registering new criterion."""
        criterion = ClinicalCriteria(
            domain=ClinicalDomain.THERAPEUTIC_INTERVENTION,
            criterion_id="custom_criterion",
            name="Custom Criterion",
            description="Custom test criterion",
        )

        initial_count = len(framework.criteria_registry)
        framework.register_criterion(criterion)

        assert len(framework.criteria_registry) == initial_count + 1
        assert "custom_criterion" in framework.criteria_registry
        assert framework.criteria_registry["custom_criterion"] == criterion

    def test_get_criteria_by_domain(self, framework):
        """Test getting criteria by domain."""
        dsm5_criteria = framework.get_criteria_by_domain(ClinicalDomain.DSM5_DIAGNOSTIC)
        crisis_criteria = framework.get_criteria_by_domain(ClinicalDomain.CRISIS_MANAGEMENT)

        assert len(dsm5_criteria) > 0
        assert len(crisis_criteria) > 0
        assert all(c.domain == ClinicalDomain.DSM5_DIAGNOSTIC for c in dsm5_criteria)
        assert all(c.domain == ClinicalDomain.CRISIS_MANAGEMENT for c in crisis_criteria)

    def test_create_assessment(self, framework):
        """Test creating new assessment."""
        assessment_id = framework.create_assessment(
            content_id="test_content_001",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            assessor_id="expert_001",
            metadata={"test": "data"},
        )

        assert assessment_id in framework.assessments
        assessment = framework.assessments[assessment_id]
        assert assessment.content_id == "test_content_001"
        assert assessment.domain == ClinicalDomain.DSM5_DIAGNOSTIC
        assert "expert_001" in assessment.assessor_ids
        assert assessment.status == ValidationStatus.PENDING
        assert assessment.metadata["test"] == "data"

    def test_conduct_assessment(self, framework):
        """Test conducting assessment."""
        # Create assessment
        assessment_id = framework.create_assessment(
            content_id="test_content_002",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            assessor_id="expert_001",
        )

        # Conduct assessment
        scores = {"dsm5_diagnostic_accuracy": 0.85, "differential_diagnosis": 0.78}
        feedback = {
            "dsm5_diagnostic_accuracy": "Good diagnostic accuracy",
            "differential_diagnosis": "Could improve differential consideration",
        }

        assessment = framework.conduct_assessment(
            assessment_id=assessment_id,
            content="Test therapeutic response",
            assessor_id="expert_001",
            individual_scores=scores,
            feedback=feedback,
        )

        assert assessment.status == ValidationStatus.COMPLETED
        assert assessment.overall_score > 0
        assert len(assessment.individual_results) > 0
        assert assessment.completed_at is not None

    def test_conduct_assessment_invalid_id(self, framework):
        """Test conducting assessment with invalid ID."""
        with pytest.raises(ValueError, match="Assessment .* not found"):
            framework.conduct_assessment(
                assessment_id="invalid_id",
                content="Test",
                assessor_id="expert_001",
                individual_scores={},
                feedback={},
            )

    def test_determine_accuracy_level(self, framework):
        """Test accuracy level determination."""
        assert framework._determine_accuracy_level(0.95) == AccuracyLevel.EXCELLENT
        assert framework._determine_accuracy_level(0.85) == AccuracyLevel.GOOD
        assert framework._determine_accuracy_level(0.75) == AccuracyLevel.ADEQUATE
        assert framework._determine_accuracy_level(0.65) == AccuracyLevel.NEEDS_IMPROVEMENT
        assert framework._determine_accuracy_level(0.55) == AccuracyLevel.INADEQUATE

    def test_get_assessment(self, framework):
        """Test getting assessment by ID."""
        assessment_id = framework.create_assessment(
            content_id="test_content_003",
            domain=ClinicalDomain.CRISIS_MANAGEMENT,
            assessor_id="expert_001",
        )

        retrieved = framework.get_assessment(assessment_id)
        assert retrieved is not None
        assert retrieved.assessment_id == assessment_id

        # Test non-existent assessment
        assert framework.get_assessment("nonexistent") is None

    def test_get_assessments_by_domain(self, framework):
        """Test getting assessments by domain."""
        # Create assessments in different domains
        dsm5_id = framework.create_assessment(
            content_id="dsm5_content",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            assessor_id="expert_001",
        )
        crisis_id = framework.create_assessment(
            content_id="crisis_content",
            domain=ClinicalDomain.CRISIS_MANAGEMENT,
            assessor_id="expert_001",
        )

        dsm5_assessments = framework.get_assessments_by_domain(ClinicalDomain.DSM5_DIAGNOSTIC)
        crisis_assessments = framework.get_assessments_by_domain(ClinicalDomain.CRISIS_MANAGEMENT)

        assert len(dsm5_assessments) >= 1
        assert len(crisis_assessments) >= 1
        assert all(a.domain == ClinicalDomain.DSM5_DIAGNOSTIC for a in dsm5_assessments)
        assert all(a.domain == ClinicalDomain.CRISIS_MANAGEMENT for a in crisis_assessments)

    def test_get_assessment_statistics_empty(self, framework):
        """Test getting statistics with no assessments."""
        stats = framework.get_assessment_statistics(ClinicalDomain.THERAPEUTIC_INTERVENTION)

        assert stats["total"] == 0
        assert stats["average_score"] == 0.0
        assert stats["accuracy_distribution"] == {}

    def test_get_assessment_statistics_with_data(self, framework):
        """Test getting statistics with assessment data."""
        # Create and complete assessment
        assessment_id = framework.create_assessment(
            content_id="stats_test", domain=ClinicalDomain.DSM5_DIAGNOSTIC, assessor_id="expert_001"
        )

        framework.conduct_assessment(
            assessment_id=assessment_id,
            content="Test content",
            assessor_id="expert_001",
            individual_scores={"dsm5_diagnostic_accuracy": 0.85},
            feedback={"dsm5_diagnostic_accuracy": "Good work"},
        )

        stats = framework.get_assessment_statistics(ClinicalDomain.DSM5_DIAGNOSTIC)

        assert stats["total"] >= 1
        assert stats["completed"] >= 1
        assert stats["average_score"] > 0
        assert "good" in stats["accuracy_distribution"]

    def test_generate_assessment_report(self, framework):
        """Test generating assessment report."""
        # Create and complete assessment
        assessment_id = framework.create_assessment(
            content_id="report_test",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            assessor_id="expert_001",
        )

        framework.conduct_assessment(
            assessment_id=assessment_id,
            content="Test content",
            assessor_id="expert_001",
            individual_scores={"dsm5_diagnostic_accuracy": 0.85},
            feedback={"dsm5_diagnostic_accuracy": "Good diagnostic work"},
        )

        report = framework.generate_assessment_report(assessment_id)

        assert report["assessment_id"] == assessment_id
        assert report["content_id"] == "report_test"
        assert report["domain"] == "dsm5_diagnostic"
        assert "overall_score" in report
        assert "criterion_analysis" in report
        assert "recommendations" in report
        assert len(report["criterion_analysis"]) > 0

    def test_generate_assessment_report_invalid_id(self, framework):
        """Test generating report with invalid assessment ID."""
        with pytest.raises(ValueError, match="Assessment .* not found"):
            framework.generate_assessment_report("invalid_id")

    def test_generate_recommendations(self, framework):
        """Test recommendation generation."""
        # Create assessment with low scores
        assessment = ClinicalAssessment(
            assessment_id="test_recommendations",
            content_id="test_content",
            domain=ClinicalDomain.CRISIS_MANAGEMENT,
            overall_score=0.6,
            accuracy_level=AccuracyLevel.NEEDS_IMPROVEMENT,
            status=ValidationStatus.COMPLETED,
            individual_results=[
                AssessmentResult(
                    criterion_id="crisis_recognition",
                    score=0.6,
                    accuracy_level=AccuracyLevel.NEEDS_IMPROVEMENT,
                    feedback="Needs improvement",
                )
            ],
        )

        recommendations = framework._generate_recommendations(assessment)

        assert len(recommendations) > 0
        assert any("improvement needed" in rec.lower() for rec in recommendations)

    @pytest.mark.asyncio
    async def test_batch_assess(self, framework):
        """Test batch assessment creation."""
        content_items = [
            {"content_id": "batch_001", "metadata": {"type": "diagnostic"}},
            {"content_id": "batch_002", "metadata": {"type": "intervention"}},
            {"content_id": "batch_003", "metadata": {"type": "crisis"}},
        ]

        assessment_ids = await framework.batch_assess(
            content_items=content_items,
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            assessor_id="expert_batch",
        )

        assert len(assessment_ids) == 3
        for assessment_id in assessment_ids:
            assert assessment_id in framework.assessments
            assessment = framework.assessments[assessment_id]
            assert assessment.domain == ClinicalDomain.DSM5_DIAGNOSTIC
            assert "expert_batch" in assessment.assessor_ids

    def test_export_assessments(self, framework):
        """Test exporting assessments to file."""
        # Create and complete assessment
        assessment_id = framework.create_assessment(
            content_id="export_test",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            assessor_id="expert_001",
        )

        framework.conduct_assessment(
            assessment_id=assessment_id,
            content="Test content",
            assessor_id="expert_001",
            individual_scores={"dsm5_diagnostic_accuracy": 0.85},
            feedback={"dsm5_diagnostic_accuracy": "Good work"},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_path = f.name

        try:
            framework.export_assessments(export_path, ClinicalDomain.DSM5_DIAGNOSTIC)

            # Verify export file
            with open(export_path, "r") as f:
                export_data = json.load(f)

            assert "export_timestamp" in export_data
            assert export_data["domain_filter"] == "dsm5_diagnostic"
            assert export_data["total_assessments"] >= 1
            assert len(export_data["assessments"]) >= 1
        finally:
            Path(export_path).unlink()


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.fixture
    def framework(self):
        """Create framework for integration testing."""
        return ClinicalAccuracyAssessmentFramework()

    def test_complete_assessment_workflow(self, framework):
        """Test complete assessment workflow from creation to reporting."""
        # Step 1: Create assessment
        assessment_id = framework.create_assessment(
            content_id="integration_test_001",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            assessor_id="expert_integration",
            metadata={"test_scenario": "complete_workflow"},
        )

        # Step 2: Conduct assessment
        scores = {"dsm5_diagnostic_accuracy": 0.88, "differential_diagnosis": 0.82}
        feedback = {
            "dsm5_diagnostic_accuracy": "Excellent diagnostic reasoning with clear rationale",
            "differential_diagnosis": "Good consideration of alternatives, could expand slightly",
        }

        completed_assessment = framework.conduct_assessment(
            assessment_id=assessment_id,
            content="Sample diagnostic response with DSM-5 criteria application",
            assessor_id="expert_integration",
            individual_scores=scores,
            feedback=feedback,
        )

        # Step 3: Verify assessment completion
        assert completed_assessment.status == ValidationStatus.COMPLETED
        assert completed_assessment.overall_score > 0.8
        assert completed_assessment.accuracy_level in [AccuracyLevel.GOOD, AccuracyLevel.EXCELLENT]

        # Step 4: Generate and verify report
        report = framework.generate_assessment_report(assessment_id)
        assert report["overall_score"] > 0.8
        assert len(report["criterion_analysis"]) == 2
        assert len(report["recommendations"]) >= 0

        # Step 5: Check statistics
        stats = framework.get_assessment_statistics(ClinicalDomain.DSM5_DIAGNOSTIC)
        assert stats["completed"] >= 1
        assert stats["average_score"] > 0

    def test_multi_domain_assessment_comparison(self, framework):
        """Test assessments across multiple domains."""
        domains_to_test = [
            ClinicalDomain.DSM5_DIAGNOSTIC,
            ClinicalDomain.CRISIS_MANAGEMENT,
            ClinicalDomain.THERAPEUTIC_INTERVENTION,
        ]

        assessment_ids = []

        # Create assessments for each domain
        for domain in domains_to_test:
            assessment_id = framework.create_assessment(
                content_id=f"multi_domain_{domain.value}", domain=domain, assessor_id="expert_multi"
            )
            assessment_ids.append((assessment_id, domain))

        # Conduct assessments with different score patterns
        for i, (assessment_id, domain) in enumerate(assessment_ids):
            # Get criteria for this domain
            criteria = framework.get_criteria_by_domain(domain)
            scores = {c.criterion_id: 0.8 + (i * 0.05) for c in criteria}
            feedback = {c.criterion_id: f"Assessment for {c.name}" for c in criteria}

            framework.conduct_assessment(
                assessment_id=assessment_id,
                content=f"Sample content for {domain.value}",
                assessor_id="expert_multi",
                individual_scores=scores,
                feedback=feedback,
            )

        # Verify domain-specific statistics
        for domain in domains_to_test:
            stats = framework.get_assessment_statistics(domain)
            assert stats["completed"] >= 1

        # Verify overall statistics
        overall_stats = framework.get_assessment_statistics()
        assert overall_stats["completed"] >= len(domains_to_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
