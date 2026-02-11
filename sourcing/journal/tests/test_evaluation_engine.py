"""
Unit tests for the Dataset Evaluation Engine.
"""

import pytest
from datetime import datetime

from ai.sourcing.journal.evaluation.evaluation_engine import (
    DatasetEvaluationEngine,
    EvaluationConfig,
)
from ai.sourcing.journal.models.dataset_models import DatasetSource, DatasetEvaluation


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_default_config_valid(self):
        """Test that default config is valid."""
        config = EvaluationConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_invalid_weight_sum(self):
        """Test that invalid weight sum raises error."""
        config = EvaluationConfig(
            therapeutic_relevance_weight=0.5,
            data_structure_quality_weight=0.5,
            training_integration_weight=0.5,
            ethical_accessibility_weight=0.5,
        )
        errors = config.validate()
        assert len(errors) > 0
        assert "sum to 1.0" in errors[0]

    def test_invalid_threshold_order(self):
        """Test that invalid threshold order raises error."""
        config = EvaluationConfig(
            high_priority_threshold=5.0,
            medium_priority_threshold=7.5,
        )
        errors = config.validate()
        assert len(errors) > 0
        assert "high_priority_threshold must be greater" in errors[0]


class TestDatasetEvaluationEngine:
    """Tests for DatasetEvaluationEngine."""

    @pytest.fixture
    def engine(self):
        """Create a test evaluation engine."""
        return DatasetEvaluationEngine()

    @pytest.fixture
    def sample_source(self):
        """Create a sample dataset source."""
        return DatasetSource(
            source_id="test-001",
            title="Therapeutic Conversation Dataset for Mental Health",
            authors=["Smith, J.", "Doe, A."],
            publication_date=datetime(2024, 1, 1),
            source_type="journal",
            url="https://example.com/dataset",
            doi="10.1000/test",
            abstract="A comprehensive dataset of therapy session transcripts for training AI models in mental health counseling.",
            keywords=["therapy", "counseling", "mental health", "transcripts"],
            open_access=True,
            data_availability="available",
            discovery_date=datetime.now(),
            discovery_method="pubmed_search",
        )

    def test_evaluate_dataset_basic(self, engine, sample_source):
        """Test basic dataset evaluation."""
        evaluation = engine.evaluate_dataset(sample_source, evaluator="test")

        assert evaluation.source_id == sample_source.source_id
        assert 1 <= evaluation.therapeutic_relevance <= 10
        assert 1 <= evaluation.data_structure_quality <= 10
        assert 1 <= evaluation.training_integration <= 10
        assert 1 <= evaluation.ethical_accessibility <= 10
        assert 0 <= evaluation.overall_score <= 10
        assert evaluation.priority_tier in ["high", "medium", "low"]
        assert evaluation.evaluator == "test"

    def test_evaluate_dataset_high_quality(self, engine):
        """Test evaluation of a high-quality dataset."""
        source = DatasetSource(
            source_id="high-quality-001",
            title="Evidence-Based CBT Therapy Transcripts Dataset",
            authors=["Researcher, A."],
            publication_date=datetime(2024, 1, 1),
            source_type="repository",
            url="https://example.com/high-quality",
            doi="10.1000/high-quality",
            abstract="A large dataset of Cognitive Behavioral Therapy session transcripts with comprehensive metadata and anonymized patient data. Includes outcome measures and treatment protocols.",
            keywords=["cbt", "cognitive behavioral therapy", "transcripts", "evidence-based"],
            open_access=True,
            data_availability="available",
            discovery_date=datetime.now(),
            discovery_method="repository_api",
        )

        evaluation = engine.evaluate_dataset(source)

        # High-quality dataset should score well
        assert evaluation.therapeutic_relevance >= 7
        assert evaluation.data_structure_quality >= 7
        assert evaluation.training_integration >= 7
        assert evaluation.ethical_accessibility >= 7
        assert evaluation.overall_score >= 7.0
        assert evaluation.priority_tier in ["high", "medium"]
        assert len(evaluation.competitive_advantages) > 0

    def test_evaluate_dataset_low_quality(self, engine):
        """Test evaluation of a low-quality dataset."""
        source = DatasetSource(
            source_id="low-quality-001",
            title="Some Dataset",
            authors=[],
            publication_date=datetime(2024, 1, 1),
            source_type="training_material",
            url="",
            doi=None,
            abstract="Some abstract",
            keywords=[],
            open_access=False,
            data_availability="unknown",
            discovery_date=datetime.now(),
            discovery_method="citation",
        )

        evaluation = engine.evaluate_dataset(source)

        # Low-quality dataset should score lower
        assert evaluation.overall_score < 7.0
        assert evaluation.priority_tier in ["low", "medium"]

    def test_calculate_priority_tier(self, engine):
        """Test priority tier calculation."""
        assert engine._calculate_priority_tier(8.0) == "high"
        assert engine._calculate_priority_tier(7.5) == "high"
        assert engine._calculate_priority_tier(6.0) == "medium"
        assert engine._calculate_priority_tier(5.0) == "medium"
        assert engine._calculate_priority_tier(4.0) == "low"

    def test_rank_datasets(self, engine):
        """Test dataset ranking."""
        evaluations = [
            DatasetEvaluation(
                source_id="low",
                therapeutic_relevance=5,
                data_structure_quality=5,
                training_integration=5,
                ethical_accessibility=5,
                overall_score=5.0,
                priority_tier="medium",
            ),
            DatasetEvaluation(
                source_id="high",
                therapeutic_relevance=9,
                data_structure_quality=9,
                training_integration=9,
                ethical_accessibility=9,
                overall_score=9.0,
                priority_tier="high",
            ),
            DatasetEvaluation(
                source_id="medium",
                therapeutic_relevance=7,
                data_structure_quality=7,
                training_integration=7,
                ethical_accessibility=7,
                overall_score=7.0,
                priority_tier="high",
            ),
        ]

        ranked = engine.rank_datasets(evaluations)

        assert ranked[0].source_id == "high"
        assert ranked[1].source_id == "medium"
        assert ranked[2].source_id == "low"

    def test_generate_evaluation_report(self, engine, sample_source):
        """Test evaluation report generation."""
        evaluation = engine.evaluate_dataset(sample_source)
        report = engine.generate_evaluation_report(evaluation, sample_source)

        assert "# Dataset Evaluation Report" in report
        assert sample_source.source_id in report
        assert str(evaluation.overall_score) in report
        assert evaluation.priority_tier.upper() in report
        assert "Therapeutic Relevance" in report
        assert "Data Structure Quality" in report
        assert "Training Integration" in report
        assert "Ethical Accessibility" in report

    def test_therapeutic_relevance_assessment(self, engine):
        """Test therapeutic relevance assessment."""
        # High relevance source
        source = DatasetSource(
            source_id="test",
            title="Therapy Session Transcripts",
            authors=[],
            publication_date=datetime.now(),
            source_type="journal",
            url="https://example.com",
            abstract="Cognitive behavioral therapy transcripts with patient-counselor dialogues",
            keywords=["therapy", "cbt", "transcripts"],
            open_access=True,
            data_availability="available",
        )

        score, notes = engine._assess_therapeutic_relevance(source)
        assert 1 <= score <= 10
        assert "therapy" in notes.lower() or "transcript" in notes.lower()

    def test_data_structure_quality_assessment(self, engine):
        """Test data structure quality assessment."""
        # High quality source
        source = DatasetSource(
            source_id="test",
            title="Test Dataset",
            authors=["Author 1"],
            publication_date=datetime.now(),
            source_type="repository",
            url="https://example.com",
            doi="10.1000/test",
            abstract="Test abstract",
            keywords=["test"],
            open_access=True,
            data_availability="available",
        )

        score, notes = engine._assess_data_structure_quality(source)
        assert 1 <= score <= 10
        assert "doi" in notes.lower() or "repository" in notes.lower()

    def test_training_integration_assessment(self, engine):
        """Test training integration assessment."""
        # Good integration source
        source = DatasetSource(
            source_id="test",
            title="Test Dataset",
            authors=[],
            publication_date=datetime.now(),
            source_type="repository",
            url="https://example.com",
            abstract="Test",
            keywords=[],
            open_access=True,
            data_availability="available",
        )

        score, notes = engine._assess_training_integration(source)
        assert 1 <= score <= 10
        assert "available" in notes.lower()

    def test_ethical_accessibility_assessment(self, engine):
        """Test ethical accessibility assessment."""
        # Good ethical source
        source = DatasetSource(
            source_id="test",
            title="Test Dataset",
            authors=[],
            publication_date=datetime.now(),
            source_type="journal",
            url="https://example.com",
            doi="10.1000/test",
            abstract="Test",
            keywords=[],
            open_access=True,
            data_availability="available",
        )

        score, notes = engine._assess_ethical_accessibility(source)
        assert 1 <= score <= 10
        assert "open access" in notes.lower()

    def test_competitive_advantages_identification(self, engine, sample_source):
        """Test competitive advantages identification."""
        evaluation = engine.evaluate_dataset(sample_source)

        # Should identify some advantages for a good dataset
        assert isinstance(evaluation.competitive_advantages, list)
        # The sample source should have some advantages
        if evaluation.therapeutic_relevance >= 7:
            assert len(evaluation.competitive_advantages) > 0

