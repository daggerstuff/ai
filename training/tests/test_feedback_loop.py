#!/usr/bin/env python3
"""Tests for evaluation-to-data feedback loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.training.feedback_loop import (
    DatasetIntervention,
    EvaluationParser,
    FailurePattern,
    FeedbackLoop,
    FeedbackReport,
    InterventionGenerator,
    InterventionType,
    UpstreamCauseMapper,
    UpstreamDomain,
    UpstreamMapping,
)


@pytest.fixture
def sample_evaluation_report() -> dict:
    """Sample evaluation report with memory and quality metrics."""
    return {
        "evaluated_examples": 1000,
        "overall_score": 0.65,
        "memory_metrics": {
            "avg_recall_recall": 0.45,  # Below threshold 0.6
            "avg_recall_precision": 0.52,  # Below threshold 0.5
        },
        "quality_metrics": {
            "avg_context_relevance": 0.55,  # Below threshold 0.6
            "avg_reflection_quality": 0.48,  # Below threshold 0.5
            "avg_generation_quality": 0.70,  # Above threshold
        },
    }


@pytest.fixture
def sample_evaluation_report_file(sample_evaluation_report: dict, tmp_path: Path) -> Path:
    """Create a temporary evaluation report file."""
    report_path = tmp_path / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(sample_evaluation_report, f)
    return report_path


class TestEvaluationParser:
    """Test evaluation parsing and failure pattern identification."""

    def test_parse_identifies_memory_recall_issue(self, sample_evaluation_report: dict):
        """Parser should identify memory recall deficiency."""
        parser = EvaluationParser()
        patterns = parser.parse(sample_evaluation_report)

        # Should find patterns for metrics below threshold
        assert len(patterns) > 0

        # Check for memory_recall_low pattern
        memory_pattern = next(
            (p for p in patterns if p.pattern_id == "pattern_memory_recall_low"),
            None,
        )
        assert memory_pattern is not None
        assert memory_pattern.pattern_type == "memory_deficiency"
        assert memory_pattern.severity in ["critical", "high", "medium", "low"]
        assert memory_pattern.frequency > 0
        assert "memory_recall_recall" in memory_pattern.metrics_impacted

    def test_parse_identifies_memory_irrelevant_issue(self, sample_evaluation_report: dict):
        """Parser should identify memory noise issue."""
        parser = EvaluationParser()
        patterns = parser.parse(sample_evaluation_report)

        # Note: memory_irrelevant has threshold 0.5, and 0.52 is above it,
        # so this pattern should NOT be identified
        memory_irrelevant = next(
            (p for p in patterns if p.pattern_id == "pattern_memory_irrelevant"),
            None,
        )
        assert memory_irrelevant is None  # 0.52 > 0.5 threshold

    def test_parse_identifies_context_drift(self, sample_evaluation_report: dict):
        """Parser should identify context alignment issue."""
        parser = EvaluationParser()
        patterns = parser.parse(sample_evaluation_report)

        context_pattern = next(
            (p for p in patterns if p.pattern_id == "pattern_context_drift"),
            None,
        )
        assert context_pattern is not None
        assert context_pattern.pattern_type == "context_alignment"

    def test_parse_identifies_reflection_absent(self, sample_evaluation_report: dict):
        """Parser should identify reflection quality issue."""
        parser = EvaluationParser()
        patterns = parser.parse(sample_evaluation_report)

        reflection_pattern = next(
            (p for p in patterns if p.pattern_id == "pattern_reflection_absent"),
            None,
        )
        assert reflection_pattern is not None
        assert reflection_pattern.pattern_type == "reflection_quality"

    def test_parse_does_not_identify_above_threshold(self, sample_evaluation_report: dict):
        """Parser should not identify issues above threshold."""
        parser = EvaluationParser()
        patterns = parser.parse(sample_evaluation_report)

        # generation_incoherent should NOT be identified (0.7 > 0.6 threshold)
        generation_pattern = next(
            (p for p in patterns if p.pattern_id == "pattern_generation_incoherent"),
            None,
        )
        assert generation_pattern is None

    def test_determine_severity_critical(self):
        """Parser should classify critical severity correctly."""
        parser = EvaluationParser()
        # 50% below threshold = critical (< threshold * 0.5)
        severity = parser._determine_severity(0.25, 0.6)
        assert severity == "critical"

    def test_determine_severity_high(self):
        """Parser should classify high severity correctly."""
        parser = EvaluationParser()
        # 70% of threshold = high (< threshold * 0.7)
        severity = parser._determine_severity(0.40, 0.6)
        assert severity == "high"

    def test_determine_severity_medium(self):
        """Parser should classify medium severity correctly."""
        parser = EvaluationParser()
        # 90% of threshold = medium (< threshold * 0.9)
        severity = parser._determine_severity(0.52, 0.6)
        assert severity == "medium"

    def test_determine_severity_low(self):
        """Parser should classify low severity correctly."""
        parser = EvaluationParser()
        # Just below threshold = low
        severity = parser._determine_severity(0.59, 0.6)
        assert severity == "low"


class TestUpstreamCauseMapper:
    """Test failure-to-upstream mapping."""

    @pytest.fixture
    def sample_failure_patterns(self) -> list[FailurePattern]:
        """Create sample failure patterns for testing."""
        return [
            FailurePattern(
                pattern_id="pattern_memory_recall_low",
                pattern_type="memory_deficiency",
                description="Model fails to recall relevant memories",
                affected_examples=["ex1", "ex2"],
                severity="critical",
                frequency=0.25,
                metrics_impacted=["memory_recall_recall"],
            ),
            FailurePattern(
                pattern_id="pattern_memory_irrelevant",
                pattern_type="memory_noise",
                description="Model retrieves irrelevant memories",
                affected_examples=["ex3"],
                severity="high",
                frequency=0.15,
                metrics_impacted=["memory_recall_precision"],
            ),
            FailurePattern(
                pattern_id="pattern_context_drift",
                pattern_type="context_alignment",
                description="Responses drift from context",
                affected_examples=["ex4", "ex5"],
                severity="medium",
                frequency=0.08,
                metrics_impacted=["context_relevance"],
            ),
        ]

    def test_map_maps_memory_deficiency_to_acquisition(
        self, sample_failure_patterns: list[FailurePattern]
    ):
        """Mapper should map memory deficiency to acquisition domain."""
        mapper = UpstreamCauseMapper()
        mappings = mapper.map(sample_failure_patterns)

        memory_mapping = next(
            (m for m in mappings if m.failure_pattern.pattern_id == "pattern_memory_recall_low"),
            None,
        )
        assert memory_mapping is not None
        assert memory_mapping.likely_upstream_domain == UpstreamDomain.ACQUISITION
        assert "source data" in memory_mapping.root_cause_hypothesis.lower()
        assert memory_mapping.confidence > 0

    def test_map_maps_memory_noise_to_curation(
        self, sample_failure_patterns: list[FailurePattern]
    ):
        """Mapper should map memory noise to curation domain."""
        mapper = UpstreamCauseMapper()
        mappings = mapper.map(sample_failure_patterns)

        noise_mapping = next(
            (m for m in mappings if m.failure_pattern.pattern_id == "pattern_memory_irrelevant"),
            None,
        )
        assert noise_mapping is not None
        assert noise_mapping.likely_upstream_domain == UpstreamDomain.CURATION
        assert "curation" in noise_mapping.root_cause_hypothesis.lower()

    def test_map_maps_context_alignment_to_curation(
        self, sample_failure_patterns: list[FailurePattern]
    ):
        """Mapper should map context alignment to curation domain."""
        mapper = UpstreamCauseMapper()
        mappings = mapper.map(sample_failure_patterns)

        context_mapping = next(
            (m for m in mappings if m.failure_pattern.pattern_id == "pattern_context_drift"),
            None,
        )
        assert context_mapping is not None
        assert context_mapping.likely_upstream_domain == UpstreamDomain.CURATION

    def test_map_calculates_confidence(self, sample_failure_patterns: list[FailurePattern]):
        """Mapper should calculate confidence based on severity and frequency."""
        mapper = UpstreamCauseMapper()
        mappings = mapper.map(sample_failure_patterns)

        # Critical severity should have high confidence
        critical_mapping = next(
            (m for m in mappings if m.failure_pattern.severity == "critical"),
            None,
        )
        assert critical_mapping is not None
        assert critical_mapping.confidence >= 0.8

        # High severity should have high confidence
        high_mapping = next(
            (m for m in mappings if m.failure_pattern.severity == "high"),
            None,
        )
        assert high_mapping is not None
        assert high_mapping.confidence >= 0.7


class TestInterventionGenerator:
    """Test intervention generation."""

    @pytest.fixture
    def sample_mappings(self) -> list[UpstreamMapping]:
        """Create sample upstream mappings for testing."""
        return [
            UpstreamMapping(
                failure_pattern=FailurePattern(
                    pattern_id="pattern_memory_recall_low",
                    pattern_type="memory_deficiency",
                    description="Model fails to recall memories",
                    affected_examples=["ex1"],
                    severity="critical",
                    frequency=0.25,
                    metrics_impacted=["memory_recall_recall"],
                ),
                likely_upstream_domain=UpstreamDomain.ACQUISITION,
                confidence=0.9,
                root_cause_hypothesis="Source data lacks quality memory-context pairs",
                evidence=["acquisition_logs"],
            ),
            UpstreamMapping(
                failure_pattern=FailurePattern(
                    pattern_id="pattern_memory_irrelevant",
                    pattern_type="memory_noise",
                    description="Retrieves irrelevant memories",
                    affected_examples=["ex2"],
                    severity="high",
                    frequency=0.15,
                    metrics_impacted=["memory_recall_precision"],
                ),
                likely_upstream_domain=UpstreamDomain.CURATION,
                confidence=0.85,
                root_cause_hypothesis="Curation rules allow low-relevance content",
                evidence=["curation_rules"],
            ),
        ]

    def test_generate_creates_interventions(self, sample_mappings: list[UpstreamMapping]):
        """Generator should create interventions from mappings."""
        generator = InterventionGenerator()
        interventions = generator.generate(sample_mappings)

        assert len(interventions) > 0

    def test_generate_creates_acquisition_interventions(
        self, sample_mappings: list[UpstreamMapping]
    ):
        """Generator should create acquisition domain interventions."""
        generator = InterventionGenerator()
        interventions = generator.generate(sample_mappings)

        acquisition_interventions = [
            i for i in interventions if i.upstream_domain == UpstreamDomain.ACQUISITION
        ]
        assert len(acquisition_interventions) > 0

        # Check intervention has required fields
        intervention = acquisition_interventions[0]
        assert intervention.intervention_type in [
            InterventionType.PRIORITY_CHANGE,
            InterventionType.THRESHOLD_CHANGE,
        ]
        assert len(intervention.title) > 0
        assert len(intervention.description) > 0
        assert intervention.priority == "critical"
        assert len(intervention.validation_criteria) > 0

    def test_generate_creates_curation_interventions(
        self, sample_mappings: list[UpstreamMapping]
    ):
        """Generator should create curation domain interventions."""
        generator = InterventionGenerator()
        interventions = generator.generate(sample_mappings)

        curation_interventions = [
            i for i in interventions if i.upstream_domain == UpstreamDomain.CURATION
        ]
        assert len(curation_interventions) > 0

        intervention = curation_interventions[0]
        assert intervention.intervention_type in [
            InterventionType.RULE_CHANGE,
            InterventionType.DATASET_FILTER,
        ]

    def test_generate_includes_implementation_details(
        self, sample_mappings: list[UpstreamMapping]
    ):
        """Interventions should include implementation details."""
        generator = InterventionGenerator()
        interventions = generator.generate(sample_mappings)

        intervention = interventions[0]
        assert "root_cause" in intervention.implementation_details
        assert "confidence" in intervention.implementation_details
        assert intervention.implementation_details["confidence"] > 0


class TestFeedbackLoop:
    """Test complete feedback loop execution."""

    def test_run_creates_feedback_report(
        self, sample_evaluation_report_file: Path, tmp_path: Path
    ):
        """Feedback loop should create a complete feedback report."""
        loop = FeedbackLoop()
        output_dir = tmp_path / "feedback_output"

        report = loop.run(sample_evaluation_report_file, output_dir)

        assert isinstance(report, FeedbackReport)
        assert report.evaluation_source == str(sample_evaluation_report_file)
        assert report.total_evaluated == 1000
        assert report.overall_score == 0.65
        assert len(report.failure_patterns) > 0
        assert len(report.upstream_mappings) > 0
        assert len(report.interventions) > 0

    def test_run_saves_feedback_report_json(
        self, sample_evaluation_report_file: Path, tmp_path: Path
    ):
        """Feedback loop should save report to JSON file."""
        loop = FeedbackLoop()
        output_dir = tmp_path / "feedback_output"

        loop.run(sample_evaluation_report_file, output_dir)

        report_path = output_dir / "feedback_report.json"
        assert report_path.exists()

        with open(report_path) as f:
            saved_report = json.load(f)

        assert saved_report["total_evaluated"] == 1000
        assert len(saved_report["failure_patterns"]) > 0
        assert len(saved_report["interventions"]) > 0

    def test_run_creates_linear_issue_templates(
        self, sample_evaluation_report_file: Path, tmp_path: Path
    ):
        """Feedback loop should create Linear issue templates."""
        loop = FeedbackLoop()
        output_dir = tmp_path / "feedback_output"

        report = loop.run(sample_evaluation_report_file, output_dir)

        issues_dir = output_dir / "linear_issues"
        assert issues_dir.exists()

        # Should have one issue file per intervention
        issue_files = list(issues_dir.glob("*.md"))
        # Each mapping can generate up to 2 interventions
        assert len(issue_files) >= 1
        assert len(issue_files) <= len(report.interventions) + 2

        # Check issue file format
        first_issue = issue_files[0]
        content = first_issue.read_text()
        assert "## Description" in content
        assert "## Validation Criteria" in content
        assert "## Root Cause" in content

    def test_run_calculates_summary_statistics(
        self, sample_evaluation_report_file: Path, tmp_path: Path
    ):
        """Feedback loop should calculate summary statistics."""
        loop = FeedbackLoop()
        output_dir = tmp_path / "feedback_output"

        report = loop.run(sample_evaluation_report_file, output_dir)

        assert report.critical_issues >= 0
        assert report.high_priority_issues >= 0
        assert report.recommended_actions == len(report.interventions)

        # Verify counts match patterns
        assert report.critical_issues == sum(
            1 for p in report.failure_patterns if p.severity == "critical"
        )
        assert report.high_priority_issues == sum(
            1 for p in report.failure_patterns if p.severity == "high"
        )

    def test_feedback_report_to_dict(self, sample_evaluation_report_file: Path, tmp_path: Path):
        """FeedbackReport should serialize to dict correctly."""
        loop = FeedbackLoop()
        output_dir = tmp_path / "feedback_output"

        report = loop.run(sample_evaluation_report_file, output_dir)
        report_dict = report.to_dict()

        assert "evaluation_source" in report_dict
        assert "generated_at" in report_dict
        assert "failure_patterns" in report_dict
        assert "upstream_mappings" in report_dict
        assert "interventions" in report_dict
        assert "summary" in report_dict

        # Check nested structures
        assert isinstance(report_dict["failure_patterns"], list)
        assert isinstance(report_dict["interventions"], list)


class TestFailurePatternDataclass:
    """Test FailurePattern dataclass functionality."""

    def test_to_dict(self):
        """FailurePattern should serialize correctly."""
        pattern = FailurePattern(
            pattern_id="test_pattern",
            pattern_type="test_type",
            description="Test description",
            affected_examples=["ex1", "ex2"],
            severity="high",
            frequency=0.5,
            metrics_impacted=["metric1"],
        )

        pattern_dict = pattern.to_dict()

        assert pattern_dict["pattern_id"] == "test_pattern"
        assert pattern_dict["pattern_type"] == "test_type"
        assert pattern_dict["severity"] == "high"
        assert pattern_dict["frequency"] == 0.5


class TestDatasetInterventionDataclass:
    """Test DatasetIntervention dataclass functionality."""

    def test_to_dict(self):
        """DatasetIntervention should serialize correctly."""
        intervention = DatasetIntervention(
            intervention_id="test_intervention",
            intervention_type=InterventionType.RULE_CHANGE,
            title="Test intervention",
            description="Test description",
            upstream_domain=UpstreamDomain.CURATION,
            priority="critical",
            expected_impact="Improve metric by 20%",
            implementation_details={"key": "value"},
            validation_criteria=["Criterion 1", "Criterion 2"],
            related_patterns=["pattern1"],
        )

        intervention_dict = intervention.to_dict()

        assert intervention_dict["intervention_id"] == "test_intervention"
        assert intervention_dict["intervention_type"] == "rule_change"
        assert intervention_dict["upstream_domain"] == "curation"
        assert intervention_dict["priority"] == "critical"


class TestUpstreamMappingDataclass:
    """Test UpstreamMapping dataclass functionality."""

    def test_to_dict(self):
        """UpstreamMapping should serialize correctly."""
        mapping = UpstreamMapping(
            failure_pattern=FailurePattern(
                pattern_id="test",
                pattern_type="test_type",
                description="Test",
                affected_examples=[],
                severity="medium",
                frequency=0.3,
                metrics_impacted=["metric"],
            ),
            likely_upstream_domain=UpstreamDomain.REVIEW,
            confidence=0.75,
            root_cause_hypothesis="Test hypothesis",
            evidence=["source1", "source2"],
        )

        mapping_dict = mapping.to_dict()

        assert mapping_dict["upstream_domain"] == "review"
        assert mapping_dict["confidence"] == 0.75
        assert mapping_dict["root_cause_hypothesis"] == "Test hypothesis"
        assert "failure_pattern" in mapping_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
