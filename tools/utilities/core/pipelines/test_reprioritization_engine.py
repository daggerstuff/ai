"""Tests for evidence-based reprioritization engine (PIX-536)."""

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from ai.tools.utilities.core.pipelines.reprioritization_engine import (
    BacklogItem,
    EvidenceAccumulation,
    EvidenceAccumulator,
    EvidencePoint,
    EvidenceSeverity,
    InterventionType,
    PriorityCalculator,
    PriorityChange,
    PriorityTier,
    ReprioritizationEngine,
    ReprioritizationReport,
    UpstreamDomain,
    _generate_item_id,
    _generate_title,
    _severity_weight,
    create_engine,
    run_reprioritization_from_report,
)

SAMPLE_FEEDBACK_REPORT = {
    "evaluation_source": "test_eval",
    "generated_at": "2026-05-13T14:52:18+00:00",
    "total_evaluated": 515749,
    "overall_score": 0.45,
    "failure_patterns": [
        {
            "pattern_id": "pattern_memory_recall_low",
            "pattern_type": "memory_deficiency",
            "description": "Model fails to recall relevant memories",
            "severity": "medium",
            "frequency": 0.3,
            "metrics_impacted": ["memory_recall_recall"],
        },
        {
            "pattern_id": "pattern_context_drift",
            "pattern_type": "context_alignment",
            "description": "Responses drift from conversation context",
            "severity": "high",
            "frequency": 0.4,
            "metrics_impacted": ["context_relevance"],
        },
        {
            "pattern_id": "pattern_privacy_risk",
            "pattern_type": "privacy_risk",
            "description": "PII detected in training data",
            "severity": "critical",
            "frequency": 0.05,
            "metrics_impacted": ["privacy_score"],
        },
    ],
    "upstream_mappings": [
        {
            "failure_pattern": {"pattern_id": "pattern_memory_recall_low"},
            "upstream_domain": "acquisition",
            "confidence": 0.7,
            "root_cause_hypothesis": "Source data lacks memory-context pairs",
        },
        {
            "failure_pattern": {"pattern_id": "pattern_context_drift"},
            "upstream_domain": "curation",
            "confidence": 0.8,
            "root_cause_hypothesis": "Normalization loses context boundaries",
        },
        {
            "failure_pattern": {"pattern_id": "pattern_privacy_risk"},
            "upstream_domain": "privacy",
            "confidence": 0.95,
            "root_cause_hypothesis": "PII scrubber missing new pattern types",
        },
    ],
    "interventions": [
        {
            "pattern_id": "pattern_memory_recall_low",
            "domain": "acquisition",
            "type": "source_intake",
            "description": "Acquire higher-quality memory-context paired data",
        },
    ],
}


class TestEvidenceSeverity(unittest.TestCase):
    def test_severity_weights(self):
        assert _severity_weight(EvidenceSeverity.CRITICAL) == 4.0
        assert _severity_weight(EvidenceSeverity.HIGH) == 3.0
        assert _severity_weight(EvidenceSeverity.MEDIUM) == 2.0
        assert _severity_weight(EvidenceSeverity.LOW) == 1.0

    def test_unknown_severity_defaults(self):
        assert _severity_weight("unknown") == 1.0


class TestEvidencePoint(unittest.TestCase):
    def test_to_dict_serializes_all_fields(self):
        point = EvidencePoint(
            pattern_id="test_pattern",
            pattern_type="memory_deficiency",
            description="Test description",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.3,
            confidence=0.8,
            root_cause_hypothesis="Test hypothesis",
            metrics_impacted=["metric_a", "metric_b"],
        )
        d = point.to_dict()
        assert d["pattern_id"] == "test_pattern"
        assert d["domain"] == "acquisition"
        assert d["severity"] == "high"
        assert d["frequency"] == 0.3
        assert d["confidence"] == 0.8
        assert d["metrics_impacted"] == ["metric_a", "metric_b"]


class TestEvidenceAccumulation(unittest.TestCase):
    def test_add_single_evidence_point(self):
        acc = EvidenceAccumulation(
            pattern_id="p1",
            domain=UpstreamDomain.ACQUISITION,
            description="Test",
            action_threshold=Decimal("0.5"),
        )
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="memory_deficiency",
            description="Test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            confidence=0.8,
            root_cause_hypothesis="Test",
        )
        acc.add_evidence(point)
        assert len(acc.evidence_points) == 1
        assert acc.first_seen == point.timestamp
        assert acc.last_seen == point.timestamp

    def test_weight_calculation_with_high_severity(self):
        acc = EvidenceAccumulation(
            pattern_id="p1",
            domain=UpstreamDomain.ACQUISITION,
            description="Test",
            action_threshold=Decimal("0.5"),
        )
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="memory_deficiency",
            description="Test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            confidence=0.8,
            root_cause_hypothesis="Test",
        )
        acc.add_evidence(point)
        assert acc.total_weight > 0
        assert acc.is_actionable is True

    def test_low_weight_does_not_cross_threshold(self):
        acc = EvidenceAccumulation(
            pattern_id="p1",
            domain=UpstreamDomain.ACQUISITION,
            description="Test",
            action_threshold=Decimal("10"),
        )
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="memory_deficiency",
            description="Test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.LOW,
            frequency=0.01,
            confidence=0.2,
            root_cause_hypothesis="Test",
        )
        acc.add_evidence(point)
        assert acc.is_actionable is False

    def test_multiple_evidence_points_accumulate(self):
        acc = EvidenceAccumulation(
            pattern_id="p1",
            domain=UpstreamDomain.ACQUISITION,
            description="Test",
            action_threshold=Decimal("0.5"),
        )
        for _i in range(3):
            point = EvidencePoint(
                pattern_id="p1",
                pattern_type="memory_deficiency",
                description="Test",
                domain=UpstreamDomain.ACQUISITION,
                severity=EvidenceSeverity.MEDIUM,
                frequency=0.3,
                confidence=0.7,
                root_cause_hypothesis="Test",
            )
            acc.add_evidence(point)
        assert len(acc.evidence_points) == 3
        assert acc.is_actionable is True

    def test_to_dict(self):
        acc = EvidenceAccumulation(
            pattern_id="p1",
            domain=UpstreamDomain.CURATION,
            description="Test",
            action_threshold=Decimal("1"),
        )
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="test",
            description="Test",
            domain=UpstreamDomain.CURATION,
            severity=EvidenceSeverity.MEDIUM,
            frequency=0.5,
            confidence=0.5,
            root_cause_hypothesis="Test",
        )
        acc.add_evidence(point)
        d = acc.to_dict()
        assert d["pattern_id"] == "p1"
        assert d["evidence_count"] == 1
        assert d["is_actionable"] is False


class TestEvidenceAccumulator(unittest.TestCase):
    def setUp(self):
        self.accumulator = EvidenceAccumulator(action_threshold=Decimal("0.5"))

    def test_ingest_feedback_dict(self):
        points = self.accumulator.ingest_feedback_dict(SAMPLE_FEEDBACK_REPORT)
        assert len(points) == 3
        assert points[0].pattern_id == "pattern_memory_recall_low"
        assert points[0].domain == UpstreamDomain.ACQUISITION
        assert points[1].domain == UpstreamDomain.CURATION
        assert points[2].domain == UpstreamDomain.PRIVACY
        # Verify evidence was actually recorded in the accumulator
        accumulations = self.accumulator.get_all_accumulations()
        assert len(accumulations) == 3
        assert "pattern_memory_recall_low" in accumulations
        assert len(accumulations["pattern_memory_recall_low"].evidence_points) == 1

    def test_ingest_feedback_report_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_FEEDBACK_REPORT, f)
            f.flush()
            points = self.accumulator.ingest_feedback_report(f.name)
        assert len(points) == 3
        # Verify evidence was recorded
        assert len(self.accumulator.get_all_accumulations()) == 3

    def test_record_evidence_creates_accumulation(self):
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="test",
            description="Test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            confidence=0.8,
            root_cause_hypothesis="Test",
        )
        acc = self.accumulator.record_evidence(point)
        assert acc.pattern_id == "p1"
        assert len(acc.evidence_points) == 1

    def test_record_evidence_accumulates_same_pattern(self):
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="test",
            description="Test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            confidence=0.8,
            root_cause_hypothesis="Test",
        )
        self.accumulator.record_evidence(point)
        self.accumulator.record_evidence(point)
        acc = self.accumulator.get_accumulation("p1")
        assert acc is not None
        assert len(acc.evidence_points) == 2

    def test_get_actionable_patterns(self):
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="test",
            description="Test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            confidence=0.8,
            root_cause_hypothesis="Test",
        )
        self.accumulator.record_evidence(point)
        actionable = self.accumulator.get_actionable_patterns()
        assert len(actionable) == 1
        assert actionable[0].pattern_id == "p1"

    def test_get_all_accumulations(self):
        point1 = EvidencePoint(
            pattern_id="p1",
            pattern_type="test",
            description="Test 1",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            confidence=0.8,
            root_cause_hypothesis="Test",
        )
        point2 = EvidencePoint(
            pattern_id="p2",
            pattern_type="test",
            description="Test 2",
            domain=UpstreamDomain.CURATION,
            severity=EvidenceSeverity.MEDIUM,
            frequency=0.3,
            confidence=0.6,
            root_cause_hypothesis="Test",
        )
        self.accumulator.record_evidence(point1)
        self.accumulator.record_evidence(point2)
        all_acc = self.accumulator.get_all_accumulations()
        assert len(all_acc) == 2
        assert "p1" in all_acc
        assert "p2" in all_acc

    def test_clear(self):
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="test",
            description="Test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            confidence=0.8,
            root_cause_hypothesis="Test",
        )
        self.accumulator.record_evidence(point)
        self.accumulator.clear()
        assert len(self.accumulator.get_all_accumulations()) == 0

    def test_summary(self):
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="test",
            description="Test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            confidence=0.8,
            root_cause_hypothesis="Test",
        )
        self.accumulator.record_evidence(point)
        summary = self.accumulator.summary()
        assert summary["total_patterns"] == 1
        assert summary["actionable_patterns"] == 1
        assert summary["total_evidence_points"] == 1
        assert "acquisition" in summary["by_domain"]

    def test_unknown_domain_defaults_to_curation(self):
        report = {
            "failure_patterns": [
                {
                    "pattern_id": "p1",
                    "pattern_type": "test",
                    "description": "Test",
                    "severity": "medium",
                    "frequency": 0.3,
                    "metrics_impacted": [],
                }
            ],
            "upstream_mappings": [
                {
                    "failure_pattern": {"pattern_id": "p1"},
                    "upstream_domain": "unknown_domain",
                    "confidence": 0.5,
                    "root_cause_hypothesis": "Test",
                }
            ],
            "interventions": [],
        }
        points = self.accumulator.ingest_feedback_dict(report)
        assert len(points) == 1
        assert points[0].domain == UpstreamDomain.CURATION


class TestPriorityCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = PriorityCalculator()

    def test_urgent_priority(self):
        score, tier = self.calculator.calculate_priority(
            evidence_weight=Decimal("5"),
            severity=EvidenceSeverity.CRITICAL,
            frequency=0.8,
            domain=UpstreamDomain.PRIVACY,
        )
        assert tier == PriorityTier.URGENT
        assert score >= self.calculator.urgent_threshold

    def test_high_priority(self):
        _score, tier = self.calculator.calculate_priority(
            evidence_weight=Decimal("4"),
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            domain=UpstreamDomain.ACQUISITION,
        )
        assert tier == PriorityTier.HIGH

    def test_medium_priority(self):
        _score, tier = self.calculator.calculate_priority(
            evidence_weight=Decimal("2.5"),
            severity=EvidenceSeverity.MEDIUM,
            frequency=0.3,
            domain=UpstreamDomain.CURATION,
        )
        assert tier == PriorityTier.MEDIUM

    def test_low_priority(self):
        _score, tier = self.calculator.calculate_priority(
            evidence_weight=Decimal("1.2"),
            severity=EvidenceSeverity.LOW,
            frequency=0.1,
            domain=UpstreamDomain.PACKAGING,
        )
        assert tier == PriorityTier.LOW

    def test_backlog_priority(self):
        _score, tier = self.calculator.calculate_priority(
            evidence_weight=Decimal("0.1"),
            severity=EvidenceSeverity.LOW,
            frequency=0.05,
            domain=UpstreamDomain.PACKAGING,
        )
        assert tier == PriorityTier.BACKLOG

    def test_coverage_gap_increases_priority(self):
        score_no_gap, _ = self.calculator.calculate_priority(
            evidence_weight=Decimal("1"),
            severity=EvidenceSeverity.MEDIUM,
            frequency=0.3,
            domain=UpstreamDomain.CURATION,
            coverage_gap=Decimal("0"),
        )
        score_with_gap, _ = self.calculator.calculate_priority(
            evidence_weight=Decimal("1"),
            severity=EvidenceSeverity.MEDIUM,
            frequency=0.3,
            domain=UpstreamDomain.CURATION,
            coverage_gap=Decimal("0.8"),
        )
        assert score_with_gap > score_no_gap

    def test_privacy_domain_has_higher_urgency(self):
        score_privacy, _ = self.calculator.calculate_priority(
            evidence_weight=Decimal("1"),
            severity=EvidenceSeverity.MEDIUM,
            frequency=0.3,
            domain=UpstreamDomain.PRIVACY,
        )
        score_packaging, _ = self.calculator.calculate_priority(
            evidence_weight=Decimal("1"),
            severity=EvidenceSeverity.MEDIUM,
            frequency=0.3,
            domain=UpstreamDomain.PACKAGING,
        )
        assert score_privacy > score_packaging

    def test_intervention_type_for_privacy_domain(self):
        intervention = self.calculator.calculate_intervention_type(
            domain=UpstreamDomain.PRIVACY,
            pattern_type="anything",
            severity=EvidenceSeverity.HIGH,
        )
        assert intervention == InterventionType.RULE_UPDATE

    def test_intervention_type_for_acquisition_domain(self):
        intervention = self.calculator.calculate_intervention_type(
            domain=UpstreamDomain.ACQUISITION,
            pattern_type="anything",
            severity=EvidenceSeverity.HIGH,
        )
        assert intervention == InterventionType.SOURCE_INTAKE

    def test_intervention_type_memory_deficiency(self):
        intervention = self.calculator.calculate_intervention_type(
            domain=UpstreamDomain.CURATION,
            pattern_type="memory_deficiency",
            severity=EvidenceSeverity.MEDIUM,
        )
        assert intervention == InterventionType.SOURCE_INTAKE

    def test_intervention_type_memory_noise(self):
        intervention = self.calculator.calculate_intervention_type(
            domain=UpstreamDomain.CURATION,
            pattern_type="memory_noise",
            severity=EvidenceSeverity.LOW,
        )
        assert intervention == InterventionType.NORMALIZATION_UPDATE

    def test_intervention_type_unknown_pattern(self):
        intervention = self.calculator.calculate_intervention_type(
            domain=UpstreamDomain.CURATION,
            pattern_type="unknown_pattern",
            severity=EvidenceSeverity.MEDIUM,
        )
        assert intervention == InterventionType.PRIORITY_CHANGE


class TestBacklogItem(unittest.TestCase):
    def test_to_dict(self):
        item = BacklogItem(
            item_id="test-item-1",
            domain=UpstreamDomain.ACQUISITION,
            intervention_type=InterventionType.SOURCE_INTAKE,
            title="Test item",
            description="Test description",
            priority_tier=PriorityTier.HIGH,
            priority_score=Decimal("2.5"),
            evidence_pattern_ids=["p1"],
            root_cause_hypothesis="Test",
            validation_criteria=["Criterion 1"],
        )
        d = item.to_dict()
        assert d["item_id"] == "test-item-1"
        assert d["domain"] == "acquisition"
        assert d["priority_tier"] == "high"
        assert d["priority_score"] == 2.5
        assert d["validation_criteria"] == ["Criterion 1"]

    def test_default_values(self):
        item = BacklogItem(
            item_id="test-item-2",
            domain=UpstreamDomain.CURATION,
            intervention_type=InterventionType.RULE_UPDATE,
            title="Test",
            description="Test",
            priority_tier=PriorityTier.MEDIUM,
            priority_score=Decimal("1"),
            evidence_pattern_ids=["p1"],
            root_cause_hypothesis="Test",
        )
        assert item.previous_priority_tier is None
        assert item.reason_for_change == ""
        assert item.created_at is not None


class TestPriorityChange(unittest.TestCase):
    def test_to_dict(self):
        change = PriorityChange(
            item_id="test-item-1",
            domain=UpstreamDomain.ACQUISITION,
            previous_tier=PriorityTier.MEDIUM,
            new_tier=PriorityTier.HIGH,
            previous_score=Decimal("1"),
            new_score=Decimal("2.5"),
            reason="Evidence accumulated",
            evidence_pattern_ids=["p1"],
        )
        d = change.to_dict()
        assert d["previous_tier"] == "medium"
        assert d["new_tier"] == "high"
        assert d["reason"] == "Evidence accumulated"

    def test_none_previous_tier(self):
        change = PriorityChange(
            item_id="test-item-1",
            domain=UpstreamDomain.ACQUISITION,
            previous_tier=None,
            new_tier=PriorityTier.HIGH,
            previous_score=Decimal("0"),
            new_score=Decimal("2.5"),
            reason="New item",
            evidence_pattern_ids=["p1"],
        )
        d = change.to_dict()
        assert d["previous_tier"] is None


class TestReprioritizationReport(unittest.TestCase):
    def test_to_dict(self):
        report = ReprioritizationReport(
            run_id="run-test",
            timestamp="2026-05-13T14:52:18+00:00",
            evidence_sources_consumed=3,
            total_evidence_points=5,
            actionable_patterns=2,
            backlog_items_created=1,
            backlog_items_reprioritized=1,
            priority_changes=[],
            new_backlog_items=[],
            reprioritized_items=[],
            unchanged_items=[],
            by_domain={"acquisition": {"total_items": 1}},
        )
        d = report.to_dict()
        assert d["run_id"] == "run-test"
        assert d["evidence_sources_consumed"] == 3
        assert d["by_domain"] == {"acquisition": {"total_items": 1}}

    def test_save_to_file(self):
        report = ReprioritizationReport(
            run_id="run-test",
            timestamp="2026-05-13T14:52:18+00:00",
            evidence_sources_consumed=1,
            total_evidence_points=1,
            actionable_patterns=1,
            backlog_items_created=0,
            backlog_items_reprioritized=0,
            priority_changes=[],
            new_backlog_items=[],
            reprioritized_items=[],
            unchanged_items=[],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            report.save(f.name)
            f.flush()
            with open(f.name) as rf:
                loaded = json.load(rf)
        assert loaded["run_id"] == "run-test"


class TestReprioritizationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ReprioritizationEngine(action_threshold=Decimal("0.3"))

    def test_load_feedback_report(self):
        points = self.engine.load_feedback_dict(SAMPLE_FEEDBACK_REPORT)
        assert len(points) == 3

    def test_run_reprioritization_creates_items(self):
        self.engine.load_feedback_dict(SAMPLE_FEEDBACK_REPORT)
        report = self.engine.run_reprioritization()
        assert report.backlog_items_created > 0
        assert report.evidence_sources_consumed == 3

    def test_run_reprioritization_report_has_run_id(self):
        self.engine.load_feedback_dict(SAMPLE_FEEDBACK_REPORT)
        report = self.engine.run_reprioritization()
        assert report.run_id.startswith("run-")

    def test_run_reprioritization_report_has_timestamp(self):
        self.engine.load_feedback_dict(SAMPLE_FEEDBACK_REPORT)
        report = self.engine.run_reprioritization()
        assert report.timestamp is not None

    def test_run_reprioritization_tracks_priority_changes(self):
        self.engine.load_feedback_dict(SAMPLE_FEEDBACK_REPORT)
        report = self.engine.run_reprioritization()
        assert report.actionable_patterns > 0

    def test_run_reprioritization_builds_domain_summary(self):
        self.engine.load_feedback_dict(SAMPLE_FEEDBACK_REPORT)
        report = self.engine.run_reprioritization()
        assert len(report.by_domain) > 0
        assert "acquisition" in report.by_domain

    def test_get_backlog_returns_sorted_items(self):
        self.engine.load_feedback_dict(SAMPLE_FEEDBACK_REPORT)
        self.engine.run_reprioritization()
        backlog = self.engine.get_backlog()
        assert len(backlog) > 0
        for i in range(len(backlog) - 1):
            assert backlog[i].priority_score >= backlog[i + 1].priority_score

    def test_get_backlog_by_domain(self):
        self.engine.load_feedback_dict(SAMPLE_FEEDBACK_REPORT)
        self.engine.run_reprioritization()
        acquisition_items = self.engine.get_backlog_by_domain(UpstreamDomain.ACQUISITION)
        assert len(acquisition_items) > 0
        for item in acquisition_items:
            assert item.domain == UpstreamDomain.ACQUISITION

    def test_add_existing_backlog_and_reprioritize(self):
        item_id = _generate_item_id("existing-item", UpstreamDomain.CURATION)
        existing = BacklogItem(
            item_id=item_id,
            domain=UpstreamDomain.CURATION,
            intervention_type=InterventionType.RULE_UPDATE,
            title="Existing item",
            description="Existing",
            priority_tier=PriorityTier.LOW,
            priority_score=Decimal("0.5"),
            evidence_pattern_ids=["old_pattern"],
            root_cause_hypothesis="Old",
        )
        self.engine.add_existing_backlog([existing])

        point = EvidencePoint(
            pattern_id="existing-item",
            pattern_type="context_alignment",
            description="Context drift detected",
            domain=UpstreamDomain.CURATION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.6,
            confidence=0.9,
            root_cause_hypothesis="Normalization issue",
        )
        self.engine.accumulator.record_evidence(point)

        report = self.engine.run_reprioritization()
        reprioritized_ids = [i.item_id for i in report.reprioritized_items]
        assert item_id in reprioritized_ids

    def test_churn_prevention_no_change(self):
        existing = BacklogItem(
            item_id="stable-item",
            domain=UpstreamDomain.CURATION,
            intervention_type=InterventionType.RULE_UPDATE,
            title="Stable item",
            description="Stable",
            priority_tier=PriorityTier.HIGH,
            priority_score=Decimal("2.5"),
            evidence_pattern_ids=["p1"],
            root_cause_hypothesis="Test",
        )
        self.engine.add_existing_backlog([existing])

        point = EvidencePoint(
            pattern_id="stable-item",
            pattern_type="context_alignment",
            description="Minor context issue",
            domain=UpstreamDomain.CURATION,
            severity=EvidenceSeverity.MEDIUM,
            frequency=0.1,
            confidence=0.3,
            root_cause_hypothesis="Minor",
        )
        self.engine.accumulator.record_evidence(point)

        report = self.engine.run_reprioritization()
        unchanged_ids = [i.item_id for i in report.unchanged_items]
        assert "stable-item" in unchanged_ids

    def test_run_reprioritization_from_report_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_FEEDBACK_REPORT, f)
            f.flush()
            report = run_reprioritization_from_report(f.name, action_threshold=Decimal("0.5"))
        assert report.backlog_items_created > 0

    def test_run_reprioritization_from_report_with_output(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_FEEDBACK_REPORT, f)
            f.flush()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as out:
                report = run_reprioritization_from_report(f.name, output_path=out.name)
                out.flush()
                with open(out.name) as rf:
                    loaded = json.load(rf)
                assert loaded["run_id"] == report.run_id

    def test_create_engine(self):
        engine = create_engine(action_threshold=Decimal("2"), churn_prevention_window_days=14)
        assert engine.accumulator._config.action_threshold == Decimal("2")
        assert engine._churn_window.days == 14

    def test_empty_report_produces_empty_report(self):
        empty_report = {
            "failure_patterns": [],
            "upstream_mappings": [],
            "interventions": [],
        }
        self.engine.load_feedback_dict(empty_report)
        report = self.engine.run_reprioritization()
        assert report.backlog_items_created == 0
        assert report.backlog_items_reprioritized == 0
        assert report.actionable_patterns == 0

    def test_item_id_is_stable(self):
        id1 = _generate_item_id("pattern_1", UpstreamDomain.ACQUISITION)
        id2 = _generate_item_id("pattern_1", UpstreamDomain.ACQUISITION)
        assert id1 == id2

    def test_item_id_differs_by_domain(self):
        id1 = _generate_item_id("pattern_1", UpstreamDomain.ACQUISITION)
        id2 = _generate_item_id("pattern_1", UpstreamDomain.CURATION)
        assert id1 != id2

    def test_item_id_differs_by_pattern(self):
        id1 = _generate_item_id("pattern_1", UpstreamDomain.ACQUISITION)
        id2 = _generate_item_id("pattern_2", UpstreamDomain.ACQUISITION)
        assert id1 != id2

    def test_title_generation(self):
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="test",
            description="Memory recall failing",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.3,
            confidence=0.8,
            root_cause_hypothesis="Test",
        )
        acc = EvidenceAccumulation(
            pattern_id="p1",
            domain=UpstreamDomain.ACQUISITION,
            description="Test",
        )
        acc.add_evidence(point)
        title = _generate_title(point, acc)
        assert "HIGH" in title
        assert "Acquisition" in title
        assert "Memory recall failing" in title

    def test_validation_criteria_generated(self):
        from ai.tools.utilities.core.pipelines.reprioritization_engine import _generate_validation_criteria

        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="test",
            description="Test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            confidence=0.8,
            root_cause_hypothesis="Test",
        )
        criteria = _generate_validation_criteria(point, InterventionType.SOURCE_INTAKE)
        assert len(criteria) > 0
        assert any("source" in c.lower() for c in criteria)

    def test_reprioritization_report_save_creates_directory(self):
        report = ReprioritizationReport(
            run_id="run-test",
            timestamp="2026-05-13T14:52:18+00:00",
            evidence_sources_consumed=1,
            total_evidence_points=1,
            actionable_patterns=1,
            backlog_items_created=0,
            backlog_items_reprioritized=0,
            priority_changes=[],
            new_backlog_items=[],
            reprioritized_items=[],
            unchanged_items=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "report.json"
            report.save(output_path)
            assert output_path.exists()

    def test_evidence_accumulation_decay_over_time(self):
        acc = EvidenceAccumulation(
            pattern_id="p1",
            domain=UpstreamDomain.ACQUISITION,
            description="Test",
            action_threshold=Decimal("0.5"),
        )
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="test",
            description="Test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            confidence=0.8,
            root_cause_hypothesis="Test",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        acc.add_evidence(point)
        old_weight = acc.total_weight

        point2 = EvidencePoint(
            pattern_id="p1",
            pattern_type="test",
            description="Test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.5,
            confidence=0.8,
            root_cause_hypothesis="Test",
        )
        acc.add_evidence(point2)
        new_weight = acc.total_weight

        assert new_weight > old_weight

    def test_get_priority_changes(self):
        self.engine.load_feedback_dict(SAMPLE_FEEDBACK_REPORT)
        self.engine.run_reprioritization()
        changes = self.engine.get_priority_changes()
        assert isinstance(changes, list)


class TestDecimalPrecision(unittest.TestCase):
    def test_small_weight_preserves_precision(self):
        """Very small values that lose precision in float stay exact in Decimal."""
        acc = EvidenceAccumulation(
            pattern_id="p1",
            domain=UpstreamDomain.ACQUISITION,
            description="Small values",
            action_threshold=Decimal("0.00001"),
        )
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="memory_deficiency",
            description="Small value test",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.CRITICAL,
            frequency=0.0001,
            confidence=0.0001,
            root_cause_hypothesis="Test",
        )
        acc.add_evidence(point)
        assert isinstance(acc.total_weight, Decimal)
        for _ in range(100):
            acc.add_evidence(point)
        assert isinstance(acc.total_weight, Decimal)
        assert acc.total_weight == acc.total_weight.quantize(Decimal("0.0001"))

    def test_high_frequency_accumulation_no_drift(self):
        """100 identical evidence points accumulate without drift."""
        acc = EvidenceAccumulation(
            pattern_id="p1",
            domain=UpstreamDomain.ACQUISITION,
            description="High frequency",
            action_threshold=Decimal("100"),
        )
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="memory_deficiency",
            description="Hundred points",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.LOW,
            frequency=1.0,
            confidence=1.0,
            root_cause_hypothesis="Test",
        )
        acc.add_evidence(point)
        single_weight = acc.total_weight
        for _ in range(99):
            acc.add_evidence(point)
        assert acc.total_weight > single_weight * 99
        self.assertAlmostEqual(float(acc.total_weight), 100.0 * float(single_weight), delta=0.02 * float(single_weight))

    def test_to_dict_decimal_roundtrip(self):
        """to_dict() converts Decimal→float without exceptions."""
        acc = EvidenceAccumulation(
            pattern_id="p1",
            domain=UpstreamDomain.ACQUISITION,
            description="Roundtrip test",
            action_threshold=Decimal("0.5"),
        )
        point = EvidencePoint(
            pattern_id="p1",
            pattern_type="memory_deficiency",
            description="Roundtrip",
            domain=UpstreamDomain.ACQUISITION,
            severity=EvidenceSeverity.HIGH,
            frequency=0.75,
            confidence=0.9,
            root_cause_hypothesis="Test",
        )
        acc.add_evidence(point)
        d = acc.to_dict()
        assert isinstance(d["total_weight"], float)
        assert isinstance(d["action_threshold"], float)
        assert d["total_weight"] > 0
        assert d["action_threshold"] == 0.5

    def test_to_dict_large_priority_score(self):
        """PriorityCalculator produces Decimal scores; to_dict handles large values."""
        calculator = PriorityCalculator()
        score, _ = calculator.calculate_priority(
            evidence_weight=Decimal("1000"),
            severity=EvidenceSeverity.CRITICAL,
            frequency=1.0,
            domain=UpstreamDomain.PRIVACY,
        )
        assert isinstance(score, Decimal)
        item = BacklogItem(
            item_id="test-large",
            domain=UpstreamDomain.PRIVACY,
            intervention_type=InterventionType.RULE_UPDATE,
            title="Large score",
            description="Large score test",
            priority_tier=PriorityTier.URGENT,
            priority_score=score,
            evidence_pattern_ids=["p1"],
            root_cause_hypothesis="Test",
        )
        d = item.to_dict()
        assert isinstance(d["priority_score"], float)
        assert d["priority_score"] > 0

    def test_severity_weight_returns_decimal(self):
        """_severity_weight returns Decimal (not float) for all severities."""
        for severity in EvidenceSeverity:
            w = _severity_weight(severity)
            assert isinstance(w, Decimal)
            assert w > 0


if __name__ == "__main__":
    unittest.main()
