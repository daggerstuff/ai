"""Tests for steering integration module (PIX-537)."""

import json
import tempfile
import unittest
from pathlib import Path

from ai.tools.utilities.pipelines.reprioritization_engine import (
    BacklogItem,
    InterventionType,
    PriorityTier,
    ReprioritizationEngine,
    UpstreamDomain,
)
from ai.tools.utilities.pipelines.steering_integration import (
    ApplicationStatus,
    SteeringAction,
    SteeringActionType,
    SteeringIntegration,
    SteeringReport,
    Workstream,
    WorkstreamState,
    _domain_to_workstream,
    _generate_action_details,
    _generate_action_id,
    _intervention_to_action_type,
    create_steering_integration,
    run_steering_from_report,
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
    "interventions": [],
}


class TestDomainMapping(unittest.TestCase):
    def test_acquisition_maps_to_acquisition_workstream(self):
        assert _domain_to_workstream(UpstreamDomain.ACQUISITION) == Workstream.ACQUISITION

    def test_curation_maps_to_curation_workstream(self):
        assert _domain_to_workstream(UpstreamDomain.CURATION) == Workstream.CURATION

    def test_privacy_maps_to_quality_handling(self):
        assert _domain_to_workstream(UpstreamDomain.PRIVACY) == Workstream.QUALITY_HANDLING

    def test_review_maps_to_quality_handling(self):
        assert _domain_to_workstream(UpstreamDomain.REVIEW) == Workstream.QUALITY_HANDLING

    def test_packaging_maps_to_curation(self):
        assert _domain_to_workstream(UpstreamDomain.PACKAGING) == Workstream.CURATION

    def test_unknown_domain_defaults_to_curation(self):
        assert _domain_to_workstream("unknown") == Workstream.CURATION


class TestInterventionMapping(unittest.TestCase):
    def test_source_intake_maps_correctly(self):
        assert (
            _intervention_to_action_type(InterventionType.SOURCE_INTAKE, UpstreamDomain.ACQUISITION)
            == SteeringActionType.ADD_SOURCE_PRIORITY
        )

    def test_privacy_domain_overrides_to_rule_update(self):
        assert (
            _intervention_to_action_type(InterventionType.NORMALIZATION_UPDATE, UpstreamDomain.PRIVACY)
            == SteeringActionType.UPDATE_PRIVACY_RULE
        )

    def test_acquisition_domain_overrides_to_source_priority(self):
        assert (
            _intervention_to_action_type(InterventionType.RULE_UPDATE, UpstreamDomain.ACQUISITION)
            == SteeringActionType.ADD_SOURCE_PRIORITY
        )

    def test_normalization_update_maps_correctly(self):
        assert (
            _intervention_to_action_type(InterventionType.NORMALIZATION_UPDATE, UpstreamDomain.CURATION)
            == SteeringActionType.UPDATE_NORMALIZATION_RULE
        )

    def test_review_focus_maps_correctly(self):
        assert (
            _intervention_to_action_type(InterventionType.REVIEW_FOCUS, UpstreamDomain.REVIEW)
            == SteeringActionType.UPDATE_REVIEW_FOCUS
        )

    def test_threshold_adjustment_maps_correctly(self):
        assert (
            _intervention_to_action_type(InterventionType.THRESHOLD_ADJUSTMENT, UpstreamDomain.CURATION)
            == SteeringActionType.ADJUST_THRESHOLD
        )

    def test_dataset_filter_maps_correctly(self):
        assert (
            _intervention_to_action_type(InterventionType.DATASET_FILTER, UpstreamDomain.CURATION)
            == SteeringActionType.UPDATE_DATASET_FILTER
        )

    def test_validation_gate_update_maps_correctly(self):
        assert (
            _intervention_to_action_type(InterventionType.VALIDATION_GATE_UPDATE, UpstreamDomain.CURATION)
            == SteeringActionType.UPDATE_VALIDATION_GATE
        )


class TestActionGeneration(unittest.TestCase):
    def test_generate_action_id_is_stable(self):
        id1 = _generate_action_id("item1", Workstream.ACQUISITION)
        id2 = _generate_action_id("item1", Workstream.ACQUISITION)
        assert id1 == id2

    def test_generate_action_id_differs_by_workstream(self):
        id1 = _generate_action_id("item1", Workstream.ACQUISITION)
        id2 = _generate_action_id("item1", Workstream.CURATION)
        assert id1 != id2

    def test_generate_action_details_for_source_intake(self):
        item = BacklogItem(
            item_id="test-item",
            domain=UpstreamDomain.ACQUISITION,
            intervention_type=InterventionType.SOURCE_INTAKE,
            title="Test item",
            description="Test",
            priority_tier=PriorityTier.HIGH,
            priority_score=2.5,
            evidence_pattern_ids=["p1"],
            root_cause_hypothesis="Test root cause",
            validation_criteria=["Criterion 1"],
        )
        details = _generate_action_details(item, SteeringActionType.ADD_SOURCE_PRIORITY)
        assert "root_cause" in details
        assert details["root_cause"] == "Test root cause"
        assert details["domain"] == "acquisition"

    def test_generate_action_details_for_privacy_rule(self):
        item = BacklogItem(
            item_id="test-item",
            domain=UpstreamDomain.PRIVACY,
            intervention_type=InterventionType.RULE_UPDATE,
            title="Test item",
            description="Test",
            priority_tier=PriorityTier.URGENT,
            priority_score=3.5,
            evidence_pattern_ids=["p1"],
            root_cause_hypothesis="PII issue",
            validation_criteria=["Criterion 1"],
        )
        details = _generate_action_details(item, SteeringActionType.UPDATE_PRIVACY_RULE)
        assert details["domain"] == "quality"
        assert "action" in details


class TestSteeringAction(unittest.TestCase):
    def test_to_dict_serializes_all_fields(self):
        action = SteeringAction(
            action_id="steer-acq-item1",
            workstream=Workstream.ACQUISITION,
            action_type=SteeringActionType.ADD_SOURCE_PRIORITY,
            description="Test action",
            details={"key": "value"},
            source_item_id="item1",
            source_pattern_id="p1",
            evidence_weight=2.5,
            priority_tier=PriorityTier.HIGH,
        )
        d = action.to_dict()
        assert d["action_id"] == "steer-acq-item1"
        assert d["workstream"] == "acquisition"
        assert d["action_type"] == "add_source_priority"
        assert d["priority_tier"] == "high"
        assert d["status"] == "pending"

    def test_default_status_is_pending(self):
        action = SteeringAction(
            action_id="test",
            workstream=Workstream.CURATION,
            action_type=SteeringActionType.UPDATE_NORMALIZATION_RULE,
            description="Test",
            details={},
            source_item_id="item1",
            source_pattern_id="p1",
            evidence_weight=1.0,
            priority_tier=PriorityTier.MEDIUM,
        )
        assert action.status == ApplicationStatus.PENDING
        assert action.applied_at is None
        assert action.rejection_reason is None


class TestWorkstreamState(unittest.TestCase):
    def test_to_dict(self):
        state = WorkstreamState(
            workstream=Workstream.ACQUISITION,
            active_rules=[{"rule": "test"}],
            pending_actions=["action1"],
            applied_actions=["action2"],
        )
        d = state.to_dict()
        assert d["workstream"] == "acquisition"
        assert len(d["active_rules"]) == 1
        assert d["pending_actions"] == ["action1"]
        assert d["applied_actions"] == ["action2"]


class TestSteeringIntegration(unittest.TestCase):
    def setUp(self):
        self.steering = create_steering_integration()
        self.engine = ReprioritizationEngine(action_threshold=0.3)
        self.engine.load_feedback_dict(SAMPLE_FEEDBACK_REPORT)
        self.report = self.engine.run_reprioritization()

    def test_process_report_generates_actions(self):
        steering_report = self.steering.process_report(self.report)
        assert steering_report.total_actions_generated > 0

    def test_process_report_skips_duplicate_pending_actions(self):
        first = self.steering.process_report(self.report)
        action_count = len(self.steering.get_all_actions())
        assert first.total_actions_generated > 0

        second = self.steering.process_report(self.report)
        assert second.total_actions_generated == 0
        assert len(self.steering.get_all_actions()) == action_count
        skipped = [entry for entry in self.steering._audit_trail if entry.get("event") == "action_skipped_idempotent"]
        assert len(skipped) >= first.total_actions_generated

    def test_process_report_has_correct_source_id(self):
        steering_report = self.steering.process_report(self.report)
        assert steering_report.source_report_id == self.report.run_id

    def test_process_report_has_run_id(self):
        steering_report = self.steering.process_report(self.report)
        assert steering_report.run_id.startswith("steer-")

    def test_process_report_tracks_actions_by_workstream(self):
        steering_report = self.steering.process_report(self.report)
        assert len(steering_report.actions_by_workstream) > 0

    def test_process_report_tracks_actions_by_type(self):
        steering_report = self.steering.process_report(self.report)
        assert len(steering_report.actions_by_type) > 0

    def test_process_report_without_handlers_leaves_actions_pending(self):
        steering_report = self.steering.process_report(self.report)
        assert steering_report.actions_pending > 0
        assert steering_report.actions_applied == 0

    def test_process_report_with_handlers_applies_actions(self):
        def mock_handler(action):
            return {"status": "applied"}

        for action_type in SteeringActionType:
            self.steering.register_handler(action_type, mock_handler)

        steering_report = self.steering.process_report(self.report)
        assert steering_report.actions_applied > 0

    def test_process_report_with_rejecting_handler(self):
        def reject_handler(action):
            return {"status": "rejected", "reason": "Test rejection"}

        for action_type in SteeringActionType:
            self.steering.register_handler(action_type, reject_handler)

        steering_report = self.steering.process_report(self.report)
        assert steering_report.actions_rejected > 0

    def test_get_action_by_id(self):
        self.steering.process_report(self.report)
        all_actions = self.steering.get_all_actions()
        assert len(all_actions) > 0
        action = self.steering.get_action(all_actions[0].action_id)
        assert action is not None
        assert action.action_id == all_actions[0].action_id

    def test_get_actions_by_workstream(self):
        self.steering.process_report(self.report)
        acquisition_actions = self.steering.get_actions_by_workstream(Workstream.ACQUISITION)
        assert len(acquisition_actions) > 0
        for action in acquisition_actions:
            assert action.workstream == Workstream.ACQUISITION

    def test_get_actions_by_status(self):
        self.steering.process_report(self.report)
        pending_actions = self.steering.get_actions_by_status(ApplicationStatus.PENDING)
        assert len(pending_actions) > 0
        for action in pending_actions:
            assert action.status == ApplicationStatus.PENDING

    def test_get_workstream_state(self):
        self.steering.process_report(self.report)
        state = self.steering.get_workstream_state(Workstream.ACQUISITION)
        assert state.workstream == Workstream.ACQUISITION
        assert len(state.pending_actions) > 0

    def test_get_summary(self):
        self.steering.process_report(self.report)
        summary = self.steering.get_summary()
        assert summary["total_actions"] > 0
        assert "by_status" in summary
        assert "by_workstream" in summary
        assert "workstream_states" in summary

    def test_audit_trail_populated(self):
        self.steering.process_report(self.report)
        assert len(self.steering._audit_trail) > 0
        events = [e["event"] for e in self.steering._audit_trail]
        assert "steering_report_processed" in events

    def test_handler_error_keeps_action_pending(self):
        def error_handler(action):
            raise ValueError("Test error")

        self.steering.register_handler(SteeringActionType.ADD_SOURCE_PRIORITY, error_handler)
        steering_report = self.steering.process_report(self.report)
        assert steering_report.actions_pending > 0


class TestSteeringReport(unittest.TestCase):
    def test_to_dict(self):
        report = SteeringReport(
            run_id="steer-test",
            timestamp="2026-05-13T14:52:18+00:00",
            source_report_id="run-test",
            total_actions_generated=2,
            actions_by_workstream={"acquisition": 1, "curation": 1},
            actions_by_type={"add_source_priority": 1, "update_normalization_rule": 1},
            actions_applied=0,
            actions_pending=2,
            actions_rejected=0,
            actions=[],
            workstream_states={},
        )
        d = report.to_dict()
        assert d["run_id"] == "steer-test"
        assert d["total_actions_generated"] == 2

    def test_save_to_file(self):
        report = SteeringReport(
            run_id="steer-test",
            timestamp="2026-05-13T14:52:18+00:00",
            source_report_id="run-test",
            total_actions_generated=1,
            actions_by_workstream={"acquisition": 1},
            actions_by_type={"add_source_priority": 1},
            actions_applied=0,
            actions_pending=1,
            actions_rejected=0,
            actions=[],
            workstream_states={},
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            report.save(f.name)
            f.flush()
            with open(f.name) as rf:
                loaded = json.load(rf)
        assert loaded["run_id"] == "steer-test"

    def test_save_creates_directory(self):
        report = SteeringReport(
            run_id="steer-test",
            timestamp="2026-05-13T14:52:18+00:00",
            source_report_id="run-test",
            total_actions_generated=0,
            actions_by_workstream={},
            actions_by_type={},
            actions_applied=0,
            actions_pending=0,
            actions_rejected=0,
            actions=[],
            workstream_states={},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "steering_report.json"
            report.save(output_path)
            assert output_path.exists()


class TestCreateSteeringIntegration(unittest.TestCase):
    def test_create_returns_steering_integration(self):
        steering = create_steering_integration()
        assert isinstance(steering, SteeringIntegration)

    def test_fresh_instance_has_no_actions(self):
        steering = create_steering_integration()
        assert len(steering.get_all_actions()) == 0

    def test_fresh_instance_has_all_workstream_states(self):
        steering = create_steering_integration()
        for ws in Workstream:
            state = steering.get_workstream_state(ws)
            assert state.workstream == ws


class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline_from_feedback_report(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_FEEDBACK_REPORT, f)
            f.flush()
            steering_report = run_steering_from_report(
                f.name,
                action_threshold=0.3,
            )
        assert steering_report.total_actions_generated > 0
        assert steering_report.actions_pending > 0

    def test_full_pipeline_with_output_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_FEEDBACK_REPORT, f)
            f.flush()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as out:
                steering_report = run_steering_from_report(
                    f.name,
                    steering_output_path=out.name,
                    action_threshold=0.3,
                )
                out.flush()
                with open(out.name) as rf:
                    loaded = json.load(rf)
                assert loaded["run_id"] == steering_report.run_id

    def test_full_pipeline_with_handlers(self):
        def mock_handler(action):
            return {"status": "applied"}

        handlers = {
            SteeringActionType.ADD_SOURCE_PRIORITY: mock_handler,
            SteeringActionType.UPDATE_NORMALIZATION_RULE: mock_handler,
            SteeringActionType.UPDATE_PRIVACY_RULE: mock_handler,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_FEEDBACK_REPORT, f)
            f.flush()
            steering_report = run_steering_from_report(
                f.name,
                action_threshold=0.3,
                handlers=handlers,
            )
        assert steering_report.actions_applied > 0


if __name__ == "__main__":
    unittest.main()
