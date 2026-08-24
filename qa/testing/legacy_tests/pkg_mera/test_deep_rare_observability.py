"""Tests for observability layer (Phase 4 enterprise upgrade)."""

from __future__ import annotations

import json
import logging

import pytest

from ai.tools.utilities.platform.deep_rare.clinical_safety import AuditAction, AuditTrail, SafetyLevel, SafetyViolation
from ai.tools.utilities.platform.deep_rare.observability import (
    AuditExporter,
    HealthSnapshot,
    MetricsCollector,
    ObservabilityContext,
    StructuredFormatter,
    TraceContext,
)
from ai.tools.utilities.platform.deep_rare.schema import (
    DiagnosisResult,
    DifferentialDiagnosis,
    EvaluationMetrics,
    PatientCase,
    RareDiseaseState,
    SymptomProfile,
)


@pytest.fixture
def metrics() -> MetricsCollector:
    return MetricsCollector()


@pytest.fixture
def obs_context() -> ObservabilityContext:
    return ObservabilityContext()


class TestMetricsCollector:
    def test_increment_counter(self, metrics: MetricsCollector):
        metrics.increment("diagnoses_total", 1)
        metrics.increment("diagnoses_total", 2)
        assert metrics.get_counter("diagnoses_total") == 3

    def test_set_gauge(self, metrics: MetricsCollector):
        metrics.gauge("active_hypotheses", 5)
        assert metrics.get_gauge("active_hypotheses") == 5

    def test_histogram(self, metrics: MetricsCollector):
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            metrics.histogram("diagnosis_time", v)
        stats = metrics.get_histogram_stats("diagnosis_time")
        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(3.0)

    def test_timing(self, metrics: MetricsCollector):
        metrics.timing("phase_duration", 0.5)
        # timing uses histogram internally
        stats = metrics.get_histogram_stats("phase_duration")
        assert stats["count"] == 1

    def test_prometheus_export(self, metrics: MetricsCollector):
        metrics.increment("test_metric", 1)
        prom = metrics.export_prometheus()
        assert isinstance(prom, str)
        assert "test_metric" in prom

    def test_thread_safety(self, metrics: MetricsCollector):
        import threading

        def worker():
            for _ in range(100):
                metrics.increment("concurrent", 1)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert metrics.get_counter("concurrent") == 500

    def test_reset(self, metrics: MetricsCollector):
        metrics.increment("test", 1)
        metrics.reset()
        assert metrics.get_counter("test") == 0

    def test_get_all_metrics(self, metrics: MetricsCollector):
        metrics.increment("a", 1)
        metrics.gauge("b", 2)
        all_m = metrics.get_all_metrics()
        assert isinstance(all_m, dict)


class TestTraceContext:
    def test_phase_context_manager(self):
        trace = TraceContext(case_id="CASE-001")
        with trace.phase("symptom_analysis"):
            pass
        d = trace.to_dict()
        assert d["case_id"] == "CASE-001"
        assert "phases" in d
        assert "symptom_analysis" in d["phases"]

    def test_record_agent_call(self):
        trace = TraceContext(case_id="CASE-001")
        trace.record_agent_call("symptom_analyzer")
        d = trace.to_dict()
        assert "agent_calls" in d

    def test_record_error(self):
        trace = TraceContext(case_id="CASE-001")
        with trace.phase("test_phase"):
            trace.record_error("test_phase", "something went wrong")
        d = trace.to_dict()
        assert "errors" in d

    def test_total_elapsed(self):
        trace = TraceContext(case_id="CASE-001")
        assert trace.total_elapsed() >= 0


class TestAuditExporter:
    def test_to_json(self):
        trail = AuditTrail()
        trail.record(AuditAction.HYPOTHESIS_CREATED, "symptom_analyzer", "CASE-001", {"disease": "Test"})
        trail.record(AuditAction.TEST_INTERPRETED, "test_interpreter", "CASE-001", {"test": "CK"})
        json_str = AuditExporter.to_json(trail)
        data = json.loads(json_str)
        assert isinstance(data, dict)
        assert "entries" in data
        assert data["entry_count"] == 2

    def test_to_csv(self):
        trail = AuditTrail()
        trail.record(AuditAction.DIAGNOSIS_FINALIZED, "orchestrator", "CASE-001", {"result": "confirmed"})
        csv_str = AuditExporter.to_csv(trail)
        assert isinstance(csv_str, str)
        assert "CASE-001" in csv_str

    def test_to_fhir_audit_events(self):
        trail = AuditTrail()
        trail.record(AuditAction.HYPOTHESIS_CREATED, "symptom_analyzer", "CASE-001", {})
        events = AuditExporter.to_fhir_audit_events(trail)
        assert isinstance(events, list)
        assert len(events) == 1


class TestObservabilityContext:
    def _make_result(self) -> DiagnosisResult:
        return DiagnosisResult(
            case_id="CASE-001",
            differential=DifferentialDiagnosis(
                ranked_list=[],
                eliminated=[],
                total_hypotheses_considered=0,
                iterations_used=3,
                convergence_achieved=True,
                reasoning_trace="test",
            ),
            state=RareDiseaseState(max_iterations=5, convergence_window=3),
            iterations=3,
            time_seconds=12.5,
            converged=True,
            agent_outputs={},
            recommended_next_steps=[],
            clinical_confidence=0.75,
        )

    def _make_case(self) -> PatientCase:
        return PatientCase(
            case_id="CASE-001",
            patient_age=30,
            patient_sex="male",
            presenting_symptoms=[
                SymptomProfile(
                    name="muscle weakness",
                    category="musculoskeletal",
                    onset="chronic",
                    progression="worsening",
                    severity="moderate",
                ),
            ],
            medical_history=[],
            family_history=[],
            current_medications=[],
            available_tests=[],
            clinical_notes="",
        )

    def test_record_diagnosis(self, obs_context: ObservabilityContext):
        obs_context.record_diagnosis(self._make_result(), self._make_case())
        all_m = obs_context.metrics.get_all_metrics()
        assert isinstance(all_m, dict)

    def test_record_evaluation(self, obs_context: ObservabilityContext):
        metrics = EvaluationMetrics(
            recall_at_1=0.42,
            recall_at_5=0.6,
            recall_at_10=0.7,
            mrr=0.55,
            accuracy_by_organ={},
            accuracy_by_rarity={},
            accuracy_by_complexity={},
            avg_iterations=3.0,
            avg_time_seconds=12.0,
            total_cases=10,
            correct_cases=4,
        )
        obs_context.record_evaluation(metrics)
        all_m = obs_context.metrics.get_all_metrics()
        assert isinstance(all_m, dict)

    def test_record_safety_violation(self, obs_context: ObservabilityContext):
        violation = SafetyViolation(
            violation_id="sv_001",
            level=SafetyLevel.WARNING,
            rule_name="test_rule",
            description="test desc",
            context={},
            timestamp="2026-01-01T00:00:00Z",
            remediation=None,
        )
        obs_context.record_safety_violation(violation)
        all_m = obs_context.metrics.get_all_metrics()
        assert isinstance(all_m, dict)

    def test_get_health_snapshot(self, obs_context: ObservabilityContext):
        snapshot = obs_context.get_health_snapshot()
        assert isinstance(snapshot, (HealthSnapshot, dict))

    def test_reset(self, obs_context: ObservabilityContext):
        obs_context.record_diagnosis(self._make_result(), self._make_case())
        obs_context.reset()
        assert obs_context.metrics.get_counter("diagnoses_total") == 0

    def test_trace(self, obs_context: ObservabilityContext):
        with obs_context.trace("CASE-001") as t:
            assert t is not None


class TestStructuredFormatter:
    def test_format_record(self):
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="test message",
            args=None,
            exc_info=None,
        )
        record.__dict__["case_id"] = "CASE-001"
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "test message"
        assert data["case_id"] == "CASE-001"
        assert "timestamp" in data
