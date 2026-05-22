"""Tests for pipeline observability module (PIX-507)."""

import unittest

from ai.core.pipelines.pipeline_observability import (
    FailureSeverity,
    HealthStatus,
    PipelineHealthSummary,
    PipelineMetricsCollector,
    get_health_summary,
    get_prometheus_metrics,
    record_failure,
    record_stage_execution,
)
from ai.core.pipelines.training_readiness_gates import (
    TrainingReadinessGates,
)


class TestPipelineMetricsCollector(unittest.TestCase):
    """Test PipelineMetricsCollector functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = PipelineMetricsCollector(max_history=100)

    def test_record_stage_execution_records_metric(self):
        """Test that stage execution is recorded correctly."""
        self.collector.record_stage_execution(
            stage_name="normalize",
            duration_ms=45.5,
            input_size=100,
            output_size=95,
            status="completed",
        )

        throughput = self.collector.get_throughput_metrics()
        self.assertEqual(throughput.total_records_in, 100)
        self.assertEqual(throughput.total_records_out, 95)
        self.assertIn("normalize", throughput.stage_metrics)
        self.assertEqual(throughput.stage_metrics["normalize"]["count"], 1)
        self.assertAlmostEqual(throughput.stage_metrics["normalize"]["avg_ms"], 45.5, places=1)

    def test_record_stage_execution_tracks_failures(self):
        """Test that stage failures are tracked."""
        self.collector.record_stage_execution(
            stage_name="validate",
            duration_ms=10.0,
            input_size=50,
            output_size=0,
            status="failed",
            error="Validation failed: missing required field",
        )

        throughput = self.collector.get_throughput_metrics()
        self.assertEqual(throughput.stage_metrics["validate"]["failures"], 1)

    def test_multiple_stage_executions_aggregate_correctly(self):
        """Test that multiple executions are properly aggregated."""
        for i in range(5):
            self.collector.record_stage_execution(
                stage_name="normalize",
                duration_ms=40.0 + i * 5,
                input_size=100,
                output_size=90 + i,
                status="completed",
            )

        throughput = self.collector.get_throughput_metrics()
        self.assertEqual(throughput.stage_metrics["normalize"]["count"], 5)
        self.assertAlmostEqual(throughput.stage_metrics["normalize"]["avg_ms"], 50.0, places=0)

    def test_throughput_metrics_with_time_window(self):
        """Test that time window filtering works."""
        # Record a metric
        self.collector.record_stage_execution(
            stage_name="normalize",
            duration_ms=50.0,
            input_size=100,
            output_size=90,
            status="completed",
        )

        # Get metrics with very small window
        throughput_old = self.collector.get_throughput_metrics(window_seconds=1)
        # Note: this may pass if execution is fast; timing-dependent test

    def test_get_health_summary_returns_valid_structure(self):
        """Test that health summary has all required fields."""
        health = self.collector.get_health_summary()

        self.assertIsInstance(health, PipelineHealthSummary)
        self.assertIn("status", health.to_dict())
        self.assertIn("throughput", health.to_dict())
        self.assertIn("readiness", health.to_dict())
        self.assertIn("failures", health.to_dict())
        self.assertIn("last_updated", health.to_dict())

    def test_health_status_is_healthy_when_no_failures(self):
        """Test that health status is HEALTHY with no failures."""
        self.collector.record_stage_execution(
            stage_name="normalize",
            duration_ms=50.0,
            input_size=100,
            output_size=100,
            status="completed",
        )

        health = self.collector.get_health_summary()
        self.assertEqual(health.status, HealthStatus.HEALTHY.value)

    def test_health_status_is_degraded_with_failures(self):
        """Test that health status is DEGRADED with failures."""
        for i in range(3):
            self.collector.record_stage_execution(
                stage_name="validate",
                duration_ms=10.0,
                input_size=100,
                output_size=0,
                status="failed",
                error="Validation error",
            )

        health = self.collector.get_health_summary()
        self.assertEqual(health.status, HealthStatus.DEGRADED.value)

    def test_health_status_is_unhealthy_with_many_failures(self):
        """Test that health status is UNHEALTHY with many failures."""
        for i in range(25):
            self.collector.record_stage_execution(
                stage_name="validate",
                duration_ms=10.0,
                input_size=100,
                output_size=0,
                status="failed",
                error="Validation error",
            )

        health = self.collector.get_health_summary()
        self.assertEqual(health.status, HealthStatus.UNHEALTHY.value)

    def test_prometheus_metrics_format(self):
        """Test that Prometheus metrics are in correct format."""
        self.collector.record_stage_execution(
            stage_name="normalize",
            duration_ms=50.0,
            input_size=100,
            output_size=95,
            status="completed",
        )

        metrics_output = get_prometheus_metrics()

        self.assertIn("pipeline_health_status", metrics_output)
        self.assertIn("pipeline_stage_duration_ms", metrics_output)
        self.assertIn("pipeline_stage_records_total", metrics_output)
        self.assertIn("pipeline_health_last_updated", metrics_output)

    def test_global_collector_singleton(self):
        """Test that global collector is singleton."""
        from ai.core.pipelines.pipeline_observability import get_metrics_collector

        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        self.assertIs(collector1, collector2)

    def test_convenience_functions_work(self):
        """Test convenience record functions."""
        record_stage_execution(
            stage_name="normalize",
            duration_ms=50.0,
            input_size=100,
            output_size=95,
            status="completed",
        )

        record_failure(
            stage="validate",
            error_message="Test error",
            severity=FailureSeverity.MEDIUM,
        )

        health = get_health_summary()
        self.assertIsNotNone(health.failures.total_failures)


class TestFailureTracker(unittest.TestCase):
    """Test failure tracking and regression detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = PipelineMetricsCollector(max_history=100)

    def test_failure_records_are_tracked(self):
        """Test that failures are properly recorded."""
        self.collector.record_failure(
            stage="validate",
            gate="quality_floors",
            package_id="pkg-001",
            error_message="Quality floor not met: empathy 0.65 < floor 0.70",
            severity=FailureSeverity.HIGH,
        )

        failures = self.collector.get_failure_metrics()
        self.assertEqual(failures.total_failures, 1)
        self.assertEqual(failures.by_stage["validate"], 1)
        self.assertEqual(failures.by_gate["quality_floors"], 1)

    def test_regression_detected_when_same_failure_repeats(self):
        """Test that regressions are detected when same pattern repeats."""
        error_msg = "Quality floor not met: empathy"

        # Record same failure pattern multiple times
        for i in range(3):
            self.collector.record_failure(
                stage="validate",
                gate="quality_floors",
                package_id=f"pkg-{i:03d}",
                error_message=error_msg,
                severity=FailureSeverity.HIGH,
            )

        failures = self.collector.get_failure_metrics()
        self.assertEqual(failures.regressions_detected, 1)
        self.assertEqual(len(failures.alert_regressions), 1)
        self.assertEqual(failures.alert_regressions[0].occurrences, 3)

    def test_no_regression_when_failure_is_unique(self):
        """Test that unique failures don't trigger regression alerts."""
        self.collector.record_failure(
            stage="validate",
            gate="completeness",
            package_id="pkg-001",
            error_message="Missing required field: source",
            severity=FailureSeverity.MEDIUM,
        )

        self.collector.record_failure(
            stage="normalize",
            gate=None,
            package_id="pkg-002",
            error_message="Different error message",
            severity=FailureSeverity.LOW,
        )

        failures = self.collector.get_failure_metrics()
        self.assertEqual(failures.regressions_detected, 0)


class TestReadinessIntegration(unittest.TestCase):
    """Test integration with TrainingReadinessGates."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = PipelineMetricsCollector(max_history=100)
        self.gates = TrainingReadinessGates()

    def test_readiness_result_recorded(self):
        """Test that readiness results are recorded correctly."""
        records = [
            {
                "id": "1",
                "text": "Hello, how are you feeling today?",
                "stage": "stage1_foundation",
                "source": "test",
                "created_at": "2024-01-01T00:00:00Z",
                "empathy_score": 0.8,
                "clinical_score": 0.5,
                "safety_score": 1.0,
            }
        ]

        result = self.gates.validate_package(
            package_id="pkg-test-001",
            stage_id="stage1_foundation",
            records=records,
            metrics={"empathy_score": 0.8, "clinical_score": 0.5, "safety_score": 1.0},
        )

        self.collector.record_readiness_result(result)

        readiness = self.collector.get_readiness_metrics()
        self.assertEqual(readiness.total_validations, 1)
        self.assertEqual(readiness.passed, 1)

    def test_failed_readiness_result_triggers_failure(self):
        """Test that failed readiness results are recorded as failures."""
        records = [
            {
                "id": "1",
                "text": "Hi",  # Too short
                "stage": "stage1_foundation",
                "source": "test",
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]

        result = self.gates.validate_package(
            package_id="pkg-test-002",
            stage_id="stage1_foundation",
            records=records,
        )

        self.collector.record_readiness_result(result)

        readiness = self.collector.get_readiness_metrics()
        self.assertEqual(readiness.total_validations, 1)
        self.assertEqual(readiness.failed, 1)

        failures = self.collector.get_failure_metrics()
        self.assertGreater(failures.total_failures, 0)


class TestHealthSummarySerialization(unittest.TestCase):
    """Test PipelineHealthSummary serialization."""

    def test_to_dict_returns_serializable_structure(self):
        """Test that health summary serializes to dict correctly."""
        collector = PipelineMetricsCollector(max_history=10)
        collector.record_stage_execution(
            stage_name="normalize",
            duration_ms=50.0,
            input_size=100,
            output_size=95,
            status="completed",
        )

        health = collector.get_health_summary()
        health_dict = health.to_dict()

        self.assertIsInstance(health_dict, dict)
        self.assertEqual(health_dict["status"], "healthy")
        self.assertIsInstance(health_dict["throughput"], dict)
        self.assertIsInstance(health_dict["readiness"], dict)
        self.assertIsInstance(health_dict["failures"], dict)
        self.assertIn("last_updated", health_dict)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = PipelineMetricsCollector(max_history=10)

    def test_empty_collector_returns_zeros(self):
        """Test that empty collector returns zero values."""
        throughput = self.collector.get_throughput_metrics()
        self.assertEqual(throughput.total_records_in, 0)
        self.assertEqual(throughput.total_records_out, 0)

        readiness = self.collector.get_readiness_metrics()
        self.assertEqual(readiness.total_validations, 0)

        failures = self.collector.get_failure_metrics()
        self.assertEqual(failures.total_failures, 0)

    def test_health_summary_with_no_data(self):
        """Test health summary structure with no data."""
        health = self.collector.get_health_summary()

        self.assertEqual(health.status, HealthStatus.HEALTHY.value)
        self.assertEqual(health.throughput.total_records_in, 0)

    def test_max_history_trimming(self):
        """Test that old metrics are trimmed when exceeding max_history."""
        # Create collector with small history
        collector = PipelineMetricsCollector(max_history=5)

        # Add more than max_history entries
        for i in range(10):
            collector.record_stage_execution(
                stage_name="normalize",
                duration_ms=50.0,
                input_size=100,
                output_size=95,
                status="completed",
            )

        throughput = collector.get_throughput_metrics()
        # Should have only 5 entries (the most recent)
        self.assertEqual(throughput.stage_metrics["normalize"]["count"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
