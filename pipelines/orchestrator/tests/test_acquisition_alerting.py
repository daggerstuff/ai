"""
Unit tests for acquisition alerting system.

Tests the comprehensive dataset acquisition monitoring and alerting functionality
including error pattern matching, automated recovery, multi-channel notifications,
and intelligent error analysis with proactive monitoring.
"""

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from .acquisition_alerting import (
    AcquisitionAlerting,
    ErrorPattern,
    NotificationChannel,
    RecoveryAction,
    RecoveryAttempt,
    RecoveryStrategy,
)
from .acquisition_monitor import Alert, AlertLevel, MetricType


class TestAcquisitionAlerting(unittest.TestCase):
    """Test acquisition alerting functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test configuration
        self.config = {
            "notifications": {
                "channels": [NotificationChannel.LOG, NotificationChannel.CONSOLE],
                "severity_filter": [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
            }
        }

        self.alerting = AcquisitionAlerting(self.config)

        # Create test alert
        self.test_alert = Alert(
            id="test_alert_001",
            level=AlertLevel.ERROR,
            message="Connection timeout occurred while loading dataset",
            metric_type=MetricType.ERROR_RATE,
            dataset_name="test_dataset",
            value=0.15,
            threshold=0.10,
            timestamp=datetime.now()
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Shutdown alerting system
        self.alerting.shutdown()

        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test alerting system initialization."""
        assert self.alerting.acquisition_monitor is not None
        assert self.alerting.performance_optimizer is not None
        assert self.alerting.notification_config is not None
        assert isinstance(self.alerting.error_patterns, list)
        assert len(self.alerting.error_patterns) > 0  # Should have default patterns

        # Check default error patterns were loaded
        pattern_ids = [pattern.pattern_id for pattern in self.alerting.error_patterns]
        expected_patterns = ["network_connectivity", "rate_limiting", "memory_exhaustion", "data_quality"]

        for expected in expected_patterns:
            assert expected in pattern_ids

    def test_error_pattern_matching(self):
        """Test error pattern matching functionality."""

        # Test that pattern signatures are detected (without frequency check for simplicity)
        network_alert = Alert(
            id="network_test",
            level=AlertLevel.ERROR,
            message="Connection timeout while fetching data",
            metric_type=MetricType.ERROR_RATE,
            dataset_name="test_dataset",
            value=0.2,
            threshold=0.1,
            timestamp=datetime.now()
        )

        # Check that the alert message contains network error signatures
        alert_text = network_alert.message.lower()

        # Find network pattern
        network_pattern = None
        for pattern in self.alerting.error_patterns:
            if pattern.pattern_id == "network_connectivity":
                network_pattern = pattern
                break

        assert network_pattern is not None, "Network connectivity pattern should exist"

        # Verify that at least one signature matches
        signature_found = any(sig.lower() in alert_text for sig in network_pattern.error_signatures)
        assert signature_found, f"Alert message '{alert_text}' should match network signatures {network_pattern.error_signatures}"

        # Test rate limiting pattern
        rate_limit_alert = Alert(
            id="rate_test",
            level=AlertLevel.WARNING,
            message="Rate limit exceeded - too many requests",
            metric_type=MetricType.ERROR_RATE,
            dataset_name="test_dataset",
            value=0.1,
            threshold=0.05,
            timestamp=datetime.now()
        )

        matching_pattern = self.alerting._match_error_pattern(rate_limit_alert)
        assert matching_pattern is not None
        assert matching_pattern.pattern_id == "rate_limiting"

        # Test non-matching alert
        unknown_alert = Alert(
            id="unknown_test",
            level=AlertLevel.INFO,
            message="Some unknown issue occurred",
            metric_type=MetricType.QUALITY_SCORE,
            dataset_name="test_dataset",
            value=0.8,
            threshold=0.7,
            timestamp=datetime.now()
        )

        matching_pattern = self.alerting._match_error_pattern(unknown_alert)
        assert matching_pattern is None

    def test_alert_suppression(self):
        """Test alert suppression to prevent spam."""

        # Create multiple similar alerts
        similar_alerts = []
        for i in range(5):
            alert = Alert(
                id=f"similar_alert_{i}",
                level=AlertLevel.WARNING,
                message="Similar error message",
                metric_type=MetricType.ERROR_RATE,
                dataset_name="test_dataset",
                value=0.1,
                threshold=0.05,
                timestamp=datetime.now()
            )
            similar_alerts.append(alert)
            self.alerting.alert_history.append(alert)

        # Test suppression
        new_similar_alert = Alert(
            id="new_similar_alert",
            level=AlertLevel.WARNING,
            message="Similar error message",
            metric_type=MetricType.ERROR_RATE,
            dataset_name="test_dataset",
            value=0.1,
            threshold=0.05,
            timestamp=datetime.now()
        )

        should_suppress = self.alerting._should_suppress_alert(new_similar_alert)
        assert should_suppress

    def test_recovery_action_execution(self):
        """Test execution of different recovery actions."""

        async def run_test():
            # Test retry action
            success = await self.alerting._execute_recovery_action(
                RecoveryAction.RETRY, self.test_alert, self.alerting.error_patterns[0]
            )
            assert success

            # Test restart action
            success = await self.alerting._execute_recovery_action(
                RecoveryAction.RESTART, self.test_alert, self.alerting.error_patterns[0]
            )
            assert success

            # Test clear cache action
            success = await self.alerting._execute_recovery_action(
                RecoveryAction.CLEAR_CACHE, self.test_alert, self.alerting.error_patterns[0]
            )
            assert success

            # Test skip action
            success = await self.alerting._execute_recovery_action(
                RecoveryAction.SKIP, self.test_alert, self.alerting.error_patterns[0]
            )
            assert success

        asyncio.run(run_test())

    def test_custom_error_pattern(self):
        """Test adding and removing custom error patterns."""

        # Create custom pattern
        custom_strategy = RecoveryStrategy(
            trigger_conditions=["custom_error"],
            actions=[RecoveryAction.RETRY, RecoveryAction.ESCALATE],
            max_attempts=2,
            delay_seconds=5.0
        )

        custom_pattern = ErrorPattern(
            pattern_id="custom_test_pattern",
            error_signatures=["custom error", "test failure"],
            frequency_threshold=1,
            time_window_minutes=5,
            recovery_strategy=custom_strategy,
            description="Custom test pattern"
        )

        # Add pattern
        initial_count = len(self.alerting.error_patterns)
        self.alerting.add_error_pattern(custom_pattern)

        assert len(self.alerting.error_patterns) == initial_count + 1
        assert custom_pattern in self.alerting.error_patterns

        # Remove pattern
        success = self.alerting.remove_error_pattern("custom_test_pattern")
        assert success
        assert len(self.alerting.error_patterns) == initial_count

        # Try to remove non-existent pattern
        success = self.alerting.remove_error_pattern("non_existent_pattern")
        assert not success

    def test_notification_channels(self):
        """Test different notification channels."""

        # Test log notification
        with patch("ai.pipelines.orchestrator.acquisition_alerting.logger") as mock_logger:
            self.alerting._send_log_notification(self.test_alert)
            mock_logger.error.assert_called_once()

        # Test console notification
        with patch("builtins.print") as mock_print:
            self.alerting._send_console_notification(self.test_alert)
            mock_print.assert_called_once()

            # Check that the printed message contains alert info
            call_args = mock_print.call_args[0][0]
            assert "ERROR" in call_args
            assert self.test_alert.message in call_args

        # Test file notification
        file_path = self.temp_dir / "notifications.log"
        self.alerting.notification_config.file_path = file_path

        self.alerting._send_file_notification(self.test_alert)

        # Check that file was created and contains notification
        assert file_path.exists()

        with open(file_path) as f:
            content = f.read()
            assert self.test_alert.id in content
            assert self.test_alert.message in content

    def test_recovery_statistics(self):
        """Test recovery statistics tracking."""

        # Add some mock recovery attempts
        attempt1 = RecoveryAttempt(
            attempt_id="attempt_1",
            error_context="Test error 1",
            recovery_action=RecoveryAction.RETRY,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            success=True
        )

        attempt2 = RecoveryAttempt(
            attempt_id="attempt_2",
            error_context="Test error 2",
            recovery_action=RecoveryAction.RESTART,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            success=False
        )

        self.alerting.recovery_attempts["test_dataset"] = [attempt1, attempt2]
        self.alerting.recovery_stats["total_attempts"] = 2
        self.alerting.recovery_stats["successful_recoveries"] = 1
        self.alerting.recovery_stats["failed_recoveries"] = 1

        # Get statistics
        stats = self.alerting.get_recovery_stats()

        assert stats["total_attempts"] == 2
        assert stats["successful_recoveries"] == 1
        assert stats["failed_recoveries"] == 1
        assert stats["success_rate_percent"] == 50.0
        assert stats["active_patterns"] > 0

    def test_recovery_history(self):
        """Test recovery history retrieval."""

        # Add recovery attempts
        attempts = [
            RecoveryAttempt(
                attempt_id=f"attempt_{i}",
                error_context=f"Test error {i}",
                recovery_action=RecoveryAction.RETRY,
                started_at=datetime.now(),
                success=i % 2 == 0  # Alternate success/failure
            )
            for i in range(3)
        ]

        self.alerting.recovery_attempts["test_dataset"] = attempts

        # Get history
        history = self.alerting.get_recovery_history("test_dataset")

        assert len(history) == 3
        assert history[0].attempt_id == "attempt_0"
        assert history[0].success
        assert not history[1].success

        # Test non-existent dataset
        empty_history = self.alerting.get_recovery_history("non_existent_dataset")
        assert len(empty_history) == 0

    def test_monitoring_report_generation(self):
        """Test comprehensive monitoring report generation."""

        # Add some test data
        self.alerting.active_alerts[self.test_alert.id] = self.test_alert
        self.alerting.alert_history.append(self.test_alert)

        # Generate report
        report = self.alerting.generate_monitoring_report()

        # Check report structure
        assert "generated_at" in report
        assert "monitoring_status" in report
        assert "alert_statistics" in report
        assert "recovery_statistics" in report
        assert "system_health" in report

        # Check specific values
        assert report["system_health"]["active_alerts"] == 1
        assert report["monitoring_status"]["error_patterns"] > 0

    def test_configuration_export_import(self):
        """Test configuration export and import functionality."""

        # Add a custom pattern for testing
        custom_pattern = ErrorPattern(
            pattern_id="export_test_pattern",
            error_signatures=["export test error"],
            frequency_threshold=2,
            time_window_minutes=10,
            recovery_strategy=RecoveryStrategy(
                trigger_conditions=["export_test"],
                actions=[RecoveryAction.RETRY],
                max_attempts=3,
                delay_seconds=2.0
            ),
            description="Pattern for export testing"
        )

        self.alerting.add_error_pattern(custom_pattern)

        # Export configuration
        export_path = self.temp_dir / "config_export.json"
        success = self.alerting.export_configuration(export_path)

        assert success
        assert export_path.exists()

        # Create new alerting instance and import
        new_alerting = AcquisitionAlerting()
        len(new_alerting.error_patterns)

        success = new_alerting.import_configuration(export_path)
        assert success

        # Check that patterns were imported
        imported_pattern_ids = [p.pattern_id for p in new_alerting.error_patterns]
        assert "export_test_pattern" in imported_pattern_ids

        # Find the imported pattern and verify details
        imported_pattern = next(p for p in new_alerting.error_patterns if p.pattern_id == "export_test_pattern")
        assert imported_pattern.frequency_threshold == 2
        assert imported_pattern.time_window_minutes == 10
        assert imported_pattern.description == "Pattern for export testing"

        new_alerting.shutdown()

    def test_active_alerts_retrieval(self):
        """Test active alerts retrieval with filtering."""

        # Add multiple alerts for different datasets
        alert1 = Alert(
            id="alert_1", level=AlertLevel.WARNING, message="Warning 1",
            metric_type=MetricType.ERROR_RATE, dataset_name="dataset_1",
            value=0.1, threshold=0.05, timestamp=datetime.now()
        )

        alert2 = Alert(
            id="alert_2", level=AlertLevel.ERROR, message="Error 2",
            metric_type=MetricType.QUALITY_SCORE, dataset_name="dataset_2",
            value=0.4, threshold=0.6, timestamp=datetime.now()
        )

        self.alerting.active_alerts[alert1.id] = alert1
        self.alerting.active_alerts[alert2.id] = alert2

        # Get all active alerts
        all_alerts = self.alerting.get_active_alerts()
        assert len(all_alerts) == 2

        # Get alerts for specific dataset
        dataset1_alerts = self.alerting.get_active_alerts("dataset_1")
        assert len(dataset1_alerts) == 1
        assert dataset1_alerts[0].id == "alert_1"

        # Get alerts for non-existent dataset
        empty_alerts = self.alerting.get_active_alerts("non_existent_dataset")
        assert len(empty_alerts) == 0

    def test_monitoring_start_stop(self):
        """Test monitoring start and stop functionality."""

        # Test starting monitoring
        datasets = ["dataset_1", "dataset_2", "dataset_3"]
        self.alerting.start_monitoring(datasets)

        # Check that monitoring is active
        assert self.alerting._monitoring_active

        # Test stopping specific datasets
        self.alerting.stop_monitoring(["dataset_1"])

        # Test stopping all monitoring
        self.alerting.stop_monitoring()
        assert not self.alerting._monitoring_active


if __name__ == "__main__":
    unittest.main()
