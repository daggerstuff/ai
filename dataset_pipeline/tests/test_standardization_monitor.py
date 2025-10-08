"""
Tests for StandardizationMonitor.
"""

import time
from datetime import datetime

import pytest

from ai.dataset_pipeline.conversation_schema import Conversation, Message
from ai.dataset_pipeline.standardization_monitor import (
    AlertLevel,
    QualityAlert,
    QualityMetric,
    StandardizationMetrics,
    StandardizationMonitor,
)


class TestStandardizationMonitor:
    """Test cases for StandardizationMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create a StandardizationMonitor instance for testing."""
        return StandardizationMonitor(
            window_size=100,
            alert_cooldown=10,
            enable_real_time=False  # Disable for testing
        )

    @pytest.fixture
    def sample_conversation(self):
        """Create a sample conversation for testing."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!")
        ]
        return Conversation(id="test_conv", messages=messages)

    def test_initialization(self):
        """Test StandardizationMonitor initialization."""
        monitor = StandardizationMonitor(window_size=50, alert_cooldown=30)

        assert monitor.window_size == 50
        assert monitor.alert_cooldown == 30
        assert monitor.enable_real_time is True
        assert len(monitor.quality_assessors) == 0
        assert len(monitor.alerts) == 0
        assert isinstance(monitor.current_metrics, StandardizationMetrics)

    def test_register_quality_assessor(self, monitor):
        """Test registering quality assessor."""
        def dummy_assessor(conversation):
            return 0.8

        monitor.register_quality_assessor(dummy_assessor)

        assert len(monitor.quality_assessors) == 1
        assert monitor.quality_assessors[0] == dummy_assessor

    def test_set_threshold(self, monitor):
        """Test setting alert thresholds."""
        monitor.set_threshold("test_metric", 0.5)

        assert monitor.quality_thresholds["test_metric"] == 0.5

    def test_record_processing_start(self, monitor):
        """Test recording processing start."""
        monitor.record_processing_start("item_1")

        assert hasattr(monitor, "_processing_starts")
        assert "item_1" in monitor._processing_starts
        assert isinstance(monitor._processing_starts["item_1"], float)

    def test_record_processing_success(self, monitor, sample_conversation):
        """Test recording successful processing."""
        # Start processing
        monitor.record_processing_start("item_1")
        time.sleep(0.01)  # Small delay to ensure processing time > 0

        # Record success
        monitor.record_processing_success(
            "item_1",
            sample_conversation,
            "simple_messages",
            {"test": "metadata"}
        )

        metrics = monitor.get_current_metrics()
        assert metrics.total_processed == 1
        assert metrics.successful == 1
        assert metrics.failed == 0
        assert "simple_messages" in metrics.format_distribution
        assert metrics.format_distribution["simple_messages"] == 1
        assert len(monitor.processing_times) == 1
        assert monitor.processing_times[0] > 0

    def test_record_processing_failure(self, monitor):
        """Test recording processing failure."""
        # Start processing
        monitor.record_processing_start("item_1")
        time.sleep(0.01)

        # Record failure
        monitor.record_processing_failure(
            "item_1",
            "Test error",
            "unknown_format",
            {"test": "metadata"}
        )

        metrics = monitor.get_current_metrics()
        assert metrics.total_processed == 1
        assert metrics.successful == 0
        assert metrics.failed == 1
        assert "unknown_format" in metrics.format_distribution
        assert "str" in metrics.error_distribution  # Error type

    def test_record_processing_success_with_quality_assessor(self, monitor, sample_conversation):
        """Test recording success with quality assessor."""
        def quality_assessor(conversation):
            return 0.9

        monitor.register_quality_assessor(quality_assessor)
        monitor.record_processing_start("item_1")

        monitor.record_processing_success(
            "item_1",
            sample_conversation,
            "simple_messages"
        )

        assert len(monitor.quality_scores) == 1
        assert monitor.quality_scores[0] == 0.9

        metrics = monitor.get_current_metrics()
        assert "average_quality" in metrics.quality_scores
        assert metrics.quality_scores["average_quality"] == 0.9

    def test_quality_assessor_error_handling(self, monitor, sample_conversation):
        """Test quality assessor error handling."""
        def failing_assessor(conversation):
            raise ValueError("Test error")

        def working_assessor(conversation):
            return 0.8

        monitor.register_quality_assessor(failing_assessor)
        monitor.register_quality_assessor(working_assessor)

        monitor.record_processing_start("item_1")
        monitor.record_processing_success("item_1", sample_conversation, "test")

        # Should use only the working assessor
        assert len(monitor.quality_scores) == 1
        assert monitor.quality_scores[0] == 0.8

    def test_get_current_metrics(self, monitor, sample_conversation):
        """Test getting current metrics."""
        # Process some items
        for i in range(5):
            monitor.record_processing_start(f"item_{i}")
            if i < 4:  # 4 successes, 1 failure
                monitor.record_processing_success(f"item_{i}", sample_conversation, "test")
            else:
                monitor.record_processing_failure(f"item_{i}", "Test error", "test")

        metrics = monitor.get_current_metrics()

        assert metrics.total_processed == 5
        assert metrics.successful == 4
        assert metrics.failed == 1
        assert metrics.quality_scores["success_rate"] == 0.8
        assert metrics.quality_scores["error_rate"] == 0.2
        assert metrics.processing_rate > 0
        assert isinstance(metrics.timestamp, datetime)

    def test_get_quality_trends(self, monitor, sample_conversation):
        """Test getting quality trends."""
        # Record some metrics
        monitor.record_processing_success("item_1", sample_conversation, "test")
        monitor.record_processing_failure("item_2", "Error", "test")

        # Get trends for processing success
        trends = monitor.get_quality_trends("processing_success", duration_minutes=60)

        assert len(trends) == 1
        assert trends[0].name == "processing_success"
        assert trends[0].value == 1.0
        assert isinstance(trends[0].timestamp, datetime)

    def test_get_quality_trends_time_filter(self, monitor, sample_conversation):
        """Test quality trends with time filtering."""
        # Record a metric
        monitor.record_processing_success("item_1", sample_conversation, "test")

        # Get trends for very short duration (should be empty)
        trends = monitor.get_quality_trends("processing_success", duration_minutes=0)
        assert len(trends) == 0

        # Get trends for long duration (should include the metric)
        trends = monitor.get_quality_trends("processing_success", duration_minutes=60)
        assert len(trends) == 1

    def test_get_alerts(self, monitor):
        """Test getting alerts."""
        # Create some test alerts
        alert1 = QualityAlert(
            level=AlertLevel.WARNING,
            message="Test warning",
            metric_name="test_metric",
            current_value=0.5,
            threshold=0.8,
            timestamp=datetime.now()
        )
        alert2 = QualityAlert(
            level=AlertLevel.ERROR,
            message="Test error",
            metric_name="test_metric",
            current_value=0.3,
            threshold=0.8,
            timestamp=datetime.now()
        )

        monitor.alerts.extend([alert1, alert2])

        # Get all alerts
        all_alerts = monitor.get_alerts()
        assert len(all_alerts) == 2

        # Get only warnings
        warning_alerts = monitor.get_alerts(level=AlertLevel.WARNING)
        assert len(warning_alerts) == 1
        assert warning_alerts[0].level == AlertLevel.WARNING

        # Get with limit
        limited_alerts = monitor.get_alerts(limit=1)
        assert len(limited_alerts) == 1

    def test_reset_metrics(self, monitor, sample_conversation):
        """Test resetting metrics."""
        # Generate some data
        monitor.record_processing_success("item_1", sample_conversation, "test")
        monitor.alerts.append(QualityAlert(
            level=AlertLevel.INFO,
            message="Test",
            metric_name="test",
            current_value=1.0,
            threshold=0.8,
            timestamp=datetime.now()
        ))

        # Verify data exists
        assert monitor.current_metrics.total_processed == 1
        assert len(monitor.alerts) == 1

        # Reset
        monitor.reset_metrics()

        # Verify reset
        assert monitor.current_metrics.total_processed == 0
        assert len(monitor.alerts) == 0
        assert len(monitor.metrics_history) == 0
        assert len(monitor.processing_times) == 0
        assert len(monitor.quality_scores) == 0

    def test_alert_creation_success_rate(self, monitor, sample_conversation):
        """Test alert creation for low success rate."""
        # Set low threshold
        monitor.set_threshold("min_success_rate", 0.9)

        # Process items with low success rate
        for i in range(10):
            monitor.record_processing_start(f"item_{i}")
            if i < 7:  # 70% success rate
                monitor.record_processing_success(f"item_{i}", sample_conversation, "test")
            else:
                monitor.record_processing_failure(f"item_{i}", "Error", "test")

        # Should create alert
        alerts = monitor.get_alerts(level=AlertLevel.WARNING)
        success_rate_alerts = [a for a in alerts if a.metric_name == "success_rate"]
        assert len(success_rate_alerts) > 0
        assert success_rate_alerts[0].current_value == 0.7

    def test_alert_creation_quality_score(self, monitor, sample_conversation):
        """Test alert creation for low quality score."""
        # Register low-scoring assessor
        def low_quality_assessor(conversation):
            return 0.5

        monitor.register_quality_assessor(low_quality_assessor)
        monitor.set_threshold("min_quality_score", 0.8)

        # Process items
        for i in range(5):
            monitor.record_processing_start(f"item_{i}")
            monitor.record_processing_success(f"item_{i}", sample_conversation, "test")

        # Should create alert
        alerts = monitor.get_alerts(level=AlertLevel.ERROR)
        quality_alerts = [a for a in alerts if a.metric_name == "quality_score"]
        assert len(quality_alerts) > 0

    def test_alert_cooldown(self, monitor, sample_conversation):
        """Test alert cooldown mechanism."""
        monitor.set_threshold("min_success_rate", 0.9)
        monitor.alert_cooldown = 1  # 1 second cooldown

        # Trigger alert condition twice quickly
        for _ in range(2):
            for i in range(5):
                monitor.record_processing_start(f"item_{i}")
                if i < 2:  # Low success rate
                    monitor.record_processing_success(f"item_{i}", sample_conversation, "test")
                else:
                    monitor.record_processing_failure(f"item_{i}", "Error", "test")

        # Should only have one alert due to cooldown
        alerts = monitor.get_alerts()
        success_rate_alerts = [a for a in alerts if a.metric_name == "success_rate"]
        assert len(success_rate_alerts) == 1

    def test_monitoring_thread_start_stop(self):
        """Test starting and stopping monitoring thread."""
        monitor = StandardizationMonitor(enable_real_time=False)

        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring_thread is not None
        assert monitor.monitoring_thread.is_alive()

        # Stop monitoring
        monitor.stop_monitoring_thread()
        assert not monitor.monitoring_thread.is_alive()

    def test_processing_time_calculation(self, monitor, sample_conversation):
        """Test processing time calculation."""
        monitor.record_processing_start("item_1")
        time.sleep(0.05)  # 50ms delay
        monitor.record_processing_success("item_1", sample_conversation, "test")

        assert len(monitor.processing_times) == 1
        assert monitor.processing_times[0] >= 0.04  # Should be at least 40ms

        metrics = monitor.get_current_metrics()
        assert metrics.average_processing_time >= 0.04


class TestQualityMetric:
    """Test cases for QualityMetric."""

    def test_initialization(self):
        """Test QualityMetric initialization."""
        timestamp = datetime.now()
        metric = QualityMetric(
            name="test_metric",
            value=0.8,
            timestamp=timestamp,
            metadata={"test": "data"}
        )

        assert metric.name == "test_metric"
        assert metric.value == 0.8
        assert metric.timestamp == timestamp
        assert metric.metadata == {"test": "data"}


class TestQualityAlert:
    """Test cases for QualityAlert."""

    def test_initialization(self):
        """Test QualityAlert initialization."""
        timestamp = datetime.now()
        alert = QualityAlert(
            level=AlertLevel.WARNING,
            message="Test alert",
            metric_name="test_metric",
            current_value=0.5,
            threshold=0.8,
            timestamp=timestamp,
            metadata={"test": "data"}
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test alert"
        assert alert.metric_name == "test_metric"
        assert alert.current_value == 0.5
        assert alert.threshold == 0.8
        assert alert.timestamp == timestamp
        assert alert.metadata == {"test": "data"}


class TestStandardizationMetrics:
    """Test cases for StandardizationMetrics."""

    def test_initialization(self):
        """Test StandardizationMetrics initialization."""
        metrics = StandardizationMetrics()

        assert metrics.total_processed == 0
        assert metrics.successful == 0
        assert metrics.failed == 0
        assert metrics.processing_rate == 0.0
        assert metrics.average_processing_time == 0.0
        assert metrics.quality_scores == {}
        assert metrics.format_distribution == {}
        assert metrics.error_distribution == {}
        assert isinstance(metrics.timestamp, datetime)
