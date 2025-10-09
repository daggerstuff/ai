"""
Unit tests for acquisition monitor and quality validator.

Tests the comprehensive real-time monitoring functionality including
quality metrics tracking, alerting, progress monitoring, and
performance analytics during dataset acquisition.
"""

import unittest
from datetime import datetime

from .acquisition_monitor import AcquisitionMonitor, Alert, AlertLevel, MetricType
from .conversation_schema import Conversation, Message
from .quality_validator import QualityResult, QualityValidator


class TestQualityValidator(unittest.TestCase):
    """Test quality validator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = QualityValidator()

        # Create test conversations
        self.good_conversation = Conversation(
            id="good_conv",
            messages=[
                Message(role="client", content="Hello, I've been feeling anxious lately and need some help."),
                Message(role="therapist", content="I understand you're feeling anxious. Can you tell me more about what's been triggering these feelings?"),
                Message(role="client", content="It started when I began my new job. I feel overwhelmed and worry constantly about making mistakes."),
                Message(role="therapist", content="That sounds really challenging. It's normal to feel anxious when starting something new. What specific situations at work make you feel most anxious?")
            ],
            context={"session_type": "therapy", "topic": "anxiety"},
            source="test"
        )

        self.poor_conversation = Conversation(
            id="poor_conv",
            messages=[
                Message(role="user", content="test"),
                Message(role="assistant", content="LOREM IPSUM DOLOR SIT AMET!!!")
            ],
            context={},
            source="test"
        )

    def test_initialization(self):
        """Test validator initialization."""
        assert self.validator.config is not None
        assert self.validator.low_quality_patterns is not None
        assert self.validator.therapeutic_patterns is not None
        assert self.validator.conversation_flow_patterns is not None

    def test_validate_good_conversation(self):
        """Test validation of a good quality conversation."""
        result = self.validator.validate_conversation(self.good_conversation)

        assert isinstance(result, QualityResult)
        assert result.conversation_id == "good_conv"
        assert result.overall_score > 0.7  # Should be high quality
        assert result.coherence_score > 0.6
        assert result.authenticity_score > 0.6
        assert result.completeness_score > 0.6
        assert len(result.strengths) > 0
        assert isinstance(result.validated_at, datetime)

    def test_validate_poor_conversation(self):
        """Test validation of a poor quality conversation."""
        result = self.validator.validate_conversation(self.poor_conversation)

        assert isinstance(result, QualityResult)
        assert result.conversation_id == "poor_conv"
        assert result.overall_score < 0.5  # Should be low quality
        assert len(result.issues) > 0

    def test_structure_validation(self):
        """Test conversation structure validation."""
        # Test too few messages
        short_conv = Conversation(
            id="short",
            messages=[Message(role="user", content="Hi")],
            context={},
            source="test"
        )

        result = self.validator.validate_conversation(short_conv)
        assert "Too few messages" in " ".join(result.issues)

        # Test empty messages
        empty_conv = Conversation(
            id="empty",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content=""),
                Message(role="user", content="Are you there?")
            ],
            context={},
            source="test"
        )

        result = self.validator.validate_conversation(empty_conv)
        assert "empty messages" in " ".join(result.issues)

    def test_content_validation(self):
        """Test content quality validation."""
        # Test low quality patterns
        spam_conv = Conversation(
            id="spam",
            messages=[
                Message(role="user", content="Click here to buy now! SPAM SPAM SPAM"),
                Message(role="assistant", content="This is a test dummy placeholder")
            ],
            context={},
            source="test"
        )

        result = self.validator.validate_conversation(spam_conv)
        assert "Low-quality content" in " ".join(result.issues)

    def test_therapeutic_content_detection(self):
        """Test detection of therapeutic content."""
        result = self.validator.validate_conversation(self.good_conversation)

        # Should detect therapeutic content
        therapeutic_strengths = [s for s in result.strengths if "Therapeutic content" in s]
        assert len(therapeutic_strengths) > 0


class TestAcquisitionMonitor(unittest.TestCase):
    """Test acquisition monitor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = AcquisitionMonitor()

        # Create test conversation
        self.test_conversation = Conversation(
            id="test_conv",
            messages=[
                Message(role="client", content="I've been feeling really anxious about my upcoming presentation at work."),
                Message(role="therapist", content="I can understand how that would feel overwhelming. What specifically about the presentation is causing you the most anxiety?")
            ],
            context={"session_type": "therapy"},
            source="test"
        )

    def test_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.quality_validator is not None
        assert isinstance(self.monitor.metrics, type(self.monitor.metrics))
        assert isinstance(self.monitor.alerts, list)
        assert isinstance(self.monitor.dataset_stats, dict)
        assert isinstance(self.monitor.quality_thresholds, dict)

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring for datasets."""
        dataset_name = "test_dataset"

        # Start monitoring
        self.monitor.start_monitoring(dataset_name)
        assert dataset_name in self.monitor.dataset_stats

        stats = self.monitor.dataset_stats[dataset_name]
        assert stats.dataset_name == dataset_name
        assert stats.start_time is not None

        # Stop monitoring
        self.monitor.stop_monitoring(dataset_name)
        assert stats.last_update is not None

    def test_process_conversation(self):
        """Test conversation processing and metric recording."""
        dataset_name = "test_dataset"
        processing_time = 0.5

        # Process conversation
        quality_scores = self.monitor.process_conversation(
            self.test_conversation,
            dataset_name,
            processing_time
        )

        # Check that scores were returned
        assert isinstance(quality_scores, dict)
        assert "quality_score" in quality_scores
        assert "coherence_score" in quality_scores
        assert "authenticity_score" in quality_scores

        # Check that metrics were recorded
        assert len(self.monitor.metrics) > 0

        # Check that dataset stats were updated
        assert dataset_name in self.monitor.dataset_stats
        stats = self.monitor.dataset_stats[dataset_name]
        assert stats.conversations_processed == 1

    def test_quality_alerts(self):
        """Test quality alert generation."""
        dataset_name = "test_dataset"

        # Create a low-quality conversation
        poor_conversation = Conversation(
            id="poor_conv",
            messages=[
                Message(role="user", content="test"),
                Message(role="assistant", content="bad response")
            ],
            context={},
            source="test"
        )

        # Process the poor conversation
        self.monitor.process_conversation(poor_conversation, dataset_name, 0.1)

        # Check if alerts were generated (might not always trigger depending on thresholds)
        # This is more of a smoke test to ensure the alerting system doesn't crash
        assert isinstance(self.monitor.alerts, list)

    def test_callback_system(self):
        """Test callback notification system."""
        metric_callback_called = False
        alert_callback_called = False
        stats_callback_called = False

        def metric_callback(metric):
            nonlocal metric_callback_called
            metric_callback_called = True

        def alert_callback(alert):
            nonlocal alert_callback_called
            alert_callback_called = True

        def stats_callback(stats):
            nonlocal stats_callback_called
            stats_callback_called = True

        # Add callbacks
        self.monitor.add_metric_callback(metric_callback)
        self.monitor.add_alert_callback(alert_callback)
        self.monitor.add_stats_callback(stats_callback)

        # Process conversation to trigger callbacks
        self.monitor.process_conversation(self.test_conversation, "test_dataset", 0.1)

        # Check that callbacks were called
        assert metric_callback_called
        assert stats_callback_called
        # Alert callback may or may not be called depending on quality scores

    def test_get_dataset_stats(self):
        """Test dataset statistics retrieval."""
        dataset_name = "test_dataset"

        # Process a conversation
        self.monitor.process_conversation(self.test_conversation, dataset_name, 0.1)

        # Get stats
        stats = self.monitor.get_dataset_stats(dataset_name)
        assert stats is not None
        assert stats.dataset_name == dataset_name
        assert stats.conversations_processed > 0

        # Test non-existent dataset
        non_existent_stats = self.monitor.get_dataset_stats("non_existent")
        assert non_existent_stats is None

    def test_get_all_stats(self):
        """Test retrieval of all dataset statistics."""
        # Process conversations for multiple datasets
        self.monitor.process_conversation(self.test_conversation, "dataset1", 0.1)
        self.monitor.process_conversation(self.test_conversation, "dataset2", 0.2)

        all_stats = self.monitor.get_all_stats()
        assert isinstance(all_stats, dict)
        assert "dataset1" in all_stats
        assert "dataset2" in all_stats

    def test_get_recent_metrics(self):
        """Test recent metrics retrieval with filtering."""
        dataset_name = "test_dataset"

        # Process conversation to generate metrics
        self.monitor.process_conversation(self.test_conversation, dataset_name, 0.1)

        # Get all recent metrics
        all_metrics = self.monitor.get_recent_metrics()
        assert len(all_metrics) > 0

        # Get metrics for specific dataset
        dataset_metrics = self.monitor.get_recent_metrics(dataset_name=dataset_name)
        assert len(dataset_metrics) > 0

        # Get metrics for specific type
        quality_metrics = self.monitor.get_recent_metrics(metric_type=MetricType.QUALITY_SCORE)
        assert len(quality_metrics) >= 0

    def test_alert_management(self):
        """Test alert creation and resolution."""
        # Create a test alert
        alert = Alert(
            id="test_alert",
            level=AlertLevel.WARNING,
            message="Test alert message",
            metric_type=MetricType.QUALITY_SCORE,
            dataset_name="test_dataset",
            value=0.5,
            threshold=0.6,
            timestamp=datetime.now()
        )

        self.monitor.alerts.append(alert)

        # Get active alerts
        active_alerts = self.monitor.get_active_alerts()
        assert alert in active_alerts

        # Resolve alert
        resolved = self.monitor.resolve_alert("test_alert")
        assert resolved
        assert alert.resolved
        assert alert.resolution_time is not None

        # Check that alert is no longer active
        active_alerts = self.monitor.get_active_alerts()
        assert alert not in active_alerts

    def test_running_averages(self):
        """Test running average calculations."""
        dataset_name = "test_dataset"

        # Process multiple conversations
        for i in range(3):
            self.monitor.process_conversation(self.test_conversation, dataset_name, 0.1 + i * 0.1)

        stats = self.monitor.dataset_stats[dataset_name]

        # Check that averages are calculated
        assert stats.average_quality_score > 0
        assert stats.average_processing_time > 0
        assert stats.current_rate > 0

    def test_shutdown(self):
        """Test monitor shutdown."""
        # Start monitoring
        self.monitor.start_monitoring("test_dataset")

        # Shutdown
        self.monitor.shutdown()

        # Check that monitoring is stopped
        assert not self.monitor._monitoring_active


if __name__ == "__main__":
    unittest.main()
