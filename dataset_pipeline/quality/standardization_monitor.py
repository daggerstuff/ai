"""
Continuous quality monitoring during standardization with real-time metrics.
Provides comprehensive monitoring and alerting for data standardization processes.
"""

import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from conversation_schema import Conversation
from logger import get_logger


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityMetric:
    """Quality metric data point."""
    name: str
    value: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAlert:
    """Quality alert information."""
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardizationMetrics:
    """Comprehensive standardization metrics."""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    processing_rate: float = 0.0  # items per second
    average_processing_time: float = 0.0
    quality_scores: dict[str, float] = field(default_factory=dict)
    format_distribution: dict[str, int] = field(default_factory=dict)
    error_distribution: dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class StandardizationMonitor:
    """
    Continuous quality monitoring system for data standardization.

    Features:
    - Real-time quality metrics tracking
    - Configurable alerting thresholds
    - Performance monitoring and bottleneck detection
    - Quality trend analysis
    - Automated quality assessment
    - Multi-threaded monitoring with minimal overhead
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_cooldown: int = 300,  # seconds
        enable_real_time: bool = True
    ):
        """
        Initialize StandardizationMonitor.

        Args:
            window_size: Size of sliding window for metrics
            alert_cooldown: Cooldown period between similar alerts
            enable_real_time: Whether to enable real-time monitoring
        """
        self.window_size = window_size
        self.alert_cooldown = alert_cooldown
        self.enable_real_time = enable_real_time

        self.logger = get_logger(__name__)

        # Metrics storage
        self.metrics_history: deque[QualityMetric] = deque(maxlen=window_size)
        self.processing_times: deque[float] = deque(maxlen=window_size)
        self.quality_scores: deque[float] = deque(maxlen=window_size)
        self.alerts: list[QualityAlert] = []

        # Current metrics
        self.current_metrics = StandardizationMetrics()

        # Thresholds and configuration
        self.quality_thresholds = {
            "min_success_rate": 0.8,
            "max_processing_time": 5.0,  # seconds
            "min_quality_score": 0.7,
            "max_error_rate": 0.2
        }

        # Alert tracking
        self.last_alert_times: dict[str, datetime] = {}

        # Quality assessors
        self.quality_assessors: list[Callable] = []

        # Threading
        self.monitoring_thread: threading.Thread | None = None
        self.stop_monitoring = threading.Event()
        self.metrics_lock = threading.Lock()

        # Performance tracking
        self.start_time = datetime.now()
        self.last_update_time = datetime.now()

        if self.enable_real_time:
            self.start_monitoring()

        self.logger.info(f"StandardizationMonitor initialized with window size {window_size}")

    def register_quality_assessor(self, assessor: Callable[[Conversation], float]) -> None:
        """
        Register a quality assessment function.

        Args:
            assessor: Function that takes a Conversation and returns quality score (0-1)
        """
        self.quality_assessors.append(assessor)
        self.logger.info("Registered quality assessor")

    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """Set alert threshold for a metric."""
        self.quality_thresholds[metric_name] = threshold
        self.logger.info(f"Set threshold for {metric_name}: {threshold}")

    def record_processing_start(self, item_id: str) -> None:
        """Record the start of processing for an item."""
        if not hasattr(self, "_processing_starts"):
            self._processing_starts = {}
        self._processing_starts[item_id] = time.time()

    def record_processing_success(
        self,
        item_id: str,
        conversation: Conversation,
        source_format: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Record successful processing of an item.

        Args:
            item_id: Unique identifier for the item
            conversation: Resulting conversation object
            source_format: Source format of the item
            metadata: Additional metadata
        """
        processing_time = self._get_processing_time(item_id)

        with self.metrics_lock:
            # Update basic metrics
            self.current_metrics.total_processed += 1
            self.current_metrics.successful += 1
            self.current_metrics.format_distribution[source_format] = \
                self.current_metrics.format_distribution.get(source_format, 0) + 1

            # Record processing time
            if processing_time > 0:
                self.processing_times.append(processing_time)
                self._update_processing_metrics()

            # Assess quality if assessors are available
            if self.quality_assessors:
                quality_score = self._assess_quality(conversation)
                self.quality_scores.append(quality_score)
                self._update_quality_metrics()

            # Record metric
            self.metrics_history.append(QualityMetric(
                name="processing_success",
                value=1.0,
                timestamp=datetime.now(),
                metadata={
                    "source_format": source_format,
                    "processing_time": processing_time,
                    "message_count": len(conversation.messages),
                    **(metadata or {})
                }
            ))

        # Check for alerts
        self._check_alerts()

    def record_processing_failure(
        self,
        item_id: str,
        error: str,
        source_format: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Record failed processing of an item.

        Args:
            item_id: Unique identifier for the item
            error: Error message
            source_format: Source format of the item
            metadata: Additional metadata
        """
        processing_time = self._get_processing_time(item_id)

        with self.metrics_lock:
            # Update basic metrics
            self.current_metrics.total_processed += 1
            self.current_metrics.failed += 1

            # Track error types
            error_type = type(error).__name__ if isinstance(error, Exception) else "Unknown"
            self.current_metrics.error_distribution[error_type] = \
                self.current_metrics.error_distribution.get(error_type, 0) + 1

            if source_format:
                self.current_metrics.format_distribution[source_format] = \
                    self.current_metrics.format_distribution.get(source_format, 0) + 1

            # Record processing time if available
            if processing_time > 0:
                self.processing_times.append(processing_time)
                self._update_processing_metrics()

            # Record metric
            self.metrics_history.append(QualityMetric(
                name="processing_failure",
                value=0.0,
                timestamp=datetime.now(),
                metadata={
                    "error": str(error),
                    "source_format": source_format,
                    "processing_time": processing_time,
                    **(metadata or {})
                }
            ))

        # Check for alerts
        self._check_alerts()

    def get_current_metrics(self) -> StandardizationMetrics:
        """Get current standardization metrics."""
        with self.metrics_lock:
            # Update derived metrics
            if self.current_metrics.total_processed > 0:
                success_rate = self.current_metrics.successful / self.current_metrics.total_processed
                self.current_metrics.quality_scores["success_rate"] = success_rate

                error_rate = self.current_metrics.failed / self.current_metrics.total_processed
                self.current_metrics.quality_scores["error_rate"] = error_rate

            if self.quality_scores:
                self.current_metrics.quality_scores["average_quality"] = statistics.mean(self.quality_scores)

            # Update processing rate
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            if elapsed_time > 0:
                self.current_metrics.processing_rate = self.current_metrics.total_processed / elapsed_time

            self.current_metrics.timestamp = datetime.now()

            return self.current_metrics

    def get_quality_trends(self, metric_name: str, duration_minutes: int = 60) -> list[QualityMetric]:
        """
        Get quality trends for a specific metric over time.

        Args:
            metric_name: Name of the metric to analyze
            duration_minutes: Duration to look back in minutes

        Returns:
            List of quality metrics within the time window
        """
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)

        with self.metrics_lock:
            return [
                metric for metric in self.metrics_history
                if metric.name == metric_name and metric.timestamp >= cutoff_time
            ]

    def get_alerts(self, level: AlertLevel | None = None, limit: int = 100) -> list[QualityAlert]:
        """
        Get recent alerts, optionally filtered by level.

        Args:
            level: Optional alert level filter
            limit: Maximum number of alerts to return

        Returns:
            List of quality alerts
        """
        alerts = self.alerts

        if level:
            alerts = [alert for alert in alerts if alert.level == level]

        # Sort by timestamp (most recent first) and limit
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts[:limit]

    def reset_metrics(self) -> None:
        """Reset all metrics and alerts."""
        with self.metrics_lock:
            self.current_metrics = StandardizationMetrics()
            self.metrics_history.clear()
            self.processing_times.clear()
            self.quality_scores.clear()
            self.alerts.clear()
            self.last_alert_times.clear()
            self.start_time = datetime.now()

        self.logger.info("Metrics reset")

    def start_monitoring(self) -> None:
        """Start real-time monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("Started real-time monitoring")

    def stop_monitoring_thread(self) -> None:
        """Stop real-time monitoring thread."""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)

        self.logger.info("Stopped real-time monitoring")

    # Private methods

    def _get_processing_time(self, item_id: str) -> float:
        """Get processing time for an item."""
        if not hasattr(self, "_processing_starts"):
            return 0.0

        start_time = self._processing_starts.pop(item_id, None)
        if start_time:
            return time.time() - start_time
        return 0.0

    def _assess_quality(self, conversation: Conversation) -> float:
        """Assess quality of a conversation using registered assessors."""
        if not self.quality_assessors:
            return 1.0

        scores = []
        for assessor in self.quality_assessors:
            try:
                score = assessor(conversation)
                if 0 <= score <= 1:
                    scores.append(score)
            except Exception as e:
                self.logger.warning(f"Quality assessor failed: {e}")

        return statistics.mean(scores) if scores else 1.0

    def _update_processing_metrics(self) -> None:
        """Update processing-related metrics."""
        if self.processing_times:
            self.current_metrics.average_processing_time = statistics.mean(self.processing_times)

    def _update_quality_metrics(self) -> None:
        """Update quality-related metrics."""
        if self.quality_scores:
            self.current_metrics.quality_scores["current_quality"] = statistics.mean(self.quality_scores)

    def _check_alerts(self) -> None:
        """Check for alert conditions."""
        current_metrics = self.get_current_metrics()

        # Check success rate
        success_rate = current_metrics.quality_scores.get("success_rate", 1.0)
        if success_rate < self.quality_thresholds.get("min_success_rate", 0.8):
            self._create_alert(
                AlertLevel.WARNING,
                f"Success rate below threshold: {success_rate:.2f}",
                "success_rate",
                success_rate,
                self.quality_thresholds["min_success_rate"]
            )

        # Check processing time
        if (current_metrics.average_processing_time >
            self.quality_thresholds.get("max_processing_time", 5.0)):
            self._create_alert(
                AlertLevel.WARNING,
                f"Average processing time too high: {current_metrics.average_processing_time:.2f}s",
                "processing_time",
                current_metrics.average_processing_time,
                self.quality_thresholds["max_processing_time"]
            )

        # Check quality score
        avg_quality = current_metrics.quality_scores.get("average_quality", 1.0)
        if avg_quality < self.quality_thresholds.get("min_quality_score", 0.7):
            self._create_alert(
                AlertLevel.ERROR,
                f"Quality score below threshold: {avg_quality:.2f}",
                "quality_score",
                avg_quality,
                self.quality_thresholds["min_quality_score"]
            )

        # Check error rate
        error_rate = current_metrics.quality_scores.get("error_rate", 0.0)
        if error_rate > self.quality_thresholds.get("max_error_rate", 0.2):
            self._create_alert(
                AlertLevel.ERROR,
                f"Error rate too high: {error_rate:.2f}",
                "error_rate",
                error_rate,
                self.quality_thresholds["max_error_rate"]
            )

    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        metric_name: str,
        current_value: float,
        threshold: float
    ) -> None:
        """Create an alert if not in cooldown period."""
        now = datetime.now()
        alert_key = f"{level.value}_{metric_name}"

        # Check cooldown
        last_alert = self.last_alert_times.get(alert_key)
        if last_alert and (now - last_alert).total_seconds() < self.alert_cooldown:
            return

        # Create alert
        alert = QualityAlert(
            level=level,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=now
        )

        self.alerts.append(alert)
        self.last_alert_times[alert_key] = now

        # Log alert
        log_method = getattr(self.logger, level.value.lower(), self.logger.info)
        log_method(f"Quality Alert: {message}")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        while not self.stop_monitoring.is_set():
            try:
                # Perform periodic checks
                self._check_alerts()

                # Clean up old alerts (keep last 1000)
                if len(self.alerts) > 1000:
                    self.alerts = self.alerts[-1000:]

                # Sleep for monitoring interval
                self.stop_monitoring.wait(timeout=30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.stop_monitoring.wait(timeout=60)  # Wait longer on error
