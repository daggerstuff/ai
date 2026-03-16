"""
Voice processing performance monitoring with quality tracking.
Provides comprehensive monitoring and analytics for voice processing operations.
"""

import json
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from logger import get_logger
from voice_types import AuthenticityProfile, OptimizationResult


class MetricType(Enum):
    """Types of performance metrics."""

    PROCESSING_TIME = "processing_time"
    QUALITY_SCORE = "quality_score"
    AUTHENTICITY_SCORE = "authenticity_score"
    CONSISTENCY_SCORE = "consistency_score"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""

    metric_type: MetricType
    value: float
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


@dataclass
class QualityMetric:
    """Quality-specific metric data point."""

    conversation_id: str
    overall_quality: float
    authenticity_score: float
    consistency_score: float
    empathy_score: float
    processing_time: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance monitoring alert."""

    severity: AlertSeverity
    metric_type: MetricType
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""

    metrics_window_size: int = 1000
    alert_cooldown_seconds: int = 300
    quality_threshold: float = 0.8
    authenticity_threshold: float = 0.75
    consistency_threshold: float = 0.85
    max_processing_time: float = 30.0
    max_error_rate: float = 0.05
    enable_real_time_monitoring: bool = True
    monitoring_interval_seconds: int = 30
    enable_quality_tracking: bool = True
    enable_trend_analysis: bool = True


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    report_period: tuple[datetime, datetime]
    total_conversations_processed: int = 0
    average_processing_time: float = 0.0
    average_quality_score: float = 0.0
    average_authenticity_score: float = 0.0
    average_consistency_score: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    quality_trends: dict[str, list[float]] = field(default_factory=dict)
    performance_trends: dict[str, list[float]] = field(default_factory=dict)
    alerts_summary: dict[AlertSeverity, int] = field(default_factory=dict)
    top_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class VoicePerformanceMonitor:
    """
    Comprehensive voice processing performance monitor.

    Features:
    - Real-time performance metrics tracking
    - Quality score monitoring and trending
    - Authenticity and consistency tracking
    - Automated alerting with configurable thresholds
    - Performance trend analysis
    - Resource utilization monitoring
    - Comprehensive reporting and analytics
    """

    def __init__(self, config: MonitoringConfig | None = None):
        """
        Initialize VoicePerformanceMonitor.

        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        self.logger = get_logger(__name__)

        # Metrics storage
        self.performance_metrics: deque[PerformanceMetric] = deque(
            maxlen=self.config.metrics_window_size
        )
        self.quality_metrics: deque[QualityMetric] = deque(
            maxlen=self.config.metrics_window_size
        )
        self.alerts: list[PerformanceAlert] = []

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None
        self.stop_monitoring = threading.Event()

        # Alert tracking
        self.last_alert_times: dict[str, datetime] = {}

        # Performance tracking
        self.session_start_time = datetime.now()
        self.total_conversations_processed = 0
        self.total_processing_time = 0.0
        self.error_count = 0

        # Thread safety
        self.metrics_lock = threading.Lock()

        if self.config.enable_real_time_monitoring:
            self.start_monitoring()

        self.logger.info("VoicePerformanceMonitor initialized")

    def record_processing_start(
        self, conversation_id: str, context: dict[str, Any] | None = None
    ) -> None:
        """Record the start of conversation processing."""
        if not hasattr(self, "_processing_starts"):
            self._processing_starts = {}

        self._processing_starts[conversation_id] = {
            "start_time": time.time(),
            "context": context or {},
        }

    def record_processing_complete(
        self,
        conversation_id: str,
        optimization_result: OptimizationResult,
        authenticity_profile: AuthenticityProfile | None = None,
    ) -> None:
        """
        Record completion of conversation processing.

        Args:
            conversation_id: ID of processed conversation
            optimization_result: Result from voice optimization
            authenticity_profile: Optional authenticity assessment
        """
        processing_info = getattr(self, "_processing_starts", {}).get(conversation_id)
        if not processing_info:
            self.logger.warning(
                f"No processing start recorded for conversation {conversation_id}"
            )
            return

        processing_time = time.time() - processing_info["start_time"]

        with self.metrics_lock:
            # Record performance metrics
            self._record_performance_metric(
                MetricType.PROCESSING_TIME,
                processing_time,
                context={"conversation_id": conversation_id},
            )

            # Record quality metrics
            if (
                optimization_result.success
                and optimization_result.optimized_conversations
            ):
                quality_scores = optimization_result.quality_metrics or {}

                quality_metric = QualityMetric(
                    conversation_id=conversation_id,
                    overall_quality=quality_scores.get("overall_quality", 0.0),
                    authenticity_score=(
                        authenticity_profile.overall_score
                        if authenticity_profile
                        else 0.0
                    ),
                    consistency_score=quality_scores.get("average_consistency", 0.0),
                    empathy_score=quality_scores.get("average_empathy", 0.0),
                    processing_time=processing_time,
                    timestamp=datetime.now(),
                    metadata={
                        "total_processed": optimization_result.total_processed,
                        "conversations_passed": len(
                            optimization_result.optimized_conversations
                        ),
                        "filtered_count": optimization_result.filtered_count,
                    },
                )

                self.quality_metrics.append(quality_metric)

                # Record individual quality scores as performance metrics
                self._record_performance_metric(
                    MetricType.QUALITY_SCORE,
                    quality_metric.overall_quality,
                    context={"conversation_id": conversation_id},
                )

                self._record_performance_metric(
                    MetricType.AUTHENTICITY_SCORE,
                    quality_metric.authenticity_score,
                    context={"conversation_id": conversation_id},
                )

                self._record_performance_metric(
                    MetricType.CONSISTENCY_SCORE,
                    quality_metric.consistency_score,
                    context={"conversation_id": conversation_id},
                )

            # Update session statistics
            self.total_conversations_processed += 1
            self.total_processing_time += processing_time

            # Calculate and record throughput
            session_duration = (
                datetime.now() - self.session_start_time
            ).total_seconds()
            if session_duration > 0:
                throughput = self.total_conversations_processed / session_duration
                self._record_performance_metric(
                    MetricType.THROUGHPUT,
                    throughput,
                    context={"session_duration": session_duration},
                )

        # Clean up processing start record
        if hasattr(self, "_processing_starts"):
            self._processing_starts.pop(conversation_id, None)

        # Check for alerts
        self._check_performance_alerts()

    def record_processing_error(
        self,
        conversation_id: str,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a processing error.

        Args:
            conversation_id: ID of conversation that failed
            error: Exception that occurred
            context: Optional error context
        """
        with self.metrics_lock:
            self.error_count += 1

            # Calculate error rate
            if self.total_conversations_processed > 0:
                error_rate = self.error_count / (
                    self.total_conversations_processed + self.error_count
                )
                self._record_performance_metric(
                    MetricType.ERROR_RATE,
                    error_rate,
                    context={
                        "conversation_id": conversation_id,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        **(context or {}),
                    },
                )

        self.logger.error(
            f"Processing error for conversation {conversation_id}: {error}"
        )

        # Check for error rate alerts
        self._check_performance_alerts()

    def get_current_performance_summary(self) -> dict[str, Any]:
        """Get current performance summary."""
        with self.metrics_lock:
            if not self.performance_metrics:
                return {"status": "No metrics available"}

            # Calculate recent metrics (last 100 data points)
            recent_metrics = list(self.performance_metrics)[-100:]

            # Group metrics by type
            metrics_by_type = defaultdict(list)
            for metric in recent_metrics:
                metrics_by_type[metric.metric_type].append(metric.value)

            summary = {
                "session_duration": (
                    datetime.now() - self.session_start_time
                ).total_seconds(),
                "total_conversations_processed": self.total_conversations_processed,
                "total_errors": self.error_count,
                "current_metrics": {},
            }

            # Calculate statistics for each metric type
            for metric_type, values in metrics_by_type.items():
                if values:
                    summary["current_metrics"][metric_type.value] = {
                        "current": values[-1],
                        "average": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "count": len(values),
                    }

            return summary

    def get_quality_trends(self, hours: int = 24) -> dict[str, Any]:
        """
        Get quality trends over specified time period.

        Args:
            hours: Number of hours to analyze

        Returns:
            Quality trends analysis
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.metrics_lock:
            # Filter quality metrics by time
            recent_quality_metrics = [
                qm for qm in self.quality_metrics if qm.timestamp >= cutoff_time
            ]

            if not recent_quality_metrics:
                return {"status": "No quality data available for specified period"}

            # Extract trends
            [qm.timestamp for qm in recent_quality_metrics]
            overall_quality = [qm.overall_quality for qm in recent_quality_metrics]
            authenticity_scores = [
                qm.authenticity_score for qm in recent_quality_metrics
            ]
            consistency_scores = [qm.consistency_score for qm in recent_quality_metrics]
            empathy_scores = [qm.empathy_score for qm in recent_quality_metrics]
            processing_times = [qm.processing_time for qm in recent_quality_metrics]

            return {
                "period": f"Last {hours} hours",
                "data_points": len(recent_quality_metrics),
                "trends": {
                    "overall_quality": {
                        "current": overall_quality[-1] if overall_quality else 0,
                        "average": statistics.mean(overall_quality),
                        "trend": self._calculate_trend(overall_quality),
                        "min": min(overall_quality),
                        "max": max(overall_quality),
                    },
                    "authenticity": {
                        "current": (
                            authenticity_scores[-1] if authenticity_scores else 0
                        ),
                        "average": statistics.mean(authenticity_scores),
                        "trend": self._calculate_trend(authenticity_scores),
                        "min": min(authenticity_scores),
                        "max": max(authenticity_scores),
                    },
                    "consistency": {
                        "current": consistency_scores[-1] if consistency_scores else 0,
                        "average": statistics.mean(consistency_scores),
                        "trend": self._calculate_trend(consistency_scores),
                        "min": min(consistency_scores),
                        "max": max(consistency_scores),
                    },
                    "empathy": {
                        "current": empathy_scores[-1] if empathy_scores else 0,
                        "average": statistics.mean(empathy_scores),
                        "trend": self._calculate_trend(empathy_scores),
                        "min": min(empathy_scores),
                        "max": max(empathy_scores),
                    },
                    "processing_time": {
                        "current": processing_times[-1] if processing_times else 0,
                        "average": statistics.mean(processing_times),
                        "trend": self._calculate_trend(processing_times),
                        "min": min(processing_times),
                        "max": max(processing_times),
                    },
                },
            }

    def get_alerts(
        self, severity: AlertSeverity | None = None, limit: int = 50
    ) -> list[PerformanceAlert]:
        """
        Get recent alerts.

        Args:
            severity: Optional severity filter
            limit: Maximum number of alerts to return

        Returns:
            List of performance alerts
        """
        alerts = self.alerts

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        # Sort by timestamp (most recent first) and limit
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts[:limit]

    def generate_performance_report(
        self, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Args:
            start_time: Report start time (default: 24 hours ago)
            end_time: Report end time (default: now)

        Returns:
            PerformanceReport with comprehensive analysis
        """
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=24)

        with self.metrics_lock:
            # Filter metrics by time period
            period_quality_metrics = [
                qm
                for qm in self.quality_metrics
                if start_time <= qm.timestamp <= end_time
            ]

            period_performance_metrics = [
                pm
                for pm in self.performance_metrics
                if start_time <= pm.timestamp <= end_time
            ]

            period_alerts = [
                alert
                for alert in self.alerts
                if start_time <= alert.timestamp <= end_time
            ]

            # Calculate report statistics
            report = PerformanceReport(
                report_period=(start_time, end_time),
                total_conversations_processed=len(period_quality_metrics),
            )

            if period_quality_metrics:
                report.average_processing_time = statistics.mean(
                    [qm.processing_time for qm in period_quality_metrics]
                )
                report.average_quality_score = statistics.mean(
                    [qm.overall_quality for qm in period_quality_metrics]
                )
                report.average_authenticity_score = statistics.mean(
                    [qm.authenticity_score for qm in period_quality_metrics]
                )
                report.average_consistency_score = statistics.mean(
                    [qm.consistency_score for qm in period_quality_metrics]
                )

                # Calculate throughput
                period_duration = (end_time - start_time).total_seconds()
                if period_duration > 0:
                    report.throughput = len(period_quality_metrics) / period_duration

            # Calculate error rate
            error_metrics = [
                pm
                for pm in period_performance_metrics
                if pm.metric_type == MetricType.ERROR_RATE
            ]
            if error_metrics:
                report.error_rate = error_metrics[-1].value  # Most recent error rate

            # Analyze trends
            if self.config.enable_trend_analysis:
                report.quality_trends = self._analyze_quality_trends(
                    period_quality_metrics
                )
                report.performance_trends = self._analyze_performance_trends(
                    period_performance_metrics
                )

            # Summarize alerts
            alert_counts = defaultdict(int)
            for alert in period_alerts:
                alert_counts[alert.severity] += 1
            report.alerts_summary = dict(alert_counts)

            # Generate recommendations
            report.recommendations = self._generate_recommendations(
                report, period_alerts
            )

            # Identify top issues
            report.top_issues = self._identify_top_issues(
                period_alerts, period_quality_metrics
            )

            return report

    def export_metrics(self, file_path: str, format: str = "json") -> None:
        """
        Export metrics to file.

        Args:
            file_path: Path to export file
            format: Export format ('json' or 'csv')
        """
        with self.metrics_lock:
            if format.lower() == "json":
                export_data = {
                    "performance_metrics": [
                        {
                            "metric_type": pm.metric_type.value,
                            "value": pm.value,
                            "timestamp": pm.timestamp.isoformat(),
                            "context": pm.context,
                            "source": pm.source,
                        }
                        for pm in self.performance_metrics
                    ],
                    "quality_metrics": [
                        {
                            "conversation_id": qm.conversation_id,
                            "overall_quality": qm.overall_quality,
                            "authenticity_score": qm.authenticity_score,
                            "consistency_score": qm.consistency_score,
                            "empathy_score": qm.empathy_score,
                            "processing_time": qm.processing_time,
                            "timestamp": qm.timestamp.isoformat(),
                            "metadata": qm.metadata,
                        }
                        for qm in self.quality_metrics
                    ],
                }

                with open(file_path, "w") as f:
                    json.dump(export_data, f, indent=2)

            else:
                raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Metrics exported to {file_path}")

    def start_monitoring(self) -> None:
        """Start real-time monitoring thread."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Real-time monitoring started")

    def stop_monitoring_thread(self) -> None:
        """Stop real-time monitoring thread."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        self.stop_monitoring.set()

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        self.logger.info("Real-time monitoring stopped")

    # Private methods

    def _record_performance_metric(
        self,
        metric_type: MetricType,
        value: float,
        context: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        """Record a performance metric (thread-safe)."""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            context=context or {},
            source=source,
        )

        self.performance_metrics.append(metric)

    def _check_performance_alerts(self) -> None:
        """Check for performance alert conditions."""
        datetime.now()

        # Check quality thresholds
        if self.quality_metrics:
            latest_quality = self.quality_metrics[-1]

            if latest_quality.overall_quality < self.config.quality_threshold:
                self._create_alert(
                    AlertSeverity.WARNING,
                    MetricType.QUALITY_SCORE,
                    f"Quality score below threshold: {latest_quality.overall_quality:.3f}",
                    latest_quality.overall_quality,
                    self.config.quality_threshold,
                )

            if latest_quality.authenticity_score < self.config.authenticity_threshold:
                self._create_alert(
                    AlertSeverity.WARNING,
                    MetricType.AUTHENTICITY_SCORE,
                    f"Authenticity score below threshold: {latest_quality.authenticity_score:.3f}",
                    latest_quality.authenticity_score,
                    self.config.authenticity_threshold,
                )

            if latest_quality.processing_time > self.config.max_processing_time:
                self._create_alert(
                    AlertSeverity.ERROR,
                    MetricType.PROCESSING_TIME,
                    f"Processing time exceeded threshold: {latest_quality.processing_time:.1f}s",
                    latest_quality.processing_time,
                    self.config.max_processing_time,
                )

        # Check error rate
        if self.total_conversations_processed > 0:
            current_error_rate = self.error_count / (
                self.total_conversations_processed + self.error_count
            )
            if current_error_rate > self.config.max_error_rate:
                self._create_alert(
                    AlertSeverity.CRITICAL,
                    MetricType.ERROR_RATE,
                    f"Error rate exceeded threshold: {current_error_rate:.3f}",
                    current_error_rate,
                    self.config.max_error_rate,
                )

    def _create_alert(
        self,
        severity: AlertSeverity,
        metric_type: MetricType,
        message: str,
        current_value: float,
        threshold: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Create an alert if not in cooldown period."""
        alert_key = f"{severity.value}_{metric_type.value}"
        current_time = datetime.now()

        # Check cooldown
        last_alert_time = self.last_alert_times.get(alert_key)
        if last_alert_time:
            time_since_last = (current_time - last_alert_time).total_seconds()
            if time_since_last < self.config.alert_cooldown_seconds:
                return

        # Create alert
        alert = PerformanceAlert(
            severity=severity,
            metric_type=metric_type,
            message=message,
            current_value=current_value,
            threshold=threshold,
            timestamp=current_time,
            context=context or {},
        )

        self.alerts.append(alert)
        self.last_alert_times[alert_key] = current_time

        # Log alert
        log_method = getattr(self.logger, severity.value.lower(), self.logger.info)
        log_method(f"Performance Alert: {message}")

        # Keep alerts list manageable
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]

    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        while not self.stop_monitoring.is_set():
            try:
                # Record system metrics if available
                self._record_system_metrics()

                # Check for alerts
                self._check_performance_alerts()

                # Sleep for monitoring interval
                self.stop_monitoring.wait(
                    timeout=self.config.monitoring_interval_seconds
                )

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.stop_monitoring.wait(timeout=60)  # Wait longer on error

    def _record_system_metrics(self) -> None:
        """Record system resource metrics if available."""
        try:
            import psutil

            # Record memory usage
            memory_percent = psutil.virtual_memory().percent
            self._record_performance_metric(
                MetricType.MEMORY_USAGE, memory_percent, context={"unit": "percent"}
            )

            # Record CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_performance_metric(
                MetricType.CPU_USAGE, cpu_percent, context={"unit": "percent"}
            )

        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            self.logger.warning(f"Failed to record system metrics: {e}")

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return "stable"

        # Simple linear trend calculation
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        change_percent = (
            ((second_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0
        )

        if change_percent > 5:
            return "increasing"
        if change_percent < -5:
            return "decreasing"
        return "stable"

    def _analyze_quality_trends(
        self, quality_metrics: list[QualityMetric]
    ) -> dict[str, list[float]]:
        """Analyze quality trends from metrics."""
        if not quality_metrics:
            return {}

        return {
            "overall_quality": [qm.overall_quality for qm in quality_metrics],
            "authenticity": [qm.authenticity_score for qm in quality_metrics],
            "consistency": [qm.consistency_score for qm in quality_metrics],
            "empathy": [qm.empathy_score for qm in quality_metrics],
        }

    def _analyze_performance_trends(
        self, performance_metrics: list[PerformanceMetric]
    ) -> dict[str, list[float]]:
        """Analyze performance trends from metrics."""
        trends = defaultdict(list)

        for metric in performance_metrics:
            trends[metric.metric_type.value].append(metric.value)

        return dict(trends)

    def _generate_recommendations(
        self, report: PerformanceReport, alerts: list[PerformanceAlert]
    ) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []

        # Quality-based recommendations
        if report.average_quality_score < 0.8:
            recommendations.append(
                "Consider adjusting quality thresholds or improving input data quality"
            )

        if report.average_authenticity_score < 0.75:
            recommendations.append(
                "Review authenticity scoring parameters and voice data sources"
            )

        if report.average_processing_time > 20.0:
            recommendations.append(
                "Optimize processing pipeline for better performance"
            )

        if report.error_rate > 0.02:
            recommendations.append(
                "Investigate and address recurring processing errors"
            )

        # Alert-based recommendations
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            recommendations.append(
                "Address critical alerts immediately to prevent system degradation"
            )

        return recommendations

    def _identify_top_issues(
        self, alerts: list[PerformanceAlert], quality_metrics: list[QualityMetric]
    ) -> list[str]:
        """Identify top performance issues."""
        issues = []

        # Analyze alert patterns
        alert_types = defaultdict(int)
        for alert in alerts:
            alert_types[alert.metric_type] += 1

        # Most frequent alert types are top issues
        for metric_type, count in sorted(
            alert_types.items(), key=lambda x: x[1], reverse=True
        )[:3]:
            issues.append(f"Frequent {metric_type.value} alerts ({count} occurrences)")

        # Analyze quality patterns
        if quality_metrics:
            low_quality_count = sum(
                1 for qm in quality_metrics if qm.overall_quality < 0.6
            )
            if low_quality_count > len(quality_metrics) * 0.2:
                issues.append(
                    f"High proportion of low-quality conversations ({low_quality_count}/{len(quality_metrics)})"
                )

        return issues
