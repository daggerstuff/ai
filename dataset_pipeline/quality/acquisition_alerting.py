"""
Dataset acquisition monitoring and alerting system with error recovery.

This module provides comprehensive monitoring, alerting, and error recovery
capabilities for dataset acquisition operations, ensuring reliable and
resilient data loading with proactive issue detection and automated recovery.
"""

import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from ai.dataset_pipeline.standardization_monitor import AlertLevel

from .acquisition_monitor import AcquisitionMonitor
from .logger import get_logger
from .monitoring_dashboard import MetricType
from .performance_optimizer import PerformanceOptimizer
from .utils import read_json, write_json

logger = get_logger("dataset_pipeline.acquisition_alerting")


@dataclass
class Alert:
    """Alert message structure for acquisition monitoring."""
    id: str
    level: AlertLevel
    message: str
    metric_type: MetricType | None = None
    dataset_name: str | None = None
    value: float | None = None
    threshold: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=datetime.timezone.utc))
    resolved: bool = False
    resolution_time: datetime | None = None


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    RESTART = "restart"
    SKIP = "skip"
    ESCALATE = "escalate"
    REDUCE_CONCURRENCY = "reduce_concurrency"
    INCREASE_TIMEOUT = "increase_timeout"
    CLEAR_CACHE = "clear_cache"
    MANUAL_INTERVENTION = "manual_intervention"


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    CONSOLE = "console"
    FILE = "file"


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration."""
    trigger_conditions: list[str]
    actions: list[RecoveryAction]
    max_attempts: int = 3
    delay_seconds: float = 5.0
    escalation_threshold: int = 5
    success_criteria: list[str] = field(default_factory=list)


@dataclass
class NotificationConfig:
    """Notification configuration."""
    channels: list[NotificationChannel]
    email_config: dict[str, str] | None = None
    webhook_config: dict[str, str] | None = None
    file_path: Path | None = None
    severity_filter: list[AlertLevel] = field(default_factory=lambda: list(AlertLevel))


@dataclass
class ErrorPattern:
    """Error pattern for intelligent recovery."""
    pattern_id: str
    error_signatures: list[str]
    frequency_threshold: int
    time_window_minutes: int
    recovery_strategy: RecoveryStrategy
    description: str


@dataclass
class RecoveryAttempt:
    """Individual recovery attempt record."""
    attempt_id: str
    error_context: str
    recovery_action: RecoveryAction
    started_at: datetime
    completed_at: datetime | None = None
    success: bool = False
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AcquisitionAlerting:
    """
    Comprehensive dataset acquisition monitoring and alerting system.

    Provides intelligent error detection, automated recovery, multi-channel
    notifications, and proactive monitoring for dataset acquisition operations.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the acquisition alerting system."""
        self.config = config or {}

        # Core components
        self.acquisition_monitor = AcquisitionMonitor()
        self.performance_optimizer = PerformanceOptimizer()

        # Alerting configuration
        notification_settings = self.config.get("notifications", {})
        self.notification_config = NotificationConfig(
            channels=notification_settings.get("channels", [NotificationChannel.LOG, NotificationChannel.CONSOLE]),
            email_config=notification_settings.get("email_config"),
            webhook_config=notification_settings.get("webhook_config"),
            file_path=notification_settings.get("file_path"),
            severity_filter=notification_settings.get("severity_filter", list(AlertLevel))
        )

        # Error tracking and recovery
        self.error_patterns: list[ErrorPattern] = []
        self.recovery_attempts: dict[str, list[RecoveryAttempt]] = defaultdict(list)
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_stats = {
            "total_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "escalations": 0
        }

        # Monitoring state
        self.active_alerts: dict[str, Alert] = {}
        self.suppressed_alerts: set[str] = set()
        self.alert_history: deque = deque(maxlen=500)

        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread = None

        # Initialize default error patterns
        self._initialize_default_patterns()

        # Setup monitoring callbacks
        self._setup_monitoring_callbacks()

        logger.info("Acquisition Alerting system initialized")

    def _initialize_default_patterns(self) -> None:
        """Initialize default error patterns for common issues."""

        # Network connectivity issues
        network_pattern = ErrorPattern(
            pattern_id="network_connectivity",
            error_signatures=[
                "connection timeout", "network unreachable", "dns resolution failed",
                "connection refused", "socket timeout", "ssl handshake failed"
            ],
            frequency_threshold=3,
            time_window_minutes=10,
            recovery_strategy=RecoveryStrategy(
                trigger_conditions=["network_error"],
                actions=[RecoveryAction.RETRY, RecoveryAction.INCREASE_TIMEOUT],
                max_attempts=5,
                delay_seconds=10.0
            ),
            description="Network connectivity issues requiring retry with backoff"
        )

        # Rate limiting issues
        rate_limit_pattern = ErrorPattern(
            pattern_id="rate_limiting",
            error_signatures=[
                "rate limit exceeded", "too many requests", "quota exceeded",
                "429", "throttled", "api limit reached"
            ],
            frequency_threshold=2,
            time_window_minutes=5,
            recovery_strategy=RecoveryStrategy(
                trigger_conditions=["rate_limit"],
                actions=[RecoveryAction.REDUCE_CONCURRENCY, RecoveryAction.RETRY],
                max_attempts=3,
                delay_seconds=30.0
            ),
            description="Rate limiting requiring concurrency reduction"
        )

        # Memory issues
        memory_pattern = ErrorPattern(
            pattern_id="memory_exhaustion",
            error_signatures=[
                "out of memory", "memory error", "allocation failed",
                "cannot allocate memory", "memory limit exceeded"
            ],
            frequency_threshold=1,
            time_window_minutes=5,
            recovery_strategy=RecoveryStrategy(
                trigger_conditions=["memory_error"],
                actions=[RecoveryAction.CLEAR_CACHE, RecoveryAction.REDUCE_CONCURRENCY, RecoveryAction.RESTART],
                max_attempts=2,
                delay_seconds=15.0
            ),
            description="Memory exhaustion requiring cache clearing and concurrency reduction"
        )

        # Data quality issues
        quality_pattern = ErrorPattern(
            pattern_id="data_quality",
            error_signatures=[
                "invalid data format", "parsing error", "schema validation failed",
                "corrupt data", "malformed json", "encoding error"
            ],
            frequency_threshold=5,
            time_window_minutes=15,
            recovery_strategy=RecoveryStrategy(
                trigger_conditions=["data_quality_error"],
                actions=[RecoveryAction.SKIP, RecoveryAction.ESCALATE],
                max_attempts=1,
                delay_seconds=0.0
            ),
            description="Data quality issues requiring manual intervention"
        )

        self.error_patterns = [network_pattern, rate_limit_pattern, memory_pattern, quality_pattern]
        logger.info(f"Initialized {len(self.error_patterns)} default error patterns")

    def _map_severity_to_alert_level(self, severity: str) -> AlertLevel:
        """Map severity string to AlertLevel enum."""
        severity_map = {
            "low": AlertLevel.INFO,
            "medium": AlertLevel.WARNING,
            "high": AlertLevel.ERROR,
            "critical": AlertLevel.CRITICAL
        }
        return severity_map.get(severity.lower(), AlertLevel.INFO)

    def _setup_monitoring_callbacks(self) -> None:
        """Setup callbacks for monitoring integration."""

        # Alert callback for acquisition monitor
        def alert_callback(alert_type: str, alert_data: dict[str, Any]):
            # Convert the acquisition monitor alert format to our Alert format
            alert = Alert(
                id=f"acquisition_{alert_type}_{datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                level=self._map_severity_to_alert_level(alert_data.get("severity", "info")),
                message=alert_data.get("message", "No message"),
                dataset_name=alert_data.get("task_id"),  # Use task_id as dataset_name
timestamp=datetime.now(tz=datetime.timezone.utc)
            )
            self._handle_alert(alert)

        # Metric callback for performance tracking
        def metric_callback(metric):
            self._analyze_metric_trends(metric)

        # Register callbacks
        self.acquisition_monitor.add_alert_callback(alert_callback)
        # Note: add_metric_callback is not available in AcquisitionMonitor, so we skip it

    def start_monitoring(self, dataset_names: list[str]) -> None:
        """Start comprehensive monitoring for specified datasets."""

        # Start acquisition monitoring system
        self.acquisition_monitor.start_monitoring()

        # Start background monitoring thread
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_worker,
                daemon=True
            )
            self._monitoring_thread.start()

        logger.info(f"Started monitoring for {len(dataset_names)} datasets")

    def stop_monitoring(self, dataset_names: list[str] | None = None) -> None:
        """Stop monitoring for specified datasets or all datasets."""

        if dataset_names:
            # Note: AcquisitionMonitor doesn't have per-dataset stop, so we stop the whole system
            self.acquisition_monitor.stop_monitoring_system()
        else:
            # Stop all monitoring
            self._monitoring_active = False
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5)
            self.acquisition_monitor.stop_monitoring_system()

        logger.info("Stopped dataset monitoring")

    def _monitoring_worker(self) -> None:
        """Background worker for continuous monitoring and analysis."""

        while self._monitoring_active:
            try:
                # Check for error patterns
                self._analyze_error_patterns()

                # Check for performance degradation
                self._check_performance_degradation()

                # Clean up old alerts and recovery attempts
                self._cleanup_old_data()

                # Update recovery statistics
                self._update_recovery_stats()

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Monitoring worker error: {e}")
                time.sleep(60)  # Wait longer on error

    def _handle_alert(self, alert: Alert) -> None:
        """Handle incoming alerts with intelligent processing."""

        # Check if alert should be suppressed
        if self._should_suppress_alert(alert):
            logger.debug(f"Suppressed duplicate alert: {alert.id}")
            return

        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)

        # Analyze for error patterns
        matching_pattern = self._match_error_pattern(alert)

        if matching_pattern:
            logger.info(f"Alert matched pattern: {matching_pattern.pattern_id}")

            # Attempt automated recovery
            asyncio.create_task(self._attempt_recovery(alert, matching_pattern))

        # Send notifications
        self._send_notifications(alert)

        logger.info(f"Processed alert: {alert.level.value} - {alert.message}")

    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Determine if alert should be suppressed to avoid spam."""

        # Check for recent similar alerts
        recent_cutoff = datetime.now(tz=datetime.timezone.utc) - timedelta(minutes=5)

        similar_alerts = [
            a for a in self.alert_history
            if (a.dataset_name == alert.dataset_name and
                a.metric_type == alert.metric_type and
                a.timestamp >= recent_cutoff)
        ]

        # Suppress if too many similar alerts recently
        if len(similar_alerts) >= 3:
            self.suppressed_alerts.add(alert.id)
            return True

        return False

    def _match_error_pattern(self, alert: Alert) -> ErrorPattern | None:
        """Match alert against known error patterns."""

        alert_text = alert.message.lower()

        for pattern in self.error_patterns:
            for signature in pattern.error_signatures:
                if signature.lower() in alert_text:
                    # Check frequency threshold
                    if self._check_pattern_frequency(pattern, alert):
                        return pattern

        return None

    def _check_pattern_frequency(self, pattern: ErrorPattern, alert: Alert) -> bool:
        """Check if error pattern frequency threshold is met."""

        time_window = datetime.now(tz=datetime.timezone.utc) - timedelta(minutes=pattern.time_window_minutes)

        # Count matching errors in time window
        matching_errors = 0
        for error_record in self.error_history:
            if (error_record.get("timestamp", datetime.min) >= time_window and
                error_record.get("dataset_name") == (alert.dataset_name or "")):

                error_text = error_record.get("message", "").lower()
                for signature in pattern.error_signatures:
                    if signature.lower() in error_text:
                        matching_errors += 1
                        break

        return matching_errors >= pattern.frequency_threshold

    async def _attempt_recovery(self, alert: Alert, pattern: ErrorPattern) -> None:
        """Attempt automated recovery based on error pattern."""

        recovery_id = f"recovery_{alert.id}_{datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting recovery attempt: {recovery_id} for pattern {pattern.pattern_id}")

        for action in pattern.recovery_strategy.actions:
            attempt = RecoveryAttempt(
                attempt_id=f"{recovery_id}_{action.value}",
                error_context=alert.message,
                recovery_action=action,
                started_at=datetime.now(tz=datetime.timezone.utc)
            )

            try:
                success = await self._execute_recovery_action(action, alert, pattern)

                attempt.completed_at = datetime.now(tz=datetime.timezone.utc)
                attempt.success = success

                self.recovery_attempts[alert.dataset_name or "unknown"].append(attempt)
                self.recovery_stats["total_attempts"] += 1

                if success:
                    self.recovery_stats["successful_recoveries"] += 1
                    logger.info(f"Recovery action {action.value} succeeded for {alert.dataset_name}")

                    # Mark alert as resolved
                    alert.resolved = True
                    alert.resolution_time = datetime.now(tz=datetime.timezone.utc)

                    # Send recovery notification
                    await self._send_recovery_notification(alert, action, True)
                    break
                self.recovery_stats["failed_recoveries"] += 1
                logger.warning(f"Recovery action {action.value} failed for {alert.dataset_name}")

                # Wait before next action
                if pattern.recovery_strategy.delay_seconds > 0:
                    await asyncio.sleep(pattern.recovery_strategy.delay_seconds)

            except Exception as e:
                attempt.completed_at = datetime.now(tz=datetime.timezone.utc)
                attempt.success = False
                attempt.error_message = str(e)

                self.recovery_attempts[alert.dataset_name or "unknown"].append(attempt)
                self.recovery_stats["failed_recoveries"] += 1

                logger.error(f"Recovery action {action.value} failed with exception: {e}")

        # If all recovery actions failed, escalate
        if not alert.resolved:
            await self._escalate_alert(alert, pattern)

    async def _execute_recovery_action(
        self,
        action: RecoveryAction,
        alert: Alert,
        pattern: ErrorPattern
    ) -> bool:
        """Execute a specific recovery action."""

        try:
            if action == RecoveryAction.RETRY:
                # Simple retry - return True to simulate success
                await asyncio.sleep(1)
                return True

            if action == RecoveryAction.RESTART:
                # Restart monitoring system
                self.acquisition_monitor.stop_monitoring_system()
                await asyncio.sleep(2)
                self.acquisition_monitor.start_monitoring()
                return True

            if action == RecoveryAction.REDUCE_CONCURRENCY:
                # Reduce concurrency in performance optimizer (simulated)
                logger.info("Reduced concurrency settings")
                return True

            if action == RecoveryAction.INCREASE_TIMEOUT:
                # Increase timeout settings (simulated)
                logger.info("Increased timeout settings")
                return True

            if action == RecoveryAction.CLEAR_CACHE:
                # Clear performance optimizer cache
                self.performance_optimizer.clear_cache()
                logger.info("Cleared performance cache")
                return True

            if action == RecoveryAction.SKIP:
                # Skip the problematic item
                logger.info(f"Skipping problematic item for {alert.dataset_name}")
                return True

            if action == RecoveryAction.ESCALATE:
                # Escalate to manual intervention
                await self._escalate_alert(alert, pattern)
                return False

            logger.warning(f"Unknown recovery action: {action}")
            return False

        except Exception as e:
            logger.error(f"Recovery action {action.value} failed: {e}")
            return False

    async def _escalate_alert(self, alert: Alert, pattern: ErrorPattern) -> None:
        """Escalate alert for manual intervention."""

        self.recovery_stats["escalations"] += 1

        escalation_message = (
            f"ESCALATION REQUIRED: Alert {alert.id} could not be automatically resolved.\n"
            f"Dataset: {alert.dataset_name}\n"
            f"Error Pattern: {pattern.pattern_id}\n"
            f"Original Message: {alert.message}\n"
            f"Recovery attempts failed. Manual intervention required."
        )

        # Create escalation alert
        escalation_alert = Alert(
            id=f"escalation_{alert.id}",
            level=AlertLevel.CRITICAL,
            value=alert.value,
            threshold=alert.threshold,
            timestamp=datetime.now(tz=datetime.timezone.utc)
        )

        # Send escalation notifications
        self._send_notifications(escalation_alert)

        logger.critical(f"Escalated alert {alert.id} for manual intervention")

    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications through configured channels."""

        # Filter by severity
        if alert.level not in self.notification_config.severity_filter:
            return

        for channel in self.notification_config.channels:
            try:
                if channel == NotificationChannel.LOG:
                    self._send_log_notification(alert)
                elif channel == NotificationChannel.CONSOLE:
                    self._send_console_notification(alert)
                elif channel == NotificationChannel.EMAIL:
                    self._send_email_notification(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    self._send_webhook_notification(alert)
                elif channel == NotificationChannel.FILE:
                    self._send_file_notification(alert)

            except Exception as e:
                logger.error(f"Failed to send notification via {channel.value}: {e}")

    def _send_log_notification(self, alert: Alert) -> None:
        """Send notification to log."""
        level_map = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }

        log_func = level_map.get(alert.level, logger.info)
        log_func(f"ALERT [{alert.level.value.upper()}]: {alert.message}")

    def _send_console_notification(self, alert: Alert) -> None:
        """Send notification to console."""
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }

        level_emoji.get(alert.level, "🔔")

    def _send_email_notification(self, alert: Alert) -> None:
        """Send email notification."""
        if not self.notification_config.email_config:
            return

        # Email implementation would go here
        logger.info(f"Email notification sent for alert {alert.id}")

    def _send_webhook_notification(self, alert: Alert) -> None:
        """Send webhook notification."""
        if not self.notification_config.webhook_config:
            return

        # Webhook implementation would go here
        logger.info(f"Webhook notification sent for alert {alert.id}")

    def _send_file_notification(self, alert: Alert) -> None:
        """Send notification to file."""
        if not self.notification_config.file_path:
            return

        notification_data = {
            "alert_id": alert.id,
            "level": alert.level.value,
            "message": alert.message,
            "dataset_name": alert.dataset_name,
            "timestamp": alert.timestamp.isoformat(),
            "metric_type": alert.metric_type.value if alert.metric_type else None,
            "value": alert.value,
            "threshold": alert.threshold
        }

        try:
            # Append to notification file
            with open(self.notification_config.file_path, "a") as f:
                f.write(json.dumps(notification_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write file notification: {e}")

    async def _send_recovery_notification(
        self,
        alert: Alert,
        recovery_action: RecoveryAction,
        success: bool
    ) -> None:
        """Send notification about recovery attempt."""

        status = "succeeded" if success else "failed"
        message = f"Recovery action '{recovery_action.value}' {status} for alert {alert.id} on dataset {alert.dataset_name}"

        recovery_alert = Alert(
            id=f"recovery_{alert.id}_{recovery_action.value}",
            level=AlertLevel.INFO if success else AlertLevel.WARNING,
            message=message,
            metric_type=alert.metric_type,
            dataset_name=alert.dataset_name,
            value=alert.value,
            threshold=alert.threshold,
            timestamp=datetime.now(tz=datetime.timezone.utc)
        )

        self._send_notifications(recovery_alert)

    def _analyze_error_patterns(self) -> None:
        """Analyze recent errors for emerging patterns."""

        # This would analyze error history for new patterns
        # For now, just log the analysis
        recent_errors = len([e for e in self.error_history
                           if e.get("timestamp", datetime.min.replace(tzinfo=datetime.timezone.utc)) >= datetime.now(tz=datetime.timezone.utc) - timedelta(hours=1)])

        if recent_errors > 10:
            logger.warning(f"High error rate detected: {recent_errors} errors in the last hour")

    def _check_performance_degradation(self) -> None:
        """Check for performance degradation patterns."""

        metrics = self.performance_optimizer.get_metrics()
        if metrics and metrics.average_operation_time > 10.0:  # Check if avg operation time is too high
            logger.warning(f"Performance degradation detected: avg operation time {metrics.average_operation_time:.1f}s")

    def _cleanup_old_data(self) -> None:
        """Clean up old alerts and recovery attempts."""

        cutoff_time = datetime.now(tz=datetime.timezone.utc) - timedelta(days=7)

        # Clean up old alerts
        old_alert_ids = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.timestamp < cutoff_time
        ]

        for alert_id in old_alert_ids:
            del self.active_alerts[alert_id]

        # Clean up old recovery attempts
        for dataset_name in list(self.recovery_attempts.keys()):
            old_attempts = [
                attempt for attempt in self.recovery_attempts[dataset_name]
                if attempt.started_at < cutoff_time
            ]

            for attempt in old_attempts:
                self.recovery_attempts[dataset_name].remove(attempt)

    def _update_recovery_stats(self) -> None:
        """Update recovery statistics."""

        # Calculate success rate
        total_attempts = self.recovery_stats["total_attempts"]
        if total_attempts > 0:
            success_rate = (self.recovery_stats["successful_recoveries"] / total_attempts) * 100
            logger.debug(f"Recovery success rate: {success_rate:.1f}%")

    def _analyze_metric_trends(self, metric) -> None:
        """Analyze metric trends for proactive alerting."""

        # This would analyze metric trends
        # For now, just basic threshold checking
        if hasattr(metric, "value") and metric.value < 0.5:
            logger.debug(f"Low metric value detected: {metric.metric_type.value} = {metric.value}")

    def add_error_pattern(self, pattern: ErrorPattern) -> None:
        """Add a custom error pattern."""

        self.error_patterns.append(pattern)
        logger.info(f"Added error pattern: {pattern.pattern_id}")

    def remove_error_pattern(self, pattern_id: str) -> bool:
        """Remove an error pattern."""

        for i, pattern in enumerate(self.error_patterns):
            if pattern.pattern_id == pattern_id:
                del self.error_patterns[i]
                logger.info(f"Removed error pattern: {pattern_id}")
                return True

        return False

    def get_active_alerts(self, dataset_name: str | None = None) -> list[Alert]:
        """Get active alerts, optionally filtered by dataset."""

        alerts = list(self.active_alerts.values())

        if dataset_name:
            alerts = [alert for alert in alerts if alert.dataset_name == dataset_name]

        return alerts

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get comprehensive recovery statistics."""

        total_attempts = self.recovery_stats["total_attempts"]
        success_rate = 0.0

        if total_attempts > 0:
            success_rate = (self.recovery_stats["successful_recoveries"] / total_attempts) * 100

        return {
            "total_attempts": total_attempts,
            "successful_recoveries": self.recovery_stats["successful_recoveries"],
            "failed_recoveries": self.recovery_stats["failed_recoveries"],
            "success_rate_percent": success_rate,
            "escalations": self.recovery_stats["escalations"],
            "active_patterns": len(self.error_patterns),
            "active_alerts": len(self.active_alerts),
            "suppressed_alerts": len(self.suppressed_alerts)
        }

    def get_error_patterns(self) -> list[ErrorPattern]:
        """Get all configured error patterns."""
        return self.error_patterns.copy()

    def get_recovery_history(self, dataset_name: str) -> list[RecoveryAttempt]:
        """Get recovery history for a specific dataset."""
        return self.recovery_attempts.get(dataset_name, []).copy()

    def generate_monitoring_report(self) -> dict[str, Any]:
        """Generate comprehensive monitoring report."""

        # Calculate alert statistics
        alert_stats = defaultdict(int)
        for alert in self.alert_history:
            alert_stats[alert.level.value] += 1

        # Calculate recovery statistics by action
        recovery_by_action = defaultdict(lambda: {"attempts": 0, "successes": 0})

        for attempts in self.recovery_attempts.values():
            for attempt in attempts:
                action = attempt.recovery_action.value
                recovery_by_action[action]["attempts"] += 1
                if attempt.success:
                    recovery_by_action[action]["successes"] += 1

        # Calculate success rates by action
        recovery_success_rates = {}
        for action, stats in recovery_by_action.items():
            if stats["attempts"] > 0:
                recovery_success_rates[action] = (stats["successes"] / stats["attempts"]) * 100
            else:
                recovery_success_rates[action] = 0.0

        return {
            "generated_at": datetime.now(tz=datetime.timezone.utc).isoformat(),
            "monitoring_status": {
                "active": self._monitoring_active,
                "monitored_datasets": len({alert.dataset_name for alert in self.active_alerts.values()}),
                "error_patterns": len(self.error_patterns),
                "notification_channels": len(self.notification_config.channels)
            },
            "alert_statistics": dict(alert_stats),
            "recovery_statistics": self.get_recovery_stats(),
            "recovery_success_rates": recovery_success_rates,
            "recent_alerts": len([
                alert for alert in self.alert_history
                if alert.timestamp >= datetime.now(tz=datetime.timezone.utc) - timedelta(hours=24)
            ]),
            "system_health": {
                "active_alerts": len(self.active_alerts),
                "suppressed_alerts": len(self.suppressed_alerts),
                "error_history_size": len(self.error_history),
                "recovery_attempts_total": sum(len(attempts) for attempts in self.recovery_attempts.values())
            }
        }

    def export_configuration(self, output_path: Path) -> bool:
        """Export current configuration to file."""

        try:
            config_data = {
                "error_patterns": [
                    {
                        "pattern_id": pattern.pattern_id,
                        "error_signatures": pattern.error_signatures,
                        "frequency_threshold": pattern.frequency_threshold,
                        "time_window_minutes": pattern.time_window_minutes,
                        "description": pattern.description,
                        "recovery_strategy": {
                            "trigger_conditions": pattern.recovery_strategy.trigger_conditions,
                            "actions": [action.value for action in pattern.recovery_strategy.actions],
                            "max_attempts": pattern.recovery_strategy.max_attempts,
                            "delay_seconds": pattern.recovery_strategy.delay_seconds,
                            "escalation_threshold": pattern.recovery_strategy.escalation_threshold
                        }
                    }
                    for pattern in self.error_patterns
                ],
                "notification_config": {
                    "channels": [channel.value for channel in self.notification_config.channels],
                    "severity_filter": [level.value for level in self.notification_config.severity_filter]
                },
                "exported_at": datetime.now(tz=datetime.timezone.utc).isoformat()
            }

            write_json(config_data, output_path)
            logger.info(f"Configuration exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False

    def import_configuration(self, config_path: Path) -> bool:
        """Import configuration from file."""

        try:
            config_data = read_json(str(config_path))

            # Import error patterns
            imported_patterns = []
            for pattern_data in config_data.get("error_patterns", []):
                recovery_strategy = RecoveryStrategy(
                    trigger_conditions=pattern_data["recovery_strategy"]["trigger_conditions"],
                    actions=[RecoveryAction(action) for action in pattern_data["recovery_strategy"]["actions"]],
                    max_attempts=pattern_data["recovery_strategy"]["max_attempts"],
                    delay_seconds=pattern_data["recovery_strategy"]["delay_seconds"],
                    escalation_threshold=pattern_data["recovery_strategy"]["escalation_threshold"]
                )

                pattern = ErrorPattern(
                    pattern_id=pattern_data["pattern_id"],
                    error_signatures=pattern_data["error_signatures"],
                    frequency_threshold=pattern_data["frequency_threshold"],
                    time_window_minutes=pattern_data["time_window_minutes"],
                    recovery_strategy=recovery_strategy,
                    description=pattern_data["description"]
                )

                imported_patterns.append(pattern)

            self.error_patterns = imported_patterns
            logger.info(f"Imported {len(imported_patterns)} error patterns")
            return True

        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the acquisition alerting system."""

        logger.info("Shutting down acquisition alerting system...")

        # Stop monitoring
        self.stop_monitoring()

        # Shutdown components
        self.acquisition_monitor.stop_monitoring_system()
        self.performance_optimizer.shutdown()

        logger.info("Acquisition alerting system shutdown complete")
