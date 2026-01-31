"""
Simple Alerting System

Simplified alerting system without email dependencies for basic notifications.
"""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import requests
from logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""

    WEBHOOK = "webhook"
    CONSOLE = "console"
    FILE = "file"


@dataclass
class Alert:
    """Alert message structure."""

    id: str
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    condition: Callable[[dict[str, Any]], bool]
    severity: AlertSeverity
    channels: list[AlertChannel]
    cooldown_minutes: int = 5
    max_alerts_per_hour: int = 10


class SimpleAlertingSystem:
    """Simplified alerting system."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.alerts: list[Alert] = []
        self.rules: list[AlertRule] = []
        self.alert_history: dict[str, list[datetime]] = {}
        self.alerts_lock = threading.Lock()

        logger.info("SimpleAlertingSystem initialized")

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def check_conditions(self, metrics: dict[str, Any]) -> None:
        """Check all alert conditions."""
        current_time = datetime.now()

        for rule in self.rules:
            try:
                if rule.condition(metrics):
                    if not self._is_in_cooldown(rule.name, current_time):
                        alert = Alert(
                            id=f"{rule.name}_{int(current_time.timestamp())}",
                            title=f"Alert: {rule.name}",
                            message=f"Condition triggered for rule: {rule.name}",
                            severity=rule.severity,
                            source="AlertingSystem",
                            metadata=metrics,
                        )

                        self.send_alert(alert, rule.channels)
                        self._record_alert(rule.name, current_time)

            except Exception as e:
                logger.error(f"Error checking rule {rule.name}: {e}")

    def send_alert(self, alert: Alert, channels: list[AlertChannel]) -> None:
        """Send alert through specified channels."""
        with self.alerts_lock:
            self.alerts.append(alert)

        for channel_type in channels:
            if channel_type == AlertChannel.CONSOLE:
                self._send_console_alert(alert)
            elif channel_type == AlertChannel.FILE:
                self._send_file_alert(alert)
            elif channel_type == AlertChannel.WEBHOOK:
                self._send_webhook_alert(alert)

    def _send_console_alert(self, alert: Alert) -> None:
        """Send alert to console."""
        severity_symbols = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨",
        }

        severity_symbols.get(alert.severity, "📢")

    def _send_file_alert(self, alert: Alert, file_path: str = "alerts.log") -> None:
        """Send alert to file."""
        try:
            alert_data = {
                "id": alert.id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata,
            }

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_data) + "\n")

        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")

    def _send_webhook_alert(self, alert: Alert, webhook_url: str | None = None) -> None:
        """Send alert via webhook."""
        if not webhook_url:
            return

        try:
            payload = {
                "id": alert.id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata,
            }

            response = requests.post(webhook_url, json=payload, timeout=30)
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def _is_in_cooldown(self, rule_name: str, current_time: datetime) -> bool:
        """Check if rule is in cooldown period."""
        rule = next((r for r in self.rules if r.name == rule_name), None)
        if not rule or rule_name not in self.alert_history:
            return False

        last_alerts = self.alert_history[rule_name]
        if not last_alerts:
            return False

        last_alert_time = max(last_alerts)
        cooldown_period = timedelta(minutes=rule.cooldown_minutes)

        return current_time - last_alert_time < cooldown_period

    def _record_alert(self, rule_name: str, alert_time: datetime) -> None:
        """Record alert for cooldown tracking."""
        if rule_name not in self.alert_history:
            self.alert_history[rule_name] = []

        self.alert_history[rule_name].append(alert_time)

        # Keep only last 24 hours
        cutoff_time = alert_time - timedelta(hours=24)
        self.alert_history[rule_name] = [
            t for t in self.alert_history[rule_name] if t > cutoff_time
        ]

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary."""
        with self.alerts_lock:
            total_alerts = len(self.alerts)
            active_alerts = len([a for a in self.alerts if not a.resolved])

            severity_counts = {}
            for severity in AlertSeverity:
                severity_counts[severity.value] = len(
                    [
                        a
                        for a in self.alerts
                        if a.severity == severity and not a.resolved
                    ]
                )

            return {
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "severity_breakdown": severity_counts,
                "last_alert": (
                    self.alerts[-1].timestamp.isoformat() if self.alerts else None
                ),
            }


# Create default rules
def create_default_rules() -> list[AlertRule]:
    """Create default alert rules."""
    return [
        AlertRule(
            name="high_failure_rate",
            condition=lambda m: m.get("failure_rate", 0) > 0.5,
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.CONSOLE, AlertChannel.FILE],
        ),
        AlertRule(
            name="low_disk_space",
            condition=lambda m: m.get("disk_usage_percent", 0) > 90,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.CONSOLE, AlertChannel.FILE],
        ),
    ]


# Example usage
if __name__ == "__main__":
    alerting = SimpleAlertingSystem()

    # Add default rules
    for rule in create_default_rules():
        alerting.add_rule(rule)

    # Test alert
    test_metrics = {"failure_rate": 0.6, "disk_usage_percent": 95}
    alerting.check_conditions(test_metrics)

    # Show summary
    summary = alerting.get_alert_summary()

    # Clean up
    import os

    if os.path.exists("alerts.log"):
        os.remove("alerts.log")
