"""Monitoring module for dataset pipeline Phase 02.

Tracks validation rates (success/fail), quarantine growth, and emits alerts.
Uses simple in-memory counters for now (not persistent); in prod, integrate with
Sentry/Prometheus exporter. Alerts via simple thresholds (e.g., >10% fail rate
or quarantine >100 records in 1h).

Exports metrics for alerting: validation_success_total, validation_fail_total,
quarantine_insert_total, quarantine_size_gauge.

Simple alert function that logs or raises if thresholds breached.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


# Simple in-memory store; replace with Prometheus or Sentry in prod
class PipelineMetrics:
    def __init__(self):
        self.validation_success = 0
        self.validation_fail = 0
        self.quarantine_inserts = 0
        self.quarantine_size = 0  # Current count
        self.last_reset = datetime.now(tz=datetime.timezone.utc)
        self.alert_threshold_fail_rate = 0.1  # 10% fail rate
        self.alert_threshold_quarantine_growth = 100  # >100 in 1h

    def increment_validation_success(self) -> None:
        """Increment success counter."""
        self.validation_success += 1

    def increment_validation_fail(self) -> None:
        """Increment fail counter."""
        self.validation_fail += 1

    def increment_quarantine_insert(self) -> None:
        """Increment quarantine insert counter."""
        self.quarantine_inserts += 1
        self.quarantine_size += 1

    def get_validation_rate(self) -> float:
        """Get recent fail rate (last hour)."""
        total = self.validation_success + self.validation_fail
        if total == 0:
            return 0.0
        return self.validation_fail / total

    def check_alerts(self, alert_func: callable | None = None) -> dict[str, bool]:
        """Check for alerts and call alert_func if breached."""
        alerts = {}
        now = datetime.now(tz=datetime.timezone.utc)
        if now - self.last_reset > timedelta(hours=1):
            # Reset hourly
            self.validation_success = 0
            self.validation_fail = 0
            self.quarantine_inserts = 0
            # Don't reset size, use growth
            self.last_reset = now

        fail_rate = self.get_validation_rate()
        if fail_rate > self.alert_threshold_fail_rate:
            alerts["high_fail_rate"] = True
            if alert_func:
                alert_func(f"High validation fail rate: {fail_rate:.2%}")

        growth = self.quarantine_inserts
        if growth > self.alert_threshold_quarantine_growth:
            alerts["high_quarantine_growth"] = True
            if alert_func:
                alert_func(f"High quarantine growth: {growth}")

        # Log metrics periodically (in prod, export to Sentry/Prometheus)

        return alerts

    def export(self) -> dict[str, Any]:
        """Export current metrics."""
        return {
            "validation_success_total": self.validation_success,
            "validation_fail_total": self.validation_fail,
            "quarantine_insert_total": self.quarantine_inserts,
            "quarantine_size_gauge": self.quarantine_size,
            "validation_fail_rate": self.get_validation_rate(),
        }


# Global metrics instance
_metrics = PipelineMetrics()


def get_metrics() -> PipelineMetrics:
    return _metrics


def log_validation_success():
    """Log successful validation."""
    metrics = get_metrics()
    metrics.increment_validation_success()
    metrics.check_alerts()


def log_validation_fail():
    """Log failed validation."""
    metrics = get_metrics()
    metrics.increment_validation_fail()
    metrics.check_alerts()


def log_quarantine_insert():
    """Log quarantine insertion."""
    metrics = get_metrics()
    metrics.increment_quarantine_insert()
    metrics.check_alerts()


__all__ = [
    "PipelineMetrics",
    "get_metrics",
    "log_quarantine_insert",
    "log_validation_fail",
    "log_validation_success",
]
