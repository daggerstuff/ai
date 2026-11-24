"""Tests for the safety monitoring system."""

import asyncio

import pytest

from ai.tools.scripts.safety_monitoring_system import SafetyMonitoringSystem


@pytest.fixture
def monitor() -> SafetyMonitoringSystem:
    monitor = SafetyMonitoringSystem()
    monitor.start_monitoring()
    return monitor


def test_a_stopped_monitor_does_not_score_input() -> None:
    """Nothing is analysed until monitoring is started."""
    stopped = SafetyMonitoringSystem()

    result = asyncio.run(stopped.process_user_input("hello", "u1"))

    assert result == {"status": "monitoring_disabled"}


def test_monitoring_toggles() -> None:
    """Starting and stopping monitoring is reflected in the metrics."""
    monitor = SafetyMonitoringSystem()
    assert monitor.get_safety_metrics()["monitoring_active"] is False

    monitor.start_monitoring()
    assert monitor.get_safety_metrics()["monitoring_active"] is True

    monitor.stop_monitoring()
    assert monitor.get_safety_metrics()["monitoring_active"] is False


def test_ordinary_input_is_monitored_without_intervention(monitor: SafetyMonitoringSystem) -> None:
    """Everyday text is scored, not escalated."""
    result = asyncio.run(monitor.process_user_input("hello there", "u1"))

    assert result["safety_status"] == "monitored"
    assert result["requires_intervention"] is False
    assert result["incident_created"] is False


def test_crisis_input_raises_an_incident_with_resources(monitor: SafetyMonitoringSystem) -> None:
    """Explicit self-harm language produces an incident and crisis resources."""
    result = asyncio.run(monitor.process_user_input("I want to kill myself", "u2"))

    assert result["crisis_level"] == "critical"
    assert result["requires_intervention"] is True
    assert result["incident_created"] is True
    assert result["resources"]["hotline"]

    assert monitor.get_safety_metrics()["crisis_detections"] == 1
    assert len(monitor.get_recent_incidents()) == 1


def test_metrics_report_the_counters(monitor: SafetyMonitoringSystem) -> None:
    """The metrics payload carries the counters the dashboard reads."""
    metrics = monitor.get_safety_metrics()

    for key in ("total_interactions", "crisis_detections", "false_positives", "false_negatives", "monitoring_active"):
        assert key in metrics

    assert metrics["total_interactions"] == 0
    assert metrics["crisis_detections"] == 0


def test_incident_log_starts_empty(monitor: SafetyMonitoringSystem) -> None:
    """A fresh monitor has no incidents to report."""
    assert monitor.get_recent_incidents() == []
