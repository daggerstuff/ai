"""Tests for the analytics dashboard: metric collection and reporting."""

import pytest

from ai.tools.utilities.pipelines.analytics_dashboard import AnalyticsDashboard


@pytest.fixture
def dashboard() -> AnalyticsDashboard:
    return AnalyticsDashboard()


def test_collect_metrics_records_a_sample(dashboard):
    result = dashboard.collect_metrics("response_time", {"response_time": 0.4})

    assert result["success"] is True
    assert result["metric_type"] == "response_time"
    assert result["metrics_collected"] == 1
    assert result["active_alerts"] == []


def test_collect_metrics_flags_a_threshold_breach(dashboard):
    result = dashboard.collect_metrics("response_time", {"response_time": 2.5})

    assert result["success"] is True
    assert result["active_alerts"]
    assert "response_time" in result["active_alerts"][0]


def test_collect_metrics_rejects_bad_input(dashboard):
    assert dashboard.collect_metrics("", {"response_time": 1.0})["success"] is False
    assert dashboard.collect_metrics("response_time", {})["success"] is False


def test_report_requires_metrics(dashboard):
    report = dashboard.generate_performance_report()

    assert report["success"] is False
    assert report["report"] is None


def test_report_summarises_collected_metrics(dashboard):
    dashboard.collect_metrics("response_time", {"response_time": 0.4, "latency": 12})
    dashboard.collect_metrics("error_rate", {"error_rate": 0.01, "latency": 8})

    report = dashboard.generate_performance_report()

    assert report["success"] is True
    payload = report["report"]
    assert payload["summary"]["latency"] == pytest.approx(10.0)
    assert len(payload["data"]) == 2


def test_export_requires_a_report(dashboard):
    assert dashboard.export_report()["success"] is False

    dashboard.collect_metrics("response_time", {"response_time": 0.4})
    dashboard.generate_performance_report()
    exported = dashboard.export_report("json")

    assert exported["success"] is True
    assert exported["records_exported"] == 1
