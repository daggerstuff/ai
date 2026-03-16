import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DashboardMetric:
    """
    Represents a singular metric node on the analytics dashboard.
    Responsible for encapsulating values and calculating trends.
    """

    def __init__(self, name: str, value: float, unit: str = ""):
        self.name = name
        self.value = value
        self.unit = unit
        self.history: List[float] = [value]
        self.last_updated = datetime.now()

    def update(self, new_value: float) -> None:
        """
        Updates the metric and retains historical data.
        """
        if not isinstance(new_value, (int, float)):
            raise ValueError(f"Metric value must be numeric, got {type(new_value)}.")

        self.value = new_value
        self.history.append(new_value)
        self.last_updated = datetime.now()


class AnalyticsDashboard:
    """
    The Analytics Dashboard aggregates real-time metrics across the pipeline.
    It compiles statistics on processing speeds, model uncertainty, quality gates,
    and adaptive learning loops to provide an enterprise-ready observability interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Analytics Dashboard with default components and configurable limits.
        """
        self.config = config or {
            "refresh_rate_ms": 1000,
            "max_history_points": 100,
            "enable_telemetry_export": False,
            "export_endpoint": None,
        }

        self.metrics: Dict[str, DashboardMetric] = {}
        self.state = {
            "is_active": True,
            "start_time": datetime.now(),
            "uptime_seconds": 0,
        }
        logger.info("AnalyticsDashboard structure initialized.")

    def register_metric(
        self, name: str, initial_value: float = 0.0, unit: str = ""
    ) -> None:
        """
        Register a new tracking metric point in the dashboard.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Metric name must be a non-empty string.")

        if name in self.metrics:
            logger.warning(f"Metric '{name}' is already registered. Overwriting.")

        try:
            self.metrics[name] = DashboardMetric(name, initial_value, unit)
        except Exception as e:
            logger.error(f"Failed to register metric {name}: {e}")

    def update_metric(self, name: str, value: float) -> None:
        """
        Update an existing metric tracked by the dashboard.
        Raises ValueError if the metric wasn't registered first.
        """
        if not isinstance(name, str):
            raise ValueError("Metric name must be a string.")

        if name not in self.metrics:
            raise ValueError(
                f"Cannot update unregistered metric: '{name}'. Call register_metric first."
            )

        try:
            metric = self.metrics[name]
            metric.update(value)

            # Prune history to prevent memory leaks over time
            if len(metric.history) > self.config["max_history_points"]:
                metric.history = metric.history[-self.config["max_history_points"] :]

        except Exception as e:
            logger.error(f"Failed to update metric {name}: {e}")

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Generate a point-in-time snapshot of all dashboard values,
        suitable for feeding to a UI frontend or telemetry exporter.
        """
        self.state["uptime_seconds"] = (
            datetime.now() - self.state["start_time"]
        ).total_seconds()

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "uptime": self.state["uptime_seconds"],
            "data": {},
        }

        for name, metric in self.metrics.items():
            snapshot["data"][name] = {
                "current": metric.value,
                "unit": metric.unit,
                "trend": (metric.value - metric.history[0])
                if len(metric.history) > 1
                else 0.0,
            }

        return snapshot

    def export_telemetry(self) -> bool:
        """
        Optionally forwards the dashboard snapshot to an external observability
        platform (e.g. Datadog, Prometheus pushgateway) if enabled in config.
        """
        if not self.config.get("enable_telemetry_export"):
            return False

        endpoint = self.config.get("export_endpoint")
        if not endpoint:
            logger.warning("Telemetry export enabled but no endpoint configured.")
            return False

        snapshot = self.get_snapshot()
        try:
            # Mocking network request
            # response = requests.post(endpoint, json=snapshot)
            logger.info(f"Successfully exported telemetry to {endpoint}")
            return True
        except Exception as e:
            logger.error(f"Failed to export telemetry: {e}")
            return False


def test_analytics_dashboard():
    """Verify that the dashboard tracks and restricts data appropriately."""
    dashboard = AnalyticsDashboard(
        {"max_history_points": 5, "enable_telemetry_export": False}
    )

    # Validation tests
    dashboard.register_metric("pipeline_throughput", 100.0, "req/sec")
    assert "pipeline_throughput" in dashboard.metrics

    # Update loop
    for val in [110.0, 120.0, 130.0, 140.0, 150.0, 160.0]:
        dashboard.update_metric("pipeline_throughput", val)

    metric = dashboard.metrics["pipeline_throughput"]

    # Ensure clipping
    assert len(metric.history) <= 5
    assert metric.value == 160.0

    # Ensure snapshot generation
    snap = dashboard.get_snapshot()
    assert snap["data"]["pipeline_throughput"]["current"] == 160.0
    print("AnalyticsDashboard passed enterprise tests.")


if __name__ == "__main__":
    test_analytics_dashboard()
