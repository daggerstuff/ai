"""
Comprehensive Monitoring Dashboard System

Real-time monitoring dashboard for dataset pipeline operations with
metrics collection, visualization, and alerting integration.
"""

import json
import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: datetime
    value: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition and data."""

    name: str
    metric_type: MetricType
    description: str
    unit: str
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""

    id: str
    title: str
    widget_type: str  # 'line_chart', 'gauge', 'counter', 'table', 'status'
    metrics: list[str]
    config: dict[str, Any] = field(default_factory=dict)
    position: dict[str, int] = field(default_factory=dict)  # x, y, width, height


@dataclass
class Dashboard:
    """Dashboard configuration."""

    id: str
    name: str
    description: str
    widgets: list[DashboardWidget]
    refresh_interval: int = 30  # seconds
    auto_refresh: bool = True


class MetricsCollector:
    """Collects and stores metrics data."""

    def __init__(self, retention_hours: int = 24):
        self.metrics: dict[str, Metric] = {}
        self.retention_hours = retention_hours
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        logger.info("MetricsCollector initialized")

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: str = "",
        labels: dict[str, str] | None = None,
    ):
        """Register a new metric."""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(
                    name=name,
                    metric_type=metric_type,
                    description=description,
                    unit=unit,
                    labels=labels or {},
                )
                logger.info(f"Registered metric: {name}")

    def record_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ):
        """Record counter metric."""
        self._record_metric(name, MetricType.COUNTER, value, labels)

    def record_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ):
        """Record gauge metric."""
        self._record_metric(name, MetricType.GAUGE, value, labels)

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ):
        """Record histogram metric."""
        self._record_metric(name, MetricType.HISTOGRAM, value, labels)

    def record_timer(
        self, name: str, duration: float, labels: dict[str, str] | None = None
    ):
        """Record timer metric."""
        self._record_metric(name, MetricType.TIMER, duration, labels)

    def _record_metric(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        labels: dict[str, str] | None = None,
    ):
        """Internal method to record metric."""
        with self.lock:
            if name not in self.metrics:
                # Auto-register metric
                self.register_metric(
                    name, metric_type, f"Auto-registered {metric_type.value}"
                )

            metric = self.metrics[name]
            point = MetricPoint(
                timestamp=datetime.now(), value=value, labels=labels or {}
            )
            metric.data_points.append(point)

    def get_metric_data(
        self,
        name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[MetricPoint]:
        """Get metric data points within time range."""
        with self.lock:
            if name not in self.metrics:
                return []

            metric = self.metrics[name]
            points = list(metric.data_points)

            if start_time or end_time:
                filtered_points = []
                for point in points:
                    if start_time and point.timestamp < start_time:
                        continue
                    if end_time and point.timestamp > end_time:
                        continue
                    filtered_points.append(point)
                return filtered_points

            return points

    def get_metric_summary(
        self, name: str, duration_minutes: int = 60
    ) -> dict[str, Any]:
        """Get metric summary statistics."""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration_minutes)

        points = self.get_metric_data(name, start_time, end_time)

        if not points:
            return {}

        values = [p.value for p in points]

        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "latest": values[-1] if values else 0,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }

    def list_metrics(self) -> list[dict[str, Any]]:
        """List all registered metrics."""
        with self.lock:
            return [
                {
                    "name": metric.name,
                    "type": metric.metric_type.value,
                    "description": metric.description,
                    "unit": metric.unit,
                    "data_points": len(metric.data_points),
                    "labels": metric.labels,
                }
                for metric in self.metrics.values()
            ]

    def _cleanup_loop(self):
        """Background cleanup of old metrics."""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

                with self.lock:
                    for metric in self.metrics.values():
                        # Remove old data points
                        while (
                            metric.data_points
                            and metric.data_points[0].timestamp < cutoff_time
                        ):
                            metric.data_points.popleft()

                # Sleep for 1 hour before next cleanup
                time.sleep(3600)

            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                time.sleep(300)  # Wait 5 minutes on error


class SystemMonitor:
    """Monitors system resources and pipeline health."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = get_logger(__name__)
        self.monitoring_active = False
        self.monitor_thread = None

        # Register system metrics
        self._register_system_metrics()

    def _register_system_metrics(self):
        """Register system monitoring metrics."""
        system_metrics = [
            ("system.cpu_usage", MetricType.GAUGE, "CPU usage percentage", "%"),
            ("system.memory_usage", MetricType.GAUGE, "Memory usage percentage", "%"),
            ("system.disk_usage", MetricType.GAUGE, "Disk usage percentage", "%"),
            (
                "pipeline.active_tasks",
                MetricType.GAUGE,
                "Number of active tasks",
                "count",
            ),
            (
                "pipeline.completed_tasks",
                MetricType.COUNTER,
                "Completed tasks",
                "count",
            ),
            ("pipeline.failed_tasks", MetricType.COUNTER, "Failed tasks", "count"),
            (
                "pipeline.processing_time",
                MetricType.HISTOGRAM,
                "Task processing time",
                "seconds",
            ),
            ("dataset.download_speed", MetricType.GAUGE, "Download speed", "MB/s"),
            (
                "dataset.quality_score",
                MetricType.GAUGE,
                "Dataset quality score",
                "score",
            ),
            ("dataset.total_size", MetricType.GAUGE, "Total dataset size", "bytes"),
        ]

        for name, metric_type, description, unit in system_metrics:
            self.metrics.register_metric(name, metric_type, description, unit)

    def start_monitoring(self, interval_seconds: int = 30):
        """Start system monitoring."""
        if self.monitoring_active:
            logger.warning("System monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval_seconds,), daemon=True
        )
        self.monitor_thread.start()

        logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("System monitoring stopped")

    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Sleep until next collection
                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute on error

    def _collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.record_gauge("system.cpu_usage", cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.record_gauge("system.memory_usage", memory.percent)

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            self.metrics.record_gauge("system.disk_usage", disk_percent)

        except ImportError:
            # psutil not available, use mock data
            import random

            self.metrics.record_gauge("system.cpu_usage", random.uniform(10, 80))
            self.metrics.record_gauge("system.memory_usage", random.uniform(30, 70))
            self.metrics.record_gauge("system.disk_usage", random.uniform(20, 60))

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")


class DashboardRenderer:
    """Renders dashboard data for display."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = get_logger(__name__)

    def render_dashboard(self, dashboard: Dashboard) -> dict[str, Any]:
        """Render complete dashboard data."""
        dashboard_data = {
            "id": dashboard.id,
            "name": dashboard.name,
            "description": dashboard.description,
            "refresh_interval": dashboard.refresh_interval,
            "timestamp": datetime.now().isoformat(),
            "widgets": [],
        }

        for widget in dashboard.widgets:
            widget_data = self.render_widget(widget)
            dashboard_data["widgets"].append(widget_data)

        return dashboard_data

    def render_widget(self, widget: DashboardWidget) -> dict[str, Any]:
        """Render individual widget data."""
        widget_data = {
            "id": widget.id,
            "title": widget.title,
            "type": widget.widget_type,
            "position": widget.position,
            "config": widget.config,
            "data": {},
        }

        if widget.widget_type == "line_chart":
            widget_data["data"] = self._render_line_chart(widget)
        elif widget.widget_type == "gauge":
            widget_data["data"] = self._render_gauge(widget)
        elif widget.widget_type == "counter":
            widget_data["data"] = self._render_counter(widget)
        elif widget.widget_type == "table":
            widget_data["data"] = self._render_table(widget)
        elif widget.widget_type == "status":
            widget_data["data"] = self._render_status(widget)

        return widget_data

    def _render_line_chart(self, widget: DashboardWidget) -> dict[str, Any]:
        """Render line chart data."""
        duration_minutes = widget.config.get("duration_minutes", 60)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration_minutes)

        series_data = []

        for metric_name in widget.metrics:
            points = self.metrics.get_metric_data(metric_name, start_time, end_time)

            series = {
                "name": metric_name,
                "data": [
                    {"x": point.timestamp.isoformat(), "y": point.value}
                    for point in points
                ],
            }
            series_data.append(series)

        return {
            "series": series_data,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }

    def _render_gauge(self, widget: DashboardWidget) -> dict[str, Any]:
        """Render gauge data."""
        if not widget.metrics:
            return {"value": 0, "max": 100}

        metric_name = widget.metrics[0]
        summary = self.metrics.get_metric_summary(metric_name, 5)  # Last 5 minutes

        return {
            "value": summary.get("latest", 0),
            "max": widget.config.get("max_value", 100),
            "min": widget.config.get("min_value", 0),
            "unit": widget.config.get("unit", ""),
            "thresholds": widget.config.get("thresholds", {}),
        }

    def _render_counter(self, widget: DashboardWidget) -> dict[str, Any]:
        """Render counter data."""
        counters = {}

        for metric_name in widget.metrics:
            summary = self.metrics.get_metric_summary(metric_name, 60)
            counters[metric_name] = {
                "value": summary.get("latest", 0),
                "change": summary.get("max", 0) - summary.get("min", 0),
            }

        return {"counters": counters}

    def _render_table(self, widget: DashboardWidget) -> dict[str, Any]:
        """Render table data."""
        rows = []

        for metric_name in widget.metrics:
            summary = self.metrics.get_metric_summary(metric_name, 60)
            if summary:
                rows.append(
                    {
                        "metric": metric_name,
                        "current": summary.get("latest", 0),
                        "average": summary.get("mean", 0),
                        "min": summary.get("min", 0),
                        "max": summary.get("max", 0),
                    }
                )

        return {"columns": ["metric", "current", "average", "min", "max"], "rows": rows}

    def _render_status(self, widget: DashboardWidget) -> dict[str, Any]:
        """Render status data."""
        statuses = {}

        for metric_name in widget.metrics:
            summary = self.metrics.get_metric_summary(metric_name, 5)
            current_value = summary.get("latest", 0)

            # Determine status based on thresholds
            thresholds = widget.config.get("thresholds", {})
            status = "unknown"

            if "critical" in thresholds and current_value >= thresholds["critical"]:
                status = "critical"
            elif "warning" in thresholds and current_value >= thresholds["warning"]:
                status = "warning"
            elif "good" in thresholds and current_value >= thresholds["good"]:
                status = "good"
            else:
                status = "normal"

            statuses[metric_name] = {"value": current_value, "status": status}

        return {"statuses": statuses}


class MonitoringDashboard:
    """Main monitoring dashboard system."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.dashboard_renderer = DashboardRenderer(self.metrics_collector)

        # Dashboard storage
        self.dashboards: dict[str, Dashboard] = {}

        # Create default dashboard
        self._create_default_dashboard()

        self.logger = get_logger(__name__)
        logger.info("MonitoringDashboard initialized")

    def start_monitoring(self):
        """Start all monitoring components."""
        self.system_monitor.start_monitoring()
        logger.info("Monitoring dashboard started")

    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.system_monitor.stop_monitoring()
        logger.info("Monitoring dashboard stopped")

    def register_dashboard(self, dashboard: Dashboard):
        """Register a new dashboard."""
        self.dashboards[dashboard.id] = dashboard
        logger.info(f"Registered dashboard: {dashboard.id}")

    def get_dashboard_data(self, dashboard_id: str) -> dict[str, Any] | None:
        """Get rendered dashboard data."""
        if dashboard_id not in self.dashboards:
            return None

        dashboard = self.dashboards[dashboard_id]
        return self.dashboard_renderer.render_dashboard(dashboard)

    def list_dashboards(self) -> list[dict[str, Any]]:
        """List all available dashboards."""
        return [
            {
                "id": dashboard.id,
                "name": dashboard.name,
                "description": dashboard.description,
                "widget_count": len(dashboard.widgets),
            }
            for dashboard in self.dashboards.values()
        ]

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: str = "gauge",
        labels: dict[str, str] | None = None,
    ):
        """Record a metric value."""
        if metric_type == "counter":
            self.metrics_collector.record_counter(name, value, labels)
        elif metric_type == "gauge":
            self.metrics_collector.record_gauge(name, value, labels)
        elif metric_type == "histogram":
            self.metrics_collector.record_histogram(name, value, labels)
        elif metric_type == "timer":
            self.metrics_collector.record_timer(name, value, labels)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get overall metrics summary."""
        metrics_list = self.metrics_collector.list_metrics()

        return {
            "total_metrics": len(metrics_list),
            "metrics_by_type": {
                metric_type.value: len(
                    [m for m in metrics_list if m["type"] == metric_type.value]
                )
                for metric_type in MetricType
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _create_default_dashboard(self):
        """Create default system monitoring dashboard."""
        widgets = [
            DashboardWidget(
                id="system_overview",
                title="System Overview",
                widget_type="gauge",
                metrics=["system.cpu_usage"],
                config={
                    "max_value": 100,
                    "unit": "%",
                    "thresholds": {"warning": 70, "critical": 90},
                },
                position={"x": 0, "y": 0, "width": 4, "height": 3},
            ),
            DashboardWidget(
                id="memory_usage",
                title="Memory Usage",
                widget_type="gauge",
                metrics=["system.memory_usage"],
                config={
                    "max_value": 100,
                    "unit": "%",
                    "thresholds": {"warning": 80, "critical": 95},
                },
                position={"x": 4, "y": 0, "width": 4, "height": 3},
            ),
            DashboardWidget(
                id="pipeline_status",
                title="Pipeline Status",
                widget_type="status",
                metrics=["pipeline.active_tasks", "pipeline.failed_tasks"],
                config={"thresholds": {"warning": 5, "critical": 10}},
                position={"x": 8, "y": 0, "width": 4, "height": 3},
            ),
            DashboardWidget(
                id="performance_chart",
                title="Performance Metrics",
                widget_type="line_chart",
                metrics=["system.cpu_usage", "system.memory_usage"],
                config={"duration_minutes": 60},
                position={"x": 0, "y": 3, "width": 12, "height": 4},
            ),
            DashboardWidget(
                id="metrics_table",
                title="Metrics Summary",
                widget_type="table",
                metrics=[
                    "pipeline.processing_time",
                    "dataset.quality_score",
                    "dataset.download_speed",
                ],
                position={"x": 0, "y": 7, "width": 12, "height": 3},
            ),
        ]

        default_dashboard = Dashboard(
            id="default",
            name="System Monitoring",
            description="Default system and pipeline monitoring dashboard",
            widgets=widgets,
            refresh_interval=30,
        )

        self.register_dashboard(default_dashboard)

    def export_dashboard_config(self, dashboard_id: str, output_path: str) -> bool:
        """Export dashboard configuration to file."""
        if dashboard_id not in self.dashboards:
            return False

        dashboard = self.dashboards[dashboard_id]

        config = {
            "id": dashboard.id,
            "name": dashboard.name,
            "description": dashboard.description,
            "refresh_interval": dashboard.refresh_interval,
            "auto_refresh": dashboard.auto_refresh,
            "widgets": [
                {
                    "id": widget.id,
                    "title": widget.title,
                    "widget_type": widget.widget_type,
                    "metrics": widget.metrics,
                    "config": widget.config,
                    "position": widget.position,
                }
                for widget in dashboard.widgets
            ],
        }

        try:
            with open(output_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Dashboard config exported: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export dashboard config: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize dashboard
    dashboard = MonitoringDashboard()
    dashboard.start_monitoring()

    # Simulate some metrics
    import random
    import time

    for i in range(10):
        # Record some sample metrics
        dashboard.record_metric("pipeline.active_tasks", random.randint(1, 10))
        dashboard.record_metric("pipeline.completed_tasks", i + 1, "counter")
        dashboard.record_metric("dataset.quality_score", random.uniform(0.7, 0.95))
        dashboard.record_metric("dataset.download_speed", random.uniform(5, 50))

        time.sleep(2)

    # Get dashboard data
    dashboard_data = dashboard.get_dashboard_data("default")

    # Get metrics summary
    summary = dashboard.get_metrics_summary()

    # Export dashboard config
    dashboard.export_dashboard_config("default", "dashboard_config.json")

    dashboard.stop_monitoring()

    # Clean up
    import os

    if os.path.exists("dashboard_config.json"):
        os.remove("dashboard_config.json")
