#!/usr/bin/env python3
"""
Real-Time Streaming Dashboard

This module provides a comprehensive dashboard for monitoring real-time streaming
data processing, including metrics visualization, alert management, and system health.
"""

import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Web dashboard imports
try:
    from flask import Flask, jsonify, render_template, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from real_time_processor import StreamingProcessor


@dataclass
class DashboardMetrics:
    """Dashboard-specific metrics."""
    total_events: int = 0
    events_per_second: float = 0.0
    average_quality: float = 0.0
    quality_distribution: dict[str, int] = None
    error_rate: float = 0.0
    active_sources: int = 0
    system_health: str = "healthy"  # healthy, warning, critical
    alerts: list[dict[str, Any]] = None
    processing_latency: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

    def __post_init__(self):
        if self.quality_distribution is None:
            self.quality_distribution = {"high": 0, "medium": 0, "low": 0}
        if self.alerts is None:
            self.alerts = []


class StreamingDashboard:
    """Real-time streaming dashboard with web interface."""

    def __init__(self, processor: StreamingProcessor, port: int = 5000):
        self.processor = processor
        self.port = port
        self.metrics_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.dashboard_metrics = DashboardMetrics()

        # Flask app setup
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self.app.config["SECRET_KEY"] = "streaming_dashboard_secret"
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")
            self._setup_routes()

        # Monitoring thread
        self.monitoring_thread = None
        self.is_monitoring = False

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _setup_routes(self):
        """Setup Flask routes for the dashboard."""

        @self.app.route("/")
        def dashboard():
            """Main dashboard page."""
            return render_template("streaming_dashboard.html")

        @self.app.route("/api/metrics")
        def get_metrics():
            """Get current metrics."""
            return jsonify(asdict(self.dashboard_metrics))

        @self.app.route("/api/metrics/history")
        def get_metrics_history():
            """Get metrics history."""
            return jsonify(list(self.metrics_history))

        @self.app.route("/api/alerts")
        def get_alerts():
            """Get current alerts."""
            return jsonify(list(self.alerts))

        @self.app.route("/api/sources")
        def get_sources():
            """Get data source information."""
            processor_metrics = self.processor.get_metrics()
            return jsonify(processor_metrics.get("source_metrics", {}))

        @self.app.route("/api/system/health")
        def get_system_health():
            """Get system health status."""
            return jsonify({
                "status": self.dashboard_metrics.system_health,
                "timestamp": datetime.now().isoformat(),
                "uptime": self._get_uptime(),
                "memory_usage": self.dashboard_metrics.memory_usage,
                "cpu_usage": self.dashboard_metrics.cpu_usage
            })

        @self.socketio.on("connect")
        def handle_connect():
            """Handle client connection."""
            self.logger.info("Dashboard client connected")
            emit("status", {"message": "Connected to streaming dashboard"})

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Handle client disconnection."""
            self.logger.info("Dashboard client disconnected")

    def start_monitoring(self):
        """Start the monitoring thread."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("Started dashboard monitoring")

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Stopped dashboard monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Update metrics
                self._update_metrics()

                # Check for alerts
                self._check_alerts()

                # Broadcast updates to connected clients
                if FLASK_AVAILABLE:
                    self.socketio.emit("metrics_update", asdict(self.dashboard_metrics))

                # Sleep for update interval
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")

    def _update_metrics(self):
        """Update dashboard metrics."""
        try:
            # Get processor metrics
            processor_metrics = self.processor.get_metrics()
            global_metrics = processor_metrics.get("global_metrics", {})

            # Update dashboard metrics
            self.dashboard_metrics.total_events = global_metrics.get("events_processed", 0)
            self.dashboard_metrics.events_per_second = global_metrics.get("events_per_second", 0.0)
            self.dashboard_metrics.processing_latency = global_metrics.get("average_processing_time", 0.0)

            # Calculate average quality
            quality_scores = global_metrics.get("quality_scores", [])
            if quality_scores:
                self.dashboard_metrics.average_quality = statistics.mean(quality_scores)

                # Update quality distribution
                high_count = sum(1 for q in quality_scores if q >= 0.8)
                medium_count = sum(1 for q in quality_scores if 0.6 <= q < 0.8)
                low_count = sum(1 for q in quality_scores if q < 0.6)

                self.dashboard_metrics.quality_distribution = {
                    "high": high_count,
                    "medium": medium_count,
                    "low": low_count
                }

            # Calculate error rate
            total_events = self.dashboard_metrics.total_events
            error_count = global_metrics.get("error_count", 0)
            if total_events > 0:
                self.dashboard_metrics.error_rate = (error_count / total_events) * 100

            # Count active sources
            source_metrics = processor_metrics.get("source_metrics", {})
            self.dashboard_metrics.active_sources = sum(
                1 for metrics in source_metrics.values()
                if metrics.get("active_connections", 0) > 0
            )

            # Determine system health
            self.dashboard_metrics.system_health = self._calculate_system_health()

            # Add to history
            self.metrics_history.append({
                "timestamp": datetime.now().isoformat(),
                "events_per_second": self.dashboard_metrics.events_per_second,
                "average_quality": self.dashboard_metrics.average_quality,
                "error_rate": self.dashboard_metrics.error_rate,
                "processing_latency": self.dashboard_metrics.processing_latency
            })

        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    def _calculate_system_health(self) -> str:
        """Calculate overall system health."""
        # Health criteria
        if self.dashboard_metrics.error_rate > 10:
            return "critical"
        if (self.dashboard_metrics.error_rate > 5 or
              self.dashboard_metrics.processing_latency > 5.0 or
              self.dashboard_metrics.average_quality < 0.5):
            return "warning"
        return "healthy"

    def _check_alerts(self):
        """Check for alert conditions."""
        current_time = datetime.now()

        # High error rate alert
        if self.dashboard_metrics.error_rate > 5:
            alert = {
                "id": f"error_rate_{int(time.time())}",
                "type": "error_rate",
                "severity": "warning" if self.dashboard_metrics.error_rate < 10 else "critical",
                "message": f"High error rate: {self.dashboard_metrics.error_rate:.2f}%",
                "timestamp": current_time.isoformat(),
                "data": {"error_rate": self.dashboard_metrics.error_rate}
            }
            self._add_alert(alert)

        # Low quality alert
        if self.dashboard_metrics.average_quality < 0.5:
            alert = {
                "id": f"low_quality_{int(time.time())}",
                "type": "quality",
                "severity": "warning",
                "message": f"Low average quality: {self.dashboard_metrics.average_quality:.3f}",
                "timestamp": current_time.isoformat(),
                "data": {"average_quality": self.dashboard_metrics.average_quality}
            }
            self._add_alert(alert)

        # High latency alert
        if self.dashboard_metrics.processing_latency > 5.0:
            alert = {
                "id": f"high_latency_{int(time.time())}",
                "type": "latency",
                "severity": "warning",
                "message": f"High processing latency: {self.dashboard_metrics.processing_latency:.3f}s",
                "timestamp": current_time.isoformat(),
                "data": {"processing_latency": self.dashboard_metrics.processing_latency}
            }
            self._add_alert(alert)

        # No active sources alert
        if self.dashboard_metrics.active_sources == 0:
            alert = {
                "id": f"no_sources_{int(time.time())}",
                "type": "sources",
                "severity": "critical",
                "message": "No active data sources",
                "timestamp": current_time.isoformat(),
                "data": {"active_sources": 0}
            }
            self._add_alert(alert)

    def _add_alert(self, alert: dict[str, Any]):
        """Add an alert to the queue."""
        # Check if similar alert already exists (avoid spam)
        existing_alert = None
        for existing in self.alerts:
            if (existing["type"] == alert["type"] and
                existing["severity"] == alert["severity"]):
                existing_alert = existing
                break

        if not existing_alert:
            self.alerts.append(alert)
            self.dashboard_metrics.alerts = list(self.alerts)

            # Broadcast alert to connected clients
            if FLASK_AVAILABLE:
                self.socketio.emit("new_alert", alert)

            self.logger.warning(f"Alert: {alert['message']}")

    def _get_uptime(self) -> str:
        """Get system uptime."""
        # This is a placeholder - in a real system, you'd track actual uptime
        return "Running"

    def run_dashboard(self, debug: bool = False):
        """Run the web dashboard."""
        if not FLASK_AVAILABLE:
            self.logger.error("Flask not available. Cannot run web dashboard.")
            return

        self.start_monitoring()

        try:
            self.logger.info(f"Starting dashboard on port {self.port}")
            self.socketio.run(self.app, host="0.0.0.0", port=self.port, debug=debug)
        except KeyboardInterrupt:
            self.logger.info("Dashboard interrupted by user")
        finally:
            self.stop_monitoring()


# HTML template for the dashboard
DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Streaming Data Processing Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-value {
            font-weight: bold;
            font-size: 1.2em;
        }
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .alert {
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            border-left: 4px solid;
        }
        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
        .alert-critical {
            background-color: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 Real-Time Streaming Data Processing Dashboard</h1>
        <p>Monitoring therapeutic conversation processing in real-time</p>
    </div>

    <div class="dashboard">
        <!-- System Health Card -->
        <div class="card">
            <h3>🏥 System Health</h3>
            <div class="metric">
                <span>Status:</span>
                <span id="system-status" class="metric-value">
                    <span class="status-indicator"></span>
                    <span id="status-text">Loading...</span>
                </span>
            </div>
            <div class="metric">
                <span>Active Sources:</span>
                <span id="active-sources" class="metric-value">0</span>
            </div>
            <div class="metric">
                <span>Error Rate:</span>
                <span id="error-rate" class="metric-value">0.00%</span>
            </div>
            <div class="metric">
                <span>Processing Latency:</span>
                <span id="processing-latency" class="metric-value">0.000s</span>
            </div>
        </div>

        <!-- Processing Metrics Card -->
        <div class="card">
            <h3>📊 Processing Metrics</h3>
            <div class="metric">
                <span>Total Events:</span>
                <span id="total-events" class="metric-value">0</span>
            </div>
            <div class="metric">
                <span>Events/Second:</span>
                <span id="events-per-second" class="metric-value">0.00</span>
            </div>
            <div class="metric">
                <span>Average Quality:</span>
                <span id="average-quality" class="metric-value">0.000</span>
            </div>
        </div>

        <!-- Quality Distribution Card -->
        <div class="card">
            <h3>🎯 Quality Distribution</h3>
            <div class="metric">
                <span>High Quality (≥0.8):</span>
                <span id="quality-high" class="metric-value status-healthy">0</span>
            </div>
            <div class="metric">
                <span>Medium Quality (0.6-0.8):</span>
                <span id="quality-medium" class="metric-value status-warning">0</span>
            </div>
            <div class="metric">
                <span>Low Quality (<0.6):</span>
                <span id="quality-low" class="metric-value status-critical">0</span>
            </div>
        </div>

        <!-- Alerts Card -->
        <div class="card">
            <h3>🚨 Active Alerts</h3>
            <div id="alerts-container">
                <p>No active alerts</p>
            </div>
        </div>

        <!-- Performance Chart -->
        <div class="card" style="grid-column: 1 / -1;">
            <h3>📈 Performance Trends</h3>
            <div class="chart-container">
                <canvas id="performance-chart"></canvas>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();

        // Chart setup
        const ctx = document.getElementById('performance-chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Events/Second',
                        data: [],
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Average Quality',
                        data: [],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Events/Second'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Quality Score'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });

        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to dashboard');
        });

        socket.on('metrics_update', function(metrics) {
            updateMetrics(metrics);
        });

        socket.on('new_alert', function(alert) {
            addAlert(alert);
        });

        // Update metrics display
        function updateMetrics(metrics) {
            // System health
            const statusElement = document.getElementById('status-text');
            const statusIndicator = document.querySelector('.status-indicator');
            statusElement.textContent = metrics.system_health;
            statusElement.className = `metric-value status-${metrics.system_health}`;
            statusIndicator.className = `status-indicator status-${metrics.system_health}`;
            statusIndicator.style.backgroundColor = getStatusColor(metrics.system_health);

            // Update metrics
            document.getElementById('active-sources').textContent = metrics.active_sources;
            document.getElementById('error-rate').textContent = metrics.error_rate.toFixed(2) + '%';
            document.getElementById('processing-latency').textContent = metrics.processing_latency.toFixed(3) + 's';
            document.getElementById('total-events').textContent = metrics.total_events.toLocaleString();
            document.getElementById('events-per-second').textContent = metrics.events_per_second.toFixed(2);
            document.getElementById('average-quality').textContent = metrics.average_quality.toFixed(3);

            // Quality distribution
            document.getElementById('quality-high').textContent = metrics.quality_distribution.high;
            document.getElementById('quality-medium').textContent = metrics.quality_distribution.medium;
            document.getElementById('quality-low').textContent = metrics.quality_distribution.low;

            // Update chart
            updateChart(metrics);
        }

        function updateChart(metrics) {
            const now = new Date().toLocaleTimeString();

            // Add new data point
            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(metrics.events_per_second);
            chart.data.datasets[1].data.push(metrics.average_quality);

            // Keep only last 20 points
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
                chart.data.datasets[1].data.shift();
            }

            chart.update('none');
        }

        function addAlert(alert) {
            const container = document.getElementById('alerts-container');

            // Clear "no alerts" message
            if (container.innerHTML.includes('No active alerts')) {
                container.innerHTML = '';
            }

            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${alert.severity}`;
            alertDiv.innerHTML = `
                <strong>${alert.type.toUpperCase()}:</strong> ${alert.message}
                <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
            `;

            container.insertBefore(alertDiv, container.firstChild);

            // Keep only last 5 alerts
            while (container.children.length > 5) {
                container.removeChild(container.lastChild);
            }
        }

        function getStatusColor(status) {
            switch (status) {
                case 'healthy': return '#28a745';
                case 'warning': return '#ffc107';
                case 'critical': return '#dc3545';
                default: return '#6c757d';
            }
        }

        // Initial data load
        fetch('/api/metrics')
            .then(response => response.json())
            .then(metrics => updateMetrics(metrics))
            .catch(error => console.error('Error loading initial metrics:', error));
    </script>
</body>
</html>
"""

# Create templates directory and save HTML template
def create_dashboard_template():
    """Create the dashboard HTML template."""
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)

    template_path = templates_dir / "streaming_dashboard.html"
    with open(template_path, "w") as f:
        f.write(DASHBOARD_HTML_TEMPLATE)



if __name__ == "__main__":
    # Create dashboard template
    create_dashboard_template()

    # Example usage
    from real_time_processor import FileWatcherDataSource, StreamingProcessor

    # Create processor
    processor = StreamingProcessor()

    # Add data source
    file_source = FileWatcherDataSource(
        source_id="file_watcher",
        config={
            "directory": "data/streaming",
            "patterns": ["*.jsonl", "*.json"]
        }
    )
    processor.add_data_source(file_source)

    # Create and run dashboard
    dashboard = StreamingDashboard(processor, port=5000)


    dashboard.run_dashboard(debug=True)
