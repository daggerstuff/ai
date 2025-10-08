#!/usr/bin/env python3
"""
Real-Time Training Metrics Dashboard for Pixel LLM Training

This module provides a comprehensive real-time dashboard for monitoring
multi-objective Pixel training with live metrics visualization, component-wise
loss tracking, EQ progression, clinical accuracy, and interactive web interface.

Features:
- Live metrics streaming from training process
- Multi-objective loss component visualization
- EQ progression tracking across 5 domains
- Clinical accuracy monitoring (DSM-5/PDM-2)
- Persona switching performance metrics
- Interactive web dashboard with real-time updates
- Training anomaly detection and alerting
- Performance trend analysis and forecasting
"""

import asyncio
import json
import logging
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Union

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import redis
import websockets
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots


@dataclass
class TrainingMetrics:
    """Container for training metrics data"""

    timestamp: datetime
    step: int
    epoch: int
    total_loss: float
    component_losses: Dict[str, float]
    eq_scores: Dict[str, float]
    clinical_accuracy: Dict[str, float]
    persona_metrics: Dict[str, float]
    empathy_scores: Dict[str, float]
    conversational_quality: Dict[str, float]
    therapeutic_appropriateness: Dict[str, float]
    gradient_norms: Dict[str, float]
    learning_rates: Dict[str, float]
    memory_usage: Dict[str, float]
    throughput_metrics: Dict[str, float]


@dataclass
class DashboardConfig:
    """Configuration for the real-time dashboard"""

    update_interval: int = 1  # seconds
    max_history_points: int = 1000
    redis_host: str = "localhost"
    redis_port: int = 6379
    websocket_port: int = 8765
    dashboard_port: int = 8050
    enable_alerts: bool = True
    alert_thresholds: Union[Dict[str, float], None] = None

    def __post_init__(self):
        if not self.alert_thresholds:
            self.alert_thresholds = {
                "loss_spike": 2.0,  # 2x increase
                "gradient_norm": 10.0,  # gradient explosion
                "memory_usage": 0.95,  # 95% GPU memory
                "eq_regression": -0.1,  # 10% EQ score drop
                "clinical_accuracy_drop": -0.05,  # 5% accuracy drop
            }


class MetricsCollector:
    """Collects and manages training metrics data"""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.metrics_history: deque[TrainingMetrics] = deque(maxlen=config.max_history_points)
        self.redis_client = None
        self.logger = logging.getLogger(__name__)

        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=config.redis_host, port=config.redis_port, decode_responses=True
            )
            self.redis_client.ping()
            self.logger.info("Connected to Redis for metrics storage")
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")

    def add_metrics(self, metrics: TrainingMetrics) -> None:
        """Add new metrics to collection"""
        self.metrics_history.append(metrics)

        # Store in Redis if available
        if self.redis_client:
            try:
                metrics_dict = {
                    "timestamp": metrics.timestamp.isoformat(),
                    "step": metrics.step,
                    "epoch": metrics.epoch,
                    "total_loss": metrics.total_loss,
                    "component_losses": json.dumps(metrics.component_losses),
                    "eq_scores": json.dumps(metrics.eq_scores),
                    "clinical_accuracy": json.dumps(metrics.clinical_accuracy),
                    "persona_metrics": json.dumps(metrics.persona_metrics),
                    "gradient_norms": json.dumps(metrics.gradient_norms),
                    "learning_rates": json.dumps(metrics.learning_rates),
                    "memory_usage": json.dumps(metrics.memory_usage),
                    "throughput_metrics": json.dumps(metrics.throughput_metrics),
                }

                self.redis_client.hset(f"training_metrics:{metrics.step}", mapping=metrics_dict)

                # Keep only recent metrics in Redis
                self.redis_client.expire(f"training_metrics:{metrics.step}", 3600)  # 1 hour

            except Exception as e:
                self.logger.error(f"Failed to store metrics in Redis: {e}")

    def get_recent_metrics(self, last_n: int = 100) -> List[TrainingMetrics]:
        """Get recent metrics from history"""
        return list(self.metrics_history)[-last_n:]

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame for analysis"""
        if not self.metrics_history:
            return pd.DataFrame()

        data = []
        for metrics in self.metrics_history:
            row = {
                "timestamp": metrics.timestamp,
                "step": metrics.step,
                "epoch": metrics.epoch,
                "total_loss": metrics.total_loss,
                **metrics.component_losses,
                **{f"eq_{k}": v for k, v in metrics.eq_scores.items()},
                **{f"clinical_{k}": v for k, v in metrics.clinical_accuracy.items()},
                **{f"persona_{k}": v for k, v in metrics.persona_metrics.items()},
                **{f"grad_{k}": v for k, v in metrics.gradient_norms.items()},
                **{f"lr_{k}": v for k, v in metrics.learning_rates.items()},
                **{f"mem_{k}": v for k, v in metrics.memory_usage.items()},
                **{f"throughput_{k}": v for k, v in metrics.throughput_metrics.items()},
            }
            data.append(row)

        return pd.DataFrame(data)


class AnomalyDetector:
    """Detects training anomalies and generates alerts"""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.baseline_metrics: Dict[str, float] = {}
        self.alert_history: List[Dict[str, Any]] = []

    def update_baseline(self, metrics: TrainingMetrics) -> None:
        """Update baseline metrics for anomaly detection"""
        self.baseline_metrics["total_loss"] = metrics.total_loss
        self.baseline_metrics["eq_average"] = np.mean(list(metrics.eq_scores.values()))
        self.baseline_metrics["clinical_average"] = np.mean(
            list(metrics.clinical_accuracy.values())
        )
        self.baseline_metrics["gradient_norm"] = np.mean(list(metrics.gradient_norms.values()))
        self.baseline_metrics["memory_usage"] = max(metrics.memory_usage.values())

    def detect_anomalies(self, metrics: TrainingMetrics) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics"""
        anomalies = []

        if not self.baseline_metrics:
            self.update_baseline(metrics)
            return anomalies

        # Loss spike detection
        if (
            metrics.total_loss
            > self.baseline_metrics["total_loss"] * self.config.alert_thresholds["loss_spike"]
        ):
            anomalies.append(
                {
                    "type": "loss_spike",
                    "severity": "high",
                    "message": f"Loss spike detected: {metrics.total_loss:.4f} (baseline: {self.baseline_metrics['total_loss']:.4f})",
                    "timestamp": metrics.timestamp,
                    "step": metrics.step,
                }
            )

        # Gradient explosion detection
        current_grad_norm = np.mean(list(metrics.gradient_norms.values()))
        if current_grad_norm > self.config.alert_thresholds["gradient_norm"]:
            anomalies.append(
                {
                    "type": "gradient_explosion",
                    "severity": "critical",
                    "message": f"Gradient explosion detected: {current_grad_norm:.4f}",
                    "timestamp": metrics.timestamp,
                    "step": metrics.step,
                }
            )

        # Memory usage alert
        max_memory = max(metrics.memory_usage.values())
        if max_memory > self.config.alert_thresholds["memory_usage"]:
            anomalies.append(
                {
                    "type": "high_memory",
                    "severity": "medium",
                    "message": f"High memory usage: {max_memory:.2%}",
                    "timestamp": metrics.timestamp,
                    "step": metrics.step,
                }
            )

        # EQ regression detection
        current_eq = np.mean(list(metrics.eq_scores.values()))
        eq_change = current_eq - self.baseline_metrics["eq_average"]
        if eq_change < self.config.alert_thresholds["eq_regression"]:
            anomalies.append(
                {
                    "type": "eq_regression",
                    "severity": "medium",
                    "message": f"EQ regression detected: {eq_change:.4f}",
                    "timestamp": metrics.timestamp,
                    "step": metrics.step,
                }
            )

        # Clinical accuracy drop detection
        current_clinical = np.mean(list(metrics.clinical_accuracy.values()))
        clinical_change = current_clinical - self.baseline_metrics["clinical_average"]
        if clinical_change < self.config.alert_thresholds["clinical_accuracy_drop"]:
            anomalies.append(
                {
                    "type": "clinical_regression",
                    "severity": "high",
                    "message": f"Clinical accuracy drop: {clinical_change:.4f}",
                    "timestamp": metrics.timestamp,
                    "step": metrics.step,
                }
            )

        # Store anomalies
        for anomaly in anomalies:
            self.alert_history.append(anomaly)
            self.logger.warning(f"Anomaly detected: {anomaly['message']}")

        return anomalies


class VisualizationEngine:
    """Creates interactive visualizations for training metrics"""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff7f0e",
            "info": "#17a2b8",
        }

    def create_loss_components_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create component-wise loss visualization"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["Total Loss", "Language Loss", "EQ Loss", "Clinical Loss"],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        if df.empty:
            return fig

        # Total loss
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["total_loss"],
                name="Total Loss",
                line=dict(color=self.colors["primary"], width=2),
            ),
            row=1,
            col=1,
        )

        # Component losses
        loss_components = ["language_loss", "eq_loss", "clinical_loss"]
        component_colors = [
            self.colors["secondary"],
            self.colors["success"],
            self.colors["danger"],
        ]

        for i, (component, color) in enumerate(zip(loss_components, component_colors)):
            if component in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["step"],
                        y=df[component],
                        name=component.replace("_", " ").title(),
                        line=dict(color=color, width=2),
                    ),
                    row=1 + i // 2,
                    col=1 + i % 2,
                )

        fig.update_layout(title="Loss Components Over Time", showlegend=True, height=600)

        return fig

    def create_eq_progression_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create EQ progression visualization"""
        fig = go.Figure()

        if df.empty:
            return fig

        eq_domains = [
            "emotional_awareness",
            "empathy_recognition",
            "emotional_regulation",
            "social_cognition",
            "interpersonal_skills",
        ]
        colors = px.colors.qualitative.Set3[: len(eq_domains)]

        for domain, color in zip(eq_domains, colors):
            col_name = f"eq_{domain}"
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["step"],
                        y=df[col_name],
                        name=domain.replace("_", " ").title(),
                        line=dict(color=color, width=2),
                        mode="lines+markers",
                    )
                )

        fig.update_layout(
            title="EQ Domain Progression",
            xaxis_title="Training Step",
            yaxis_title="EQ Score",
            height=400,
            showlegend=True,
        )

        return fig

    def create_clinical_accuracy_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create clinical accuracy visualization"""
        fig = go.Figure()

        if df.empty:
            return fig

        clinical_metrics = [
            "dsm5_accuracy",
            "pdm2_accuracy",
            "therapeutic_appropriateness",
        ]
        colors = [
            self.colors["primary"],
            self.colors["success"],
            self.colors["warning"],
        ]

        for metric, color in zip(clinical_metrics, colors):
            col_name = f"clinical_{metric}"
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["step"],
                        y=df[col_name],
                        name=metric.replace("_", " ").title(),
                        line=dict(color=color, width=2),
                        mode="lines+markers",
                    )
                )

        fig.update_layout(
            title="Clinical Accuracy Metrics",
            xaxis_title="Training Step",
            yaxis_title="Accuracy",
            height=400,
            showlegend=True,
        )

        return fig

    def create_system_metrics_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create system performance metrics visualization"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "GPU Memory Usage",
                "Gradient Norms",
                "Learning Rates",
                "Throughput",
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        if df.empty:
            return fig

        # GPU Memory
        memory_cols = [col for col in df.columns if col.startswith("mem_")]
        for i, col in enumerate(memory_cols[:3]):  # Limit to 3 GPUs for display
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df["step"], y=df[col], name=f"GPU {i}", line=dict(width=2)),
                    row=1,
                    col=1,
                )

        # Gradient Norms
        grad_cols = [col for col in df.columns if col.startswith("grad_")]
        for col in grad_cols[:3]:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["step"],
                        y=df[col],
                        name=col.replace("grad_", "").title(),
                        line=dict(width=2),
                    ),
                    row=1,
                    col=2,
                )

        # Learning Rates
        lr_cols = [col for col in df.columns if col.startswith("lr_")]
        for col in lr_cols[:3]:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["step"],
                        y=df[col],
                        name=col.replace("lr_", "").title(),
                        line=dict(width=2),
                    ),
                    row=2,
                    col=1,
                )

        # Throughput
        throughput_cols = [col for col in df.columns if col.startswith("throughput_")]
        for col in throughput_cols[:3]:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["step"],
                        y=df[col],
                        name=col.replace("throughput_", "").title(),
                        line=dict(width=2),
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(title="System Performance Metrics", showlegend=False, height=600)

        return fig


class WebSocketServer:
    """WebSocket server for real-time metrics streaming"""

    def __init__(self, config: DashboardConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        self.clients = set()
        self.logger = logging.getLogger(__name__)

    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        self.logger.info(f"Client registered. Total clients: {len(self.clients)}")

        # Send recent metrics to new client
        recent_metrics = self.metrics_collector.get_recent_metrics(50)
        for metrics in recent_metrics:
            await self.send_metrics_to_client(websocket, metrics)

    async def unregister_client(self, websocket):
        """Unregister a WebSocket client"""
        self.clients.discard(websocket)
        self.logger.info(f"Client unregistered. Total clients: {len(self.clients)}")

    async def send_metrics_to_client(self, websocket, metrics: TrainingMetrics):
        """Send metrics to a specific client"""
        try:
            message = {
                "type": "metrics_update",
                "timestamp": metrics.timestamp.isoformat(),
                "step": metrics.step,
                "epoch": metrics.epoch,
                "total_loss": metrics.total_loss,
                "component_losses": metrics.component_losses,
                "eq_scores": metrics.eq_scores,
                "clinical_accuracy": metrics.clinical_accuracy,
                "persona_metrics": metrics.persona_metrics,
                "gradient_norms": metrics.gradient_norms,
                "learning_rates": metrics.learning_rates,
                "memory_usage": metrics.memory_usage,
                "throughput_metrics": metrics.throughput_metrics,
            }

            await websocket.send(json.dumps(message))

        except Exception as e:
            self.logger.error(f"Failed to send metrics to client: {e}")

    async def broadcast_metrics(self, metrics: TrainingMetrics):
        """Broadcast metrics to all connected clients"""
        if not self.clients:
            return

        disconnected = set()
        for client in self.clients:
            try:
                await self.send_metrics_to_client(client, metrics)
            except Exception as e:
                self.logger.error(f"Client disconnected: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        await self.register_client(websocket)
        try:
            async for _ in websocket:
                # Handle incoming messages if needed
                pass
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            await self.unregister_client(websocket)

    def start_server(self):
        """Start the WebSocket server"""
        return websockets.serve(self.handle_client, "localhost", self.config.websocket_port)


class TrainingDashboard:
    """Main dashboard application using Dash"""

    def __init__(
        self,
        config: DashboardConfig,
        metrics_collector: MetricsCollector,
        anomaly_detector: AnomalyDetector,
        visualization_engine: VisualizationEngine,
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        self.anomaly_detector = anomaly_detector
        self.viz_engine = visualization_engine
        self.logger = logging.getLogger(__name__)

        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "Pixel LLM Training Dashboard",
                                    className="text-center mb-4",
                                ),
                                html.Hr(),
                            ]
                        )
                    ]
                ),
                # Status cards
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Training Status",
                                                    className="card-title",
                                                ),
                                                html.H2(
                                                    id="training-status",
                                                    children="Initializing...",
                                                    className="text-primary",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Current Step",
                                                    className="card-title",
                                                ),
                                                html.H2(
                                                    id="current-step",
                                                    children="0",
                                                    className="text-info",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4("Total Loss", className="card-title"),
                                                html.H2(
                                                    id="total-loss",
                                                    children="N/A",
                                                    className="text-warning",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4("EQ Score", className="card-title"),
                                                html.H2(
                                                    id="eq-score",
                                                    children="N/A",
                                                    className="text-success",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                    ],
                    className="mb-4",
                ),
                # Alert section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Alert(
                                    id="alert-container",
                                    children="No alerts",
                                    color="success",
                                    dismissable=True,
                                    is_open=False,
                                )
                            ]
                        )
                    ],
                    className="mb-4",
                ),
                # Main plots
                dbc.Row(
                    [dbc.Col([dcc.Graph(id="loss-components-plot")], width=12)],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col([dcc.Graph(id="eq-progression-plot")], width=6),
                        dbc.Col([dcc.Graph(id="clinical-accuracy-plot")], width=6),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [dbc.Col([dcc.Graph(id="system-metrics-plot")], width=12)],
                    className="mb-4",
                ),
                # Auto-refresh interval
                dcc.Interval(
                    id="interval-component",
                    interval=self.config.update_interval * 1000,  # milliseconds
                    n_intervals=0,
                ),
            ],
            fluid=True,
        )

    def setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            [
                Output("training-status", "children"),
                Output("current-step", "children"),
                Output("total-loss", "children"),
                Output("eq-score", "children"),
                Output("loss-components-plot", "figure"),
                Output("eq-progression-plot", "figure"),
                Output("clinical-accuracy-plot", "figure"),
                Output("system-metrics-plot", "figure"),
                Output("alert-container", "children"),
                Output("alert-container", "color"),
                Output("alert-container", "is_open"),
            ],
            [Input("interval-component", "n_intervals")],
        )
        def update_dashboard(n):
            """Update all dashboard components"""
            df = self.metrics_collector.get_metrics_dataframe()

            if df.empty:
                return (
                    "Waiting for data...",
                    "0",
                    "N/A",
                    "N/A",
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    "No data available",
                    "info",
                    True,
                )

            # Get latest metrics
            latest = df.iloc[-1]

            # Calculate average EQ score
            eq_cols = [col for col in df.columns if col.startswith("eq_")]
            eq_score = latest[eq_cols].mean() if eq_cols else 0

            # Status
            status = "Training" if n > 0 else "Initializing"

            # Create plots
            loss_plot = self.viz_engine.create_loss_components_plot(df)
            eq_plot = self.viz_engine.create_eq_progression_plot(df)
            clinical_plot = self.viz_engine.create_clinical_accuracy_plot(df)
            system_plot = self.viz_engine.create_system_metrics_plot(df)

            # Check for recent alerts
            recent_alerts = [
                alert
                for alert in self.anomaly_detector.alert_history
                if alert["timestamp"] > datetime.now() - timedelta(minutes=5)
            ]

            if recent_alerts:
                alert_msg = f"{len(recent_alerts)} recent alerts detected"
                alert_color = "danger"
                alert_open = True
            else:
                alert_msg = "All systems normal"
                alert_color = "success"
                alert_open = False

            return (
                status,
                f"{latest['step']:,}",
                f"{latest['total_loss']:.4f}",
                f"{eq_score:.3f}",
                loss_plot,
                eq_plot,
                clinical_plot,
                system_plot,
                alert_msg,
                alert_color,
                alert_open,
            )

    def run(self):
        """Run the dashboard"""
        self.app.run_server(host="127.0.0.1", port=self.config.dashboard_port, debug=False)


class RealTimeDashboardManager:
    """Main manager for the real-time training dashboard"""

    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.logger = self._setup_logging()

        # Initialize components
        self.metrics_collector = MetricsCollector(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.viz_engine = VisualizationEngine(self.config)
        self.websocket_server = WebSocketServer(self.config, self.metrics_collector)
        self.dashboard = TrainingDashboard(
            self.config, self.metrics_collector, self.anomaly_detector, self.viz_engine
        )

        # Background tasks
        self.background_tasks = []
        self.running = False

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)

    def add_metrics(self, metrics: TrainingMetrics) -> None:
        """Add new training metrics"""
        self.metrics_collector.add_metrics(metrics)

        # Detect anomalies
        if self.config.enable_alerts:
            anomalies = self.anomaly_detector.detect_anomalies(metrics)
            if anomalies:
                self.logger.warning(f"Detected {len(anomalies)} anomalies")

        # Broadcast to WebSocket clients
        if self.running:
            asyncio.create_task(self.websocket_server.broadcast_metrics(metrics))

    def start(self):
        """Start the dashboard and all background services"""
        self.logger.info("Starting Real-Time Training Dashboard...")
        self.running = True

        # Start WebSocket server
        def start_websocket():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            start_server = self.websocket_server.start_server()
            loop.run_until_complete(start_server)
            loop.run_forever()

        websocket_thread = threading.Thread(target=start_websocket, daemon=True)
        websocket_thread.start()
        self.background_tasks.append(websocket_thread)

        self.logger.info(f"WebSocket server started on port {self.config.websocket_port}")

        # Start dashboard (blocking)
        self.logger.info(f"Starting dashboard on port {self.config.dashboard_port}")
        if self.running:
            self.dashboard.run()

    def stop(self):
        """Stop the dashboard and cleanup"""
        self.logger.info("Stopping Real-Time Training Dashboard...")
        self.running = False

        # Cleanup background tasks
        for task in self.background_tasks:
            if task.is_alive():
                task.join(timeout=1)

    def save_metrics_history(self, filepath: Union[str, Path]) -> None:
        """Save metrics history to file"""
        df = self.metrics_collector.get_metrics_dataframe()
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved metrics history to {filepath}")

    def load_metrics_history(self, filepath: Union[str, Path]) -> None:
        """Load metrics history from file"""
        try:
            df = pd.read_csv(filepath)
            # Convert back to TrainingMetrics objects
            for _, row in df.iterrows():
                metrics = TrainingMetrics(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    step=int(row["step"]),
                    epoch=int(row["epoch"]),
                    total_loss=float(row["total_loss"]),
                    component_losses={
                        k: v
                        for k, v in row.items()
                        if not k.startswith(
                            (
                                "timestamp",
                                "step",
                                "epoch",
                                "total_loss",
                                "eq_",
                                "clinical_",
                                "persona_",
                                "empathy_",
                                "conversational_quality_",
                                "therapeutic_appropriateness_",
                                "grad_",
                                "lr_",
                                "mem_",
                                "throughput_",
                            )
                        )
                    },
                    eq_scores={
                        k.replace("eq_", ""): v for k, v in row.items() if k.startswith("eq_")
                    },
                    clinical_accuracy={
                        k.replace("clinical_", ""): v
                        for k, v in row.items()
                        if k.startswith("clinical_")
                    },
                    persona_metrics={
                        k.replace("persona_", ""): v
                        for k, v in row.items()
                        if k.startswith("persona_")
                    },
                    empathy_scores={
                        k.replace("empathy_", ""): v
                        for k, v in row.items()
                        if k.startswith("empathy_")
                    },
                    conversational_quality={
                        k.replace("conversational_quality_", ""): v
                        for k, v in row.items()
                        if k.startswith("conversational_quality_")
                    },
                    therapeutic_appropriateness={
                        k.replace("therapeutic_appropriateness_", ""): v
                        for k, v in row.items()
                        if k.startswith("therapeutic_appropriateness_")
                    },
                    gradient_norms={
                        k.replace("grad_", ""): v for k, v in row.items() if k.startswith("grad_")
                    },
                    learning_rates={
                        k.replace("lr_", ""): v for k, v in row.items() if k.startswith("lr_")
                    },
                    memory_usage={
                        k.replace("mem_", ""): v for k, v in row.items() if k.startswith("mem_")
                    },
                    throughput_metrics={
                        k.replace("throughput_", ""): v
                        for k, v in row.items()
                        if k.startswith("throughput_")
                    },
                )
                self.metrics_collector.add_metrics(metrics)

            self.logger.info(f"Loaded {len(df)} metrics from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load metrics history: {e}")


def create_sample_metrics(step: int) -> TrainingMetrics:
    """Create sample metrics for testing"""
    # Use secrets module for cryptographically secure random numbers
    rng = secrets.SystemRandom()

    base_loss = 2.5 + rng.uniform(-0.5, 0.5)

    return TrainingMetrics(
        timestamp=datetime.now(),
        step=step,
        epoch=step // 1000,
        total_loss=base_loss,
        component_losses={
            "language_loss": base_loss * 0.6 + rng.uniform(-0.1, 0.1),
            "eq_loss": base_loss * 0.2 + rng.uniform(-0.05, 0.05),
            "clinical_loss": base_loss * 0.15 + rng.uniform(-0.05, 0.05),
            "persona_loss": base_loss * 0.05 + rng.uniform(-0.02, 0.02),
        },
        eq_scores={
            "emotional_awareness": 0.7 + rng.uniform(-0.1, 0.1),
            "empathy_recognition": 0.75 + rng.uniform(-0.1, 0.1),
            "emotional_regulation": 0.68 + rng.uniform(-0.1, 0.1),
            "social_cognition": 0.72 + rng.uniform(-0.1, 0.1),
            "interpersonal_skills": 0.69 + rng.uniform(-0.1, 0.1),
        },
        clinical_accuracy={
            "dsm5_accuracy": 0.82 + rng.uniform(-0.05, 0.05),
            "pdm2_accuracy": 0.78 + rng.uniform(-0.05, 0.05),
            "therapeutic_appropriateness": 0.85 + rng.uniform(-0.05, 0.05),
        },
        persona_metrics={
            "switching_accuracy": 0.92 + rng.uniform(-0.03, 0.03),
            "consistency_score": 0.88 + rng.uniform(-0.05, 0.05),
            "transition_smoothness": 0.85 + rng.uniform(-0.05, 0.05),
        },
        empathy_scores={
            "empathy_vs_simulation": 0.8 + rng.uniform(-0.1, 0.1),
            "progressive_empathy": 0.75 + rng.uniform(-0.1, 0.1),
            "empathy_consistency": 0.7 + rng.uniform(-0.1, 0.1),
            "empathy_calibration": 0.78 + rng.uniform(-0.1, 0.1),
            "empathy_progression": 0.76 + rng.uniform(-0.1, 0.1),
        },
        conversational_quality={
            "coherence": 0.85 + rng.uniform(-0.05, 0.05),
            "authenticity": 0.88 + rng.uniform(-0.05, 0.05),
            "naturalness": 0.87 + rng.uniform(-0.05, 0.05),
            "therapeutic_flow": 0.86 + rng.uniform(-0.05, 0.05),
            "quality_benchmark": 0.84 + rng.uniform(-0.05, 0.05),
        },
        therapeutic_appropriateness={
            "intervention_appropriateness": 0.9 + rng.uniform(-0.05, 0.05),
            "therapeutic_boundary": 0.92 + rng.uniform(-0.05, 0.05),
            "crisis_handling": 0.91 + rng.uniform(-0.05, 0.05),
            "ethical_compliance": 0.93 + rng.uniform(-0.05, 0.05),
            "therapeutic_effectiveness": 0.89 + rng.uniform(-0.05, 0.05),
        },
        gradient_norms={
            "total_norm": 2.5 + rng.uniform(-0.5, 0.5),
            "eq_head_norm": 1.8 + rng.uniform(-0.3, 0.3),
            "clinical_head_norm": 2.2 + rng.uniform(-0.4, 0.4),
        },
        learning_rates={
            "base_lr": 1e-4,
            "eq_head_lr": 2e-4,
            "clinical_head_lr": 1.5e-4,
        },
        memory_usage={
            "gpu_0": 0.85 + rng.uniform(-0.1, 0.1),
            "gpu_1": 0.82 + rng.uniform(-0.1, 0.1),
            "gpu_2": 0.87 + rng.uniform(-0.1, 0.1),
        },
        throughput_metrics={
            "samples_per_second": 150 + rng.uniform(-20, 20),
            "tokens_per_second": 12000 + rng.uniform(-1000, 1000),
        },
    )


if __name__ == "__main__":
    # Example usage and testing
    config = DashboardConfig(update_interval=2, dashboard_port=8050, websocket_port=8765)

    dashboard_manager = RealTimeDashboardManager(config)

    # Add some sample data for testing
    for i in range(100):
        metrics = create_sample_metrics(i * 10)
        dashboard_manager.add_metrics(metrics)
        time.sleep(0.01)  # Small delay to simulate real training

    # Start dashboard
    try:
        dashboard_manager.start()
    except KeyboardInterrupt:
        dashboard_manager.stop()
