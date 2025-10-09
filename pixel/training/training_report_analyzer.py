#!/usr/bin/env python3
"""
Comprehensive Training Reporting and Analysis System

This module provides comprehensive training reporting and analysis capabilities for the Pixel LLM training system.
It aggregates data from all monitoring components (dashboard, validation, anomaly detection, checkpoints)
and generates detailed reports with visualizations and insights.

Features:
- Multi-objective training progress analysis
- Emotional intelligence progression tracking
- Clinical accuracy assessment
- Anomaly pattern analysis
- Checkpoint performance comparison
- Automated report generation with visualizations
- Export capabilities (PDF, HTML, JSON)
- Trend analysis and forecasting
- Performance benchmarking
"""

import json
import logging
import sqlite3
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from jinja2 import Template
from matplotlib.backends.backend_pdf import PdfPages
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


@dataclass
class ReportConfig:
    """Configuration for training report generation."""

    # Data sources
    metrics_db_path: str = "training_metrics.db"
    validation_db_path: str = "validation_results.db"
    anomaly_db_path: str = "anomaly_alerts.db"
    checkpoint_dir: str = "checkpoints/"

    # Report settings
    report_name: str = "training_report"
    output_dir: str = "reports/"
    include_visualizations: bool = True
    include_anomaly_analysis: bool = True
    include_checkpoint_analysis: bool = True
    include_trends: bool = True

    # Time range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    last_n_days: Optional[int] = 7

    # Export formats
    export_pdf: bool = True
    export_html: bool = True
    export_json: bool = True

    # Analysis settings
    trend_window: int = 100  # Steps for moving average
    anomaly_threshold: float = 2.0  # Standard deviations
    benchmark_comparison: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.end_date is None:
            self.end_date = datetime.now()

        if self.start_date is None and self.last_n_days:
            self.start_date = self.end_date - timedelta(days=self.last_n_days)

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingMetrics:
    """Training metrics data structure."""

    step: int
    epoch: int
    timestamp: datetime
    total_loss: float
    eq_loss: float
    clinical_loss: float
    persona_loss: float
    eq_score: float
    clinical_accuracy: float
    persona_consistency: float
    learning_rate: float
    gpu_memory_used: float
    throughput_tokens_per_sec: float
    gradient_norm: float


@dataclass
class ValidationResult:
    """Validation result data structure."""

    step: int
    timestamp: datetime
    validation_loss: float
    eq_validation_score: float
    clinical_validation_accuracy: float
    persona_validation_consistency: float
    early_stopping_patience: int
    best_score: float
    improved: bool


@dataclass
class AnomalyReport:
    """Anomaly detection report data structure."""

    timestamp: datetime
    anomaly_type: str
    severity: str
    description: str
    affected_metrics: List[str]
    remediation_suggestions: List[str]
    resolved: bool


@dataclass
class CheckpointInfo:
    """Checkpoint information data structure."""

    step: int
    timestamp: datetime
    file_path: str
    file_size_mb: float
    validation_score: float
    is_best: bool
    save_reason: str


@dataclass
class TrainingReport:
    """Comprehensive training report data structure."""

    # Report metadata
    report_id: str
    generation_time: datetime
    time_range: Tuple[datetime, datetime]
    total_training_time: timedelta

    # Summary statistics
    total_steps: int
    total_epochs: int
    final_metrics: TrainingMetrics
    best_validation: ValidationResult

    # Progress analysis
    loss_progression: Dict[str, Any]
    eq_progression: Dict[str, Any]
    clinical_progression: Dict[str, Any]
    persona_progression: Dict[str, Any]

    # Anomaly analysis
    anomaly_summary: Dict[str, int]
    critical_anomalies: List[AnomalyReport]
    anomaly_patterns: Dict[str, Any]

    # Checkpoint analysis
    checkpoint_summary: Dict[str, Any]
    best_checkpoints: List[CheckpointInfo]

    # Performance analysis
    training_efficiency: Dict[str, float]
    resource_utilization: Dict[str, float]
    trend_analysis: Dict[str, Any]

    # Recommendations
    recommendations: List[str]
    next_steps: List[str]


class DatabaseConnector:
    """Database connection and query utilities."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None

    def __enter__(self):
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()

    def query_to_dataframe(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute query and return results as DataFrame."""
        with self as conn:
            return pd.read_sql_query(query, conn, params=params)


class MetricsAnalyzer:
    """Training metrics analysis utilities."""

    @staticmethod
    def calculate_progression(
        data: pd.DataFrame, metric_col: str, window: int = 100
    ) -> Dict[str, Any]:
        """Calculate progression statistics for a metric."""
        if data.empty:
            return {}

        values = data[metric_col].values
        steps = data["step"].values

        # Calculate trends
        moving_avg = pd.Series(values).rolling(window=min(window, len(values))).mean()

        # Calculate improvement rate
        if len(values) > 1:
            improvement_rate = (values[-1] - values[0]) / len(values)
        else:
            improvement_rate = 0.0

        # Calculate volatility
        volatility = np.std(values) if len(values) > 1 else 0.0

        # Identify trends
        if len(moving_avg) > 2:
            recent_trend = np.polyfit(range(len(moving_avg[-10:])), moving_avg[-10:], 1)[0]
        else:
            recent_trend = 0.0

        return {
            "initial_value": float(values[0]) if len(values) > 0 else 0.0,
            "final_value": float(values[-1]) if len(values) > 0 else 0.0,
            "best_value": float(np.max(values)) if len(values) > 0 else 0.0,
            "worst_value": float(np.min(values)) if len(values) > 0 else 0.0,
            "improvement_rate": float(improvement_rate),
            "volatility": float(volatility),
            "recent_trend": float(recent_trend),
            "moving_average": moving_avg.tolist(),
            "total_steps": len(values),
        }

    @staticmethod
    def detect_plateaus(
        data: pd.DataFrame,
        metric_col: str,
        plateau_threshold: float = 0.001,
        min_length: int = 50,
    ) -> List[Dict]:
        """Detect plateau periods in training metrics."""
        if data.empty or len(data) < min_length:
            return []

        values = data[metric_col].values
        steps = data["step"].values

        plateaus = []
        plateau_start = None

        for i in range(1, len(values)):
            change = abs(values[i] - values[i - 1])

            if change < plateau_threshold:
                if plateau_start is None:
                    plateau_start = i - 1
            else:
                if plateau_start is not None and (i - plateau_start) >= min_length:
                    plateaus.append(
                        {
                            "start_step": int(steps[plateau_start]),
                            "end_step": int(steps[i - 1]),
                            "length": i - plateau_start,
                            "value": float(np.mean(values[plateau_start:i])),
                        }
                    )
                plateau_start = None

        # Check for plateau at the end
        if plateau_start is not None and (len(values) - plateau_start) >= min_length:
            plateaus.append(
                {
                    "start_step": int(steps[plateau_start]),
                    "end_step": int(steps[-1]),
                    "length": len(values) - plateau_start,
                    "value": float(np.mean(values[plateau_start:])),
                }
            )

        return plateaus


class VisualizationEngine:
    """Training visualization utilities."""

    def __init__(self, style: str = "plotly_white"):
        self.style = style
        pio.templates.default = style

        # Set color palette
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff8c00",
            "info": "#17a2b8",
        }

    def create_metrics_overview(self, metrics_df: pd.DataFrame) -> go.Figure:
        """Create comprehensive metrics overview plot."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Loss Progression",
                "EQ Score Progression",
                "Clinical Accuracy",
                "Persona Consistency",
            ],
            vertical_spacing=0.08,
        )

        # Loss progression
        fig.add_trace(
            go.Scatter(
                x=metrics_df["step"],
                y=metrics_df["total_loss"],
                name="Total Loss",
                line=dict(color=self.colors["primary"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=metrics_df["step"],
                y=metrics_df["eq_loss"],
                name="EQ Loss",
                line=dict(color=self.colors["secondary"]),
            ),
            row=1,
            col=1,
        )

        # EQ Score progression
        fig.add_trace(
            go.Scatter(
                x=metrics_df["step"],
                y=metrics_df["eq_score"],
                name="EQ Score",
                line=dict(color=self.colors["success"]),
            ),
            row=1,
            col=2,
        )

        # Clinical accuracy
        fig.add_trace(
            go.Scatter(
                x=metrics_df["step"],
                y=metrics_df["clinical_accuracy"],
                name="Clinical Accuracy",
                line=dict(color=self.colors["info"]),
            ),
            row=2,
            col=1,
        )

        # Persona consistency
        fig.add_trace(
            go.Scatter(
                x=metrics_df["step"],
                y=metrics_df["persona_consistency"],
                name="Persona Consistency",
                line=dict(color=self.colors["warning"]),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(title="Training Metrics Overview", showlegend=False, height=600)

        return fig

    def create_anomaly_timeline(self, anomalies: List[AnomalyReport]) -> go.Figure:
        """Create anomaly timeline visualization."""
        if not anomalies:
            fig = go.Figure()
            fig.add_annotation(
                text="No anomalies detected",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        df = pd.DataFrame([asdict(a) for a in anomalies])

        # Create color mapping for severity
        severity_colors = {
            "low": self.colors["info"],
            "medium": self.colors["warning"],
            "high": self.colors["secondary"],
            "critical": self.colors["danger"],
        }

        fig = go.Figure()

        for severity in df["severity"].unique():
            severity_data = df[df["severity"] == severity]
            fig.add_trace(
                go.Scatter(
                    x=severity_data["timestamp"],
                    y=severity_data["anomaly_type"],
                    mode="markers",
                    name=f"{severity.title()} Severity",
                    marker=dict(color=severity_colors[severity], size=10, symbol="circle"),
                    text=severity_data["description"],
                    hovertemplate="<b>%{y}</b><br>%{text}<br>%{x}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Training Anomalies Timeline",
            xaxis_title="Time",
            yaxis_title="Anomaly Type",
            showlegend=True,
        )

        return fig

    def create_checkpoint_analysis(self, checkpoints: List[CheckpointInfo]) -> go.Figure:
        """Create checkpoint analysis visualization."""
        if not checkpoints:
            fig = go.Figure()
            fig.add_annotation(
                text="No checkpoint data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        df = pd.DataFrame([asdict(c) for c in checkpoints])

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Checkpoint Performance", "File Size Analysis"],
        )

        # Performance over time
        colors = ["red" if is_best else "blue" for is_best in df["is_best"]]
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["validation_score"],
                mode="markers+lines",
                name="Validation Score",
                marker=dict(color=colors, size=8),
                text=df["save_reason"],
                hovertemplate="<b>Step %{x}</b><br>Score: %{y:.4f}<br>%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # File size analysis
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["file_size_mb"],
                mode="markers+lines",
                name="File Size (MB)",
                marker=dict(color=self.colors["secondary"]),
            ),
            row=1,
            col=2,
        )

        fig.update_layout(title="Checkpoint Analysis", showlegend=False)

        return fig


class TrainingReportAnalyzer:
    """Main training report analyzer class."""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.db_metrics = DatabaseConnector(config.metrics_db_path)
        self.db_validation = DatabaseConnector(config.validation_db_path)
        self.db_anomaly = DatabaseConnector(config.anomaly_db_path)
        self.visualization_engine = VisualizationEngine()

        logger.info(f"Initialized TrainingReportAnalyzer with config: {config.report_name}")

    def load_training_metrics(self) -> pd.DataFrame:
        """Load training metrics from database."""
        query = """
        SELECT * FROM training_metrics 
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY step
        """

        try:
            df = self.db_metrics.query_to_dataframe(
                query, (self.config.start_date, self.config.end_date)
            )
            logger.info(f"Loaded {len(df)} training metric records")
            return df
        except Exception as e:
            logger.warning(f"Could not load training metrics: {e}")
            return pd.DataFrame()

    def load_validation_results(self) -> pd.DataFrame:
        """Load validation results from database."""
        query = """
        SELECT * FROM validation_results 
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY step
        """

        try:
            df = self.db_validation.query_to_dataframe(
                query, (self.config.start_date, self.config.end_date)
            )
            logger.info(f"Loaded {len(df)} validation records")
            return df
        except Exception as e:
            logger.warning(f"Could not load validation results: {e}")
            return pd.DataFrame()

    def load_anomaly_reports(self) -> List[AnomalyReport]:
        """Load anomaly reports from database."""
        query = """
        SELECT * FROM anomaly_alerts 
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp
        """

        try:
            df = self.db_anomaly.query_to_dataframe(
                query, (self.config.start_date, self.config.end_date)
            )

            anomalies = []
            for _, row in df.iterrows():
                anomalies.append(
                    AnomalyReport(
                        timestamp=row["timestamp"],
                        anomaly_type=row["anomaly_type"],
                        severity=row["severity"],
                        description=row["description"],
                        affected_metrics=json.loads(row["affected_metrics"]),
                        remediation_suggestions=json.loads(row["remediation_suggestions"]),
                        resolved=bool(row["resolved"]),
                    )
                )

            logger.info(f"Loaded {len(anomalies)} anomaly reports")
            return anomalies
        except Exception as e:
            logger.warning(f"Could not load anomaly reports: {e}")
            return []

    def load_checkpoint_info(self) -> List[CheckpointInfo]:
        """Load checkpoint information from directory."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return []

        checkpoints = []
        for checkpoint_file in checkpoint_dir.glob("*.pt"):
            try:
                # Extract step from filename (assuming format: checkpoint_step_XXXXX.pt)
                step = int(checkpoint_file.stem.split("_")[-1])

                # Get file stats
                stat = checkpoint_file.stat()
                file_size_mb = stat.st_size / (1024 * 1024)
                timestamp = datetime.fromtimestamp(stat.st_mtime)

                # Check if within time range
                if self.config.start_date <= timestamp <= self.config.end_date:
                    checkpoints.append(
                        CheckpointInfo(
                            step=step,
                            timestamp=timestamp,
                            file_path=str(checkpoint_file),
                            file_size_mb=file_size_mb,
                            validation_score=0.0,  # Would need to load from metadata
                            is_best=False,  # Would need to determine from metadata
                            save_reason="regular_interval",
                        )
                    )
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse checkpoint file {checkpoint_file}: {e}")

        logger.info(f"Loaded {len(checkpoints)} checkpoint records")
        return sorted(checkpoints, key=lambda x: x.step)

    def analyze_training_efficiency(self, metrics_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze training efficiency metrics."""
        if metrics_df.empty:
            return {}

        return {
            "avg_throughput": float(metrics_df["throughput_tokens_per_sec"].mean()),
            "max_throughput": float(metrics_df["throughput_tokens_per_sec"].max()),
            "avg_gpu_memory": float(metrics_df["gpu_memory_used"].mean()),
            "max_gpu_memory": float(metrics_df["gpu_memory_used"].max()),
            "avg_gradient_norm": float(metrics_df["gradient_norm"].mean()),
            "gradient_stability": float(1.0 / (1.0 + metrics_df["gradient_norm"].std())),
            "learning_rate_final": float(metrics_df["learning_rate"].iloc[-1]),
        }

    def generate_recommendations(
        self, metrics_df: pd.DataFrame, anomalies: List[AnomalyReport]
    ) -> Tuple[List[str], List[str]]:
        """Generate training recommendations and next steps."""
        recommendations = []
        next_steps = []

        if not metrics_df.empty:
            # Analyze loss progression
            final_loss = metrics_df["total_loss"].iloc[-1]
            initial_loss = metrics_df["total_loss"].iloc[0]
            loss_reduction = (initial_loss - final_loss) / initial_loss

            if loss_reduction < 0.1:
                recommendations.append(
                    "Loss reduction is minimal. Consider adjusting learning rate or architecture."
                )

            # Analyze EQ progression
            eq_progress = metrics_df["eq_score"].iloc[-1] - metrics_df["eq_score"].iloc[0]
            if eq_progress < 0.05:
                recommendations.append(
                    "EQ score improvement is slow. Review EQ training data quality."
                )

            # Analyze gradient norms
            avg_grad_norm = metrics_df["gradient_norm"].mean()
            if avg_grad_norm > 1.0:
                recommendations.append("High gradient norms detected. Consider gradient clipping.")
            elif avg_grad_norm < 0.1:
                recommendations.append(
                    "Low gradient norms detected. Consider increasing learning rate."
                )

        # Analyze anomalies
        critical_anomalies = [a for a in anomalies if a.severity == "critical"]
        if critical_anomalies:
            recommendations.append(
                f"Address {len(critical_anomalies)} critical anomalies immediately."
            )

        # Generate next steps
        next_steps.extend(
            [
                "Continue monitoring training progress with current configuration",
                "Validate model performance on held-out test set",
                "Consider hyperparameter tuning if performance plateaus",
                "Prepare for next phase of training pipeline",
            ]
        )

        return recommendations, next_steps

    def generate_report(self) -> TrainingReport:
        """Generate comprehensive training report."""
        logger.info("Generating comprehensive training report...")

        # Load all data
        metrics_df = self.load_training_metrics()
        validation_df = self.load_validation_results()
        anomalies = self.load_anomaly_reports()
        checkpoints = self.load_checkpoint_info()

        # Generate report ID
        report_id = f"{self.config.report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate time range
        time_range = (self.config.start_date, self.config.end_date)
        total_training_time = self.config.end_date - self.config.start_date

        # Analyze metrics progression
        analyzer = MetricsAnalyzer()

        loss_progression = analyzer.calculate_progression(
            metrics_df, "total_loss", self.config.trend_window
        )
        eq_progression = analyzer.calculate_progression(
            metrics_df, "eq_score", self.config.trend_window
        )
        clinical_progression = analyzer.calculate_progression(
            metrics_df, "clinical_accuracy", self.config.trend_window
        )
        persona_progression = analyzer.calculate_progression(
            metrics_df, "persona_consistency", self.config.trend_window
        )

        # Analyze anomalies
        anomaly_summary = {}
        for anomaly in anomalies:
            anomaly_summary[anomaly.severity] = anomaly_summary.get(anomaly.severity, 0) + 1

        critical_anomalies = [a for a in anomalies if a.severity == "critical"]

        # Analyze checkpoints
        checkpoint_summary = {
            "total_checkpoints": len(checkpoints),
            "total_size_mb": sum(c.file_size_mb for c in checkpoints),
            "avg_size_mb": (
                sum(c.file_size_mb for c in checkpoints) / len(checkpoints) if checkpoints else 0.0
            ),
        }

        best_checkpoints = sorted(checkpoints, key=lambda x: x.validation_score, reverse=True)[:5]

        # Analyze training efficiency
        training_efficiency = self.analyze_training_efficiency(metrics_df)

        # Generate recommendations
        recommendations, next_steps = self.generate_recommendations(metrics_df, anomalies)

        # Create final metrics and best validation
        final_metrics = None
        best_validation = None

        if not metrics_df.empty:
            final_row = metrics_df.iloc[-1]
            final_metrics = TrainingMetrics(
                step=int(final_row["step"]),
                epoch=int(final_row["epoch"]),
                timestamp=final_row["timestamp"],
                total_loss=float(final_row["total_loss"]),
                eq_loss=float(final_row["eq_loss"]),
                clinical_loss=float(final_row["clinical_loss"]),
                persona_loss=float(final_row["persona_loss"]),
                eq_score=float(final_row["eq_score"]),
                clinical_accuracy=float(final_row["clinical_accuracy"]),
                persona_consistency=float(final_row["persona_consistency"]),
                learning_rate=float(final_row["learning_rate"]),
                gpu_memory_used=float(final_row["gpu_memory_used"]),
                throughput_tokens_per_sec=float(final_row["throughput_tokens_per_sec"]),
                gradient_norm=float(final_row["gradient_norm"]),
            )

        if not validation_df.empty:
            best_row = validation_df.loc[validation_df["validation_loss"].idxmin()]
            best_validation = ValidationResult(
                step=int(best_row["step"]),
                timestamp=best_row["timestamp"],
                validation_loss=float(best_row["validation_loss"]),
                eq_validation_score=float(best_row["eq_validation_score"]),
                clinical_validation_accuracy=float(best_row["clinical_validation_accuracy"]),
                persona_validation_consistency=float(best_row["persona_validation_consistency"]),
                early_stopping_patience=int(best_row["early_stopping_patience"]),
                best_score=float(best_row["best_score"]),
                improved=bool(best_row["improved"]),
            )

        report = TrainingReport(
            report_id=report_id,
            generation_time=datetime.now(),
            time_range=time_range,
            total_training_time=total_training_time,
            total_steps=len(metrics_df),
            total_epochs=int(metrics_df["epoch"].max()) if not metrics_df.empty else 0,
            final_metrics=final_metrics,
            best_validation=best_validation,
            loss_progression=loss_progression,
            eq_progression=eq_progression,
            clinical_progression=clinical_progression,
            persona_progression=persona_progression,
            anomaly_summary=anomaly_summary,
            critical_anomalies=critical_anomalies,
            anomaly_patterns={},  # Could be expanded with pattern analysis
            checkpoint_summary=checkpoint_summary,
            best_checkpoints=best_checkpoints,
            training_efficiency=training_efficiency,
            resource_utilization={},  # Could be expanded with detailed resource analysis
            trend_analysis={},  # Could be expanded with advanced trend analysis
            recommendations=recommendations,
            next_steps=next_steps,
        )

        logger.info(f"Generated comprehensive training report: {report_id}")
        return report

    def export_report(self, report: TrainingReport) -> Dict[str, str]:
        """Export report in multiple formats."""
        output_files = {}

        # Export JSON
        if self.config.export_json:
            json_path = Path(self.config.output_dir) / f"{report.report_id}.json"
            with open(json_path, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)
            output_files["json"] = str(json_path)
            logger.info(f"Exported JSON report: {json_path}")

        # Export HTML
        if self.config.export_html:
            html_path = Path(self.config.output_dir) / f"{report.report_id}.html"
            html_content = self._generate_html_report(report)
            with open(html_path, "w") as f:
                f.write(html_content)
            output_files["html"] = str(html_path)
            logger.info(f"Exported HTML report: {html_path}")

        # Export PDF
        if self.config.export_pdf:
            pdf_path = Path(self.config.output_dir) / f"{report.report_id}.pdf"
            self._generate_pdf_report(report, pdf_path)
            output_files["pdf"] = str(pdf_path)
            logger.info(f"Exported PDF report: {pdf_path}")

        return output_files

    def _generate_html_report(self, report: TrainingReport) -> str:
        """Generate HTML report content."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Report: {{ report.report_id }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }
                .recommendation { background-color: #d4edda; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .critical { background-color: #f8d7da; padding: 10px; margin: 5px 0; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Training Report: {{ report.report_id }}</h1>
                <p>Generated: {{ report.generation_time }}</p>
                <p>Time Range: {{ report.time_range[0] }} to {{ report.time_range[1] }}</p>
                <p>Total Training Time: {{ report.total_training_time }}</p>
            </div>
            
            <div class="section">
                <h2>Training Summary</h2>
                <div class="metric">Total Steps: {{ report.total_steps }}</div>
                <div class="metric">Total Epochs: {{ report.total_epochs }}</div>
                {% if report.final_metrics %}
                <div class="metric">Final Loss: {{ "%.4f"|format(report.final_metrics.total_loss) }}</div>
                <div class="metric">Final EQ Score: {{ "%.4f"|format(report.final_metrics.eq_score) }}</div>
                <div class="metric">Final Clinical Accuracy: {{ "%.4f"|format(report.final_metrics.clinical_accuracy) }}</div>
                {% endif %}
            </div>
            
            <div class="section">
                <h2>Progress Analysis</h2>
                <h3>Loss Progression</h3>
                <p>Initial: {{ "%.4f"|format(report.loss_progression.initial_value) }}</p>
                <p>Final: {{ "%.4f"|format(report.loss_progression.final_value) }}</p>
                <p>Improvement Rate: {{ "%.6f"|format(report.loss_progression.improvement_rate) }}</p>
                
                <h3>EQ Progression</h3>
                <p>Initial: {{ "%.4f"|format(report.eq_progression.initial_value) }}</p>
                <p>Final: {{ "%.4f"|format(report.eq_progression.final_value) }}</p>
                <p>Best: {{ "%.4f"|format(report.eq_progression.best_value) }}</p>
            </div>
            
            <div class="section">
                <h2>Anomaly Analysis</h2>
                {% for severity, count in report.anomaly_summary.items() %}
                <div class="metric">{{ severity.title() }}: {{ count }}</div>
                {% endfor %}
                
                {% if report.critical_anomalies %}
                <h3>Critical Anomalies</h3>
                {% for anomaly in report.critical_anomalies %}
                <div class="critical">
                    <strong>{{ anomaly.anomaly_type }}</strong> - {{ anomaly.description }}
                    <br>Time: {{ anomaly.timestamp }}
                </div>
                {% endfor %}
                {% endif %}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {% for rec in report.recommendations %}
                <div class="recommendation">{{ rec }}</div>
                {% endfor %}
                
                <h3>Next Steps</h3>
                {% for step in report.next_steps %}
                <div class="recommendation">{{ step }}</div>
                {% endfor %}
            </div>
        </body>
        </html>
        """

        template = Template(html_template)
        return template.render(report=report)

    def _generate_pdf_report(self, report: TrainingReport, output_path: Path):
        """Generate PDF report with visualizations."""
        with PdfPages(str(output_path)) as pdf:
            # Create summary page
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(f"Training Report: {report.report_id}", fontsize=16)

            # Summary statistics
            summary_text = f"""
            Training Summary:
            • Total Steps: {report.total_steps}
            • Total Epochs: {report.total_epochs}
            • Training Time: {report.total_training_time}
            
            Final Metrics:
            • Total Loss: {report.final_metrics.total_loss:.4f if report.final_metrics else 'N/A'}
            • EQ Score: {report.final_metrics.eq_score:.4f if report.final_metrics else 'N/A'}
            • Clinical Accuracy: {report.final_metrics.clinical_accuracy:.4f if report.final_metrics else 'N/A'}
            """

            axes[0, 0].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment="center")
            axes[0, 0].set_title("Training Summary")
            axes[0, 0].axis("off")

            # Anomaly summary
            if report.anomaly_summary:
                severities = list(report.anomaly_summary.keys())
                counts = list(report.anomaly_summary.values())
                axes[0, 1].bar(severities, counts)
                axes[0, 1].set_title("Anomalies by Severity")
                axes[0, 1].set_ylabel("Count")
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "No anomalies detected",
                    ha="center",
                    va="center",
                    transform=axes[0, 1].transAxes,
                )
                axes[0, 1].set_title("Anomalies by Severity")
                axes[0, 1].axis("off")

            # Training efficiency
            if report.training_efficiency:
                metrics = list(report.training_efficiency.keys())[:4]  # Top 4 metrics
                values = [report.training_efficiency[m] for m in metrics]
                axes[1, 0].bar(range(len(metrics)), values)
                axes[1, 0].set_xticks(range(len(metrics)))
                axes[1, 0].set_xticklabels(metrics, rotation=45, ha="right")
                axes[1, 0].set_title("Training Efficiency")
            else:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "No efficiency data",
                    ha="center",
                    va="center",
                    transform=axes[1, 0].transAxes,
                )
                axes[1, 0].set_title("Training Efficiency")
                axes[1, 0].axis("off")

            # Recommendations
            rec_text = "Recommendations:\n" + "\n".join(
                [f"• {rec}" for rec in report.recommendations[:5]]
            )
            axes[1, 1].text(0.1, 0.5, rec_text, fontsize=9, verticalalignment="center")
            axes[1, 1].set_title("Key Recommendations")
            axes[1, 1].axis("off")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()


def main():
    """Main function for testing the reporting system."""
    # Create test configuration
    config = ReportConfig(
        report_name="test_training_report", last_n_days=7, output_dir="test_reports/"
    )

    # Initialize analyzer
    analyzer = TrainingReportAnalyzer(config)

    # Generate report
    report = analyzer.generate_report()

    # Export report
    output_files = analyzer.export_report(report)

    print("Training report generated successfully!")
    print(f"Report ID: {report.report_id}")
    print("Output files:")
    for format_type, file_path in output_files.items():
        print(f"  {format_type.upper()}: {file_path}")


if __name__ == "__main__":
    main()
