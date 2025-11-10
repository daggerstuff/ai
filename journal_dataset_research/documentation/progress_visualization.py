"""
Progress Visualization

Generates progress metrics charts, timeline visualizations, and quality score
distributions for research activities.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from ai.journal_dataset_research.models.dataset_models import (
    DatasetEvaluation,
    ResearchProgress,
    ResearchSession,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


class ProgressVisualization:
    """
    Generates visualizations for research progress.

    Creates charts for progress metrics, timeline visualizations, and
    quality score distributions. Exports visualizations as images or HTML.
    """

    def __init__(self, output_directory: str = "visualizations"):
        """
        Initialize the progress visualization generator.

        Args:
            output_directory: Directory to store generated visualizations
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        if not MATPLOTLIB_AVAILABLE:
            import warnings
            warnings.warn(
                "matplotlib is not available. Visualization features will be limited. "
                "Install it with: pip install matplotlib",
                ImportWarning
            )

    def generate_progress_metrics_chart(
        self,
        progress_history: List[Dict[str, any]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate a chart showing progress metrics over time.

        Args:
            progress_history: List of progress snapshots with timestamps
            output_path: Optional output path (auto-generated if not provided)

        Returns:
            Path to the generated chart
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for this visualization. "
                "Install it with: pip install matplotlib"
            )
        if output_path is None:
            output_path = (
                self.output_directory
                / f"progress_metrics_{datetime.now().strftime('%Y%m%d')}.png"
            )

        # Extract data
        timestamps = [
            datetime.fromisoformat(entry["timestamp"]) for entry in progress_history
        ]
        sources = [entry.get("sources_identified", 0) for entry in progress_history]
        evaluated = [
            entry.get("datasets_evaluated", 0) for entry in progress_history
        ]
        acquired = [entry.get("datasets_acquired", 0) for entry in progress_history]

        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, sources, label="Sources Identified", marker="o")
        ax.plot(timestamps, evaluated, label="Datasets Evaluated", marker="s")
        ax.plot(timestamps, acquired, label="Datasets Acquired", marker="^")

        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.set_title("Research Progress Metrics Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def generate_timeline_visualization(
        self,
        session: ResearchSession,
        phase_transitions: List[Dict[str, any]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate a timeline visualization for research phases.

        Args:
            session: Research session information
            phase_transitions: List of phase transitions with timestamps
            output_path: Optional output path (auto-generated if not provided)

        Returns:
            Path to the generated visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for this visualization. "
                "Install it with: pip install matplotlib"
            )
        if output_path is None:
            output_path = (
                self.output_directory
                / f"timeline_{session.session_id}_{datetime.now().strftime('%Y%m%d')}.png"
            )

        # Create timeline data
        phases = ["discovery", "evaluation", "acquisition", "integration"]
        phase_colors = {
            "discovery": "#3498db",
            "evaluation": "#e74c3c",
            "acquisition": "#2ecc71",
            "integration": "#f39c12",
        }

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot phase transitions
        y_positions = {phase: i for i, phase in enumerate(phases)}
        for i, transition in enumerate(phase_transitions):
            phase = transition.get("phase", session.current_phase)
            timestamp = datetime.fromisoformat(transition.get("timestamp", datetime.now().isoformat()))
            y_pos = y_positions.get(phase, 0)

            ax.scatter(
                timestamp, y_pos, c=phase_colors.get(phase, "#000000"), s=100, zorder=3
            )
            ax.text(
                timestamp,
                y_pos + 0.2,
                phase.title(),
                ha="center",
                fontsize=8,
                rotation=45,
            )

        # Draw timeline line
        if phase_transitions:
            start_time = datetime.fromisoformat(
                phase_transitions[0].get("timestamp", session.start_date.isoformat())
            )
            end_time = datetime.now()
            ax.plot([start_time, end_time], [1, 1], "k-", linewidth=2, alpha=0.3)

        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels([phase.title() for phase in phases])
        ax.set_xlabel("Date")
        ax.set_title(f"Research Phase Timeline - Session {session.session_id}")
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def generate_quality_score_distribution(
        self,
        evaluations: List[DatasetEvaluation],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate a distribution chart of quality scores.

        Args:
            evaluations: List of dataset evaluations
            output_path: Optional output path (auto-generated if not provided)

        Returns:
            Path to the generated chart
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for this visualization. "
                "Install it with: pip install matplotlib"
            )
        if output_path is None:
            output_path = (
                self.output_directory
                / f"quality_scores_{datetime.now().strftime('%Y%m%d')}.png"
            )

        # Extract scores
        overall_scores = [eval.overall_score for eval in evaluations]
        therapeutic_scores = [eval.therapeutic_relevance for eval in evaluations]
        structure_scores = [eval.data_structure_quality for eval in evaluations]
        integration_scores = [eval.training_integration for eval in evaluations]
        ethical_scores = [eval.ethical_accessibility for eval in evaluations]

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Overall score distribution
        axes[0].hist(overall_scores, bins=20, edgecolor="black", alpha=0.7)
        axes[0].set_title("Overall Score Distribution")
        axes[0].set_xlabel("Score")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True, alpha=0.3)

        # Therapeutic relevance
        axes[1].hist(therapeutic_scores, bins=10, edgecolor="black", alpha=0.7, color="#e74c3c")
        axes[1].set_title("Therapeutic Relevance Distribution")
        axes[1].set_xlabel("Score")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True, alpha=0.3)

        # Data structure quality
        axes[2].hist(structure_scores, bins=10, edgecolor="black", alpha=0.7, color="#3498db")
        axes[2].set_title("Data Structure Quality Distribution")
        axes[2].set_xlabel("Score")
        axes[2].set_ylabel("Frequency")
        axes[2].grid(True, alpha=0.3)

        # Training integration
        axes[3].hist(integration_scores, bins=10, edgecolor="black", alpha=0.7, color="#2ecc71")
        axes[3].set_title("Training Integration Distribution")
        axes[3].set_xlabel("Score")
        axes[3].set_ylabel("Frequency")
        axes[3].grid(True, alpha=0.3)

        # Ethical accessibility
        axes[4].hist(ethical_scores, bins=10, edgecolor="black", alpha=0.7, color="#f39c12")
        axes[4].set_title("Ethical Accessibility Distribution")
        axes[4].set_xlabel("Score")
        axes[4].set_ylabel("Frequency")
        axes[4].grid(True, alpha=0.3)

        # Priority tier pie chart
        priority_counts: Dict[str, int] = {}
        for eval in evaluations:
            priority_counts[eval.priority_tier] = (
                priority_counts.get(eval.priority_tier, 0) + 1
            )

        if priority_counts:
            axes[5].pie(
                priority_counts.values(),
                labels=[tier.upper() for tier in priority_counts.keys()],
                autopct="%1.1f%%",
                startangle=90,
            )
            axes[5].set_title("Priority Tier Distribution")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def generate_html_dashboard(
        self,
        session: ResearchSession,
        progress: ResearchProgress,
        evaluations: List[DatasetEvaluation],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate an HTML dashboard with embedded visualizations.

        Args:
            session: Research session information
            progress: Current research progress
            evaluations: List of dataset evaluations
            output_path: Optional output path (auto-generated if not provided)

        Returns:
            Path to the generated HTML dashboard
        """
        if output_path is None:
            output_path = (
                self.output_directory
                / f"dashboard_{session.session_id}_{datetime.now().strftime('%Y%m%d')}.html"
            )

        # Generate individual charts
        progress_chart_path = self._generate_simple_progress_chart(progress)
        score_chart_path = (
            self._generate_simple_score_chart(evaluations) if evaluations else None
        )

        # Create HTML template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Research Progress Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #3498db; color: white; padding: 20px; border-radius: 5px; }
        .metrics { display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }
        .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; flex: 1; min-width: 200px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .metric-label { color: #666; margin-top: 5px; }
        .chart { margin: 20px 0; text-align: center; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Research Progress Dashboard</h1>
        <p>Session ID: {{ session_id }}</p>
        <p>Generated: {{ generated_date }}</p>
    </div>

    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{{ sources_identified }}</div>
            <div class="metric-label">Sources Identified</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ datasets_evaluated }}</div>
            <div class="metric-label">Datasets Evaluated</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ datasets_acquired }}</div>
            <div class="metric-label">Datasets Acquired</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ integration_plans }}</div>
            <div class="metric-label">Integration Plans</div>
        </div>
    </div>

    {% if progress_chart %}
    <div class="chart">
        <h2>Progress Metrics</h2>
        <img src="{{ progress_chart }}" alt="Progress Metrics Chart">
    </div>
    {% endif %}

    {% if score_chart %}
    <div class="chart">
        <h2>Quality Score Distribution</h2>
        <img src="{{ score_chart }}" alt="Quality Score Distribution">
    </div>
    {% endif %}
</body>
</html>
"""

        if JINJA2_AVAILABLE:
            template = Template(html_template)
            html_content = template.render(
                session_id=session.session_id,
                generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                sources_identified=progress.sources_identified,
                datasets_evaluated=progress.datasets_evaluated,
                datasets_acquired=progress.datasets_acquired,
                integration_plans=progress.integration_plans_created,
                progress_chart=progress_chart_path.name if progress_chart_path else None,
                score_chart=score_chart_path.name if score_chart_path else None,
            )
        else:
            # Fallback to simple string replacement
            html_content = html_template.replace("{{ session_id }}", session.session_id)
            html_content = html_content.replace(
                "{{ generated_date }}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            html_content = html_content.replace(
                "{{ sources_identified }}", str(progress.sources_identified)
            )
            html_content = html_content.replace(
                "{{ datasets_evaluated }}", str(progress.datasets_evaluated)
            )
            html_content = html_content.replace(
                "{{ datasets_acquired }}", str(progress.datasets_acquired)
            )
            html_content = html_content.replace(
                "{{ integration_plans }}", str(progress.integration_plans_created)
            )
            html_content = html_content.replace(
                "{{ progress_chart }}", progress_chart_path.name if progress_chart_path else ""
            )
            html_content = html_content.replace(
                "{{ score_chart }}", score_chart_path.name if score_chart_path else ""
            )

        output_path.write_text(html_content, encoding="utf-8")
        return output_path

    def _generate_simple_progress_chart(self, progress: ResearchProgress) -> Path:
        """Generate a simple progress chart."""
        output_path = (
            self.output_directory
            / f"progress_simple_{datetime.now().strftime('%Y%m%d')}.png"
        )

        metrics = [
            "Sources\nIdentified",
            "Datasets\nEvaluated",
            "Access\nEstablished",
            "Datasets\nAcquired",
            "Integration\nPlans",
        ]
        values = [
            progress.sources_identified,
            progress.datasets_evaluated,
            progress.access_established,
            progress.datasets_acquired,
            progress.integration_plans_created,
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(metrics, values, color="#3498db", alpha=0.7)
        ax.set_ylabel("Count")
        ax.set_title("Research Progress Metrics")
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _generate_simple_score_chart(
        self, evaluations: List[DatasetEvaluation]
    ) -> Path:
        """Generate a simple score distribution chart."""
        output_path = (
            self.output_directory
            / f"scores_simple_{datetime.now().strftime('%Y%m%d')}.png"
        )

        scores = [eval.overall_score for eval in evaluations]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(scores, bins=20, edgecolor="black", alpha=0.7, color="#e74c3c")
        ax.set_xlabel("Overall Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Dataset Evaluation Score Distribution")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

