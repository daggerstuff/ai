#!/usr/bin/env python3
"""
Quality Comparison System
Compares quality metrics across tiers, datasets, and time periods
"""

import json
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.simplefilter("default")


@dataclass
class QualityComparison:
    """Quality comparison results"""

    category_type: str  # 'tier', 'dataset', 'time_period'
    category_a: str
    category_b: str
    metric: str
    value_a: float
    value_b: float
    difference: float
    percentage_difference: float
    statistical_significance: float
    effect_size: float
    interpretation: str


class QualityComparisonSystem:
    """Enterprise-grade quality comparison system"""

    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/quality_comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Quality metrics to compare
        self.quality_metrics = [
            "conversation_length",
            "content_richness",
            "processing_efficiency",
        ]

        # Comparison categories
        self.comparison_categories = ["tier", "dataset_source"]

    def perform_comprehensive_comparisons(self) -> dict[str, list[QualityComparison]]:
        """Perform comprehensive quality comparisons"""

        try:
            # Get data from database
            data = self._get_comparison_data()

            if not data:
                return {}

            df = pd.DataFrame(data)

            # Perform comparisons for each category
            all_comparisons = {}

            for category in self.comparison_categories:
                if category in df.columns:
                    comparisons = self._compare_by_category(df, category)
                    if comparisons:
                        all_comparisons[category] = comparisons

            sum(len(comps) for comps in all_comparisons.values())
            return all_comparisons

        except Exception:
            return {}

    def _get_comparison_data(self) -> list[dict]:
        """Get data for quality comparisons"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
            SELECT
                dataset_source,
                tier,
                turn_count,
                word_count,
                processing_status
            FROM conversations
            WHERE turn_count IS NOT NULL
            AND word_count IS NOT NULL
            LIMIT 5000
            """

            cursor = conn.execute(query)
            columns = [desc[0] for desc in cursor.description]

            data = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row, strict=False))
                data.append(record)

            conn.close()
            return data

        except Exception:
            return []

    def _compare_by_category(self, df: pd.DataFrame, category: str) -> list[QualityComparison]:
        """Compare quality metrics by category"""
        try:
            comparisons = []

            # Get unique category values
            category_values = df[category].dropna().unique()

            if len(category_values) < 2:
                return comparisons

            # Compare each pair of categories for each metric
            for i, cat_a in enumerate(category_values):
                for j, cat_b in enumerate(category_values):
                    if i >= j:  # Avoid duplicate comparisons
                        continue

                    for metric in self.quality_metrics:
                        comparison = self._compare_categories(df, category, cat_a, cat_b, metric)
                        if comparison:
                            comparisons.append(comparison)

            return comparisons

        except Exception:
            return []

    def _compare_categories(
        self, df: pd.DataFrame, category_type: str, cat_a: str, cat_b: str, metric: str
    ) -> QualityComparison | None:
        """Compare two categories for a specific metric"""
        try:
            # Get data for each category
            data_a = df[df[category_type] == cat_a]
            data_b = df[df[category_type] == cat_b]

            if len(data_a) < 10 or len(data_b) < 10:  # Need sufficient data
                return None

            # Calculate metric values
            values_a = self._calculate_metric_values(data_a, metric)
            values_b = self._calculate_metric_values(data_b, metric)

            if not values_a or not values_b:
                return None

            # Calculate statistics
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            difference = mean_b - mean_a
            percentage_difference = (difference / mean_a) * 100 if mean_a != 0 else 0

            # Statistical test (Mann-Whitney U)
            try:
                _statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            except:
                p_value = 1.0

            # Effect size (Cohen's d)
            try:
                pooled_std = np.sqrt(
                    ((len(values_a) - 1) * np.var(values_a) + (len(values_b) - 1) * np.var(values_b))
                    / (len(values_a) + len(values_b) - 2)
                )
                effect_size = abs(difference) / pooled_std if pooled_std > 0 else 0
            except:
                effect_size = 0

            # Interpretation
            interpretation = self._interpret_comparison(
                difference,
                percentage_difference,
                p_value,
                effect_size,
            )

            return QualityComparison(
                category_type=category_type,
                category_a=str(cat_a),
                category_b=str(cat_b),
                metric=metric,
                value_a=mean_a,
                value_b=mean_b,
                difference=difference,
                percentage_difference=percentage_difference,
                statistical_significance=p_value,
                effect_size=effect_size,
                interpretation=interpretation,
            )

        except Exception:
            return None

    def _calculate_metric_values(self, data: pd.DataFrame, metric: str) -> list[float]:
        """Calculate metric values for a dataset"""
        try:
            if metric == "conversation_length":
                return data["turn_count"].dropna().astype(float).tolist()
            if metric == "content_richness":
                return data["word_count"].dropna().astype(float).tolist()
            if metric == "processing_efficiency":
                total = len(data)
                successful = len(data[data["processing_status"] == "processed"])
                return [100.0 * successful / total] if total > 0 else []
            return []

        except Exception:
            return []

    def _interpret_comparison(
        self,
        difference: float,
        percentage_difference: float,
        p_value: float,
        effect_size: float,
    ) -> str:
        """Interpret comparison results"""
        try:
            # Statistical significance
            if p_value < 0.001:
                significance = "highly significant"
            elif p_value < 0.01:
                significance = "very significant"
            elif p_value < 0.05:
                significance = "significant"
            else:
                significance = "not significant"

            # Effect size interpretation
            if effect_size < 0.2:
                effect_desc = "negligible"
            elif effect_size < 0.5:
                effect_desc = "small"
            elif effect_size < 0.8:
                effect_desc = "medium"
            else:
                effect_desc = "large"

            # Direction
            direction = "higher" if difference > 0 else "lower"

            return f"{significance} difference ({direction} by {abs(percentage_difference):.1f}%) with {effect_desc} effect size"

        except Exception:
            return "Unable to interpret"

    def create_comparison_visualizations(self, comparisons: dict[str, list[QualityComparison]]) -> dict[str, str]:
        """Create comparison visualizations"""

        viz_files = {}

        try:
            # Set style
            plt.style.use("default")
            sns.set_palette("husl")

            # Create comparison dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("Quality Comparison Analysis Dashboard", fontsize=16, fontweight="bold")

            # Tier comparisons (if available)
            if "tier" in comparisons:
                ax = axes[0, 0]
                tier_comps = comparisons["tier"]

                # Group by metric
                metrics = {}
                for comp in tier_comps:
                    if comp.metric not in metrics:
                        metrics[comp.metric] = []
                    metrics[comp.metric].append(comp.percentage_difference)

                if metrics:
                    metric_names = list(metrics.keys())
                    avg_diffs = [np.mean(diffs) for diffs in metrics.values()]

                    bars = ax.bar(range(len(metric_names)), avg_diffs, alpha=0.7)
                    ax.set_title("Average Tier Differences by Metric")
                    ax.set_xlabel("Metrics")
                    ax.set_ylabel("Average % Difference")
                    ax.set_xticks(range(len(metric_names)))
                    ax.set_xticklabels(
                        [m.replace("_", " ").title() for m in metric_names],
                        rotation=45,
                        ha="right",
                    )
                    ax.grid(True, alpha=0.3)

                    # Add value labels
                    for bar, diff in zip(bars, avg_diffs, strict=False):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + (0.5 if height >= 0 else -1),
                            f"{diff:.1f}%",
                            ha="center",
                            va="bottom" if height >= 0 else "top",
                        )

            # Dataset comparisons (if available)
            if "dataset_source" in comparisons:
                ax = axes[0, 1]
                dataset_comps = comparisons["dataset_source"]

                # Show distribution of percentage differences
                percentage_diffs = [comp.percentage_difference for comp in dataset_comps]
                ax.hist(percentage_diffs, bins=20, alpha=0.7, edgecolor="black")
                ax.set_title("Distribution of Dataset Differences")
                ax.set_xlabel("Percentage Difference")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

                # Add mean line
                mean_diff = np.mean(percentage_diffs)
                ax.axvline(
                    mean_diff,
                    color="red",
                    linestyle="--",
                    label=f"Mean: {mean_diff:.1f}%",
                )
                ax.legend()

            # Statistical significance distribution
            ax = axes[1, 0]
            all_p_values = []
            for category_comps in comparisons.values():
                all_p_values.extend([comp.statistical_significance for comp in category_comps])

            if all_p_values:
                ax.hist(all_p_values, bins=20, alpha=0.7, edgecolor="black")
                ax.set_title("Statistical Significance Distribution")
                ax.set_xlabel("P-value")
                ax.set_ylabel("Frequency")
                ax.axvline(0.05, color="red", linestyle="--", label="α = 0.05")
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Effect sizes
            ax = axes[1, 1]
            all_effect_sizes = []
            for category_comps in comparisons.values():
                all_effect_sizes.extend([comp.effect_size for comp in category_comps])

            if all_effect_sizes:
                ax.hist(all_effect_sizes, bins=20, alpha=0.7, edgecolor="black")
                ax.set_title("Effect Size Distribution")
                ax.set_xlabel("Effect Size (Cohen's d)")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

                # Add interpretation lines
                ax.axvline(0.2, color="orange", linestyle="--", alpha=0.7, label="Small")
                ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Medium")
                ax.axvline(0.8, color="darkred", linestyle="--", alpha=0.7, label="Large")
                ax.legend()

            plt.tight_layout()

            # Save dashboard
            dashboard_file = self.output_dir / "quality_comparison_dashboard.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches="tight")
            plt.close()

            viz_files["dashboard"] = str(dashboard_file)

            return viz_files

        except Exception:
            return {}

    def export_comparison_report(
        self,
        comparisons: dict[str, list[QualityComparison]],
        visualizations: dict[str, str],
    ) -> str:
        """Export comprehensive comparison report"""

        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"quality_comparison_report_{timestamp}.json"

            # Create summary statistics
            summary_stats = self._create_comparison_summary(comparisons)

            # Prepare export data
            export_data = {
                "report_metadata": {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "analyzer_version": "1.0.0",
                    "total_comparisons": sum(len(comps) for comps in comparisons.values()),
                    "categories_analyzed": list(comparisons.keys()),
                },
                "summary_statistics": summary_stats,
                "detailed_comparisons": {
                    category: [
                        {
                            "category_a": comp.category_a,
                            "category_b": comp.category_b,
                            "metric": comp.metric,
                            "value_a": comp.value_a,
                            "value_b": comp.value_b,
                            "difference": comp.difference,
                            "percentage_difference": comp.percentage_difference,
                            "statistical_significance": comp.statistical_significance,
                            "effect_size": comp.effect_size,
                            "interpretation": comp.interpretation,
                        }
                        for comp in category_comparisons
                    ]
                    for category, category_comparisons in comparisons.items()
                },
                "visualizations": visualizations,
            }

            # Save report
            with open(report_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            return str(report_file)

        except Exception:
            return ""

    def _create_comparison_summary(self, comparisons: dict[str, list[QualityComparison]]) -> dict[str, Any]:
        """Create comparison summary statistics"""
        try:
            summary = {}

            for category, category_comparisons in comparisons.items():
                if not category_comparisons:
                    continue

                # Calculate summary statistics
                percentage_diffs = [comp.percentage_difference for comp in category_comparisons]
                [comp.statistical_significance for comp in category_comparisons]
                effect_sizes = [comp.effect_size for comp in category_comparisons]

                significant_comparisons = [
                    comp for comp in category_comparisons if comp.statistical_significance < 0.05
                ]

                summary[category] = {
                    "total_comparisons": len(category_comparisons),
                    "significant_comparisons": len(significant_comparisons),
                    "significance_rate": len(significant_comparisons) / len(category_comparisons) * 100,
                    "average_percentage_difference": float(np.mean(percentage_diffs)),
                    "max_percentage_difference": float(np.max(np.abs(percentage_diffs))),
                    "average_effect_size": float(np.mean(effect_sizes)),
                    "large_effect_count": len([es for es in effect_sizes if es > 0.8]),
                }

            return summary

        except Exception:
            return {}


def main():
    """Main execution function"""

    # Initialize system
    system = QualityComparisonSystem()

    # Perform comparisons
    comparisons = system.perform_comprehensive_comparisons()

    if not comparisons:
        return

    # Create visualizations
    visualizations = system.create_comparison_visualizations(comparisons)

    # Export report
    system.export_comparison_report(comparisons, visualizations)

    # Display summary
    sum(len(comps) for comps in comparisons.values())

    # Show key findings
    for _category, category_comparisons in comparisons.items():
        len([c for c in category_comparisons if c.statistical_significance < 0.05])
        np.mean([c.percentage_difference for c in category_comparisons])


if __name__ == "__main__":
    main()
