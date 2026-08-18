#!/usr/bin/env python3
"""
Simple Quality Distribution Analysis System
Analyzes quality score distributions across datasets, tiers, and time periods
"""

import json
import sqlite3
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.simplefilter("default")


class SimpleQualityDistributionAnalyzer:
    """Simple quality distribution analysis system"""

    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/quality_distributions")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_distributions(self) -> dict[str, Any]:
        """Analyze quality distributions"""

        try:
            # Get data from database
            data = self._get_data()

            if not data:
                return {}

            df = pd.DataFrame(data)

            # Analyze different metrics
            return {
                "conversation_length": self._analyze_metric(df, "turn_count", "Conversation Length"),
                "content_richness": self._analyze_metric(df, "word_count", "Content Richness"),
                "tier_distribution": self._analyze_categorical(df, "tier", "Tier Distribution"),
                "dataset_distribution": self._analyze_categorical(df, "dataset_source", "Dataset Distribution"),
            }


        except Exception:
            return {}

    def _get_data(self) -> list[dict]:
        """Get data from database"""
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

    def _analyze_metric(self, df: pd.DataFrame, column: str, title: str) -> dict[str, Any]:
        """Analyze a numeric metric"""
        try:
            values = df[column].dropna().astype(float)

            if len(values) == 0:
                return {"error": "No valid values"}

            # Calculate statistics
            stats_dict = {
                "count": len(values),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "skewness": float(stats.skew(values)),
                "kurtosis": float(stats.kurtosis(values)),
            }

            # Calculate percentiles
            percentiles = {
                "p25": float(values.quantile(0.25)),
                "p50": float(values.quantile(0.50)),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "p95": float(values.quantile(0.95)),
            }

            # Find outliers using IQR
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            outliers = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]

            # Distribution by tier
            tier_stats = {}
            for tier in df["tier"].unique():
                if pd.notna(tier):
                    tier_values = df[df["tier"] == tier][column].dropna().astype(float)
                    if len(tier_values) > 0:
                        tier_stats[str(tier)] = {
                            "count": len(tier_values),
                            "mean": float(tier_values.mean()),
                            "std": float(tier_values.std()),
                        }

            return {
                "title": title,
                "statistics": stats_dict,
                "percentiles": percentiles,
                "outlier_count": len(outliers),
                "tier_breakdown": tier_stats,
                "values": values.tolist()[:100],  # Sample for visualization
            }

        except Exception as e:
            return {"error": str(e)}

    def _analyze_categorical(self, df: pd.DataFrame, column: str, title: str) -> dict[str, Any]:
        """Analyze a categorical variable"""
        try:
            value_counts = df[column].value_counts()

            return {
                "title": title,
                "total_count": len(df),
                "unique_values": len(value_counts),
                "distribution": {str(k): int(v) for k, v in value_counts.items()},
                "percentages": {str(k): float(v / len(df) * 100) for k, v in value_counts.items()},
            }

        except Exception as e:
            return {"error": str(e)}

    def create_visualizations(self, results: dict[str, Any]) -> dict[str, str]:
        """Create distribution visualizations"""

        viz_files = {}

        try:
            # Set style
            plt.style.use("default")
            sns.set_palette("husl")

            # Create summary dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                "Quality Distribution Analysis Dashboard",
                fontsize=16,
                fontweight="bold",
            )

            # Conversation Length Distribution
            if "conversation_length" in results and "values" in results["conversation_length"]:
                ax = axes[0, 0]
                values = results["conversation_length"]["values"]
                ax.hist(values, bins=30, alpha=0.7, edgecolor="black")
                ax.set_title("Conversation Length Distribution")
                ax.set_xlabel("Turn Count")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

                # Add statistics
                mean_val = results["conversation_length"]["statistics"]["mean"]
                ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.1f}")
                ax.legend()

            # Content Richness Distribution
            if "content_richness" in results and "values" in results["content_richness"]:
                ax = axes[0, 1]
                values = results["content_richness"]["values"]
                ax.hist(values, bins=30, alpha=0.7, edgecolor="black")
                ax.set_title("Content Richness Distribution")
                ax.set_xlabel("Word Count")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

                # Add statistics
                mean_val = results["content_richness"]["statistics"]["mean"]
                ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.1f}")
                ax.legend()

            # Tier Distribution
            if "tier_distribution" in results and "distribution" in results["tier_distribution"]:
                ax = axes[1, 0]
                dist = results["tier_distribution"]["distribution"]
                ax.pie(dist.values(), labels=dist.keys(), autopct="%1.1f%%")
                ax.set_title("Tier Distribution")

            # Dataset Distribution
            if "dataset_distribution" in results and "distribution" in results["dataset_distribution"]:
                ax = axes[1, 1]
                dist = results["dataset_distribution"]["distribution"]
                # Show top 10 datasets
                top_datasets = dict(list(dist.items())[:10])
                ax.bar(range(len(top_datasets)), list(top_datasets.values()))
                ax.set_title("Top 10 Dataset Sources")
                ax.set_xlabel("Dataset")
                ax.set_ylabel("Count")
                ax.set_xticks(range(len(top_datasets)))
                ax.set_xticklabels(list(top_datasets.keys()), rotation=45, ha="right")
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save dashboard
            dashboard_file = self.output_dir / "distribution_dashboard.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches="tight")
            plt.close()

            viz_files["dashboard"] = str(dashboard_file)

            return viz_files

        except Exception:
            return {}

    def export_report(self, results: dict[str, Any], visualizations: dict[str, str]) -> str:
        """Export distribution analysis report"""

        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"distribution_report_{timestamp}.json"

            # Prepare export data
            export_data = {
                "report_metadata": {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "analyzer_version": "1.0.0",
                },
                "analysis_results": results,
                "visualizations": visualizations,
                "summary": self._create_summary(results),
            }

            # Save report
            with open(report_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            return str(report_file)

        except Exception:
            return ""

    def _create_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Create analysis summary"""
        summary = {}

        for metric, data in results.items():
            if "error" in data:
                summary[metric] = {"status": "error", "message": data["error"]}
            elif "statistics" in data:
                summary[metric] = {
                    "status": "success",
                    "count": data["statistics"]["count"],
                    "mean": data["statistics"]["mean"],
                    "std": data["statistics"]["std"],
                    "outliers": data.get("outlier_count", 0),
                }
            elif "distribution" in data:
                summary[metric] = {
                    "status": "success",
                    "total_count": data["total_count"],
                    "unique_values": data["unique_values"],
                }

        return summary


def main():
    """Main execution function"""

    # Initialize analyzer
    analyzer = SimpleQualityDistributionAnalyzer()

    # Analyze distributions
    results = analyzer.analyze_distributions()

    if not results:
        return

    # Create visualizations
    visualizations = analyzer.create_visualizations(results)

    # Export report
    analyzer.export_report(results, visualizations)

    # Display summary

    # Show key findings
    for _metric, data in results.items():
        if "error" in data:
            pass
        elif "statistics" in data:
            data["statistics"]
        elif "distribution" in data:
            pass


if __name__ == "__main__":
    main()
