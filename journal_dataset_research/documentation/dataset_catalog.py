"""
Dataset Catalog

Provides functionality to export dataset catalogs in multiple formats
(markdown, CSV, JSON) with statistics and summaries.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ai.journal_dataset_research.models.dataset_models import (
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
)


class DatasetCatalog:
    """
    Manages dataset catalog export in multiple formats.

    Generates markdown catalogs, CSV exports, and JSON exports with
    comprehensive statistics and summaries.
    """

    def __init__(self):
        """Initialize the dataset catalog."""
        self.sources: List[DatasetSource] = []
        self.evaluations: List[DatasetEvaluation] = []
        self.acquired_datasets: List[AcquiredDataset] = []
        self.integration_plans: List[IntegrationPlan] = []

    def add_source(self, source: DatasetSource) -> None:
        """Add a dataset source to the catalog."""
        self.sources.append(source)

    def add_evaluation(self, evaluation: DatasetEvaluation) -> None:
        """Add a dataset evaluation to the catalog."""
        self.evaluations.append(evaluation)

    def add_acquired_dataset(self, dataset: AcquiredDataset) -> None:
        """Add an acquired dataset to the catalog."""
        self.acquired_datasets.append(dataset)

    def add_integration_plan(self, plan: IntegrationPlan) -> None:
        """Add an integration plan to the catalog."""
        self.integration_plans.append(plan)

    def export_markdown(self, output_path: Path) -> None:
        """
        Export catalog to markdown format.

        Args:
            output_path: Path to write the markdown catalog
        """
        lines = [
            "# Dataset Catalog",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Statistics",
            "",
        ]

        # Add statistics
        stats = self.get_statistics()
        lines.extend(self._format_statistics_markdown(stats))
        lines.append("")

        # Add sources section
        lines.extend(["## Dataset Sources", ""])
        if self.sources:
            for source in self.sources:
                lines.extend(self._format_source_markdown(source))
                lines.append("")
        else:
            lines.append("No sources in catalog.")
            lines.append("")

        # Add evaluations section
        if self.evaluations:
            lines.extend(["## Dataset Evaluations", ""])
            for evaluation in self.evaluations:
                lines.extend(self._format_evaluation_markdown(evaluation))
                lines.append("")

        # Add acquired datasets section
        if self.acquired_datasets:
            lines.extend(["## Acquired Datasets", ""])
            for dataset in self.acquired_datasets:
                lines.extend(self._format_acquired_dataset_markdown(dataset))
                lines.append("")

        # Add integration plans section
        if self.integration_plans:
            lines.extend(["## Integration Plans", ""])
            for plan in self.integration_plans:
                lines.extend(self._format_integration_plan_markdown(plan))
                lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def export_csv(self, output_path: Path) -> None:
        """
        Export catalog to CSV format.

        Args:
            output_path: Path to write the CSV catalog
        """
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write sources
            if self.sources:
                writer.writerow(["## Dataset Sources"])
                writer.writerow(
                    [
                        "Source ID",
                        "Title",
                        "Authors",
                        "Publication Date",
                        "Source Type",
                        "URL",
                        "DOI",
                        "Open Access",
                        "Data Availability",
                        "Discovery Date",
                    ]
                )
                for source in self.sources:
                    writer.writerow(
                        [
                            source.source_id,
                            source.title,
                            "; ".join(source.authors),
                            source.publication_date.strftime("%Y-%m-%d"),
                            source.source_type,
                            source.url,
                            source.doi or "",
                            source.open_access,
                            source.data_availability,
                            source.discovery_date.strftime("%Y-%m-%d %H:%M:%S"),
                        ]
                    )
                writer.writerow([])

            # Write evaluations
            if self.evaluations:
                writer.writerow(["## Dataset Evaluations"])
                writer.writerow(
                    [
                        "Source ID",
                        "Therapeutic Relevance",
                        "Data Structure Quality",
                        "Training Integration",
                        "Ethical Accessibility",
                        "Overall Score",
                        "Priority Tier",
                        "Evaluation Date",
                    ]
                )
                for evaluation in self.evaluations:
                    writer.writerow(
                        [
                            evaluation.source_id,
                            evaluation.therapeutic_relevance,
                            evaluation.data_structure_quality,
                            evaluation.training_integration,
                            evaluation.ethical_accessibility,
                            evaluation.overall_score,
                            evaluation.priority_tier,
                            evaluation.evaluation_date.strftime("%Y-%m-%d %H:%M:%S"),
                        ]
                    )
                writer.writerow([])

            # Write acquired datasets
            if self.acquired_datasets:
                writer.writerow(["## Acquired Datasets"])
                writer.writerow(
                    [
                        "Source ID",
                        "Acquisition Date",
                        "Storage Path",
                        "File Format",
                        "File Size (MB)",
                        "License",
                        "Attribution Required",
                    ]
                )
                for dataset in self.acquired_datasets:
                    writer.writerow(
                        [
                            dataset.source_id,
                            dataset.acquisition_date.strftime("%Y-%m-%d %H:%M:%S"),
                            dataset.storage_path,
                            dataset.file_format,
                            dataset.file_size_mb,
                            dataset.license,
                            dataset.attribution_required,
                        ]
                    )

    def export_json(self, output_path: Path) -> None:
        """
        Export catalog to JSON format.

        Args:
            output_path: Path to write the JSON catalog
        """
        catalog_data = {
            "generated": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "sources": [self._source_to_dict(source) for source in self.sources],
            "evaluations": [
                self._evaluation_to_dict(eval) for eval in self.evaluations
            ],
            "acquired_datasets": [
                self._acquired_dataset_to_dict(ds) for ds in self.acquired_datasets
            ],
            "integration_plans": [
                self._integration_plan_to_dict(plan) for plan in self.integration_plans
            ],
        }

        output_path.write_text(
            json.dumps(catalog_data, indent=2, default=str), encoding="utf-8"
        )

    def get_statistics(self) -> Dict[str, any]:
        """
        Get catalog statistics.

        Returns:
            Dictionary with catalog statistics
        """
        stats = {
            "total_sources": len(self.sources),
            "total_evaluations": len(self.evaluations),
            "total_acquired": len(self.acquired_datasets),
            "total_integration_plans": len(self.integration_plans),
        }

        # Source type breakdown
        source_types: Dict[str, int] = {}
        for source in self.sources:
            source_types[source.source_type] = (
                source_types.get(source.source_type, 0) + 1
            )
        stats["sources_by_type"] = source_types

        # Data availability breakdown
        availability_counts: Dict[str, int] = {}
        for source in self.sources:
            availability_counts[source.data_availability] = (
                availability_counts.get(source.data_availability, 0) + 1
            )
        stats["sources_by_availability"] = availability_counts

        # Evaluation score statistics
        if self.evaluations:
            scores = [eval.overall_score for eval in self.evaluations]
            stats["evaluation_score_stats"] = {
                "average": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
            }

            # Priority tier breakdown
            priority_counts: Dict[str, int] = {}
            for eval in self.evaluations:
                priority_counts[eval.priority_tier] = (
                    priority_counts.get(eval.priority_tier, 0) + 1
                )
            stats["evaluations_by_priority"] = priority_counts

        # Total dataset size
        total_size_mb = sum(ds.file_size_mb for ds in self.acquired_datasets)
        stats["total_acquired_size_mb"] = total_size_mb

        return stats

    def _format_statistics_markdown(self, stats: Dict[str, any]) -> List[str]:
        """Format statistics as markdown."""
        lines = [
            f"- **Total Sources**: {stats['total_sources']}",
            f"- **Total Evaluations**: {stats['total_evaluations']}",
            f"- **Total Acquired**: {stats['total_acquired']}",
            f"- **Total Integration Plans**: {stats['total_integration_plans']}",
            "",
        ]

        if stats.get("sources_by_type"):
            lines.append("### Sources by Type")
            for source_type, count in stats["sources_by_type"].items():
                lines.append(f"- {source_type}: {count}")
            lines.append("")

        if stats.get("evaluation_score_stats"):
            score_stats = stats["evaluation_score_stats"]
            lines.append("### Evaluation Score Statistics")
            lines.append(f"- Average: {score_stats['average']:.2f}")
            lines.append(f"- Min: {score_stats['min']:.2f}")
            lines.append(f"- Max: {score_stats['max']:.2f}")
            lines.append("")

        if stats.get("total_acquired_size_mb", 0) > 0:
            lines.append(
                f"- **Total Acquired Dataset Size**: {stats['total_acquired_size_mb']:.2f} MB"
            )

        return lines

    def _format_source_markdown(self, source: DatasetSource) -> List[str]:
        """Format a source as markdown."""
        lines = [
            f"### {source.title}",
            "",
            f"- **Source ID**: {source.source_id}",
            f"- **Authors**: {', '.join(source.authors)}",
            f"- **Publication Date**: {source.publication_date.strftime('%Y-%m-%d')}",
            f"- **Source Type**: {source.source_type}",
            f"- **URL**: {source.url}",
        ]

        if source.doi:
            lines.append(f"- **DOI**: {source.doi}")

        lines.extend(
            [
                f"- **Open Access**: {source.open_access}",
                f"- **Data Availability**: {source.data_availability}",
                f"- **Discovery Date**: {source.discovery_date.strftime('%Y-%m-%d %H:%M:%S')}",
            ]
        )

        if source.abstract:
            lines.extend(["", f"**Abstract**: {source.abstract[:200]}..."])

        if source.keywords:
            lines.append(f"**Keywords**: {', '.join(source.keywords)}")

        return lines

    def _format_evaluation_markdown(self, evaluation: DatasetEvaluation) -> List[str]:
        """Format an evaluation as markdown."""
        lines = [
            f"### Evaluation: {evaluation.source_id}",
            "",
            f"- **Therapeutic Relevance**: {evaluation.therapeutic_relevance}/10",
            f"- **Data Structure Quality**: {evaluation.data_structure_quality}/10",
            f"- **Training Integration**: {evaluation.training_integration}/10",
            f"- **Ethical Accessibility**: {evaluation.ethical_accessibility}/10",
            f"- **Overall Score**: {evaluation.overall_score:.2f}/10",
            f"- **Priority Tier**: {evaluation.priority_tier}",
        ]

        if evaluation.competitive_advantages:
            lines.append(f"- **Competitive Advantages**: {', '.join(evaluation.competitive_advantages)}")

        return lines

    def _format_acquired_dataset_markdown(self, dataset: AcquiredDataset) -> List[str]:
        """Format an acquired dataset as markdown."""
        lines = [
            f"### Acquired Dataset: {dataset.source_id}",
            "",
            f"- **Acquisition Date**: {dataset.acquisition_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Storage Path**: {dataset.storage_path}",
            f"- **File Format**: {dataset.file_format}",
            f"- **File Size**: {dataset.file_size_mb:.2f} MB",
            f"- **License**: {dataset.license}",
            f"- **Attribution Required**: {dataset.attribution_required}",
        ]

        if dataset.usage_restrictions:
            lines.append(f"- **Usage Restrictions**: {', '.join(dataset.usage_restrictions)}")

        return lines

    def _format_integration_plan_markdown(self, plan: IntegrationPlan) -> List[str]:
        """Format an integration plan as markdown."""
        lines = [
            f"### Integration Plan: {plan.source_id}",
            "",
            f"- **Dataset Format**: {plan.dataset_format}",
            f"- **Complexity**: {plan.complexity}",
            f"- **Estimated Effort**: {plan.estimated_effort_hours} hours",
            f"- **Integration Priority**: {plan.integration_priority}",
        ]

        if plan.schema_mapping:
            lines.append("")
            lines.append("**Schema Mapping**:")
            for dataset_field, pipeline_field in plan.schema_mapping.items():
                lines.append(f"- {dataset_field} -> {pipeline_field}")

        if plan.required_transformations:
            lines.append("")
            lines.append("**Required Transformations**:")
            for transformation in plan.required_transformations:
                lines.append(f"- {transformation}")

        return lines

    def _source_to_dict(self, source: DatasetSource) -> Dict:
        """Convert a source to a dictionary."""
        return {
            "source_id": source.source_id,
            "title": source.title,
            "authors": source.authors,
            "publication_date": source.publication_date.isoformat(),
            "source_type": source.source_type,
            "url": source.url,
            "doi": source.doi,
            "abstract": source.abstract,
            "keywords": source.keywords,
            "open_access": source.open_access,
            "data_availability": source.data_availability,
            "discovery_date": source.discovery_date.isoformat(),
            "discovery_method": source.discovery_method,
        }

    def _evaluation_to_dict(self, evaluation: DatasetEvaluation) -> Dict:
        """Convert an evaluation to a dictionary."""
        return {
            "source_id": evaluation.source_id,
            "therapeutic_relevance": evaluation.therapeutic_relevance,
            "data_structure_quality": evaluation.data_structure_quality,
            "training_integration": evaluation.training_integration,
            "ethical_accessibility": evaluation.ethical_accessibility,
            "overall_score": evaluation.overall_score,
            "priority_tier": evaluation.priority_tier,
            "evaluation_date": evaluation.evaluation_date.isoformat(),
            "competitive_advantages": evaluation.competitive_advantages,
        }

    def _acquired_dataset_to_dict(self, dataset: AcquiredDataset) -> Dict:
        """Convert an acquired dataset to a dictionary."""
        return {
            "source_id": dataset.source_id,
            "acquisition_date": dataset.acquisition_date.isoformat(),
            "storage_path": dataset.storage_path,
            "file_format": dataset.file_format,
            "file_size_mb": dataset.file_size_mb,
            "license": dataset.license,
            "usage_restrictions": dataset.usage_restrictions,
            "attribution_required": dataset.attribution_required,
            "checksum": dataset.checksum,
        }

    def _integration_plan_to_dict(self, plan: IntegrationPlan) -> Dict:
        """Convert an integration plan to a dictionary."""
        return {
            "source_id": plan.source_id,
            "dataset_format": plan.dataset_format,
            "schema_mapping": plan.schema_mapping,
            "required_transformations": plan.required_transformations,
            "preprocessing_steps": plan.preprocessing_steps,
            "complexity": plan.complexity,
            "estimated_effort_hours": plan.estimated_effort_hours,
            "dependencies": plan.dependencies,
            "integration_priority": plan.integration_priority,
            "created_date": plan.created_date.isoformat(),
        }

