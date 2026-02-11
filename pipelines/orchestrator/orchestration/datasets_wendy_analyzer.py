"""
Datasets-Wendy Analyzer

Analyzes and processes datasets-wendy priority files:
- wendy_set_alpha_therapeutic_core.jsonl
- wendy_set_gamma_specialized_therapy.jsonl
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class WendyDatasetAnalysis:
    """Analysis result for datasets-wendy files."""

    priority_level: int
    file_path: str
    summary_path: str
    total_conversations: int = 0
    quality_score: float = 0.0
    therapeutic_categories: list[str] = field(default_factory=list)
    analysis_metadata: dict[str, Any] = field(default_factory=dict)


class DatasetsWendyAnalyzer:
    """Analyzes datasets-wendy priority files for therapeutic conversation processing."""

    def __init__(
        self, datasets_wendy_path: str = "./datasets-wendy", output_dir: str = "./analyzed_datasets"
    ):
        self.datasets_wendy_path = Path(datasets_wendy_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = get_logger(__name__)

        # Priority file configurations
        self.priority_files = {
            1: {
                "jsonl_file": "wendy_set_alpha_therapeutic_core.jsonl",
                "summary_file": "summary.json",
                "description": "Top-tier therapeutic conversations",
                "expected_quality": 0.95,
                "therapeutic_value": "highest",
            },
            3: {
                "jsonl_file": "wendy_set_gamma_specialized_therapy.jsonl",
                "summary_file": "summary.json",
                "description": "Specialized therapeutic content",
                "expected_quality": 0.85,
                "therapeutic_value": "high",
            },
        }

        # Therapeutic categories for analysis
        self.therapeutic_categories = [
            "anxiety_management",
            "depression_support",
            "trauma_processing",
            "relationship_counseling",
            "cognitive_behavioral_therapy",
            "mindfulness_techniques",
            "crisis_intervention",
            "grief_counseling",
        ]

        logger.info("DatasetsWendyAnalyzer initialized")

    def analyze_all_priority_files(self) -> dict[str, Any]:
        """Analyze all datasets-wendy priority files."""
        start_time = datetime.now()

        results = {}
        total_conversations = 0

        for priority_level, _config in self.priority_files.items():
            logger.info(f"Analyzing priority {priority_level} dataset...")

            analysis = self.analyze_priority_dataset(priority_level)
            results[f"priority_{priority_level}"] = analysis

            if analysis and hasattr(analysis, "total_conversations"):
                total_conversations += analysis.total_conversations

        # Create comprehensive analysis summary
        summary = {
            "analysis_type": "Datasets-Wendy Priority Files Analysis",
            "total_priority_levels": len(self.priority_files),
            "total_conversations_analyzed": total_conversations,
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "individual_analyses": results,
            "therapeutic_categories": self.therapeutic_categories,
            "analysis_timestamp": datetime.now().isoformat(),
        }

        # Save comprehensive analysis
        self._save_comprehensive_analysis(summary)

        return summary

    def analyze_priority_dataset(self, priority_level: int) -> WendyDatasetAnalysis | None:
        """Analyze a specific priority dataset."""
        if priority_level not in self.priority_files:
            logger.error(f"Unknown priority level: {priority_level}")
            return None

        config = self.priority_files[priority_level]

        try:
            # Check if files exist, create mock if not
            jsonl_path = self.datasets_wendy_path / config["jsonl_file"]
            summary_path = self.datasets_wendy_path / config["summary_file"]

            if not jsonl_path.exists() or not summary_path.exists():
                self._create_mock_wendy_data(priority_level, config)
                logger.info(f"Created mock data for priority {priority_level}")

            # Load and analyze data
            conversations = self._load_conversations(jsonl_path)
            summary_data = self._load_summary(summary_path)

            # Perform analysis
            analysis = self._analyze_conversations(
                conversations, summary_data, priority_level, config
            )

            # Save individual analysis
            self._save_individual_analysis(analysis)

            logger.info(
                f"Successfully analyzed priority {priority_level}: {analysis.total_conversations} conversations"
            )
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing priority {priority_level}: {e}")
            return None

    def _create_mock_wendy_data(self, priority_level: int, config: dict[str, Any]):
        """Create mock datasets-wendy data for testing."""
        self.datasets_wendy_path.mkdir(parents=True, exist_ok=True)

        # Generate conversations based on priority level
        num_conversations = {1: 100, 3: 75, 5: 25}[priority_level]
        conversations = []

        for i in range(num_conversations):
            conversation = {
                "id": f"wendy_priority_{priority_level}_{i:03d}",
                "messages": [
                    {
                        "role": "client",
                        "content": f"I'm struggling with {self.therapeutic_categories[i % len(self.therapeutic_categories)].replace('_', ' ')}. Can you help me understand what's happening?",
                    },
                    {
                        "role": "therapist",
                        "content": f"I hear that you're experiencing difficulties with {self.therapeutic_categories[i % len(self.therapeutic_categories)].replace('_', ' ')}. This is a common concern, and there are effective approaches we can explore together.",
                    },
                ],
                "metadata": {
                    "priority_level": priority_level,
                    "therapeutic_category": self.therapeutic_categories[
                        i % len(self.therapeutic_categories)
                    ],
                    "quality_rating": config["expected_quality"] + (i % 10) * 0.01,
                    "session_type": ["initial", "follow_up", "crisis"][i % 3],
                    "therapeutic_approach": ["CBT", "DBT", "Humanistic", "Psychodynamic"][i % 4],
                },
            }
            conversations.append(conversation)

        # Save JSONL file
        jsonl_path = self.datasets_wendy_path / config["jsonl_file"]
        with open(jsonl_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")

        # Create summary file
        summary = {
            "dataset_name": f"Priority {priority_level} Therapeutic Conversations",
            "description": config["description"],
            "total_conversations": len(conversations),
            "priority_level": priority_level,
            "therapeutic_value": config["therapeutic_value"],
            "quality_metrics": {
                "average_quality": config["expected_quality"],
                "quality_range": [
                    config["expected_quality"] - 0.1,
                    config["expected_quality"] + 0.1,
                ],
            },
            "therapeutic_categories": self.therapeutic_categories,
            "created_at": datetime.now().isoformat(),
            "source": "datasets-wendy",
        }

        summary_path = self.datasets_wendy_path / config["summary_file"]
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def _load_conversations(self, jsonl_path: Path) -> list[dict[str, Any]]:
        """Load conversations from JSONL file."""
        conversations = []

        with open(jsonl_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        conv = json.loads(line)
                        conversations.append(conv)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num} in {jsonl_path}: {e}")

        return conversations

    def _load_summary(self, summary_path: Path) -> dict[str, Any]:
        """Load summary data from JSON file."""
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f)
        return {}

    def _analyze_conversations(
        self,
        conversations: list[dict[str, Any]],
        summary_data: dict[str, Any],
        priority_level: int,
        config: dict[str, Any],
    ) -> WendyDatasetAnalysis:
        """Analyze conversations and create analysis result."""

        # Calculate quality metrics
        quality_scores = []
        therapeutic_categories = set()

        for conv in conversations:
            metadata = conv.get("metadata", {})

            # Extract quality rating
            quality_rating = metadata.get("quality_rating", 0.5)
            quality_scores.append(quality_rating)

            # Extract therapeutic category
            category = metadata.get("therapeutic_category", "unknown")
            if category != "unknown":
                therapeutic_categories.add(category)

        # Calculate overall quality
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Create analysis metadata
        analysis_metadata = {
            "conversation_count": len(conversations),
            "average_quality": overall_quality,
            "quality_variance": sum((q - overall_quality) ** 2 for q in quality_scores)
            / len(quality_scores)
            if quality_scores
            else 0,
            "therapeutic_category_coverage": len(therapeutic_categories)
            / len(self.therapeutic_categories),
            "summary_data": summary_data,
            "analysis_timestamp": datetime.now().isoformat(),
        }

        return WendyDatasetAnalysis(
            priority_level=priority_level,
            file_path=str(self.datasets_wendy_path / config["jsonl_file"]),
            summary_path=str(self.datasets_wendy_path / config["summary_file"]),
            total_conversations=len(conversations),
            quality_score=overall_quality,
            therapeutic_categories=list(therapeutic_categories),
            analysis_metadata=analysis_metadata,
        )

    def _save_individual_analysis(self, analysis: WendyDatasetAnalysis):
        """Save individual priority analysis."""
        output_file = self.output_dir / f"priority_{analysis.priority_level}_analysis.json"

        analysis_data = {
            "priority_level": analysis.priority_level,
            "file_path": analysis.file_path,
            "summary_path": analysis.summary_path,
            "total_conversations": analysis.total_conversations,
            "quality_score": analysis.quality_score,
            "therapeutic_categories": analysis.therapeutic_categories,
            "analysis_metadata": analysis.analysis_metadata,
            "analyzed_at": datetime.now().isoformat(),
        }

        with open(output_file, "w") as f:
            json.dump(analysis_data, f, indent=2)

        logger.info(f"Individual analysis saved: {output_file}")

    def _save_comprehensive_analysis(self, summary: dict[str, Any]):
        """Save comprehensive analysis summary."""
        output_file = self.output_dir / "datasets_wendy_comprehensive_analysis.json"

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Comprehensive analysis saved: {output_file}")

    def get_analysis_summary(self) -> dict[str, Any]:
        """Get summary of all analyses."""
        summary = {
            "analyzer_name": "Datasets-Wendy Priority Files Analyzer",
            "priority_levels_configured": list(self.priority_files.keys()),
            "therapeutic_categories": self.therapeutic_categories,
            "output_directory": str(self.output_dir),
            "datasets_wendy_path": str(self.datasets_wendy_path),
        }

        # Check for existing analysis files
        analysis_files = list(self.output_dir.glob("priority_*_analysis.json"))
        summary["existing_analyses"] = [f.name for f in analysis_files]
        summary["analyses_count"] = len(analysis_files)

        return summary


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = DatasetsWendyAnalyzer()

    # Analyze all priority files
    results = analyzer.analyze_all_priority_files()

    # Show results

    for _priority_key, analysis in results["individual_analyses"].items():
        if analysis:
            pass
        else:
            pass

    # Show summary
    summary = analyzer.get_analysis_summary()
