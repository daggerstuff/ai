#!/usr/bin/env python3
"""
Seamless Edge Case Integration Script
Automatically integrates edge case generation with main training pipeline
"""

import json
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.edge_case_pipeline_standalone.edge_case_generator import (
    EdgeCaseGenerator,
)

from scripts.enhanced_conversion_pipeline import EnhancedConversionPipeline


class EdgeCaseIntegrator:
    """Seamlessly integrate edge cases into main training pipeline"""

    def __init__(
        self,
        edge_cases_per_category: int = 10,
        max_edge_conversations: int = 250,
        output_dir: str = "training_data_consolidated",
    ):
        self.edge_cases_per_category = edge_cases_per_category
        self.max_edge_conversations = max_edge_conversations
        self.output_dir = Path(output_dir)

        # Setup generators
        self.edge_generator = EdgeCaseGenerator(
            api_provider="ollama",
            model_name="artifish/llama3.2-uncensored",
            output_dir=self.output_dir / "edge_cases" / "generated",
        )

        self.conversion_pipeline = EnhancedConversionPipeline()

    def generate_and_integrate_edge_cases(self) -> dict:
        """Generate edge cases and integrate into main training data"""

        # Step 1: Generate edge case prompts
        edge_prompts = self.edge_generator.generate_prompts(scenarios_per_category=self.edge_cases_per_category)

        # Step 2: Generate conversations from prompts
        edge_conversations = self.edge_generator.generate_conversations(
            edge_prompts, max_conversations=self.max_edge_conversations
        )

        # Step 3: Convert to training format
        training_format = self.edge_generator.create_training_format(edge_conversations)

        # Step 4: Integrate with main training data
        integration_stats = self._integrate_with_main_pipeline(training_format)

        # Step 5: Generate comprehensive report
        report = self._generate_integration_report(edge_conversations, integration_stats)

        return {
            "edge_conversations": len(edge_conversations),
            "training_pairs": len(training_format),
            "integration_stats": integration_stats,
            "report": report,
        }

    def _integrate_with_main_pipeline(self, edge_training_data: list[dict]) -> dict:
        """Integrate edge cases with main training data"""

        # Get main training data location
        main_training_dir = self.output_dir / "final_datasets"
        main_training_dir.mkdir(parents=True, exist_ok=True)

        # Save edge cases as separate training file
        edge_training_file = main_training_dir / "edge_cases_training.jsonl"
        self.edge_generator._write_jsonl_file(edge_training_file, edge_training_data)

        # Merge with existing training data if available
        merged_file = main_training_dir / "combined_training_with_edge_cases.jsonl"
        existing_data = []

        # Load existing training data
        existing_files = [
            main_training_dir / "therapeutic_conversations.jsonl",
            main_training_dir / "psychological_training.jsonl",
            main_training_dir / "clinical_scenarios.jsonl",
        ]

        for file_path in existing_files:
            if file_path.exists():
                with open(file_path) as f:
                    for line in f:
                        existing_data.append(json.loads(line))

        # Combine datasets
        combined_data = existing_data + edge_training_data

        # Write combined dataset
        self.edge_generator._write_jsonl_file(merged_file, combined_data)

        return {
            "existing_training_pairs": len(existing_data),
            "edge_case_pairs": len(edge_training_data),
            "total_combined_pairs": len(combined_data),
            "edge_case_file": str(edge_training_file),
            "combined_file": str(merged_file),
        }

    def _generate_integration_report(self, conversations: list[dict], integration_stats: dict) -> str:
        """Generate comprehensive integration report"""

        # Analyze edge case distribution
        category_counts = {}
        difficulty_counts = {}
        unwinnable_count = 0

        for conv in conversations:
            cat = conv.get("category", "unknown")
            diff = conv.get("difficulty_level", "unknown")

            category_counts[cat] = category_counts.get(cat, 0) + 1
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

            if diff == "very_high":
                unwinnable_count += 1

        report = f"""# 🎯 Edge Case Integration Report
Generated: {Path().cwd().name} Training Pipeline

## 📊 Integration Summary
- **Edge Conversations Generated**: {len(conversations)}
- **Training Pairs Created**: {integration_stats["edge_case_pairs"]}
- **Existing Training Pairs**: {integration_stats["existing_training_pairs"]}
- **Total Combined Pairs**: {integration_stats["total_combined_pairs"]}

## 🔥 Edge Case Breakdown
- **"Unwinnable" Cases (Very High)**: {unwinnable_count} ({unwinnable_count / len(conversations) * 100:.1f}%)
- **High Difficulty Cases**: {difficulty_counts.get("high", 0)}
- **Moderate Difficulty Cases**: {difficulty_counts.get("moderate", 0)}

## 📂 Files Generated
- **Edge Cases Only**: `{integration_stats["edge_case_file"]}`
- **Combined Training Data**: `{integration_stats["combined_file"]}`

## 🎯 Category Distribution (Top 10)
"""

        # Add top categories
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = count / len(conversations) * 100
            report += f"- **{category}**: {count} cases ({percentage:.1f}%)\n"

        report += f"""
## 🚀 Ready for Training!
Your edge cases are now seamlessly integrated with your main training pipeline.

**Next Steps:**
1. Review generated conversations in: `{integration_stats["edge_case_file"]}`
2. Train your model using: `{integration_stats["combined_file"]}`
3. Test model performance on edge cases
4. Adjust edge case ratios if needed

**Training Command:**
```bash
cd ai/lightning
uv run python train.py --data_file="../{integration_stats["combined_file"]}" --edge_case_enhanced=true
```
"""

        # Save report
        report_file = self.output_dir / "edge_cases" / "integration_report.md"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w") as f:
            f.write(report)

        return report


def main():
    """Run edge case integration"""

    integrator = EdgeCaseIntegrator(
        edge_cases_per_category=15,  # 375 total prompts
        max_edge_conversations=300,  # Up to 300 conversations
        output_dir="training_data_consolidated",
    )

    integrator.generate_and_integrate_edge_cases()


if __name__ == "__main__":
    main()
