#!/usr/bin/env python3
"""
Task 5.1: Priority Dataset Processing (Tier 1 - Production Ready)
Processes the highest quality, curated priority datasets from datasets-wendy/
These represent the gold standard for therapeutic conversation training.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.dataset_pipeline.logger import get_logger

logger = get_logger("task_5_1_priority_datasets")


class PriorityDatasetProcessor:
    """
    Processor for Tier 1 Priority Datasets - the highest quality therapeutic conversations.
    These datasets have been pre-curated and quality-scored for production use.
    """

    def __init__(self, output_dir: str = "data/processed/priority"):
        self.output_dir = output_dir
        self.datasets_dir = "ai/datasets/datasets-wendy"

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Priority dataset configurations (in processing order)
        self.priority_configs = [
            {
                "name": "priority_1",
                "path": f"{self.datasets_dir}/priority_1/priority_1_FINAL.jsonl",
                "summary_path": f"{self.datasets_dir}/priority_1/priority_1_FINAL_summary.json",
                "tier": 1,
                "expected_quality": 0.85,
                "description": "Top-tier therapeutic conversations (102K+ samples)"
            },
            {
                "name": "priority_2",
                "path": f"{self.datasets_dir}/priority_2/priority_2_FINAL.jsonl",
                "summary_path": f"{self.datasets_dir}/priority_2/priority_2_FINAL_summary.json",
                "tier": 2,
                "expected_quality": 0.80,
                "description": "High-quality mental health conversations"
            },
            {
                "name": "priority_3",
                "path": f"{self.datasets_dir}/priority_3/priority_3_FINAL.jsonl",
                "summary_path": f"{self.datasets_dir}/priority_3/priority_3_FINAL_summary.json",
                "tier": 3,
                "expected_quality": 0.75,
                "description": "Specialized therapeutic content"
            },
            {
                "name": "priority_4",
                "path": f"{self.datasets_dir}/priority_4/priority_4_FINAL.jsonl",
                "summary_path": f"{self.datasets_dir}/priority_4/priority_4_FINAL_summary.json",
                "tier": 4,
                "expected_quality": 0.70,
                "description": "Extended training data"
            },
            {
                "name": "priority_5",
                "path": f"{self.datasets_dir}/priority_5/priority_5_FINAL.jsonl",
                "summary_path": f"{self.datasets_dir}/priority_5/priority_5_FINAL_summary.json",
                "tier": 5,
                "expected_quality": 0.65,
                "description": "Supplementary datasets"
            }
        ]

        self.stats = {
            "datasets_processed": 0,
            "total_conversations": 0,
            "total_accepted": 0,
            "quality_filtered": 0,
            "format_errors": 0,
            "processing_errors": 0,
            "tier_distribution": {}
        }

    def process_all_priority_datasets(self) -> dict:
        """Process all priority datasets in order."""
        logger.info("Starting Task 5.1: Priority Dataset Processing")
        start_time = datetime.now()

        all_conversations = []
        dataset_results = {}

        for config in self.priority_configs:
            logger.info(f"Processing {config['name']} (Tier {config['tier']}): {config['description']}")

            try:
                # Load summary first to understand the dataset
                summary = self._load_summary(config["summary_path"])
                if summary:
                    logger.info(f"Summary - Total samples: {summary.get('total_samples', 'unknown')}")
                    logger.info(f"Summary - Sources: {summary.get('sources', {})}")

                # Process the dataset
                conversations = self._process_single_priority_dataset(config)
                all_conversations.extend(conversations)

                dataset_results[config["name"]] = {
                    "conversations_processed": len(conversations),
                    "tier": config["tier"],
                    "expected_quality": config["expected_quality"],
                    "description": config["description"],
                    "success": True,
                    "summary": summary
                }

                self.stats["datasets_processed"] += 1
                self.stats["tier_distribution"][f"tier_{config['tier']}"] = len(conversations)

                logger.info(f"Successfully processed {len(conversations)} conversations from {config['name']}")

            except Exception as e:
                logger.error(f"Failed to process {config['name']}: {e}")
                dataset_results[config["name"]] = {
                    "conversations_processed": 0,
                    "tier": config["tier"],
                    "success": False,
                    "error": str(e)
                }
                self.stats["processing_errors"] += 1

        # Save consolidated priority conversations
        output_path = os.path.join(self.output_dir, "priority_conversations_consolidated.jsonl")
        self._save_conversations(all_conversations, output_path)

        # Generate comprehensive report
        processing_time = (datetime.now() - start_time).total_seconds()
        report = self._generate_report(dataset_results, all_conversations, processing_time)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_1_priority_processing_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Task 5.1 completed. Processed {len(all_conversations)} priority conversations")
        return report

    def _load_summary(self, summary_path: str) -> dict:
        """Load dataset summary if available."""
        try:
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load summary from {summary_path}: {e}")
        return {}

    def _process_single_priority_dataset(self, config: dict) -> list:
        """Process a single priority dataset."""
        dataset_path = config["path"]

        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found: {dataset_path}")
            return []

        processed_conversations = []

        try:
            with open(dataset_path) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Parse JSON line
                        item = json.loads(line.strip())
                        self.stats["total_conversations"] += 1

                        # Process conversation
                        processed = self._process_priority_conversation(item, config, line_num)
                        if processed:
                            processed_conversations.append(processed)
                            self.stats["total_accepted"] += 1

                        # Progress logging
                        if line_num % 10000 == 0:
                            logger.info(f"Processed {line_num} lines from {config['name']}")

                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num} in {config['name']}: {e}")
                        self.stats["format_errors"] += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num} in {config['name']}: {e}")
                        self.stats["processing_errors"] += 1
                        continue

        except Exception as e:
            logger.error(f"Failed to read dataset {dataset_path}: {e}")
            raise

        return processed_conversations

    def _process_priority_conversation(self, item: dict, config: dict, line_num: int) -> dict:
        """Process a single priority conversation."""
        try:
            # Extract conversation data - priority datasets should have standardized format
            conversation_data = None

            # Try different possible formats
            if "conversation" in item:
                conversation_data = item["conversation"]
            elif "messages" in item:
                conversation_data = item["messages"]
            elif "dialogue" in item:
                conversation_data = item["dialogue"]
            else:
                # If no standard format, try to extract from the item itself
                logger.warning(f"Unknown format in {config['name']} line {line_num}: {list(item.keys())}")
                self.stats["format_errors"] += 1
                return None

            if not conversation_data or not isinstance(conversation_data, list):
                self.stats["format_errors"] += 1
                return None

            # Basic quality checks
            if len(conversation_data) < 2:
                self.stats["quality_filtered"] += 1
                return None

            # Check conversation length and content quality
            total_content = " ".join([msg.get("content", "") for msg in conversation_data if isinstance(msg, dict)])
            if len(total_content) < 50:  # Too short
                self.stats["quality_filtered"] += 1
                return None

            # Extract quality score if available
            quality_score = item.get("quality_score", config["expected_quality"])

            # Filter by quality threshold
            if quality_score < (config["expected_quality"] - 0.1):  # Allow some tolerance
                self.stats["quality_filtered"] += 1
                return None

            # Create standardized conversation format
            standardized = {
                "conversation": conversation_data,
                "metadata": {
                    "category": "priority_therapeutic",
                    "tier": config["tier"],
                    "priority_level": config["name"],
                    "quality_score": quality_score,
                    "expected_quality": config["expected_quality"],
                    "source_dataset": config["name"],
                    "description": config["description"],
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_line": line_num,
                    "conversation_length": len(conversation_data),
                    "total_content_length": len(total_content),
                    "therapeutic_indicators": self._extract_therapeutic_indicators(total_content),
                    "conversation_type": self._classify_conversation_type(conversation_data),
                    "emotional_complexity": self._assess_emotional_complexity(total_content)
                }
            }

            # Add any additional metadata from the original item
            if "metadata" in item:
                standardized["metadata"]["original_metadata"] = item["metadata"]

            return standardized

        except Exception as e:
            logger.warning(f"Failed to process conversation at line {line_num} in {config['name']}: {e}")
            return None

    def _extract_therapeutic_indicators(self, content: str) -> list:
        """Extract therapeutic approach indicators from conversation content."""
        content_lower = content.lower()
        indicators = []

        therapeutic_patterns = {
            "cognitive_behavioral": ["thought", "thinking", "belief", "cognitive", "behavior", "cbt"],
            "emotion_focused": ["feeling", "emotion", "emotional", "feel", "hurt", "pain"],
            "solution_focused": ["goal", "solution", "strength", "resource", "progress"],
            "psychodynamic": ["past", "childhood", "relationship", "pattern", "unconscious"],
            "mindfulness": ["mindful", "present", "awareness", "meditation", "breathing"],
            "supportive": ["support", "understand", "listen", "here for you", "care"]
        }

        for approach, keywords in therapeutic_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                indicators.append(approach)

        return indicators if indicators else ["general_therapeutic"]

    def _classify_conversation_type(self, conversation_data: list) -> str:
        """Classify the type of therapeutic conversation."""
        if len(conversation_data) < 4:
            return "brief_interaction"
        if len(conversation_data) < 10:
            return "standard_session"
        return "extended_session"

    def _assess_emotional_complexity(self, content: str) -> float:
        """Assess the emotional complexity of the conversation."""
        content_lower = content.lower()

        # Emotional complexity indicators
        complex_emotions = ["ambivalent", "conflicted", "overwhelmed", "confused", "mixed feelings"]
        emotional_depth = ["deep", "profound", "intense", "overwhelming", "complex"]
        emotional_range = ["angry", "sad", "happy", "anxious", "excited", "frustrated", "hopeful"]

        complexity_score = 0.0

        # Count complex emotional expressions
        complexity_score += sum(0.2 for term in complex_emotions if term in content_lower)
        complexity_score += sum(0.1 for term in emotional_depth if term in content_lower)
        complexity_score += min(sum(0.05 for term in emotional_range if term in content_lower), 0.3)

        return min(complexity_score, 1.0)

    def _save_conversations(self, conversations: list, output_path: str) -> None:
        """Save conversations to JSONL file."""
        with open(output_path, "w") as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + "\n")

        logger.info(f"Saved {len(conversations)} priority conversations to {output_path}")

    def _generate_report(self, dataset_results: dict, conversations: list, processing_time: float) -> dict:
        """Generate comprehensive processing report."""
        return {
            "task": "5.1: Priority Dataset Processing (Tier 1 - Production Ready)",
            "processing_summary": {
                "total_conversations": len(conversations),
                "processing_time_seconds": processing_time,
                "datasets_processed": self.stats["datasets_processed"],
                "average_processing_rate": len(conversations) / processing_time if processing_time > 0 else 0
            },
            "quality_metrics": {
                "total_conversations_examined": self.stats["total_conversations"],
                "total_accepted": self.stats["total_accepted"],
                "acceptance_rate": (self.stats["total_accepted"] / max(self.stats["total_conversations"], 1)) * 100,
                "quality_filtered": self.stats["quality_filtered"],
                "format_errors": self.stats["format_errors"],
                "processing_errors": self.stats["processing_errors"]
            },
            "tier_distribution": self.stats["tier_distribution"],
            "dataset_results": dataset_results,
            "conversation_analysis": {
                "therapeutic_indicators": self._analyze_therapeutic_indicators(conversations),
                "conversation_types": self._analyze_conversation_types(conversations),
                "emotional_complexity_distribution": self._analyze_emotional_complexity(conversations),
                "quality_score_distribution": self._analyze_quality_scores(conversations)
            },
            "timestamp": datetime.now().isoformat()
        }

    def _analyze_therapeutic_indicators(self, conversations: list) -> dict:
        """Analyze therapeutic approach distribution."""
        indicator_counts = {}
        for conv in conversations:
            indicators = conv.get("metadata", {}).get("therapeutic_indicators", [])
            for indicator in indicators:
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
        return indicator_counts

    def _analyze_conversation_types(self, conversations: list) -> dict:
        """Analyze conversation type distribution."""
        type_counts = {}
        for conv in conversations:
            conv_type = conv.get("metadata", {}).get("conversation_type", "unknown")
            type_counts[conv_type] = type_counts.get(conv_type, 0) + 1
        return type_counts

    def _analyze_emotional_complexity(self, conversations: list) -> dict:
        """Analyze emotional complexity distribution."""
        complexity_ranges = {"low": 0, "medium": 0, "high": 0}
        for conv in conversations:
            complexity = conv.get("metadata", {}).get("emotional_complexity", 0.0)
            if complexity < 0.3:
                complexity_ranges["low"] += 1
            elif complexity < 0.7:
                complexity_ranges["medium"] += 1
            else:
                complexity_ranges["high"] += 1
        return complexity_ranges

    def _analyze_quality_scores(self, conversations: list) -> dict:
        """Analyze quality score distribution."""
        score_ranges = {"0.6-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, "0.9-1.0": 0}
        for conv in conversations:
            score = conv.get("metadata", {}).get("quality_score", 0.0)
            if 0.6 <= score < 0.7:
                score_ranges["0.6-0.7"] += 1
            elif 0.7 <= score < 0.8:
                score_ranges["0.7-0.8"] += 1
            elif 0.8 <= score < 0.9:
                score_ranges["0.8-0.9"] += 1
            elif score >= 0.9:
                score_ranges["0.9-1.0"] += 1
        return score_ranges


def main():
    """Main execution function for Task 5.1."""

    try:
        # Create processor
        processor = PriorityDatasetProcessor()

        # Process all priority datasets
        result = processor.process_all_priority_datasets()

        # Print results


        for _tier, _count in result["tier_distribution"].items():
            pass

        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
