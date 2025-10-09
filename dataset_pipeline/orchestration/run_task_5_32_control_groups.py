#!/usr/bin/env python3
"""
Task 5.32: Process control group datasets (fitness, jokes, meditation, personalfinance - non-clinical baselines)
Non-clinical baseline datasets for comparison and model training balance.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd


class ControlGroupProcessor:
    """Processor for control group (non-clinical baseline) datasets."""

    def __init__(self, output_dir: str = "ai/data/processed/phase_4_reddit_mental_health/task_5_32_control_groups"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define control group datasets
        self.control_groups = [
            "fitness", "jokes", "meditation", "personalfinance"
        ]

        # Define temporal periods
        self.temporal_periods = ["2018", "2019", "pre", "post"]

        # Processing statistics
        self.processing_stats = {
            "total_files_processed": 0,
            "total_conversations_created": 0,
            "total_posts_processed": 0,
            "control_group_distribution": defaultdict(int),
            "temporal_distribution": defaultdict(int),
            "quality_metrics": defaultdict(int)
        }

    def process_all_control_groups(self) -> dict[str, Any]:
        """Process all control group datasets."""

        all_conversations = []
        control_group_reports = {}

        for control_group in self.control_groups:
            conversations, report = self.process_control_group(control_group)

            all_conversations.extend(conversations)
            control_group_reports[control_group] = report


        # Save consolidated conversations
        consolidated_path = os.path.join(self.output_dir, "control_group_conversations.jsonl")
        with open(consolidated_path, "w", encoding="utf-8") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Generate comprehensive report
        report = self.generate_comprehensive_report(control_group_reports, all_conversations)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_32_control_groups_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


        return report

    def process_control_group(self, control_group: str) -> tuple[list[dict], dict]:
        """Process all temporal datasets for a specific control group."""
        control_conversations = []
        temporal_stats = {}

        for period in self.temporal_periods:
            # Construct file path
            if period in ["2018", "2019"]:
                filename = f"{control_group}_{period}_features_tfidf_256.csv"
            else:
                filename = f"{control_group}_{period}_features_tfidf_256.csv"

            file_path = f"../../ai/datasets/reddit_mental_health/{filename}"

            if os.path.exists(file_path):
                conversations, stats = self.process_csv_file(file_path, control_group, period)
                control_conversations.extend(conversations)
                temporal_stats[period] = stats

                self.processing_stats["total_files_processed"] += 1
                self.processing_stats["control_group_distribution"][control_group] += len(conversations)
                self.processing_stats["temporal_distribution"][period] += len(conversations)
            else:
                # Silently skip missing temporal periods - this is normal
                temporal_stats[period] = {"status": "no_data_available", "conversations": 0}

        control_report = {
            "control_group": control_group,
            "total_conversations": len(control_conversations),
            "temporal_breakdown": temporal_stats,
            "processing_timestamp": datetime.now().isoformat()
        }

        return control_conversations, control_report

    def process_csv_file(self, file_path: str, control_group: str, period: str) -> tuple[list[dict], dict]:
        """Process a single control group CSV file."""
        conversations = []
        stats = {"total_rows": 0, "processed_rows": 0, "errors": 0}

        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            stats["total_rows"] = len(df)


            # Process each row
            for idx, row in df.iterrows():
                try:
                    conversation = self.create_conversation_from_row(row, control_group, period, idx)
                    if conversation:
                        conversations.append(conversation)
                        stats["processed_rows"] += 1

                        self.processing_stats["total_posts_processed"] += 1
                        self.processing_stats["total_conversations_created"] += 1

                except Exception:
                    stats["errors"] += 1
                    continue

                # Progress indicator for large files
                if (idx + 1) % 1000 == 0:
                    pass


        except Exception as e:
            stats["file_error"] = str(e)

        return conversations, stats

    def create_conversation_from_row(self, row: pd.Series, control_group: str, period: str, idx: int) -> dict[str, Any]:
        """Create a conversation from a control group row."""
        try:
            # Extract text content
            text_content = ""
            possible_text_columns = ["text", "post", "content", "body", "title", "selftext"]

            for col in possible_text_columns:
                if col in row.index and pd.notna(row[col]):
                    text_content = str(row[col]).strip()
                    break

            if not text_content or len(text_content) < 20:
                return None

            # Extract TF-IDF features from token columns (excluding metadata columns)
            tfidf_features = {}
            metadata_columns = {"text", "post", "content", "body", "title", "selftext", "id", "author", "created_utc", "score", "num_comments"}

            for col in row.index:
                if col not in metadata_columns and pd.notna(row[col]) and isinstance(row[col], (int, float)) and row[col] != 0:
                    tfidf_features[col] = float(row[col])

            # Create non-clinical conversation
            client_message = self.adapt_control_post_to_client_message(text_content, control_group)
            therapist_response = self.generate_control_therapeutic_response(text_content, control_group, period)

            messages = [
                {
                    "role": "client",
                    "content": client_message,
                    "turn_id": 1
                },
                {
                    "role": "therapist",
                    "content": therapist_response,
                    "turn_id": 2
                }
            ]

            return {
                "conversation": messages,
                "metadata": {
                    "category": "reddit_control_group_non_clinical",
                    "tier": 4,
                    "source_dataset": f"{control_group}_{period}_reddit",
                    "control_group": control_group,
                    "temporal_period": period,
                    "original_text": text_content,
                    "tfidf_features": tfidf_features,
                    "feature_count": len(tfidf_features),
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_row_id": idx,
                    "conversation_length": len(messages),
                    "text_length": len(text_content),
                    "quality_score": self.assess_conversation_quality(text_content, control_group),
                    "therapeutic_relevance": self.assess_therapeutic_relevance(text_content, control_group),
                    "control_group_type": self.classify_control_group_type(control_group),
                    "is_clinical": False,
                    "baseline_category": self.get_baseline_category(control_group)
                }
            }

        except Exception:
            return None

    def adapt_control_post_to_client_message(self, text: str, control_group: str) -> str:
        """Adapt a control group post to sound like a client message."""
        # Clean and adapt the text
        adapted_text = text.replace("[removed]", "").replace("[deleted]", "").strip()

        # Add appropriate framing based on control group
        control_group_intros = {
            "fitness": "I wanted to share something about my fitness journey: ",
            "jokes": "I wanted to share something that made me laugh: ",
            "meditation": "I've been practicing meditation and wanted to discuss: ",
            "personalfinance": "I wanted to talk about a financial situation: "
        }

        intro = control_group_intros.get(control_group, "I wanted to share something positive: ")
        return intro + adapted_text

    def generate_control_therapeutic_response(self, text: str, control_group: str, period: str) -> str:
        """Generate an appropriate therapeutic response for control group content."""
        # Base positive response
        base_response = "Thank you for sharing this positive aspect of your life. "

        # Control group specific responses
        control_responses = {
            "fitness": "It's wonderful to hear about your commitment to physical health. Exercise and fitness can have significant positive impacts on mental wellbeing too. ",
            "jokes": "Humor and laughter are important for mental health. It's great that you're finding joy and lightness in life. ",
            "meditation": "Meditation and mindfulness practices are excellent tools for mental health and emotional regulation. ",
            "personalfinance": "Financial wellness and planning can reduce stress and contribute to overall mental wellbeing. "
        }

        control_response = control_responses.get(control_group, "It's important to maintain positive activities and interests. ")

        # Add temporal context if relevant
        temporal_context = ""
        if period == "pre":
            temporal_context = "It's good to see you engaging in positive activities. "
        elif period == "post":
            temporal_context = "Continuing with positive activities is beneficial for maintaining wellbeing. "
        elif period in ["2018", "2019"]:
            temporal_context = f"Looking back at this positive experience from {period} shows the importance of maintaining good habits. "

        return base_response + temporal_context + control_response + "How do you feel this contributes to your overall wellbeing?"

    def assess_conversation_quality(self, text: str, control_group: str) -> float:
        """Assess the quality of the control group conversation."""
        quality_score = 0.0

        # Length check
        if len(text) >= 50:
            quality_score += 0.3

        # Control group relevance
        control_keywords = {
            "fitness": ["exercise", "workout", "gym", "fitness", "health", "training"],
            "jokes": ["funny", "laugh", "humor", "joke", "hilarious", "comedy"],
            "meditation": ["meditation", "mindfulness", "peace", "calm", "breathe", "zen"],
            "personalfinance": ["money", "finance", "budget", "investment", "savings", "financial"]
        }

        keywords = control_keywords.get(control_group, [])
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)

        if keyword_matches > 0:
            quality_score += 0.4

        # Positive content indicators
        positive_indicators = ["good", "great", "happy", "positive", "success", "achievement", "progress"]
        positive_matches = sum(1 for indicator in positive_indicators if indicator in text_lower)

        if positive_matches > 0:
            quality_score += 0.3

        return min(quality_score, 1.0)

    def assess_therapeutic_relevance(self, text: str, control_group: str) -> float:
        """Assess therapeutic relevance of control group content."""
        text_lower = text.lower()

        # Wellbeing indicators (lower relevance for control groups)
        wellbeing_terms = [
            "wellbeing", "wellness", "health", "positive", "good", "beneficial",
            "helpful", "improvement", "progress", "success"
        ]

        relevance_score = 0.0
        for term in wellbeing_terms:
            if term in text_lower:
                relevance_score += 0.05  # Lower scores for control groups

        return min(relevance_score, 0.5)  # Cap at 0.5 for control groups

    def classify_control_group_type(self, control_group: str) -> str:
        """Classify the type of control group."""
        control_types = {
            "fitness": "physical_wellness",
            "jokes": "entertainment_humor",
            "meditation": "mindfulness_wellness",
            "personalfinance": "financial_wellness"
        }
        return control_types.get(control_group, "general_non_clinical")

    def get_baseline_category(self, control_group: str) -> str:
        """Get the baseline category for the control group."""
        baseline_categories = {
            "fitness": "wellness_positive",
            "jokes": "entertainment_positive",
            "meditation": "wellness_mindfulness",
            "personalfinance": "practical_life_skills"
        }
        return baseline_categories.get(control_group, "general_baseline")

    def generate_comprehensive_report(self, control_group_reports: dict, all_conversations: list) -> dict[str, Any]:
        """Generate comprehensive control group processing report."""

        # Analyze overall statistics
        total_conversations = len(all_conversations)
        avg_quality = sum(conv.get("metadata", {}).get("quality_score", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0
        avg_relevance = sum(conv.get("metadata", {}).get("therapeutic_relevance", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0

        # Feature analysis
        total_features = sum(conv.get("metadata", {}).get("feature_count", 0) for conv in all_conversations)
        avg_features = total_features / total_conversations if total_conversations > 0 else 0

        # Control group type analysis
        control_type_breakdown = {}
        for conv in all_conversations:
            control_type = conv.get("metadata", {}).get("control_group_type", "unknown")
            control_type_breakdown[control_type] = control_type_breakdown.get(control_type, 0) + 1

        return {
            "task": "5.32: Control Group Dataset Processing",
            "processing_summary": {
                "total_control_groups_processed": len(self.control_groups),
                "total_temporal_periods": len(self.temporal_periods),
                "total_files_processed": self.processing_stats["total_files_processed"],
                "total_conversations_created": total_conversations,
                "total_posts_processed": self.processing_stats["total_posts_processed"],
                "processing_timestamp": datetime.now().isoformat()
            },
            "control_group_breakdown": dict(self.processing_stats["control_group_distribution"]),
            "temporal_breakdown": dict(self.processing_stats["temporal_distribution"]),
            "control_type_breakdown": control_type_breakdown,
            "quality_metrics": {
                "average_quality_score": avg_quality,
                "average_therapeutic_relevance": avg_relevance,
                "average_tfidf_features": avg_features,
                "total_tfidf_features": total_features
            },
            "detailed_control_group_reports": control_group_reports,
            "dataset_info": {
                "tier": 4,
                "category": "reddit_control_group_non_clinical",
                "control_groups_covered": self.control_groups,
                "temporal_periods": self.temporal_periods,
                "feature_type": "tfidf_256_dimensional",
                "baseline_purpose": "Non-clinical comparison and model training balance"
            },
            "baseline_analysis": {
                "purpose": "Provide non-clinical baselines for mental health model training",
                "categories": ["physical_wellness", "entertainment_humor", "mindfulness_wellness", "financial_wellness"],
                "therapeutic_value": "Low therapeutic relevance by design - serves as control/comparison data",
                "ml_applications": ["Binary classification (clinical vs non-clinical)", "Baseline comparison", "Model calibration"]
            }
        }

def process_control_groups():
    """Main function to run control group processing."""
    processor = ControlGroupProcessor()
    return processor.process_all_control_groups()

if __name__ == "__main__":
    process_control_groups()
