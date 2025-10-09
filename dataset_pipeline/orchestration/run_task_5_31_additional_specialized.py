#!/usr/bin/env python3
"""
Task 5.31: Process additional specialized populations
ADHD women and Eating Disorders Anonymous datasets.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd


class AdditionalSpecializedProcessor:
    """Processor for additional specialized population datasets."""

    def __init__(self, output_dir: str = "ai/data/processed/task_5_31_additional_specialized"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define additional specialized datasets
        self.specialized_datasets = [
            {
                "name": "adhd_women",
                "file": "adhdwomen.csv",
                "type": "gender_specific_neurodiversity",
                "description": "ADHD experiences specific to women"
            },
            {
                "name": "ed_anonymous_2019",
                "file": "EDAnonymous_2019_features_tfidf_256.csv",
                "type": "eating_disorders",
                "description": "Eating disorders anonymous 2019 posts with TF-IDF features"
            },
            {
                "name": "ed_anonymous_post",
                "file": "EDAnonymous_post_features_tfidf_256.csv",
                "type": "eating_disorders",
                "description": "Eating disorders anonymous post-treatment features"
            },
            {
                "name": "ed_anonymous_pre",
                "file": "EDAnonymous_pre_features_tfidf_256.csv",
                "type": "eating_disorders",
                "description": "Eating disorders anonymous pre-treatment features"
            }
        ]

        # Processing statistics
        self.processing_stats = {
            "total_files_processed": 0,
            "total_conversations_created": 0,
            "total_posts_processed": 0,
            "dataset_distribution": defaultdict(int),
            "type_distribution": defaultdict(int),
            "quality_metrics": defaultdict(int)
        }

    def process_all_specialized_datasets(self) -> dict[str, Any]:
        """Process all additional specialized population datasets."""

        all_conversations = []
        dataset_reports = {}

        for dataset_config in self.specialized_datasets:
            conversations, report = self.process_specialized_dataset(dataset_config)

            all_conversations.extend(conversations)
            dataset_reports[dataset_config["name"]] = report


        # Save consolidated conversations
        consolidated_path = os.path.join(self.output_dir, "additional_specialized_conversations.jsonl")
        with open(consolidated_path, "w", encoding="utf-8") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Generate comprehensive report
        report = self.generate_comprehensive_report(dataset_reports, all_conversations)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_31_additional_specialized_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


        return report

    def process_specialized_dataset(self, dataset_config: dict) -> tuple[list[dict], dict]:
        """Process a specific specialized dataset."""
        file_path = f"../../ai/datasets/reddit_mental_health/{dataset_config['file']}"

        if not os.path.exists(file_path):
            # Silently skip missing files - this is normal
            return [], {"status": "no_data_available", "conversations": 0}

        conversations, stats = self.process_csv_file(file_path, dataset_config)

        self.processing_stats["total_files_processed"] += 1
        self.processing_stats["dataset_distribution"][dataset_config["name"]] += len(conversations)
        self.processing_stats["type_distribution"][dataset_config["type"]] += len(conversations)

        report = {
            "dataset_name": dataset_config["name"],
            "dataset_type": dataset_config["type"],
            "total_conversations": len(conversations),
            "processing_stats": stats,
            "processing_timestamp": datetime.now().isoformat()
        }

        return conversations, report

    def process_csv_file(self, file_path: str, dataset_config: dict) -> tuple[list[dict], dict]:
        """Process a single specialized CSV file."""
        conversations = []
        stats = {"total_rows": 0, "processed_rows": 0, "errors": 0}

        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            stats["total_rows"] = len(df)


            # Process each row
            for idx, row in df.iterrows():
                try:
                    conversation = self.create_specialized_conversation_from_row(row, dataset_config, idx)
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

    def create_specialized_conversation_from_row(self, row: pd.Series, dataset_config: dict, idx: int) -> dict[str, Any]:
        """Create a therapeutic conversation from a specialized population row."""
        try:
            # Extract text content
            text_content = ""
            possible_text_columns = ["text", "post", "content", "body", "title", "selftext", "comment"]

            for col in possible_text_columns:
                if col in row.index and pd.notna(row[col]):
                    text_content = str(row[col]).strip()
                    break

            if not text_content or len(text_content) < 20:
                return None

            # Extract TF-IDF features from token columns (excluding metadata columns)
            tfidf_features = {}
            metadata_columns = {"text", "post", "content", "body", "title", "selftext", "id", "author", "created_utc", "score", "num_comments", "comment"}

            for col in row.index:
                if col not in metadata_columns and pd.notna(row[col]) and isinstance(row[col], (int, float)) and row[col] != 0:
                    tfidf_features[col] = float(row[col])

            # Create therapeutic conversation
            client_message = self.adapt_specialized_post_to_client_message(text_content, dataset_config)
            therapist_response = self.generate_specialized_therapeutic_response(text_content, dataset_config)

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
                    "category": "reddit_mental_health_additional_specialized",
                    "tier": 4,
                    "source_dataset": dataset_config["name"],
                    "population_type": dataset_config["type"],
                    "original_text": text_content,
                    "tfidf_features": tfidf_features,
                    "feature_count": len(tfidf_features),
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_row_id": idx,
                    "conversation_length": len(messages),
                    "text_length": len(text_content),
                    "quality_score": self.assess_conversation_quality(text_content, dataset_config),
                    "therapeutic_relevance": self.assess_therapeutic_relevance(text_content, dataset_config),
                    "population_specificity": self.assess_population_specificity(text_content, dataset_config)
                }
            }

        except Exception:
            return None

    def adapt_specialized_post_to_client_message(self, text: str, dataset_config: dict) -> str:
        """Adapt a specialized post to sound like a client message in therapy."""
        # Clean and adapt the text
        adapted_text = text.replace("[removed]", "").replace("[deleted]", "").strip()

        # Add therapeutic framing based on population type
        if dataset_config["type"] == "gender_specific_neurodiversity":
            intro = "As a woman with ADHD, I wanted to share my experience: "
        elif dataset_config["type"] == "eating_disorders":
            if "pre" in dataset_config["name"]:
                intro = "I'm struggling with an eating disorder and wanted to talk about: "
            elif "post" in dataset_config["name"]:
                intro = "I'm in recovery from an eating disorder and reflecting on: "
            else:
                intro = "I'm dealing with eating disorder challenges and need to discuss: "
        else:
            intro = "I wanted to share something that's been affecting me: "

        return intro + adapted_text

    def generate_specialized_therapeutic_response(self, text: str, dataset_config: dict) -> str:
        """Generate an appropriate therapeutic response for specialized populations."""
        # Base therapeutic response
        base_response = "Thank you for sharing this with me. I can see this is something that's really important to you. "

        # Population-specific therapeutic approaches
        if dataset_config["type"] == "gender_specific_neurodiversity":
            specialized_response = "ADHD can present differently in women, and your experiences are valid. Let's explore strategies that work specifically for you. "
            approach = "We can discuss how ADHD affects different aspects of your life and develop personalized coping strategies. "
        elif dataset_config["type"] == "eating_disorders":
            if "pre" in dataset_config["name"]:
                specialized_response = "Eating disorders are serious mental health conditions, and reaching out for help is a brave first step. "
                approach = "Let's work together on developing a healthy relationship with food and your body. "
            elif "post" in dataset_config["name"]:
                specialized_response = "Recovery from an eating disorder is an ongoing journey, and it's normal to have ups and downs. "
                approach = "Let's focus on maintaining your progress and building resilience for continued recovery. "
            else:
                specialized_response = "Eating disorders affect both your physical and mental health. Let's address this comprehensively. "
                approach = "We can work on both the behavioral and emotional aspects of your relationship with food. "
        else:
            specialized_response = "Your unique experiences deserve specialized attention and care. "
            approach = "Let's explore approaches that are tailored to your specific situation. "

        return base_response + specialized_response + approach + "What would be most helpful to focus on today?"

    def assess_conversation_quality(self, text: str, dataset_config: dict) -> float:
        """Assess the quality of the conversation."""
        quality_score = 0.0

        # Length check
        if len(text) >= 50:
            quality_score += 0.3

        # Population relevance
        if dataset_config["type"] == "gender_specific_neurodiversity":
            keywords = ["adhd", "attention", "focus", "women", "female", "girl", "hyperactive"]
        elif dataset_config["type"] == "eating_disorders":
            keywords = ["eating", "food", "weight", "body", "anorexia", "bulimia", "binge", "restrict"]
        else:
            keywords = ["mental", "health", "support", "help"]

        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)

        if keyword_matches > 0:
            quality_score += 0.4

        # Therapeutic potential
        therapeutic_indicators = ["help", "support", "therapy", "treatment", "better", "cope", "manage", "recovery"]
        therapeutic_matches = sum(1 for indicator in therapeutic_indicators if indicator in text_lower)

        if therapeutic_matches > 0:
            quality_score += 0.3

        return min(quality_score, 1.0)

    def assess_therapeutic_relevance(self, text: str, dataset_config: dict) -> float:
        """Assess therapeutic relevance of the content."""
        text_lower = text.lower()

        # Mental health indicators
        mental_health_terms = [
            "mental health", "therapy", "counseling", "psychiatrist", "psychologist",
            "medication", "treatment", "symptoms", "diagnosis", "support", "recovery"
        ]

        relevance_score = 0.0
        for term in mental_health_terms:
            if term in text_lower:
                relevance_score += 0.1

        return min(relevance_score, 1.0)

    def assess_population_specificity(self, text: str, dataset_config: dict) -> float:
        """Assess how specific the content is to the population."""
        text_lower = text.lower()

        # Population-specific terms
        if dataset_config["type"] == "gender_specific_neurodiversity":
            specificity_terms = ["women with adhd", "female adhd", "girls with adhd", "adhd in women"]
        elif dataset_config["type"] == "eating_disorders":
            specificity_terms = ["eating disorder", "anorexia", "bulimia", "binge eating", "ed recovery"]
        else:
            specificity_terms = ["specialized", "specific", "unique"]

        specificity_score = 0.0
        for term in specificity_terms:
            if term in text_lower:
                specificity_score += 0.25

        return min(specificity_score, 1.0)

    def generate_comprehensive_report(self, dataset_reports: dict, all_conversations: list) -> dict[str, Any]:
        """Generate comprehensive processing report."""

        # Analyze overall statistics
        total_conversations = len(all_conversations)
        avg_quality = sum(conv.get("metadata", {}).get("quality_score", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0
        avg_relevance = sum(conv.get("metadata", {}).get("therapeutic_relevance", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0
        avg_specificity = sum(conv.get("metadata", {}).get("population_specificity", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0

        # Feature analysis
        total_features = sum(conv.get("metadata", {}).get("feature_count", 0) for conv in all_conversations)
        avg_features = total_features / total_conversations if total_conversations > 0 else 0

        # Population type analysis
        type_breakdown = {}
        for conv in all_conversations:
            pop_type = conv.get("metadata", {}).get("population_type", "unknown")
            type_breakdown[pop_type] = type_breakdown.get(pop_type, 0) + 1

        return {
            "task": "5.31: Additional Specialized Population Processing",
            "processing_summary": {
                "total_datasets_processed": len(self.specialized_datasets),
                "total_files_processed": self.processing_stats["total_files_processed"],
                "total_conversations_created": total_conversations,
                "total_posts_processed": self.processing_stats["total_posts_processed"],
                "processing_timestamp": datetime.now().isoformat()
            },
            "dataset_breakdown": dict(self.processing_stats["dataset_distribution"]),
            "population_type_breakdown": type_breakdown,
            "quality_metrics": {
                "average_quality_score": avg_quality,
                "average_therapeutic_relevance": avg_relevance,
                "average_population_specificity": avg_specificity,
                "average_tfidf_features": avg_features,
                "total_tfidf_features": total_features
            },
            "detailed_dataset_reports": dataset_reports,
            "dataset_info": {
                "tier": 4,
                "category": "reddit_mental_health_additional_specialized",
                "datasets_covered": [d["name"] for d in self.specialized_datasets],
                "population_types": list({d["type"] for d in self.specialized_datasets}),
                "feature_type": "tfidf_256_dimensional"
            },
            "specialized_focus": {
                "adhd_women": "Gender-specific neurodiversity experiences",
                "eating_disorders": "Pre/post treatment and 2019 temporal data",
                "therapeutic_approaches": "Population-specific interventions and support"
            }
        }

def process_additional_specialized():
    """Main function to run additional specialized population processing."""
    processor = AdditionalSpecializedProcessor()
    return processor.process_all_specialized_datasets()

if __name__ == "__main__":
    process_additional_specialized()
