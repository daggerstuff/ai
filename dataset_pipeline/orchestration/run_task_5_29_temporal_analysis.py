#!/usr/bin/env python3
"""
Task 5.29: Process temporal analysis data (2018/2019 longitudinal studies, pre/post treatment features)
Additional temporal datasets from reddit_mental_health not covered by other tasks.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd


class TemporalAnalysisProcessor:
    """Processor for additional temporal analysis datasets."""

    def __init__(self, output_dir: str = "ai/data/processed/task_5_29_temporal_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define additional temporal datasets not covered by other tasks
        self.temporal_datasets = [
            "alcoholism", "conspiracy", "guns", "legaladvice", "merged", "COVID19"
        ]

        # Temporal periods for analysis
        self.temporal_periods = ["2018", "2019", "pre", "post"]

        # Processing statistics
        self.processing_stats = {
            "total_files_processed": 0,
            "total_conversations_created": 0,
            "total_posts_processed": 0,
            "temporal_dataset_distribution": defaultdict(int),
            "temporal_period_distribution": defaultdict(int),
            "quality_metrics": defaultdict(int)
        }

    def process_temporal_analysis(self) -> dict[str, Any]:
        """Process additional temporal analysis datasets."""

        all_conversations = []
        dataset_reports = {}

        for dataset in self.temporal_datasets:
            conversations, report = self.process_temporal_dataset(dataset)

            all_conversations.extend(conversations)
            dataset_reports[dataset] = report


        # Save consolidated conversations
        consolidated_path = os.path.join(self.output_dir, "temporal_analysis_conversations.jsonl")
        with open(consolidated_path, "w", encoding="utf-8") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Generate comprehensive report
        report = self.generate_comprehensive_report(dataset_reports, all_conversations)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_29_temporal_analysis_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


        return report

    def process_temporal_dataset(self, dataset: str) -> tuple[list[dict], dict]:
        """Process all available temporal periods for a specific dataset."""
        dataset_conversations = []
        temporal_stats = {}
        files_found = 0

        # Check for all possible temporal file patterns
        possible_files = []

        # Standard temporal patterns
        for period in self.temporal_periods:
            filename = f"{dataset}_{period}_features_tfidf_256.csv"
            possible_files.append((filename, period))

        # Special cases for datasets that might have different naming
        if dataset == "COVID19":
            # COVID19 might have different naming patterns
            possible_files.extend([
                ("COVID19_support_post_features_tfidf_256.csv", "post"),
                ("COVID19_support_features_tfidf_256.csv", "general")
            ])
        elif dataset == "merged":
            # Merged dataset might have different patterns
            possible_files.extend([
                ("merged_mental_health_dataset.jsonl", "general"),
                ("mental_disorders_reddit.csv", "general")
            ])

        for filename, period in possible_files:
            file_path = f"../../ai/datasets/reddit_mental_health/{filename}"

            if os.path.exists(file_path):
                conversations, stats = self.process_csv_file(file_path, dataset, period)
                dataset_conversations.extend(conversations)
                temporal_stats[period] = stats
                files_found += 1

                self.processing_stats["total_files_processed"] += 1
                self.processing_stats["temporal_dataset_distribution"][dataset] += len(conversations)
                self.processing_stats["temporal_period_distribution"][period] += len(conversations)

        if files_found == 0:
            pass
        else:
            pass

        dataset_report = {
            "dataset": dataset,
            "total_conversations": len(dataset_conversations),
            "files_found": files_found,
            "temporal_breakdown": temporal_stats,
            "processing_timestamp": datetime.now().isoformat()
        }

        return dataset_conversations, dataset_report

    def process_csv_file(self, file_path: str, dataset: str, period: str) -> tuple[list[dict], dict]:
        """Process a single temporal file (CSV or JSONL)."""
        conversations = []
        stats = {"total_rows": 0, "processed_rows": 0, "errors": 0, "file_type": ""}

        try:
            filename = os.path.basename(file_path)

            # Handle different file formats
            if filename.endswith(".jsonl"):
                stats["file_type"] = "jsonl"
                # Process JSONL file
                with open(file_path, encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        try:
                            data = json.loads(line.strip())
                            stats["total_rows"] += 1

                            # Convert JSON to row-like format for processing
                            row_data = pd.Series(data)
                            conversation = self.create_conversation_from_row(row_data, dataset, period, idx)
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

            else:
                stats["file_type"] = "csv"
                # Process CSV file
                df = pd.read_csv(file_path)
                stats["total_rows"] = len(df)


                # Process each row
                for idx, row in df.iterrows():
                    try:
                        conversation = self.create_conversation_from_row(row, dataset, period, idx)
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

    def create_conversation_from_row(self, row: pd.Series, dataset: str, period: str, idx: int) -> dict[str, Any]:
        """Create a therapeutic conversation from a temporal dataset row."""
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

            # Create therapeutic conversation
            client_message = self.adapt_temporal_post_to_client_message(text_content, dataset)
            therapist_response = self.generate_temporal_therapeutic_response(text_content, dataset, period)

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
                    "category": "reddit_temporal_analysis",
                    "tier": 4,
                    "source_dataset": f"{dataset}_{period}_reddit",
                    "temporal_dataset": dataset,
                    "temporal_period": period,
                    "original_text": text_content,
                    "tfidf_features": tfidf_features,
                    "feature_count": len(tfidf_features),
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_row_id": idx,
                    "conversation_length": len(messages),
                    "text_length": len(text_content),
                    "quality_score": self.assess_conversation_quality(text_content, dataset),
                    "therapeutic_relevance": self.assess_therapeutic_relevance(text_content, dataset),
                    "temporal_category": self.classify_temporal_category(dataset)
                }
            }

        except Exception:
            return None

    def adapt_temporal_post_to_client_message(self, text: str, dataset: str) -> str:
        """Adapt a temporal post to sound like a client message."""
        # Clean and adapt the text
        adapted_text = text.replace("[removed]", "").replace("[deleted]", "").strip()

        # Add therapeutic framing based on dataset
        dataset_intros = {
            "alcoholism": "I've been struggling with alcohol issues and wanted to discuss: ",
            "conspiracy": "I've been having some concerning thoughts and beliefs: ",
            "guns": "I wanted to talk about some thoughts I've been having: ",
            "legaladvice": "I'm dealing with a legal situation that's affecting my mental health: ",
            "merged": "I wanted to share something from my mental health journey: ",
            "COVID19": "The pandemic has been affecting my mental health and I wanted to discuss: "
        }

        intro = dataset_intros.get(dataset, "I wanted to share something that's been on my mind: ")
        return intro + adapted_text

    def generate_temporal_therapeutic_response(self, text: str, dataset: str, period: str) -> str:
        """Generate an appropriate therapeutic response for temporal datasets."""
        # Base therapeutic response
        base_response = "Thank you for sharing this with me. I can see this is something that's been affecting you. "

        # Dataset-specific therapeutic approaches
        dataset_responses = {
            "alcoholism": "Substance use concerns are serious, and it takes courage to talk about them. Let's explore healthy coping strategies and support options. ",
            "conspiracy": "I hear that you're having some intense thoughts. Let's work together to examine these beliefs and how they're affecting your wellbeing. ",
            "guns": "I want to make sure you're feeling safe and supported. Let's talk about what's going on and how we can help you through this. ",
            "legaladvice": "Legal issues can be very stressful and impact mental health significantly. Let's discuss how to manage this stress while addressing your concerns. ",
            "merged": "I appreciate you sharing your mental health experiences. Let's work together to understand and support your wellbeing. ",
            "COVID19": "The pandemic has created unprecedented challenges for mental health. Many people are struggling with similar feelings during this time. "
        }

        dataset_response = dataset_responses.get(dataset, "Let's explore this together and find ways to support you. ")

        # Add temporal context if relevant
        temporal_context = ""
        if period == "pre":
            temporal_context = "It sounds like you're preparing for or anticipating some changes. "
        elif period == "post":
            temporal_context = "It seems like you're reflecting on recent experiences. "
        elif period in ["2018", "2019"]:
            temporal_context = f"Looking back at this experience from {period} can give us valuable perspective. "

        return base_response + temporal_context + dataset_response + "What feels most important to address right now?"

    def assess_conversation_quality(self, text: str, dataset: str) -> float:
        """Assess the quality of the temporal conversation."""
        quality_score = 0.0

        # Length check
        if len(text) >= 50:
            quality_score += 0.3

        # Dataset relevance
        dataset_keywords = {
            "alcoholism": ["alcohol", "drinking", "drunk", "sober", "addiction", "recovery"],
            "conspiracy": ["conspiracy", "theory", "believe", "truth", "government", "secret"],
            "guns": ["gun", "weapon", "firearm", "shooting", "violence", "safety"],
            "legaladvice": ["legal", "law", "lawyer", "court", "advice", "help"],
            "merged": ["mental", "health", "depression", "anxiety", "therapy", "support"],
            "COVID19": ["covid", "pandemic", "coronavirus", "lockdown", "isolation", "quarantine"]
        }

        keywords = dataset_keywords.get(dataset, [])
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)

        if keyword_matches > 0:
            quality_score += 0.4

        # Therapeutic potential
        therapeutic_indicators = ["help", "support", "advice", "guidance", "better", "cope", "manage"]
        therapeutic_matches = sum(1 for indicator in therapeutic_indicators if indicator in text_lower)

        if therapeutic_matches > 0:
            quality_score += 0.3

        return min(quality_score, 1.0)

    def assess_therapeutic_relevance(self, text: str, dataset: str) -> float:
        """Assess therapeutic relevance of temporal content."""
        text_lower = text.lower()

        # Mental health indicators
        mental_health_terms = [
            "mental health", "therapy", "counseling", "help", "support",
            "stress", "anxiety", "depression", "wellbeing", "coping"
        ]

        relevance_score = 0.0
        for term in mental_health_terms:
            if term in text_lower:
                relevance_score += 0.1

        return min(relevance_score, 1.0)

    def classify_temporal_category(self, dataset: str) -> str:
        """Classify the temporal category of the dataset."""
        categories = {
            "alcoholism": "substance_use_temporal",
            "conspiracy": "belief_system_temporal",
            "guns": "safety_concern_temporal",
            "legaladvice": "legal_stress_temporal",
            "merged": "general_mental_health_temporal",
            "COVID19": "pandemic_mental_health_temporal"
        }
        return categories.get(dataset, "general_temporal")

    def generate_comprehensive_report(self, dataset_reports: dict, all_conversations: list) -> dict[str, Any]:
        """Generate comprehensive temporal analysis processing report."""

        # Analyze overall statistics
        total_conversations = len(all_conversations)
        avg_quality = sum(conv.get("metadata", {}).get("quality_score", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0
        avg_relevance = sum(conv.get("metadata", {}).get("therapeutic_relevance", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0

        # Feature analysis
        total_features = sum(conv.get("metadata", {}).get("feature_count", 0) for conv in all_conversations)
        avg_features = total_features / total_conversations if total_conversations > 0 else 0

        return {
            "task": "5.29: Temporal Analysis Data Processing",
            "processing_summary": {
                "total_temporal_datasets_processed": len(self.temporal_datasets),
                "total_temporal_periods": len(self.temporal_periods),
                "total_files_processed": self.processing_stats["total_files_processed"],
                "total_conversations_created": total_conversations,
                "total_posts_processed": self.processing_stats["total_posts_processed"],
                "processing_timestamp": datetime.now().isoformat()
            },
            "temporal_dataset_breakdown": dict(self.processing_stats["temporal_dataset_distribution"]),
            "temporal_period_breakdown": dict(self.processing_stats["temporal_period_distribution"]),
            "quality_metrics": {
                "average_quality_score": avg_quality,
                "average_therapeutic_relevance": avg_relevance,
                "average_tfidf_features": avg_features,
                "total_tfidf_features": total_features
            },
            "detailed_dataset_reports": dataset_reports,
            "dataset_info": {
                "tier": 4,
                "category": "reddit_temporal_analysis",
                "temporal_datasets_covered": self.temporal_datasets,
                "temporal_periods": self.temporal_periods,
                "feature_type": "tfidf_256_dimensional"
            }
        }

def process_temporal_analysis():
    """Main function to run temporal analysis processing."""
    processor = TemporalAnalysisProcessor()
    return processor.process_temporal_analysis()

if __name__ == "__main__":
    process_temporal_analysis()
