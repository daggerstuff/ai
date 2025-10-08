#!/usr/bin/env python3
"""
Task 5.27: Process condition-specific Reddit mental health datasets
Massive scale processing of 9 major mental health conditions with temporal analysis.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd


class ConditionSpecificProcessor:
    """Processor for condition-specific Reddit mental health datasets."""

    def __init__(self, output_dir: str = "ai/data/processed/task_5_27_condition_specific"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define the 9 major mental health conditions
        self.conditions = [
            "addiction", "adhd", "anxiety", "autism",
            "bipolarreddit", "bpd", "depression", "ptsd", "schizophrenia"
        ]

        # Define temporal periods
        self.temporal_periods = ["2018", "2019", "pre", "post"]

        # Processing statistics
        self.processing_stats = {
            "total_files_processed": 0,
            "total_conversations_created": 0,
            "total_posts_processed": 0,
            "condition_distribution": defaultdict(int),
            "temporal_distribution": defaultdict(int),
            "quality_metrics": defaultdict(int)
        }

    def process_all_conditions(self) -> dict[str, Any]:
        """Process all condition-specific datasets."""

        all_conversations = []
        condition_reports = {}

        for condition in self.conditions:
            condition_conversations, condition_report = self.process_condition(condition)

            all_conversations.extend(condition_conversations)
            condition_reports[condition] = condition_report


        # Save consolidated conversations
        consolidated_path = os.path.join(self.output_dir, "condition_specific_conversations.jsonl")
        with open(consolidated_path, "w", encoding="utf-8") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Generate comprehensive report
        report = self.generate_comprehensive_report(condition_reports, all_conversations)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_27_condition_specific_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


        return report

    def process_condition(self, condition: str) -> tuple[list[dict], dict]:
        """Process all temporal datasets for a specific condition."""
        condition_conversations = []
        temporal_stats = {}

        for period in self.temporal_periods:
            # Construct file path
            if period in ["2018", "2019"]:
                filename = f"{condition}_{period}_features_tfidf_256.csv"
            else:
                filename = f"{condition}_{period}_features_tfidf_256.csv"

            file_path = f"../../ai/datasets/reddit_mental_health/{filename}"

            if os.path.exists(file_path):
                conversations, stats = self.process_csv_file(file_path, condition, period)
                condition_conversations.extend(conversations)
                temporal_stats[period] = stats

                self.processing_stats["total_files_processed"] += 1
                self.processing_stats["condition_distribution"][condition] += len(conversations)
                self.processing_stats["temporal_distribution"][period] += len(conversations)
            else:
                temporal_stats[period] = {"error": "File not found", "conversations": 0}

        condition_report = {
            "condition": condition,
            "total_conversations": len(condition_conversations),
            "temporal_breakdown": temporal_stats,
            "processing_timestamp": datetime.now().isoformat()
        }

        return condition_conversations, condition_report

    def process_csv_file(self, file_path: str, condition: str, period: str) -> tuple[list[dict], dict]:
        """Process a single CSV file with TF-IDF features."""
        conversations = []
        stats = {"total_rows": 0, "processed_rows": 0, "errors": 0}

        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            stats["total_rows"] = len(df)


            # Process each row
            for idx, row in df.iterrows():
                try:
                    conversation = self.create_conversation_from_row(row, condition, period, idx)
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

    def create_conversation_from_row(self, row: pd.Series, condition: str, period: str, idx: int) -> dict[str, Any]:
        """Create a therapeutic conversation from a Reddit post row."""
        try:
            # Extract text content (common column names)
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
            client_message = self.adapt_reddit_post_to_client_message(text_content, condition)
            therapist_response = self.generate_therapeutic_response(text_content, condition, period)

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
                    "category": "reddit_mental_health_condition_specific",
                    "tier": 4,
                    "source_dataset": f"{condition}_{period}_reddit",
                    "condition": condition,
                    "temporal_period": period,
                    "original_text": text_content,
                    "tfidf_features": tfidf_features,
                    "feature_count": len(tfidf_features),
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_row_id": idx,
                    "conversation_length": len(messages),
                    "text_length": len(text_content),
                    "quality_score": self.assess_conversation_quality(text_content, condition),
                    "therapeutic_relevance": self.assess_therapeutic_relevance(text_content, condition)
                }
            }

        except Exception:
            return None

    def adapt_reddit_post_to_client_message(self, text: str, condition: str) -> str:
        """Adapt a Reddit post to sound like a client message in therapy."""
        # Clean and adapt the text
        adapted_text = text.replace("[removed]", "").replace("[deleted]", "").strip()

        # Add therapeutic framing based on condition
        condition_intros = {
            "depression": "I've been struggling with depression and wanted to share what I'm going through: ",
            "anxiety": "I'm dealing with anxiety and need to talk about this: ",
            "adhd": "I have ADHD and this has been on my mind: ",
            "autism": "As someone on the autism spectrum, I wanted to discuss: ",
            "bipolarreddit": "I'm managing bipolar disorder and need to share: ",
            "bpd": "I have borderline personality disorder and this is affecting me: ",
            "addiction": "I'm in recovery and wanted to talk about: ",
            "ptsd": "I'm dealing with PTSD and this has been triggering: ",
            "schizophrenia": "I have schizophrenia and wanted to discuss: "
        }

        intro = condition_intros.get(condition, "I wanted to talk about something that's been bothering me: ")
        return intro + adapted_text

    def generate_therapeutic_response(self, text: str, condition: str, period: str) -> str:
        """Generate an appropriate therapeutic response."""
        # Base therapeutic response
        base_response = "Thank you for sharing this with me. I can hear that this is something important to you. "

        # Condition-specific therapeutic approaches
        condition_responses = {
            "depression": "Depression can make everything feel overwhelming. Let's work together to identify some coping strategies and small steps you can take. ",
            "anxiety": "Anxiety can be very challenging to manage. Let's explore some grounding techniques and ways to manage these feelings. ",
            "adhd": "ADHD affects everyone differently. Let's discuss strategies that might help you manage these challenges. ",
            "autism": "Thank you for sharing your perspective. Let's explore ways to support you in navigating these situations. ",
            "bipolarreddit": "Managing bipolar disorder requires ongoing attention. Let's discuss your current strategies and how we might adjust them. ",
            "bpd": "I appreciate you sharing these intense feelings. Let's work on some skills to help you manage these emotions. ",
            "addiction": "Recovery is a journey with ups and downs. Let's focus on your strengths and healthy coping mechanisms. ",
            "ptsd": "Trauma responses are normal reactions to abnormal experiences. Let's work on some grounding and safety techniques. ",
            "schizophrenia": "Thank you for trusting me with this. Let's discuss how we can support your mental health and wellbeing. "
        }

        condition_response = condition_responses.get(condition, "Let's explore this together and find ways to support you. ")

        # Add temporal context if relevant
        temporal_context = ""
        if period == "pre":
            temporal_context = "It sounds like you're preparing for or anticipating some changes. "
        elif period == "post":
            temporal_context = "It seems like you're reflecting on recent experiences. "
        elif period in ["2018", "2019"]:
            temporal_context = f"I notice this was from {period}, which can give us perspective on your journey. "

        return base_response + temporal_context + condition_response + "What would be most helpful to focus on right now?"

    def assess_conversation_quality(self, text: str, condition: str) -> float:
        """Assess the quality of the conversation."""
        quality_score = 0.0

        # Length check
        if len(text) >= 50:
            quality_score += 0.3

        # Condition relevance
        condition_keywords = {
            "depression": ["depressed", "sad", "hopeless", "mood", "down"],
            "anxiety": ["anxious", "worry", "panic", "fear", "nervous"],
            "adhd": ["attention", "focus", "hyperactive", "concentrate"],
            "autism": ["autism", "sensory", "social", "routine"],
            "bipolarreddit": ["bipolar", "manic", "mood", "episode"],
            "bpd": ["borderline", "emotional", "intense", "relationship"],
            "addiction": ["addiction", "substance", "recovery", "sober"],
            "ptsd": ["trauma", "ptsd", "flashback", "trigger"],
            "schizophrenia": ["schizophrenia", "psychosis", "hallucination"]
        }

        keywords = condition_keywords.get(condition, [])
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)

        if keyword_matches > 0:
            quality_score += 0.4

        # Therapeutic potential
        therapeutic_indicators = ["help", "support", "therapy", "treatment", "better", "cope", "manage"]
        therapeutic_matches = sum(1 for indicator in therapeutic_indicators if indicator in text_lower)

        if therapeutic_matches > 0:
            quality_score += 0.3

        return min(quality_score, 1.0)

    def assess_therapeutic_relevance(self, text: str, condition: str) -> float:
        """Assess therapeutic relevance of the content."""
        text_lower = text.lower()

        # Mental health indicators
        mental_health_terms = [
            "mental health", "therapy", "counseling", "psychiatrist", "psychologist",
            "medication", "treatment", "symptoms", "diagnosis", "support"
        ]

        relevance_score = 0.0
        for term in mental_health_terms:
            if term in text_lower:
                relevance_score += 0.1

        return min(relevance_score, 1.0)

    def generate_comprehensive_report(self, condition_reports: dict, all_conversations: list) -> dict[str, Any]:
        """Generate comprehensive processing report."""

        # Analyze overall statistics
        total_conversations = len(all_conversations)
        avg_quality = sum(conv.get("metadata", {}).get("quality_score", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0
        avg_relevance = sum(conv.get("metadata", {}).get("therapeutic_relevance", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0

        # Feature analysis
        total_features = sum(conv.get("metadata", {}).get("feature_count", 0) for conv in all_conversations)
        avg_features = total_features / total_conversations if total_conversations > 0 else 0

        return {
            "task": "5.27: Condition-Specific Reddit Mental Health Processing",
            "processing_summary": {
                "total_conditions_processed": len(self.conditions),
                "total_temporal_periods": len(self.temporal_periods),
                "total_files_processed": self.processing_stats["total_files_processed"],
                "total_conversations_created": total_conversations,
                "total_posts_processed": self.processing_stats["total_posts_processed"],
                "processing_timestamp": datetime.now().isoformat()
            },
            "condition_breakdown": dict(self.processing_stats["condition_distribution"]),
            "temporal_breakdown": dict(self.processing_stats["temporal_distribution"]),
            "quality_metrics": {
                "average_quality_score": avg_quality,
                "average_therapeutic_relevance": avg_relevance,
                "average_tfidf_features": avg_features,
                "total_tfidf_features": total_features
            },
            "detailed_condition_reports": condition_reports,
            "dataset_info": {
                "tier": 4,
                "category": "reddit_mental_health_condition_specific",
                "conditions_covered": self.conditions,
                "temporal_periods": self.temporal_periods,
                "feature_type": "tfidf_256_dimensional"
            }
        }

def process_condition_specific():
    """Main function to run condition-specific processing."""
    processor = ConditionSpecificProcessor()
    return processor.process_all_conditions()

if __name__ == "__main__":
    process_condition_specific()
