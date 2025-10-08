#!/usr/bin/env python3
"""
Task 5.28: Process specialized population Reddit mental health datasets
Social anxiety, health anxiety, eating disorders, loneliness, parenting stress, divorce recovery.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd


class SpecializedPopulationProcessor:
    """Processor for specialized population Reddit mental health datasets."""

    def __init__(self, output_dir: str = "ai/data/processed/task_5_28_specialized_populations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define specialized populations
        self.populations = [
            "socialanxiety", "healthanxiety", "lonely",
            "parenting", "divorce", "mentalhealth"
        ]

        # Define temporal periods
        self.temporal_periods = ["2018", "2019", "pre", "post"]

        # Processing statistics
        self.processing_stats = {
            "total_files_processed": 0,
            "total_conversations_created": 0,
            "total_posts_processed": 0,
            "population_distribution": defaultdict(int),
            "temporal_distribution": defaultdict(int),
            "quality_metrics": defaultdict(int)
        }

    def process_all_populations(self) -> dict[str, Any]:
        """Process all specialized population datasets."""

        all_conversations = []
        population_reports = {}

        for population in self.populations:
            population_conversations, population_report = self.process_population(population)

            all_conversations.extend(population_conversations)
            population_reports[population] = population_report


        # Save consolidated conversations
        consolidated_path = os.path.join(self.output_dir, "specialized_populations_conversations.jsonl")
        with open(consolidated_path, "w", encoding="utf-8") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Generate comprehensive report
        report = self.generate_comprehensive_report(population_reports, all_conversations)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_28_specialized_populations_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


        return report

    def process_population(self, population: str) -> tuple[list[dict], dict]:
        """Process all temporal datasets for a specific population."""
        population_conversations = []
        temporal_stats = {}

        for period in self.temporal_periods:
            # Construct file path
            if period in ["2018", "2019"]:
                filename = f"{population}_{period}_features_tfidf_256.csv"
            else:
                filename = f"{population}_{period}_features_tfidf_256.csv"

            file_path = f"../../ai/datasets/reddit_mental_health/{filename}"

            if os.path.exists(file_path):
                conversations, stats = self.process_csv_file(file_path, population, period)
                population_conversations.extend(conversations)
                temporal_stats[period] = stats

                self.processing_stats["total_files_processed"] += 1
                self.processing_stats["population_distribution"][population] += len(conversations)
                self.processing_stats["temporal_distribution"][period] += len(conversations)
            else:
                temporal_stats[period] = {"error": "File not found", "conversations": 0}

        population_report = {
            "population": population,
            "total_conversations": len(population_conversations),
            "temporal_breakdown": temporal_stats,
            "processing_timestamp": datetime.now().isoformat()
        }

        return population_conversations, population_report

    def process_csv_file(self, file_path: str, population: str, period: str) -> tuple[list[dict], dict]:
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
                    conversation = self.create_conversation_from_row(row, population, period, idx)
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

    def create_conversation_from_row(self, row: pd.Series, population: str, period: str, idx: int) -> dict[str, Any]:
        """Create a therapeutic conversation from a Reddit post row."""
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
            client_message = self.adapt_reddit_post_to_client_message(text_content, population)
            therapist_response = self.generate_therapeutic_response(text_content, population, period)

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
                    "category": "reddit_mental_health_specialized_population",
                    "tier": 4,
                    "source_dataset": f"{population}_{period}_reddit",
                    "population": population,
                    "temporal_period": period,
                    "original_text": text_content,
                    "tfidf_features": tfidf_features,
                    "feature_count": len(tfidf_features),
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_row_id": idx,
                    "conversation_length": len(messages),
                    "text_length": len(text_content),
                    "quality_score": self.assess_conversation_quality(text_content, population),
                    "therapeutic_relevance": self.assess_therapeutic_relevance(text_content, population),
                    "population_specificity": self.assess_population_specificity(text_content, population)
                }
            }

        except Exception:
            return None

    def adapt_reddit_post_to_client_message(self, text: str, population: str) -> str:
        """Adapt a Reddit post to sound like a client message in therapy."""
        # Clean and adapt the text
        adapted_text = text.replace("[removed]", "").replace("[deleted]", "").strip()

        # Add therapeutic framing based on population
        population_intros = {
            "socialanxiety": "I struggle with social anxiety and wanted to share what happened: ",
            "healthanxiety": "I've been dealing with health anxiety and this is really bothering me: ",
            "lonely": "I've been feeling really lonely lately and needed to talk about: ",
            "parenting": "As a parent, I'm struggling with this situation: ",
            "divorce": "I'm going through a divorce and this has been difficult: ",
            "mentalhealth": "I wanted to talk about my mental health journey: "
        }

        intro = population_intros.get(population, "I wanted to discuss something that's been on my mind: ")
        return intro + adapted_text

    def generate_therapeutic_response(self, text: str, population: str, period: str) -> str:
        """Generate an appropriate therapeutic response for specialized populations."""
        # Base therapeutic response
        base_response = "Thank you for sharing this with me. I can see this is something that's really affecting you. "

        # Population-specific therapeutic approaches
        population_responses = {
            "socialanxiety": "Social anxiety can make everyday interactions feel overwhelming. Let's work on some strategies to help you feel more comfortable in social situations. ",
            "healthanxiety": "Health anxiety can be very distressing. Let's explore some techniques to help manage these worries and develop a healthier relationship with your body and health concerns. ",
            "lonely": "Loneliness is a very real and painful experience. Let's talk about ways to build meaningful connections and address these feelings of isolation. ",
            "parenting": "Parenting brings unique challenges and stresses. Let's explore strategies to help you navigate this while taking care of your own wellbeing. ",
            "divorce": "Going through a divorce is one of life's major stressors. Let's work on coping strategies and ways to support yourself through this transition. ",
            "mentalhealth": "Taking care of your mental health is so important. Let's explore what's working for you and what additional support might be helpful. "
        }

        population_response = population_responses.get(population, "Let's explore this together and find ways to support you. ")

        # Add temporal context if relevant
        temporal_context = ""
        if period == "pre":
            temporal_context = "It sounds like you're preparing for or anticipating some changes. "
        elif period == "post":
            temporal_context = "It seems like you're reflecting on recent experiences. "
        elif period in ["2018", "2019"]:
            temporal_context = f"Looking back at this experience from {period} can give us valuable perspective. "

        return base_response + temporal_context + population_response + "What feels most important to focus on right now?"

    def assess_conversation_quality(self, text: str, population: str) -> float:
        """Assess the quality of the conversation."""
        quality_score = 0.0

        # Length check
        if len(text) >= 50:
            quality_score += 0.3

        # Population relevance
        population_keywords = {
            "socialanxiety": ["social", "anxiety", "anxious", "people", "interaction", "embarrassed"],
            "healthanxiety": ["health", "anxiety", "medical", "symptoms", "doctor", "illness"],
            "lonely": ["lonely", "alone", "isolated", "friends", "connection", "social"],
            "parenting": ["parent", "child", "kids", "family", "mother", "father"],
            "divorce": ["divorce", "separation", "ex", "custody", "marriage", "relationship"],
            "mentalhealth": ["mental", "health", "therapy", "depression", "anxiety", "wellbeing"]
        }

        keywords = population_keywords.get(population, [])
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

    def assess_therapeutic_relevance(self, text: str, population: str) -> float:
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

    def assess_population_specificity(self, text: str, population: str) -> float:
        """Assess how specific the content is to the population."""
        text_lower = text.lower()

        # Population-specific terms
        specificity_terms = {
            "socialanxiety": ["social anxiety", "social phobia", "social situations", "public speaking"],
            "healthanxiety": ["health anxiety", "hypochondria", "medical anxiety", "illness anxiety"],
            "lonely": ["loneliness", "isolation", "alone", "no friends"],
            "parenting": ["parenting", "parenthood", "raising children", "mom", "dad"],
            "divorce": ["divorce", "separation", "custody", "ex-spouse"],
            "mentalhealth": ["mental health", "psychological", "psychiatric", "emotional wellbeing"]
        }

        terms = specificity_terms.get(population, [])
        specificity_score = 0.0

        for term in terms:
            if term in text_lower:
                specificity_score += 0.25

        return min(specificity_score, 1.0)

    def generate_comprehensive_report(self, population_reports: dict, all_conversations: list) -> dict[str, Any]:
        """Generate comprehensive processing report."""

        # Analyze overall statistics
        total_conversations = len(all_conversations)
        avg_quality = sum(conv.get("metadata", {}).get("quality_score", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0
        avg_relevance = sum(conv.get("metadata", {}).get("therapeutic_relevance", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0
        avg_specificity = sum(conv.get("metadata", {}).get("population_specificity", 0) for conv in all_conversations) / total_conversations if total_conversations > 0 else 0

        # Feature analysis
        total_features = sum(conv.get("metadata", {}).get("feature_count", 0) for conv in all_conversations)
        avg_features = total_features / total_conversations if total_conversations > 0 else 0

        return {
            "task": "5.28: Specialized Population Reddit Mental Health Processing",
            "processing_summary": {
                "total_populations_processed": len(self.populations),
                "total_temporal_periods": len(self.temporal_periods),
                "total_files_processed": self.processing_stats["total_files_processed"],
                "total_conversations_created": total_conversations,
                "total_posts_processed": self.processing_stats["total_posts_processed"],
                "processing_timestamp": datetime.now().isoformat()
            },
            "population_breakdown": dict(self.processing_stats["population_distribution"]),
            "temporal_breakdown": dict(self.processing_stats["temporal_distribution"]),
            "quality_metrics": {
                "average_quality_score": avg_quality,
                "average_therapeutic_relevance": avg_relevance,
                "average_population_specificity": avg_specificity,
                "average_tfidf_features": avg_features,
                "total_tfidf_features": total_features
            },
            "detailed_population_reports": population_reports,
            "dataset_info": {
                "tier": 4,
                "category": "reddit_mental_health_specialized_population",
                "populations_covered": self.populations,
                "temporal_periods": self.temporal_periods,
                "feature_type": "tfidf_256_dimensional"
            }
        }

def process_specialized_populations():
    """Main function to run specialized population processing."""
    processor = SpecializedPopulationProcessor()
    return processor.process_all_populations()

if __name__ == "__main__":
    process_specialized_populations()
