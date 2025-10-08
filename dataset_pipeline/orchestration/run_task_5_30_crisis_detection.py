#!/usr/bin/env python3
"""
Task 5.30: Process crisis detection datasets
Suicide detection and COVID-19 support post features - CRITICAL mental health data.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd


class CrisisDetectionProcessor:
    """Processor for crisis detection datasets - suicide detection and COVID support."""

    def __init__(self, output_dir: str = "ai/data/processed/task_5_30_crisis_detection"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define crisis detection datasets
        self.crisis_datasets = [
            {
                "name": "suicide_detection",
                "file": "Suicide_Detection.csv",
                "type": "suicide_risk",
                "priority": "CRITICAL",
                "description": "Suicide detection and risk assessment data"
            },
            {
                "name": "covid19_support",
                "file": "COVID19_support_post_features_tfidf_256.csv",
                "type": "pandemic_support",
                "priority": "HIGH",
                "description": "COVID-19 mental health support posts with TF-IDF features"
            }
        ]

        # Processing statistics
        self.processing_stats = {
            "total_files_processed": 0,
            "total_conversations_created": 0,
            "total_posts_processed": 0,
            "crisis_type_distribution": defaultdict(int),
            "risk_level_distribution": defaultdict(int),
            "quality_metrics": defaultdict(int)
        }

    def process_all_crisis_datasets(self) -> dict[str, Any]:
        """Process all crisis detection datasets."""

        all_conversations = []
        dataset_reports = {}

        for dataset_config in self.crisis_datasets:
            conversations, report = self.process_crisis_dataset(dataset_config)

            all_conversations.extend(conversations)
            dataset_reports[dataset_config["name"]] = report


        # Save consolidated conversations
        consolidated_path = os.path.join(self.output_dir, "crisis_detection_conversations.jsonl")
        with open(consolidated_path, "w", encoding="utf-8") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Generate comprehensive report
        report = self.generate_comprehensive_report(dataset_reports, all_conversations)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_30_crisis_detection_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


        return report

    def process_crisis_dataset(self, dataset_config: dict) -> tuple[list[dict], dict]:
        """Process a specific crisis detection dataset."""
        file_path = f"../../ai/datasets/reddit_mental_health/{dataset_config['file']}"

        if not os.path.exists(file_path):
            # Silently skip missing files - this is normal
            return [], {"status": "no_data_available", "conversations": 0}

        conversations, stats = self.process_csv_file(file_path, dataset_config)

        self.processing_stats["total_files_processed"] += 1
        self.processing_stats["crisis_type_distribution"][dataset_config["type"]] += len(conversations)

        report = {
            "dataset_name": dataset_config["name"],
            "dataset_type": dataset_config["type"],
            "priority": dataset_config["priority"],
            "total_conversations": len(conversations),
            "processing_stats": stats,
            "processing_timestamp": datetime.now().isoformat()
        }

        return conversations, report

    def process_csv_file(self, file_path: str, dataset_config: dict) -> tuple[list[dict], dict]:
        """Process a single crisis detection CSV file."""
        conversations = []
        stats = {"total_rows": 0, "processed_rows": 0, "errors": 0, "high_risk": 0, "medium_risk": 0, "low_risk": 0}

        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            stats["total_rows"] = len(df)


            # Process each row
            for idx, row in df.iterrows():
                try:
                    conversation = self.create_crisis_conversation_from_row(row, dataset_config, idx)
                    if conversation:
                        conversations.append(conversation)
                        stats["processed_rows"] += 1

                        # Track risk levels
                        risk_level = conversation.get("metadata", {}).get("risk_level", "unknown")
                        if risk_level == "high":
                            stats["high_risk"] += 1
                        elif risk_level == "medium":
                            stats["medium_risk"] += 1
                        elif risk_level == "low":
                            stats["low_risk"] += 1

                        self.processing_stats["total_posts_processed"] += 1
                        self.processing_stats["total_conversations_created"] += 1
                        self.processing_stats["risk_level_distribution"][risk_level] += 1

                except Exception:
                    stats["errors"] += 1
                    continue

                # Progress indicator for large files
                if (idx + 1) % 1000 == 0:
                    pass


        except Exception as e:
            stats["file_error"] = str(e)

        return conversations, stats

    def create_crisis_conversation_from_row(self, row: pd.Series, dataset_config: dict, idx: int) -> dict[str, Any]:
        """Create a therapeutic conversation from a crisis detection row."""
        try:
            # Extract text content
            text_content = ""
            possible_text_columns = ["text", "post", "content", "body", "title", "selftext", "tweet"]

            for col in possible_text_columns:
                if col in row.index and pd.notna(row[col]):
                    text_content = str(row[col]).strip()
                    break

            if not text_content or len(text_content) < 10:
                return None

            # Extract labels/classifications if available
            crisis_label = None

            # Look for suicide/crisis labels
            label_columns = ["class", "label", "suicide", "crisis", "risk", "target"]
            for col in label_columns:
                if col in row.index and pd.notna(row[col]):
                    crisis_label = str(row[col]).strip()
                    break

            # Extract TF-IDF features from token columns (excluding metadata columns)
            tfidf_features = {}
            metadata_columns = {"text", "post", "content", "body", "title", "selftext", "id", "author", "created_utc", "score", "num_comments", "class", "label", "suicide", "crisis", "risk", "target", "tweet"}

            for col in row.index:
                if col not in metadata_columns and pd.notna(row[col]) and isinstance(row[col], (int, float)) and row[col] != 0:
                    tfidf_features[col] = float(row[col])

            # Assess risk level
            risk_level = self.assess_crisis_risk_level(text_content, crisis_label, dataset_config)

            # Create therapeutic conversation with crisis-appropriate response
            client_message = self.adapt_crisis_post_to_client_message(text_content, dataset_config, risk_level)
            therapist_response = self.generate_crisis_therapeutic_response(text_content, dataset_config, risk_level)

            messages = [
                {
                    "role": "client",
                    "content": client_message,
                    "turn_id": 1,
                    "crisis_indicators": self.identify_crisis_indicators(text_content)
                },
                {
                    "role": "therapist",
                    "content": therapist_response,
                    "turn_id": 2,
                    "intervention_type": self.determine_intervention_type(risk_level),
                    "safety_protocol": self.get_safety_protocol(risk_level)
                }
            ]

            return {
                "conversation": messages,
                "metadata": {
                    "category": "crisis_detection_mental_health",
                    "tier": 4,
                    "source_dataset": dataset_config["name"],
                    "crisis_type": dataset_config["type"],
                    "priority": dataset_config["priority"],
                    "risk_level": risk_level,
                    "original_text": text_content,
                    "original_label": crisis_label,
                    "tfidf_features": tfidf_features,
                    "feature_count": len(tfidf_features),
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_row_id": idx,
                    "conversation_length": len(messages),
                    "text_length": len(text_content),
                    "crisis_score": self.calculate_crisis_score(text_content),
                    "therapeutic_urgency": self.assess_therapeutic_urgency(risk_level),
                    "safety_assessment": self.assess_safety_factors(text_content)
                }
            }

        except Exception:
            return None

    def adapt_crisis_post_to_client_message(self, text: str, dataset_config: dict, risk_level: str) -> str:
        """Adapt a crisis post to sound like a client message in therapy."""
        # Clean and adapt the text
        adapted_text = text.replace("[removed]", "").replace("[deleted]", "").strip()

        # Add appropriate therapeutic framing based on crisis type and risk level
        if dataset_config["type"] == "suicide_risk":
            if risk_level == "high":
                intro = "I'm in a really dark place right now and having thoughts that scare me. "
            elif risk_level == "medium":
                intro = "I've been struggling with some difficult thoughts lately. "
            else:
                intro = "I wanted to talk about some feelings I've been having. "
        elif dataset_config["type"] == "pandemic_support":
            intro = "The pandemic has been really affecting my mental health. "
        else:
            intro = "I'm going through a crisis and need support. "

        return intro + adapted_text

    def generate_crisis_therapeutic_response(self, text: str, dataset_config: dict, risk_level: str) -> str:
        """Generate an appropriate therapeutic response for crisis situations."""
        # Base crisis response
        base_response = "Thank you for sharing this with me. I can hear that you're going through something very difficult right now. "

        # Risk-level specific responses
        if risk_level == "high":
            safety_response = "Your safety is my primary concern. Let's talk about what's happening and make sure you have immediate support. "
            intervention = "I want to help you get through this moment. Do you have someone you can call or be with right now? "
        elif risk_level == "medium":
            safety_response = "I'm concerned about what you're going through. Let's explore these feelings together. "
            intervention = "It's important that we address these thoughts. How are you taking care of yourself right now? "
        else:
            safety_response = "I hear that you're struggling, and I want to help. "
            intervention = "Let's work together to understand what you're experiencing. "

        # Crisis type specific responses
        if dataset_config["type"] == "suicide_risk":
            crisis_response = "Thoughts of suicide can feel overwhelming, but you don't have to face this alone. "
            resources = "There are people who want to help you through this. "
        elif dataset_config["type"] == "pandemic_support":
            crisis_response = "The pandemic has created unprecedented challenges for mental health. "
            resources = "Many people are struggling with similar feelings during this time. "
        else:
            crisis_response = "Crisis situations can feel overwhelming, but support is available. "
            resources = "We can work together to find ways to help you through this. "

        return base_response + safety_response + crisis_response + intervention + resources + "What feels most important to address right now?"

    def assess_crisis_risk_level(self, text: str, label: str, dataset_config: dict) -> str:
        """Assess the risk level of crisis content."""
        text_lower = text.lower()

        # If label is "non-suicide", it's automatically low risk
        if label and isinstance(label, str):
            label_lower = label.lower()
            if label_lower == "non-suicide":
                return "low"

        # For suicide-labeled content, assess severity based on text content
        # Immediate/active risk indicators (HIGH RISK)
        immediate_risk_terms = [
            "kill myself", "end my life", "want to die", "going to die", "plan to die",
            "take my own life", "end it all", "better off dead", "ready to die",
            "tonight", "today", "right now", "can't take it anymore"
        ]

        # Suicidal ideation indicators (MEDIUM RISK)
        ideation_terms = [
            "suicide", "suicidal", "kill me", "wish I was dead", "don't want to live",
            "no point living", "can't go on", "want to disappear", "end the pain"
        ]

        # Despair/hopelessness indicators (LOW-MEDIUM RISK)
        despair_terms = [
            "hopeless", "worthless", "give up", "no way out", "can't take it",
            "tired of living", "don't want to be here", "escape this pain",
            "nothing matters", "no hope"
        ]

        # Count severity indicators
        immediate_count = sum(1 for term in immediate_risk_terms if term in text_lower)
        ideation_count = sum(1 for term in ideation_terms if term in text_lower)
        despair_count = sum(1 for term in despair_terms if term in text_lower)

        # Risk assessment based on content severity
        if immediate_count >= 2 or (immediate_count >= 1 and ideation_count >= 1):
            return "high"
        if immediate_count >= 1 or ideation_count >= 2 or ideation_count >= 1 or despair_count >= 2:
            return "medium"
        if despair_count >= 1:
            return "low"
        # If labeled as "suicide" but no clear indicators, default to medium
        if label and label.lower() == "suicide":
            return "medium"
        return "low"

    def identify_crisis_indicators(self, text: str) -> list[str]:
        """Identify specific crisis indicators in the text."""
        text_lower = text.lower()
        indicators = []

        indicator_patterns = {
            "suicidal_ideation": ["kill myself", "end my life", "suicide", "want to die"],
            "hopelessness": ["hopeless", "no hope", "pointless", "no future"],
            "isolation": ["alone", "no one cares", "isolated", "abandoned"],
            "desperation": ["can't take it", "desperate", "at the end", "breaking point"],
            "self_harm": ["hurt myself", "self harm", "cut myself", "harm"],
            "substance_use": ["drinking", "drugs", "high", "numb the pain"]
        }

        for category, terms in indicator_patterns.items():
            if any(term in text_lower for term in terms):
                indicators.append(category)

        return indicators

    def determine_intervention_type(self, risk_level: str) -> str:
        """Determine the type of therapeutic intervention needed."""
        if risk_level == "high":
            return "immediate_safety_intervention"
        if risk_level == "medium":
            return "crisis_counseling"
        return "supportive_therapy"

    def get_safety_protocol(self, risk_level: str) -> dict[str, Any]:
        """Get appropriate safety protocol for risk level."""
        if risk_level == "high":
            return {
                "immediate_action": "assess_imminent_risk",
                "resources": ["crisis_hotline", "emergency_services", "safety_planning"],
                "follow_up": "immediate"
            }
        if risk_level == "medium":
            return {
                "immediate_action": "safety_assessment",
                "resources": ["crisis_hotline", "safety_planning", "support_network"],
                "follow_up": "within_24_hours"
            }
        return {
            "immediate_action": "supportive_intervention",
            "resources": ["counseling_resources", "support_groups"],
            "follow_up": "routine"
        }

    def calculate_crisis_score(self, text: str) -> float:
        """Calculate a crisis severity score."""
        text_lower = text.lower()
        score = 0.0

        # Suicide indicators (highest weight)
        suicide_terms = ["suicide", "kill myself", "end my life", "want to die"]
        score += sum(0.3 for term in suicide_terms if term in text_lower)

        # Hopelessness indicators
        hopeless_terms = ["hopeless", "no hope", "pointless", "worthless"]
        score += sum(0.2 for term in hopeless_terms if term in text_lower)

        # Desperation indicators
        desperation_terms = ["can't take it", "desperate", "breaking point", "end"]
        score += sum(0.1 for term in desperation_terms if term in text_lower)

        return min(score, 1.0)

    def assess_therapeutic_urgency(self, risk_level: str) -> str:
        """Assess the urgency of therapeutic intervention."""
        if risk_level == "high":
            return "immediate"
        if risk_level == "medium":
            return "urgent"
        return "routine"

    def assess_safety_factors(self, text: str) -> dict[str, Any]:
        """Assess safety factors in the text."""
        text_lower = text.lower()

        # Protective factors
        protective_factors = []
        protective_terms = {
            "social_support": ["family", "friends", "support", "people who care"],
            "future_orientation": ["future", "goals", "plans", "hope"],
            "help_seeking": ["help", "therapy", "counseling", "treatment"],
            "coping_skills": ["cope", "manage", "deal with", "handle"]
        }

        for factor, terms in protective_terms.items():
            if any(term in text_lower for term in terms):
                protective_factors.append(factor)

        # Risk factors
        risk_factors = []
        risk_terms = {
            "isolation": ["alone", "isolated", "no one", "abandoned"],
            "substance_use": ["drinking", "drugs", "alcohol", "high"],
            "recent_loss": ["died", "death", "lost", "breakup", "divorce"],
            "mental_illness": ["depression", "anxiety", "bipolar", "psychosis"]
        }

        for factor, terms in risk_terms.items():
            if any(term in text_lower for term in terms):
                risk_factors.append(factor)

        return {
            "protective_factors": protective_factors,
            "risk_factors": risk_factors,
            "protective_count": len(protective_factors),
            "risk_count": len(risk_factors)
        }

    def generate_comprehensive_report(self, dataset_reports: dict, all_conversations: list) -> dict[str, Any]:
        """Generate comprehensive crisis detection processing report."""

        # Analyze overall statistics
        total_conversations = len(all_conversations)

        # Risk level analysis
        risk_distribution = {}
        crisis_scores = []

        for conv in all_conversations:
            metadata = conv.get("metadata", {})
            risk_level = metadata.get("risk_level", "unknown")
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1

            crisis_score = metadata.get("crisis_score", 0.0)
            crisis_scores.append(crisis_score)

        avg_crisis_score = sum(crisis_scores) / len(crisis_scores) if crisis_scores else 0.0

        return {
            "task": "5.30: Crisis Detection Dataset Processing",
            "processing_summary": {
                "total_crisis_datasets_processed": len(self.crisis_datasets),
                "total_files_processed": self.processing_stats["total_files_processed"],
                "total_conversations_created": total_conversations,
                "total_posts_processed": self.processing_stats["total_posts_processed"],
                "processing_timestamp": datetime.now().isoformat(),
                "warning": "CONTAINS CRITICAL MENTAL HEALTH DATA - HANDLE WITH CARE"
            },
            "crisis_type_breakdown": dict(self.processing_stats["crisis_type_distribution"]),
            "risk_level_distribution": risk_distribution,
            "safety_metrics": {
                "average_crisis_score": avg_crisis_score,
                "high_risk_conversations": risk_distribution.get("high", 0),
                "medium_risk_conversations": risk_distribution.get("medium", 0),
                "low_risk_conversations": risk_distribution.get("low", 0),
                "total_at_risk": risk_distribution.get("high", 0) + risk_distribution.get("medium", 0)
            },
            "detailed_dataset_reports": dataset_reports,
            "dataset_info": {
                "tier": 4,
                "category": "crisis_detection_mental_health",
                "datasets_processed": [d["name"] for d in self.crisis_datasets],
                "priority_levels": [d["priority"] for d in self.crisis_datasets],
                "crisis_types": [d["type"] for d in self.crisis_datasets]
            },
            "ethical_considerations": {
                "data_sensitivity": "EXTREMELY HIGH - Contains suicide risk indicators",
                "usage_guidelines": "Requires specialized training and ethical oversight",
                "safety_protocols": "Must implement crisis intervention procedures",
                "privacy_requirements": "Enhanced privacy protection required"
            }
        }

def process_crisis_detection():
    """Main function to run crisis detection processing."""
    processor = CrisisDetectionProcessor()
    return processor.process_all_crisis_datasets()

if __name__ == "__main__":
    process_crisis_detection()
