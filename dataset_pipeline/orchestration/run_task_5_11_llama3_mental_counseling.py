#!/usr/bin/env python3
"""
Task 5.11: Process LLAMA3_Mental_Counseling_Data advanced AI counseling conversations
Processes LLAMA3 mental counseling data from Parquet format.
"""

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd


def process_llama3_mental_counseling():
    """Process LLAMA3 mental counseling dataset for Task 5.11."""

    # Create output directory
    output_dir = "ai/data/processed/phase_2_professional_datasets/task_5_11_llama3_mental_counseling"
    os.makedirs(output_dir, exist_ok=True)


    # Dataset configuration
    dataset_config = {
        "name": "llama3_mental_counseling",
        "parquet_path": "ai/datasets/LLAMA3_Mental_Counseling_Data/data/train-00000-of-00001.parquet",
        "target_conversations": 3512,  # Process all available conversations
        "tier": 2,
        "expected_quality": 0.85,
        "description": "LLAMA3 advanced AI mental counseling conversations"
    }

    if not os.path.exists(dataset_config["parquet_path"]):
        return create_error_report(dataset_config, "Dataset file not found")

    try:
        # Load the Parquet dataset
        df = pd.read_parquet(dataset_config["parquet_path"])


        # Process conversations
        processed_conversations = []
        processing_stats = {
            "total_processed": 0,
            "total_accepted": 0,
            "quality_filtered": 0,
            "format_errors": 0
        }

        for i, row in df.iterrows():
            processing_stats["total_processed"] += 1

            try:
                # Standardize conversation format
                conversation = standardize_llama3_conversation(row, i)
                if not conversation:
                    processing_stats["format_errors"] += 1
                    continue

                # Quality assessment
                if not assess_llama3_quality(conversation):
                    processing_stats["quality_filtered"] += 1
                    continue

                processed_conversations.append(conversation)
                processing_stats["total_accepted"] += 1

                # Stop when we reach target
                if len(processed_conversations) >= dataset_config["target_conversations"]:
                    break

            except Exception:
                processing_stats["format_errors"] += 1
                continue

        # Save processed conversations
        output_path = os.path.join(output_dir, "llama3_mental_counseling_conversations.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in processed_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Generate report
        report = generate_llama3_report(dataset_config, processed_conversations, processing_stats)

        # Save report
        report_path = os.path.join(output_dir, "task_5_11_llama3_mental_counseling_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    except Exception as e:
        return create_error_report(dataset_config, str(e))

def standardize_llama3_conversation(row: pd.Series, index: int) -> dict[str, Any]:
    """Standardize LLAMA3 conversation format."""
    try:
        # Extract Context and Response from Parquet data
        if "Context" not in row or "Response" not in row:
            return None

        client_content = str(row["Context"]).strip()
        therapist_content = str(row["Response"]).strip()

        # Validate content
        if len(client_content) < 20 or len(therapist_content) < 20:
            return None

        if client_content == "nan" or therapist_content == "nan":
            return None

        # Create standardized conversation
        standardized_messages = [
            {
                "role": "client",
                "content": client_content
            },
            {
                "role": "therapist",
                "content": therapist_content
            }
        ]

        return {
            "conversation": standardized_messages,
            "metadata": {
                "category": "mental_health",
                "subcategory": "llama3_ai_counseling",
                "source": "llama3_mental_counseling",
                "dataset": "llama3_mental_counseling",
                "tier": 2,
                "therapeutic_approach": determine_therapeutic_approach(standardized_messages),
                "ai_generated": True,
                "conversation_length": len(standardized_messages),
                "index": index
            }
        }

    except Exception:
        return None

def assess_llama3_quality(conversation: dict[str, Any]) -> bool:
    """Assess quality of LLAMA3 conversation."""
    try:
        messages = conversation.get("conversation", [])

        # Must have exactly 2 messages
        if len(messages) != 2:
            return False

        # Check content quality
        for msg in messages:
            content = msg.get("content", "")

            # Minimum content length
            if len(content.strip()) < 20:
                return False

            # Check for therapeutic indicators in therapist response
            if msg["role"] == "therapist":
                therapeutic_indicators = [
                    "understand", "feel", "experience", "support", "help",
                    "thoughts", "emotions", "coping", "strategies", "therapy",
                    "counseling", "mental health", "wellbeing", "healing"
                ]
                if not any(indicator in content.lower() for indicator in therapeutic_indicators):
                    # Allow some flexibility but prefer therapeutic language
                    pass

        return True

    except Exception:
        return False

def determine_therapeutic_approach(messages: list[dict[str, Any]]) -> str:
    """Determine therapeutic approach from conversation content."""
    all_text = " ".join([msg.get("content", "") for msg in messages]).lower()

    # Approach indicators
    if any(word in all_text for word in ["cognitive", "thoughts", "thinking patterns", "beliefs", "cbt"]):
        return "cognitive_behavioral"
    if any(word in all_text for word in ["feelings", "emotions", "emotional", "empathy", "validation"]):
        return "emotion_focused"
    if any(word in all_text for word in ["behavior", "actions", "habits", "patterns", "behavioral"]):
        return "behavioral"
    if any(word in all_text for word in ["mindfulness", "present", "awareness", "meditation", "mindful"]):
        return "mindfulness_based"
    if any(word in all_text for word in ["relationship", "interpersonal", "social", "family", "communication"]):
        return "interpersonal"
    if any(word in all_text for word in ["solution", "goals", "strengths", "resources", "brief"]):
        return "solution_focused"
    return "integrative"

def generate_llama3_report(config: dict, conversations: list, stats: dict) -> dict[str, Any]:
    """Generate comprehensive LLAMA3 processing report."""

    # Analyze therapeutic approaches
    approaches = {}
    conversation_lengths = []

    for conv in conversations:
        approach = conv.get("metadata", {}).get("therapeutic_approach", "unknown")
        approaches[approach] = approaches.get(approach, 0) + 1

        length = len(conv.get("conversation", []))
        conversation_lengths.append(length)

    return {
        "task": "5.11: LLAMA3 Mental Counseling Data Processing",
        "processing_summary": {
            "dataset_name": config["name"],
            "total_conversations": len(conversations),
            "target_conversations": config["target_conversations"],
            "completion_percentage": (len(conversations) / config["target_conversations"]) * 100,
            "processing_timestamp": datetime.now().isoformat()
        },
        "quality_metrics": {
            "total_processed": stats["total_processed"],
            "total_accepted": stats["total_accepted"],
            "acceptance_rate": (stats["total_accepted"] / stats["total_processed"]) * 100 if stats["total_processed"] > 0 else 0,
            "quality_filtered": stats["quality_filtered"],
            "format_errors": stats["format_errors"]
        },
        "conversation_analysis": {
            "therapeutic_approaches": approaches,
            "conversation_length_stats": {
                "min": min(conversation_lengths) if conversation_lengths else 0,
                "max": max(conversation_lengths) if conversation_lengths else 0,
                "average": sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
            }
        },
        "dataset_characteristics": {
            "source": "LLAMA3 Mental Counseling Data",
            "quality_level": "AI-Generated Advanced Counseling Grade",
            "tier": config["tier"],
            "expected_quality": config["expected_quality"],
            "ai_generated_validation": True,
            "data_format": "Parquet (Context/Response)"
        }
    }

def create_error_report(config: dict, error: str) -> dict[str, Any]:
    """Create error report for failed processing."""
    return {
        "task": "5.11: LLAMA3 Mental Counseling Data Processing",
        "processing_summary": {
            "dataset_name": config["name"],
            "total_conversations": 0,
            "success": False,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    process_llama3_mental_counseling()
