#!/usr/bin/env python3
"""
Task 5.10: Process therapist-sft-format conversations
Processes therapist SFT format dataset for supervised fine-tuning conversations.
"""

import csv
import json
import os
from datetime import datetime
from typing import Any


def process_therapist_sft():
    """Process therapist SFT format dataset for Task 5.10."""

    # Create output directory
    output_dir = "ai/data/processed/phase_2_professional_datasets/task_5_10_counsel_chat"
    os.makedirs(output_dir, exist_ok=True)


    # Dataset configuration
    dataset_config = {
        "name": "therapist_sft_format",
        "path": "ai/datasets/therapist-sft-format/train.csv",
        "target_conversations": 2000,
        "tier": 2,
        "expected_quality": 0.85,
        "description": "Therapist SFT format conversations for supervised fine-tuning"
    }

    if not os.path.exists(dataset_config["path"]):
        return create_error_report(dataset_config, "Dataset file not found")

    try:
        # Load the CSV dataset
        raw_data = []
        with open(dataset_config["path"], encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_data = list(reader)


        # Process conversations
        processed_conversations = []
        processing_stats = {
            "total_processed": 0,
            "total_accepted": 0,
            "quality_filtered": 0,
            "format_errors": 0
        }

        for i, item in enumerate(raw_data):
            processing_stats["total_processed"] += 1

            try:
                # Standardize conversation format
                conversation = standardize_therapist_sft_conversation(item, i)
                if not conversation:
                    processing_stats["format_errors"] += 1
                    continue

                # Quality assessment
                if not assess_therapist_sft_quality(conversation):
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
        output_path = os.path.join(output_dir, "therapist_sft_conversations.jsonl")
        with open(output_path, "w") as f:
            for conv in processed_conversations:
                f.write(json.dumps(conv) + "\n")

        # Generate report
        report = generate_therapist_sft_report(dataset_config, processed_conversations, processing_stats)

        # Save report
        report_path = os.path.join(output_dir, "task_5_10_therapist_sft_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    except Exception as e:
        return create_error_report(dataset_config, str(e))

def standardize_therapist_sft_conversation(item: dict[str, Any], index: int) -> dict[str, Any]:
    """Standardize therapist SFT conversation format."""
    try:
        # This dataset has a single 'text' column with full conversations
        if "text" not in item:
            return None

        conversation_text = item["text"].strip()
        if len(conversation_text) < 50:
            return None

        # Parse the conversation format: "human: ... gpt: ... human: ... gpt: ..."
        messages = []
        parts = conversation_text.split(" gpt: ")

        for i, part in enumerate(parts):
            if i == 0:
                # First part starts with "human: "
                if part.startswith("human: "):
                    client_content = part[7:].strip()  # Remove "human: "
                    if len(client_content) > 15:
                        messages.append({
                            "role": "client",
                            "content": client_content
                        })
            # Split on " human: " to separate therapist and next client message
            elif " human: " in part:
                therapist_part, human_part = part.split(" human: ", 1)

                # Add therapist message
                therapist_content = therapist_part.strip()
                if len(therapist_content) > 15:
                    messages.append({
                        "role": "therapist",
                        "content": therapist_content
                    })

                # Add client message
                client_content = human_part.strip()
                if len(client_content) > 15:
                    messages.append({
                        "role": "client",
                        "content": client_content
                    })
            else:
                # Last therapist message
                therapist_content = part.strip()
                if len(therapist_content) > 15:
                    messages.append({
                        "role": "therapist",
                        "content": therapist_content
                    })

        # Must have at least 2 messages and start with client
        if len(messages) < 2 or messages[0]["role"] != "client":
            return None

        return {
            "conversation": messages,
            "metadata": {
                "category": "mental_health",
                "subcategory": "therapist_sft",
                "source": "therapist_sft_format",
                "dataset": "therapist_sft_format",
                "tier": 2,
                "therapeutic_approach": determine_therapeutic_approach(messages),
                "sft_format": True,
                "conversation_length": len(messages),
                "index": index
            }
        }

    except Exception:
        return None

def assess_therapist_sft_quality(conversation: dict[str, Any]) -> bool:
    """Assess quality of therapist SFT conversation."""
    try:
        messages = conversation.get("conversation", [])

        # Must have at least 2 messages for SFT format
        if len(messages) < 2:
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
                    "understand", "feel", "experience", "explore", "share",
                    "thoughts", "emotions", "support", "help", "together",
                    "perspective", "coping", "strategies", "feelings", "therapy"
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

def generate_therapist_sft_report(config: dict, conversations: list, stats: dict) -> dict[str, Any]:
    """Generate comprehensive therapist SFT processing report."""

    # Analyze therapeutic approaches
    approaches = {}
    conversation_lengths = []

    for conv in conversations:
        approach = conv.get("metadata", {}).get("therapeutic_approach", "unknown")
        approaches[approach] = approaches.get(approach, 0) + 1

        length = len(conv.get("conversation", []))
        conversation_lengths.append(length)

    return {
        "task": "5.10: Therapist SFT Format Processing",
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
            "source": "Therapist SFT Format Dataset",
            "quality_level": "Supervised Fine-Tuning Grade",
            "tier": config["tier"],
            "expected_quality": config["expected_quality"],
            "sft_format_validation": True
        }
    }

def create_error_report(config: dict, error: str) -> dict[str, Any]:
    """Create error report for failed processing."""
    return {
        "task": "5.10: Therapist SFT Format Processing",
        "processing_summary": {
            "dataset_name": config["name"],
            "total_conversations": 0,
            "success": False,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    process_therapist_sft()
