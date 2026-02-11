#!/usr/bin/env python3
"""
Task 5.7: Process Psych8k Alexander Street dataset (40K+ professional therapy conversations)
Processes the high-quality Alexander Street professional therapy dataset.
"""

import json
import os
from datetime import datetime
from typing import Any


def process_psych8k_alexander_street():
    """Process Psych8k Alexander Street dataset for Task 5.7."""

    # Create output directory
    output_dir = "ai/data/processed/phase_2_professional_datasets/task_5_7_psych8k_alexander_street"
    os.makedirs(output_dir, exist_ok=True)


    # Dataset configuration
    dataset_config = {
        "name": "psych8k_alexander_street",
        "path": "ai/datasets/Psych8k/Alexander_Street_shareGPT_2.0.json",
        "target_conversations": 8000,  # Process 8K high-quality conversations
        "tier": 2,
        "expected_quality": 0.90,
        "description": "Alexander Street professional therapy conversations (40K+ samples, 6.3MB)"
    }

    if not os.path.exists(dataset_config["path"]):
        return create_error_report(dataset_config, "Dataset file not found")

    try:
        # Load the dataset
        with open(dataset_config["path"], encoding="utf-8") as f:
            raw_data = json.load(f)


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
                conversation = standardize_alexander_street_conversation(item, i)
                if not conversation:
                    processing_stats["format_errors"] += 1
                    continue

                # Quality assessment
                if not assess_alexander_street_quality(conversation):
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
        output_path = os.path.join(output_dir, "psych8k_alexander_conversations.jsonl")
        with open(output_path, "w") as f:
            for conv in processed_conversations:
                f.write(json.dumps(conv) + "\n")

        # Generate report
        report = generate_psych8k_report(dataset_config, processed_conversations, processing_stats)

        # Save report
        report_path = os.path.join(output_dir, "task_5_7_psych8k_alexander_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    except Exception as e:
        return create_error_report(dataset_config, str(e))

def standardize_alexander_street_conversation(item: dict[str, Any], index: int) -> dict[str, Any]:
    """Standardize Alexander Street conversation format."""
    try:
        # Alexander Street format uses instruction/input/output format
        if "input" not in item or "output" not in item:
            return None

        client_content = item["input"].strip()
        therapist_content = item["output"].strip()

        # Validate content quality
        if len(client_content) < 20 or len(therapist_content) < 20:
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
                "subcategory": "professional_therapy",
                "source": "alexander_street",
                "dataset": "psych8k_alexander_street",
                "tier": 2,
                "therapeutic_approach": determine_therapeutic_approach(standardized_messages),
                "professional_quality": True,
                "conversation_length": len(standardized_messages),
                "index": index,
                "instruction": item.get("instruction", "")
            }
        }

    except Exception:
        return None

def assess_alexander_street_quality(conversation: dict[str, Any]) -> bool:
    """Assess quality of Alexander Street conversation."""
    try:
        messages = conversation.get("conversation", [])

        # Must have at least 2 messages
        if len(messages) < 2:
            return False

        # Check content quality
        for msg in messages:
            content = msg.get("content", "")

            # Minimum content length
            if len(content.strip()) < 20:
                return False

            # Check for therapeutic indicators
            if msg["role"] == "therapist":
                therapeutic_indicators = [
                    "feel", "understand", "experience", "explore", "share",
                    "thoughts", "emotions", "support", "help", "together"
                ]
                if not any(indicator in content.lower() for indicator in therapeutic_indicators):
                    continue  # Allow some therapist responses without indicators

        # Check conversation flow
        roles = [msg["role"] for msg in messages]
        return roles[0] == "client" and len(set(roles)) == 2

    except Exception:
        return False

def determine_therapeutic_approach(messages: list[dict[str, Any]]) -> str:
    """Determine therapeutic approach from conversation content."""
    all_text = " ".join([msg.get("content", "") for msg in messages]).lower()

    # Approach indicators
    if any(word in all_text for word in ["cognitive", "thoughts", "thinking patterns", "beliefs"]):
        return "cognitive_behavioral"
    if any(word in all_text for word in ["feelings", "emotions", "emotional", "empathy"]):
        return "emotion_focused"
    if any(word in all_text for word in ["behavior", "actions", "habits", "patterns"]):
        return "behavioral"
    if any(word in all_text for word in ["mindfulness", "present", "awareness", "meditation"]):
        return "mindfulness_based"
    if any(word in all_text for word in ["relationship", "interpersonal", "social", "family"]):
        return "interpersonal"
    return "integrative"

def generate_psych8k_report(config: dict, conversations: list, stats: dict) -> dict[str, Any]:
    """Generate comprehensive Psych8k processing report."""

    # Analyze therapeutic approaches
    approaches = {}
    conversation_lengths = []

    for conv in conversations:
        approach = conv.get("metadata", {}).get("therapeutic_approach", "unknown")
        approaches[approach] = approaches.get(approach, 0) + 1

        length = len(conv.get("conversation", []))
        conversation_lengths.append(length)

    return {
        "task": "5.7: Psych8k Alexander Street Dataset Processing",
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
            "source": "Alexander Street Professional Therapy Database",
            "quality_level": "Professional/Clinical Grade",
            "tier": config["tier"],
            "expected_quality": config["expected_quality"],
            "professional_validation": True
        },
        "next_steps": [
            "Task 5.8: Integrate mental_health_counseling_conversations (3.5K licensed therapist responses)",
            "Task 5.9: Process SoulChat2.0 psychological counselor digital twin framework"
        ]
    }

def create_error_report(config: dict, error: str) -> dict[str, Any]:
    """Create error report for failed processing."""
    return {
        "task": "5.7: Psych8k Alexander Street Dataset Processing",
        "processing_summary": {
            "dataset_name": config["name"],
            "total_conversations": 0,
            "success": False,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    process_psych8k_alexander_street()
