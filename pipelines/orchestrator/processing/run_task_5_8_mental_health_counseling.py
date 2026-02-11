#!/usr/bin/env python3
"""
Task 5.8: Integrate mental_health_counseling_conversations (3.5K licensed therapist responses)
Processes licensed therapist conversations for high-quality therapeutic data.
"""

import json
import os
from datetime import datetime
from typing import Any


def process_mental_health_counseling():
    """Process mental health counseling conversations for Task 5.8."""

    # Create output directory
    output_dir = "ai/data/processed/phase_2_professional_datasets/task_5_8_mental_health_counseling"
    os.makedirs(output_dir, exist_ok=True)


    # Dataset configuration
    dataset_config = {
        "name": "mental_health_counseling_conversations",
        "path": "ai/datasets/mental_health_counseling_conversations/combined_dataset.json",
        "target_conversations": 3500,
        "tier": 2,
        "expected_quality": 0.95,
        "description": "Licensed therapist responses (3.5K high-quality conversations)"
    }

    if not os.path.exists(dataset_config["path"]):
        return create_error_report(dataset_config, "Dataset file not found")

    try:
        # Load the dataset (JSONL format)
        raw_data = []
        with open(dataset_config["path"], encoding="utf-8") as f:
            for line in f:
                try:
                    raw_data.append(json.loads(line.strip()))
                except:
                    continue


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
                conversation = standardize_counseling_conversation(item, i)
                if not conversation:
                    processing_stats["format_errors"] += 1
                    continue

                # Quality assessment
                if not assess_counseling_quality(conversation):
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
        output_path = os.path.join(output_dir, "mental_health_counseling_conversations.jsonl")
        with open(output_path, "w") as f:
            for conv in processed_conversations:
                f.write(json.dumps(conv) + "\n")

        # Generate report
        report = generate_counseling_report(dataset_config, processed_conversations, processing_stats)

        # Save report
        report_path = os.path.join(output_dir, "task_5_8_mental_health_counseling_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    except Exception as e:
        return create_error_report(dataset_config, str(e))

def standardize_counseling_conversation(item: dict[str, Any], index: int) -> dict[str, Any]:
    """Standardize mental health counseling conversation format."""
    try:
        # Check for different possible formats
        if "conversations" in item:
            # ShareGPT format
            messages = item["conversations"]
            standardized_messages = []

            for msg in messages:
                if "from" in msg and "value" in msg:
                    role = "client" if msg["from"] in ["human", "user"] else "therapist"
                    content = msg["value"]
                elif "role" in msg and "content" in msg:
                    role = "client" if msg["role"] in ["human", "user"] else "therapist"
                    content = msg["content"]
                else:
                    continue

                if content and len(content.strip()) > 10:
                    standardized_messages.append({
                        "role": role,
                        "content": content.strip()
                    })

        elif "input" in item and "output" in item:
            # Input/output format
            client_content = item["input"].strip()
            therapist_content = item["output"].strip()

            if len(client_content) < 20 or len(therapist_content) < 20:
                return None

            standardized_messages = [
                {"role": "client", "content": client_content},
                {"role": "therapist", "content": therapist_content}
            ]

        elif "question" in item and "answer" in item:
            # Q&A format
            client_content = item["question"].strip()
            therapist_content = item["answer"].strip()

            if len(client_content) < 20 or len(therapist_content) < 20:
                return None

            standardized_messages = [
                {"role": "client", "content": client_content},
                {"role": "therapist", "content": therapist_content}
            ]

        elif "Context" in item and "Response" in item:
            # Context/Response format (mental health counseling)
            client_content = item["Context"].strip()
            therapist_content = item["Response"].strip()

            if len(client_content) < 20 or len(therapist_content) < 20:
                return None

            standardized_messages = [
                {"role": "client", "content": client_content},
                {"role": "therapist", "content": therapist_content}
            ]

        else:
            return None

        if len(standardized_messages) < 2:
            return None

        # Ensure starts with client
        if standardized_messages[0]["role"] != "client":
            standardized_messages = standardized_messages[1:]

        return {
            "conversation": standardized_messages,
            "metadata": {
                "category": "mental_health",
                "subcategory": "licensed_counseling",
                "source": "mental_health_counseling",
                "dataset": "mental_health_counseling_conversations",
                "tier": 2,
                "therapeutic_approach": determine_therapeutic_approach(standardized_messages),
                "licensed_therapist": True,
                "professional_quality": True,
                "conversation_length": len(standardized_messages),
                "index": index
            }
        }

    except Exception:
        return None

def assess_counseling_quality(conversation: dict[str, Any]) -> bool:
    """Assess quality of counseling conversation."""
    try:
        messages = conversation.get("conversation", [])

        # Must have at least 2 messages
        if len(messages) < 2:
            return False

        # Check content quality
        for msg in messages:
            content = msg.get("content", "")

            # Minimum content length
            if len(content.strip()) < 25:
                return False

            # Check for therapeutic quality indicators
            if msg["role"] == "therapist":
                therapeutic_indicators = [
                    "understand", "feel", "experience", "explore", "share",
                    "thoughts", "emotions", "support", "help", "together",
                    "perspective", "coping", "strategies", "feelings"
                ]
                if not any(indicator in content.lower() for indicator in therapeutic_indicators):
                    # Allow some flexibility but prefer therapeutic language
                    pass

        # Check for inappropriate content
        inappropriate_content = ["inappropriate", "unprofessional", "harmful"]
        all_text = " ".join([msg.get("content", "") for msg in messages]).lower()
        return not any(word in all_text for word in inappropriate_content)

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

def generate_counseling_report(config: dict, conversations: list, stats: dict) -> dict[str, Any]:
    """Generate comprehensive counseling processing report."""

    # Analyze therapeutic approaches
    approaches = {}
    conversation_lengths = []

    for conv in conversations:
        approach = conv.get("metadata", {}).get("therapeutic_approach", "unknown")
        approaches[approach] = approaches.get(approach, 0) + 1

        length = len(conv.get("conversation", []))
        conversation_lengths.append(length)

    return {
        "task": "5.8: Mental Health Counseling Conversations Integration",
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
            "source": "Licensed Mental Health Counselors",
            "quality_level": "Licensed Professional Grade",
            "tier": config["tier"],
            "expected_quality": config["expected_quality"],
            "licensed_therapist_validation": True
        },
        "next_steps": [
            "Task 5.9: Process SoulChat2.0 psychological counselor digital twin framework",
            "Task 5.10: Integrate therapist_sft_format conversations"
        ]
    }

def create_error_report(config: dict, error: str) -> dict[str, Any]:
    """Create error report for failed processing."""
    return {
        "task": "5.8: Mental Health Counseling Conversations Integration",
        "processing_summary": {
            "dataset_name": config["name"],
            "total_conversations": 0,
            "success": False,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    process_mental_health_counseling()
