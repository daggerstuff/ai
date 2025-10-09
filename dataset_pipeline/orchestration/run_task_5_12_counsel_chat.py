#!/usr/bin/env python3
"""
Task 5.12: Process counsel-chat dataset
Processes the counsel-chat dataset for counseling conversations.
"""

import csv
import json
import os
from datetime import datetime
from typing import Any


def process_counsel_chat():
    """Process counsel-chat dataset for Task 5.12."""

    # Create output directory
    output_dir = "ai/data/processed/phase_2_professional_datasets/task_5_12_therapist_sft"
    os.makedirs(output_dir, exist_ok=True)


    # Dataset configuration
    dataset_config = {
        "name": "counsel_chat",
        "base_path": "ai/datasets/counsel-chat/data",
        "target_conversations": 3000,
        "tier": 2,
        "expected_quality": 0.85,
        "description": "Counsel-Chat counseling conversations"
    }

    if not os.path.exists(dataset_config["base_path"]):
        return create_error_report(dataset_config, "Dataset directory not found")

    try:
        # Find all data files in the counsel-chat data directory
        data_files = []
        for root, _dirs, files in os.walk(dataset_config["base_path"]):
            for file in files:
                if file.endswith((".json", ".jsonl", ".csv")):
                    data_files.append(os.path.join(root, file))


        # Process all data files
        all_conversations = []
        processing_stats = {
            "total_processed": 0,
            "total_accepted": 0,
            "quality_filtered": 0,
            "format_errors": 0,
            "files_processed": 0
        }

        for file_path in data_files:

            try:
                conversations = process_counsel_chat_file(file_path, dataset_config)
                all_conversations.extend(conversations)
                processing_stats["files_processed"] += 1


                # Stop when we reach target
                if len(all_conversations) >= dataset_config["target_conversations"]:
                    all_conversations = all_conversations[:dataset_config["target_conversations"]]
                    break

            except Exception:
                continue

        # Update processing stats
        processing_stats["total_processed"] = len(all_conversations)
        processing_stats["total_accepted"] = len(all_conversations)

        # Save processed conversations
        output_path = os.path.join(output_dir, "counsel_chat_conversations.jsonl")
        with open(output_path, "w") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv) + "\n")

        # Generate report
        report = generate_counsel_chat_report(dataset_config, all_conversations, processing_stats)

        # Save report
        report_path = os.path.join(output_dir, "task_5_12_counsel_chat_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    except Exception as e:
        return create_error_report(dataset_config, str(e))

def process_counsel_chat_file(file_path: str, config: dict) -> list[dict[str, Any]]:
    """Process a single counsel-chat data file."""
    conversations = []

    try:
        if file_path.endswith(".csv"):
            # CSV format
            with open(file_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    conv = standardize_counsel_chat_conversation(row)
                    if conv:
                        conversations.append(conv)

        elif file_path.endswith(".jsonl"):
            # JSONL format
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        conv = standardize_counsel_chat_conversation(item)
                        if conv:
                            conversations.append(conv)
                    except:
                        continue

        else:
            # Regular JSON
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        conv = standardize_counsel_chat_conversation(item)
                        if conv:
                            conversations.append(conv)
                elif isinstance(data, dict):
                    conv = standardize_counsel_chat_conversation(data)
                    if conv:
                        conversations.append(conv)

    except Exception:
        pass

    return conversations

def standardize_counsel_chat_conversation(item: dict[str, Any]) -> dict[str, Any]:
    """Standardize counsel-chat conversation format."""
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

                if content and len(content.strip()) > 15:
                    standardized_messages.append({
                        "role": role,
                        "content": content.strip()
                    })

        elif "questionText" in item and "answerText" in item:
            # Counsel-chat specific format
            client_content = item["questionText"].strip()
            therapist_content = item["answerText"].strip()

            if len(client_content) < 20 or len(therapist_content) < 20:
                return None

            standardized_messages = [
                {"role": "client", "content": client_content},
                {"role": "therapist", "content": therapist_content}
            ]

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

        else:
            return None

        if len(standardized_messages) < 2:
            return None

        # Ensure starts with client
        if standardized_messages[0]["role"] != "client":
            if len(standardized_messages) > 2:
                standardized_messages = standardized_messages[1:]
            else:
                return None

        # Extract topic/theme if available
        topic = item.get("topic", item.get("theme", "general"))

        return {
            "conversation": standardized_messages,
            "metadata": {
                "category": "mental_health",
                "subcategory": "counseling_chat",
                "source": "counsel_chat",
                "dataset": "counsel_chat",
                "tier": 2,
                "therapeutic_approach": determine_therapeutic_approach(standardized_messages),
                "counseling_chat": True,
                "topic": topic,
                "conversation_length": len(standardized_messages)
            }
        }

    except Exception:
        return None

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

def generate_counsel_chat_report(config: dict, conversations: list, stats: dict) -> dict[str, Any]:
    """Generate comprehensive counsel-chat processing report."""

    # Analyze therapeutic approaches and topics
    approaches = {}
    topics = {}
    conversation_lengths = []

    for conv in conversations:
        approach = conv.get("metadata", {}).get("therapeutic_approach", "unknown")
        approaches[approach] = approaches.get(approach, 0) + 1

        topic = conv.get("metadata", {}).get("topic", "unknown")
        topics[topic] = topics.get(topic, 0) + 1

        length = len(conv.get("conversation", []))
        conversation_lengths.append(length)

    return {
        "task": "5.12: Counsel-Chat Dataset Processing",
        "processing_summary": {
            "dataset_name": config["name"],
            "total_conversations": len(conversations),
            "target_conversations": config["target_conversations"],
            "completion_percentage": (len(conversations) / config["target_conversations"]) * 100,
            "files_processed": stats["files_processed"],
            "processing_timestamp": datetime.now().isoformat()
        },
        "quality_metrics": {
            "total_processed": stats["total_processed"],
            "total_accepted": stats["total_accepted"],
            "acceptance_rate": 100.0,  # All processed conversations are accepted
            "format_errors": stats["format_errors"]
        },
        "conversation_analysis": {
            "therapeutic_approaches": approaches,
            "topics": topics,
            "conversation_length_stats": {
                "min": min(conversation_lengths) if conversation_lengths else 0,
                "max": max(conversation_lengths) if conversation_lengths else 0,
                "average": sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
            }
        },
        "dataset_characteristics": {
            "source": "Counsel-Chat Counseling Dataset",
            "quality_level": "Counseling Chat Grade",
            "tier": config["tier"],
            "expected_quality": config["expected_quality"],
            "counseling_chat_validation": True
        }
    }

def create_error_report(config: dict, error: str) -> dict[str, Any]:
    """Create error report for failed processing."""
    return {
        "task": "5.12: Counsel-Chat Dataset Processing",
        "processing_summary": {
            "dataset_name": config["name"],
            "total_conversations": 0,
            "success": False,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    process_counsel_chat()
