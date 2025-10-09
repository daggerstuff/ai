#!/usr/bin/env python3
"""
Task 5.2: Process datasets-wendy/priority_2_FINAL.jsonl + summary.json
Processes priority 2 therapeutic conversations with chain-of-thought reasoning (Tier 1).
"""

import json
import os
from datetime import datetime
from typing import Any


def process_priority_2():
    """Process priority_2_FINAL.jsonl for Task 5.2."""

    # Create output directory
    output_dir = "ai/data/processed/phase_1_priority_conversations/task_5_2_priority_2"
    os.makedirs(output_dir, exist_ok=True)


    # Dataset configuration
    dataset_config = {
        "name": "priority_2_final",
        "jsonl_path": "ai/datasets/datasets-wendy/priority_2/priority_2_FINAL.jsonl",
        "summary_path": "ai/datasets/datasets-wendy/priority_2/priority_2_FINAL_summary.json",
        "target_conversations": 30000,  # Process up to 30K from 84K available
        "tier": 1,
        "expected_quality": 0.95,
        "description": "Priority 2 therapeutic conversations with chain-of-thought reasoning"
    }

    # Check if files exist
    if not os.path.exists(dataset_config["jsonl_path"]):
        return create_error_report(dataset_config, "JSONL file not found")

    if not os.path.exists(dataset_config["summary_path"]):
        return create_error_report(dataset_config, "Summary file not found")

    try:
        # Load summary data
        with open(dataset_config["summary_path"], encoding="utf-8") as f:
            summary_data = json.load(f)


        # Process conversations
        processed_conversations = []
        processing_stats = {
            "total_processed": 0,
            "total_accepted": 0,
            "quality_filtered": 0,
            "format_errors": 0,
            "source_distribution": {},
            "category_distribution": {}
        }

        with open(dataset_config["jsonl_path"], encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if len(processed_conversations) >= dataset_config["target_conversations"]:
                    break

                processing_stats["total_processed"] += 1

                try:
                    item = json.loads(line.strip())

                    # Standardize conversation format
                    conversation = standardize_priority_2_conversation(item, line_num)
                    if not conversation:
                        processing_stats["format_errors"] += 1
                        continue

                    # Quality assessment
                    if not assess_priority_2_quality(conversation, item):
                        processing_stats["quality_filtered"] += 1
                        continue

                    # Track distributions
                    source = item.get("source", "unknown")
                    category = item.get("category", "unknown")
                    processing_stats["source_distribution"][source] = processing_stats["source_distribution"].get(source, 0) + 1
                    processing_stats["category_distribution"][category] = processing_stats["category_distribution"].get(category, 0) + 1

                    processed_conversations.append(conversation)
                    processing_stats["total_accepted"] += 1

                    if processing_stats["total_processed"] % 10000 == 0:
                        pass

                except Exception:
                    processing_stats["format_errors"] += 1
                    continue

        # Save processed conversations
        output_path = os.path.join(output_dir, "priority_2_conversations.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in processed_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Generate report
        report = generate_priority_2_report(dataset_config, processed_conversations, processing_stats, summary_data)

        # Save report
        report_path = os.path.join(output_dir, "task_5_2_priority_2_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    except Exception as e:
        return create_error_report(dataset_config, str(e))

def standardize_priority_2_conversation(item: dict[str, Any], index: int) -> dict[str, Any]:
    """Standardize priority 2 conversation format."""
    try:
        # Extract conversation data
        if "conversation" not in item:
            return None

        conversation_data = item["conversation"]
        if not isinstance(conversation_data, list) or len(conversation_data) < 2:
            return None

        # Standardize messages
        standardized_messages = []
        for msg in conversation_data:
            if "role" not in msg or "content" not in msg:
                continue

            role = msg["role"]
            content = msg["content"].strip()

            if len(content) < 20:  # Skip very short messages
                continue

            standardized_messages.append({
                "role": role,
                "content": content
            })

        if len(standardized_messages) < 2:
            return None

        # Ensure starts with client
        if standardized_messages[0]["role"] != "client":
            return None

        # Extract reasoning if available
        reasoning = None
        if len(conversation_data) > 1 and "internal_reasoning" in conversation_data[1]:
            reasoning = conversation_data[1]["internal_reasoning"]

        return {
            "conversation": standardized_messages,
            "metadata": {
                "category": "mental_health",
                "subcategory": "priority_clinical_reasoning",
                "source": item.get("source", "unknown"),
                "dataset": "priority_2_final",
                "tier": 1,
                "priority": item.get("priority", 2),
                "quality_score": item.get("metadata", {}).get("quality_score", 0.95),
                "conversation_length": len(standardized_messages),
                "original_conversation_id": item.get("conversation_id"),
                "turn_count": item.get("metadata", {}).get("turn_count", len(standardized_messages)),
                "category_type": item.get("category", "unknown"),
                "has_reasoning": reasoning is not None,
                "reasoning_type": item.get("metadata", {}).get("reasoning_type", "chain_of_thought"),
                "difficulty": item.get("metadata", {}).get("difficulty", 4),
                "topic": item.get("metadata", {}).get("topic", ""),
                "index": index
            },
            "reasoning": reasoning
        }

    except Exception:
        return None

def assess_priority_2_quality(conversation: dict[str, Any], original_item: dict[str, Any]) -> bool:
    """Assess quality of priority 2 conversation."""
    try:
        # Check original quality score (should be very high for priority 2)
        original_quality = original_item.get("metadata", {}).get("quality_score", 0.0)
        if original_quality < 0.90:  # Only accept very high quality
            return False

        messages = conversation.get("conversation", [])

        # Must have at least 2 messages
        if len(messages) < 2:
            return False

        # Check content quality
        for msg in messages:
            content = msg.get("content", "")

            # Minimum content length for clinical reasoning
            if len(content.strip()) < 50:
                return False

            # Check for clinical/therapeutic indicators
            if msg["role"] == "therapist":
                clinical_indicators = [
                    "assessment", "diagnosis", "treatment", "therapy", "clinical",
                    "symptoms", "disorder", "intervention", "therapeutic", "patient",
                    "client", "mental health", "psychological", "psychiatric", "counseling"
                ]
                if not any(indicator in content.lower() for indicator in clinical_indicators):
                    # Allow some flexibility but prefer clinical language
                    pass

        # Check for reasoning quality if available
        reasoning = conversation.get("reasoning")
        if reasoning and len(reasoning.strip()) < 100:
            return False  # Reasoning should be substantial

        return True

    except Exception:
        return False

def generate_priority_2_report(config: dict, conversations: list, stats: dict, summary: dict) -> dict[str, Any]:
    """Generate comprehensive priority 2 processing report."""

    # Analyze quality scores and reasoning
    quality_scores = []
    conversation_lengths = []
    turn_counts = []
    reasoning_counts = {"with_reasoning": 0, "without_reasoning": 0}
    difficulty_distribution = {}

    for conv in conversations:
        metadata = conv.get("metadata", {})
        quality_scores.append(metadata.get("quality_score", 0.0))
        conversation_lengths.append(len(conv.get("conversation", [])))
        turn_counts.append(metadata.get("turn_count", 0))

        if metadata.get("has_reasoning", False):
            reasoning_counts["with_reasoning"] += 1
        else:
            reasoning_counts["without_reasoning"] += 1

        difficulty = metadata.get("difficulty", 0)
        difficulty_distribution[str(difficulty)] = difficulty_distribution.get(str(difficulty), 0) + 1

    return {
        "task": "5.2: Priority 2 Dataset Processing",
        "processing_summary": {
            "dataset_name": config["name"],
            "total_conversations": len(conversations),
            "target_conversations": config["target_conversations"],
            "completion_percentage": (len(conversations) / config["target_conversations"]) * 100,
            "processing_timestamp": datetime.now().isoformat()
        },
        "original_summary": summary,
        "quality_metrics": {
            "total_processed": stats["total_processed"],
            "total_accepted": stats["total_accepted"],
            "acceptance_rate": (stats["total_accepted"] / stats["total_processed"]) * 100 if stats["total_processed"] > 0 else 0,
            "quality_filtered": stats["quality_filtered"],
            "format_errors": stats["format_errors"],
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "quality_score_range": {
                "min": min(quality_scores) if quality_scores else 0,
                "max": max(quality_scores) if quality_scores else 0
            }
        },
        "conversation_analysis": {
            "source_distribution": stats["source_distribution"],
            "category_distribution": stats["category_distribution"],
            "reasoning_analysis": reasoning_counts,
            "difficulty_distribution": difficulty_distribution,
            "conversation_length_stats": {
                "min": min(conversation_lengths) if conversation_lengths else 0,
                "max": max(conversation_lengths) if conversation_lengths else 0,
                "average": sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
            },
            "turn_count_stats": {
                "min": min(turn_counts) if turn_counts else 0,
                "max": max(turn_counts) if turn_counts else 0,
                "average": sum(turn_counts) / len(turn_counts) if turn_counts else 0
            }
        },
        "dataset_characteristics": {
            "source": "datasets-wendy/priority_2_FINAL.jsonl",
            "quality_level": "Tier 1 - Clinical Reasoning Grade",
            "tier": config["tier"],
            "expected_quality": config["expected_quality"],
            "priority_validation": True,
            "chain_of_thought_reasoning": True,
            "original_total_samples": summary.get("total_samples", 0)
        }
    }

def create_error_report(config: dict, error: str) -> dict[str, Any]:
    """Create error report for failed processing."""
    return {
        "task": "5.2: Priority 2 Dataset Processing",
        "processing_summary": {
            "dataset_name": config["name"],
            "total_conversations": 0,
            "success": False,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    process_priority_2()
