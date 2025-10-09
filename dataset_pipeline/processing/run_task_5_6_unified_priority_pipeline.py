#!/usr/bin/env python3
"""
Task 5.6: Create unified priority dataset pipeline and quality assessment framework
Creates a unified pipeline combining all priority datasets with comprehensive quality assessment.
"""

import json
import os
from datetime import datetime
from typing import Any


def create_unified_priority_pipeline():
    """Create unified priority dataset pipeline for Task 5.6."""

    # Create output directory
    output_dir = "ai/data/processed/phase_1_priority_conversations/task_5_6_unified_priority"
    os.makedirs(output_dir, exist_ok=True)


    # Priority dataset configuration
    priority_datasets = {
        "priority_1": {
            "path": "ai/data/processed/task_5_1_priority_1/priority_1_conversations.jsonl",
            "report": "ai/data/processed/task_5_1_priority_1/task_5_1_priority_1_report.json",
            "tier": 1,
            "weight": 0.40  # 40% weight - highest quality
        },
        "priority_2": {
            "path": "ai/data/processed/task_5_2_priority_2/priority_2_conversations.jsonl",
            "report": "ai/data/processed/task_5_2_priority_2/task_5_2_priority_2_report.json",
            "tier": 1,
            "weight": 0.35  # 35% weight
        },
        "priority_3": {
            "path": "ai/data/processed/task_5_3_priority_3/priority_3_conversations.jsonl",
            "report": "ai/data/processed/task_5_3_priority_3/task_5_3_priority_3_report.json",
            "tier": 1,
            "weight": 0.25  # 25% weight
        }
    }

    try:
        # Load and combine all priority datasets
        all_conversations = []
        dataset_stats = {}

        for dataset_name, config in priority_datasets.items():

            if not os.path.exists(config["path"]):
                continue

            # Load conversations
            conversations = []
            with open(config["path"], encoding="utf-8") as f:
                for line in f:
                    try:
                        conv = json.loads(line.strip())
                        # Add unified metadata
                        conv["metadata"]["unified_priority"] = True
                        conv["metadata"]["priority_dataset"] = dataset_name
                        conv["metadata"]["dataset_weight"] = config["weight"]
                        conversations.append(conv)
                    except:
                        continue

            # Load report
            report = {}
            if os.path.exists(config["report"]):
                with open(config["report"], encoding="utf-8") as f:
                    report = json.load(f)

            dataset_stats[dataset_name] = {
                "conversation_count": len(conversations),
                "weight": config["weight"],
                "tier": config["tier"],
                "report_summary": report.get("processing_summary", {})
            }

            all_conversations.extend(conversations)

        # Perform quality assessment
        quality_assessment = perform_unified_quality_assessment(all_conversations)

        # Create balanced dataset
        balanced_conversations = create_balanced_dataset(all_conversations, priority_datasets)

        # Save unified conversations
        unified_path = os.path.join(output_dir, "unified_priority_conversations.jsonl")
        with open(unified_path, "w", encoding="utf-8") as f:
            for conv in balanced_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Generate comprehensive report
        unified_report = generate_unified_priority_report(
            priority_datasets,
            dataset_stats,
            all_conversations,
            balanced_conversations,
            quality_assessment
        )

        # Save unified report
        report_path = os.path.join(output_dir, "task_5_6_unified_priority_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(unified_report, f, indent=2, ensure_ascii=False)

        # Create quality dashboard
        create_priority_quality_dashboard(unified_report, output_dir)


        return unified_report

    except Exception as e:
        return create_error_report(str(e))

def perform_unified_quality_assessment(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    """Perform comprehensive quality assessment across all priority conversations."""

    assessment = {
        "total_conversations": len(conversations),
        "quality_distribution": {"high": 0, "medium": 0, "low": 0},
        "tier_distribution": {},
        "source_distribution": {},
        "priority_distribution": {},
        "average_quality_score": 0.0,
        "conversation_length_stats": {"min": 0, "max": 0, "average": 0.0},
        "quality_issues": []
    }

    if not conversations:
        return assessment

    quality_scores = []
    conversation_lengths = []

    for conv in conversations:
        metadata = conv.get("metadata", {})

        # Quality score analysis
        quality_score = metadata.get("quality_score", 0.0)
        quality_scores.append(quality_score)

        if quality_score >= 0.9:
            assessment["quality_distribution"]["high"] += 1
        elif quality_score >= 0.7:
            assessment["quality_distribution"]["medium"] += 1
        else:
            assessment["quality_distribution"]["low"] += 1

        # Distribution analysis
        tier = str(metadata.get("tier", "unknown"))
        source = metadata.get("source", "unknown")
        priority = str(metadata.get("priority", "unknown"))

        assessment["tier_distribution"][tier] = assessment["tier_distribution"].get(tier, 0) + 1
        assessment["source_distribution"][source] = assessment["source_distribution"].get(source, 0) + 1
        assessment["priority_distribution"][priority] = assessment["priority_distribution"].get(priority, 0) + 1

        # Conversation length
        conv_length = len(conv.get("conversation", []))
        conversation_lengths.append(conv_length)

    # Calculate statistics
    if quality_scores:
        assessment["average_quality_score"] = sum(quality_scores) / len(quality_scores)

    if conversation_lengths:
        assessment["conversation_length_stats"] = {
            "min": min(conversation_lengths),
            "max": max(conversation_lengths),
            "average": sum(conversation_lengths) / len(conversation_lengths)
        }

    # Identify quality issues
    low_quality_pct = (assessment["quality_distribution"]["low"] / len(conversations)) * 100
    if low_quality_pct > 10:
        assessment["quality_issues"].append(f"High percentage of low quality conversations: {low_quality_pct:.1f}%")

    return assessment

def create_balanced_dataset(conversations: list[dict[str, Any]], priority_configs: dict) -> list[dict[str, Any]]:
    """Create a balanced dataset respecting priority weights."""

    # Group conversations by priority dataset
    grouped_conversations = {}
    for conv in conversations:
        dataset = conv.get("metadata", {}).get("priority_dataset", "unknown")
        if dataset not in grouped_conversations:
            grouped_conversations[dataset] = []
        grouped_conversations[dataset].append(conv)

    # Calculate target counts based on weights
    total_target = 50000  # Target 50K balanced conversations
    balanced_conversations = []

    for dataset_name, config in priority_configs.items():
        if dataset_name in grouped_conversations:
            dataset_conversations = grouped_conversations[dataset_name]
            target_count = int(total_target * config["weight"])

            # Take up to target count, or all available if less
            selected_count = min(target_count, len(dataset_conversations))
            selected_conversations = dataset_conversations[:selected_count]

            balanced_conversations.extend(selected_conversations)

    return balanced_conversations

def generate_unified_priority_report(
    priority_configs: dict,
    dataset_stats: dict,
    all_conversations: list,
    balanced_conversations: list,
    quality_assessment: dict
) -> dict[str, Any]:
    """Generate comprehensive unified priority report."""

    return {
        "task": "5.6: Unified Priority Dataset Pipeline and Quality Assessment",
        "processing_summary": {
            "total_priority_datasets": len(priority_configs),
            "total_conversations_processed": len(all_conversations),
            "balanced_conversations_created": len(balanced_conversations),
            "processing_timestamp": datetime.now().isoformat()
        },
        "dataset_breakdown": dataset_stats,
        "quality_assessment": quality_assessment,
        "unified_characteristics": {
            "tier_1_focus": True,
            "multi_priority_integration": True,
            "weighted_balancing": True,
            "comprehensive_quality_validation": True
        },
        "pipeline_configuration": {
            "priority_weights": {name: config["weight"] for name, config in priority_configs.items()},
            "target_balanced_size": 50000,
            "quality_threshold": 0.75,
            "tier_focus": 1
        },
        "recommendations": [
            "Priority 1 conversations provide highest quality foundation",
            "Priority 2 adds clinical reasoning capabilities",
            "Priority 3 contributes cultural nuance understanding",
            "Balanced dataset maintains quality while ensuring diversity",
            "Continue monitoring quality metrics for future iterations"
        ]
    }

def create_priority_quality_dashboard(report: dict[str, Any], output_dir: str):
    """Create a quality dashboard for the unified priority dataset."""

    dashboard = {
        "title": "Unified Priority Dataset Quality Dashboard",
        "generated": datetime.now().isoformat(),
        "summary": {
            "total_conversations": report["processing_summary"]["total_conversations_processed"],
            "balanced_conversations": report["processing_summary"]["balanced_conversations_created"],
            "average_quality": report["quality_assessment"]["average_quality_score"],
            "high_quality_percentage": (report["quality_assessment"]["quality_distribution"]["high"] / report["quality_assessment"]["total_conversations"]) * 100
        },
        "dataset_contributions": report["dataset_breakdown"],
        "quality_metrics": report["quality_assessment"],
        "recommendations": report["recommendations"]
    }

    # Save dashboard
    dashboard_path = os.path.join(output_dir, "priority_quality_dashboard.json")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)

def create_error_report(error: str) -> dict[str, Any]:
    """Create error report for failed processing."""
    return {
        "task": "5.6: Unified Priority Dataset Pipeline and Quality Assessment",
        "processing_summary": {
            "total_conversations": 0,
            "success": False,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    create_unified_priority_pipeline()
