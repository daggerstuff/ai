#!/usr/bin/env python3
"""
Task 5.25: Process ToT (Tree-of-Thought) Reasoning Problem Solving Dataset V2
Advanced tree-of-thought reasoning for complex therapeutic problem solving.
"""

import json
import os
from datetime import datetime
from typing import Any


def process_tot_reasoning():
    """Process ToT Reasoning Problem Solving Dataset V2 for Task 5.25."""

    # Create output directory
    output_dir = "ai/data/processed/phase_3_cot_reasoning/task_5_25_tot_reasoning"
    os.makedirs(output_dir, exist_ok=True)


    # Dataset configuration
    dataset_config = {
        "name": "tot_reasoning_problem_solving_v2",
        "json_path": "datasets/ToT_Reasoning_Problem_Solving_Dataset_V2/ToT-RPSD-V2.json",
        "target_conversations": 5000,  # Process up to 5K conversations
        "tier": 3,
        "expected_quality": 0.8,
        "description": "Tree-of-thought reasoning for complex therapeutic problem solving"
    }

    processing_stats = {
        "total_processed": 0,
        "total_accepted": 0,
        "quality_filtered": 0,
        "format_errors": 0,
        "reasoning_complexity_distribution": {},
        "problem_type_distribution": {}
    }

    try:
        # Check if dataset exists
        if not os.path.exists(dataset_config["json_path"]):
            return create_error_report(dataset_config, "Dataset file not found")

        # Load dataset
        with open(dataset_config["json_path"], encoding="utf-8") as f:
            data = json.load(f)


        # Process conversations
        processed_conversations = []

        for i, item in enumerate(data):
            if len(processed_conversations) >= dataset_config["target_conversations"]:
                break

            processing_stats["total_processed"] += 1

            try:
                # Standardize conversation format
                conversation = standardize_tot_conversation(item, i)
                if not conversation:
                    processing_stats["format_errors"] += 1
                    continue

                # Quality assessment
                if not assess_tot_quality(conversation):
                    processing_stats["quality_filtered"] += 1
                    continue

                processed_conversations.append(conversation)
                processing_stats["total_accepted"] += 1

                # Track reasoning complexity and problem types
                metadata = conversation.get("metadata", {})
                complexity = metadata.get("reasoning_complexity", "unknown")
                problem_type = metadata.get("problem_type", "unknown")

                processing_stats["reasoning_complexity_distribution"][complexity] = \
                    processing_stats["reasoning_complexity_distribution"].get(complexity, 0) + 1
                processing_stats["problem_type_distribution"][problem_type] = \
                    processing_stats["problem_type_distribution"].get(problem_type, 0) + 1

                if processing_stats["total_processed"] % 1000 == 0:
                    pass

            except Exception:
                processing_stats["format_errors"] += 1
                continue

        # Save processed conversations
        output_path = os.path.join(output_dir, "tot_reasoning_conversations.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in processed_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Generate report
        report = generate_tot_report(dataset_config, processed_conversations, processing_stats)

        # Save report
        report_path = os.path.join(output_dir, "task_5_25_tot_reasoning_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    except Exception as e:
        return create_error_report(dataset_config, str(e))

def standardize_tot_conversation(item: dict[str, Any], index: int) -> dict[str, Any]:
    """Standardize ToT reasoning conversation format."""
    try:
        # Extract problem and solution
        problem = item.get("question", "")
        solution = item.get("answer", "")

        if not problem or not solution:
            return None

        # Extract tree-of-thought reasoning from metadata
        metadata_reasoning = item.get("metadata", {}).get("reasoning", "")
        reasoning_steps = []

        if metadata_reasoning:
            # Parse string-based reasoning into steps
            reasoning_steps = parse_reasoning_steps(metadata_reasoning)

        # Create standardized conversation with therapeutic context
        therapeutic_problem = adapt_to_therapeutic_context(problem)
        therapeutic_solution = adapt_solution_to_therapeutic_context(solution, reasoning_steps)

        messages = [
            {
                "role": "client",
                "content": therapeutic_problem.strip(),
                "turn_id": 1
            },
            {
                "role": "therapist",
                "content": therapeutic_solution.strip(),
                "turn_id": 2,
                "reasoning_steps": reasoning_steps
            }
        ]

        return {
            "conversation": messages,
            "reasoning_chain": reasoning_steps,
            "metadata": {
                "category": "tot_reasoning_therapeutic",
                "tier": 3,
                "source_dataset": "tot_reasoning_problem_solving_v2",
                "reasoning_type": "tree_of_thought",
                "expected_quality": 0.8,
                "description": "Tree-of-thought reasoning for complex therapeutic problem solving",
                "processing_timestamp": datetime.now().isoformat(),
                "original_item_id": index,
                "original_question": problem,
                "original_answer": solution,
                "conversation_length": len(messages),
                "reasoning_chain_length": len(reasoning_steps),
                "reasoning_complexity": assess_reasoning_complexity(reasoning_steps),
                "problem_type": classify_problem_type(therapeutic_problem),
                "solution_approach": classify_solution_approach(therapeutic_solution),
                "difficulty": item.get("metadata", {}).get("difficulty", 1),
                "topic": item.get("metadata", {}).get("topic", "general")
            }
        }

    except Exception:
        return None

def adapt_to_therapeutic_context(question: str) -> str:
    """Adapt a general question to therapeutic context."""
    # Simple adaptation - add therapeutic framing
    if "how" in question.lower() and "solve" in question.lower():
        return f"I'm struggling with a complex problem and need help thinking through it step by step. {question}"
    if "what" in question.lower():
        return f"I'm trying to understand something better for my personal growth. {question}"
    return f"I'm working on understanding this concept for my therapeutic journey: {question}"

def adapt_solution_to_therapeutic_context(answer: str, reasoning_steps: list[str]) -> str:
    """Adapt a solution to therapeutic context."""
    therapeutic_intro = "Let me help you work through this step by step, using a structured approach to problem-solving. "

    if reasoning_steps:
        therapeutic_process = "Here's how we can break this down systematically: "
        return therapeutic_intro + therapeutic_process + answer
    return therapeutic_intro + answer

def parse_reasoning_steps(reasoning_text: str) -> list[str]:
    """Parse reasoning text into discrete steps."""
    # Common patterns for step separation
    step_indicators = ["Step 1:", "Step 2:", "Step 3:", "First,", "Second,", "Third,", "Then,", "Finally,", "Therefore,"]

    steps = []
    current_step = ""

    for line in reasoning_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check if this line starts a new step
        is_new_step = any(line.startswith(indicator) for indicator in step_indicators)

        if is_new_step and current_step:
            steps.append(current_step.strip())
            current_step = line
        else:
            current_step += " " + line if current_step else line

    # Add the last step
    if current_step:
        steps.append(current_step.strip())

    return steps

def assess_reasoning_complexity(reasoning_steps: list[str]) -> float:
    """Assess the complexity of reasoning based on steps."""
    if not reasoning_steps:
        return 0.0

    # Base complexity on number of steps and content depth
    step_count = len(reasoning_steps)
    avg_step_length = sum(len(step) for step in reasoning_steps) / step_count

    # Complexity indicators
    complexity_keywords = [
        "consider", "analyze", "evaluate", "compare", "contrast", "synthesize",
        "integrate", "differentiate", "prioritize", "weigh", "balance"
    ]

    complexity_score = 0.0
    for step in reasoning_steps:
        step_lower = step.lower()
        keyword_count = sum(1 for keyword in complexity_keywords if keyword in step_lower)
        complexity_score += keyword_count / len(complexity_keywords)

    # Normalize and combine factors
    step_factor = min(step_count / 5.0, 1.0)  # Max at 5 steps
    length_factor = min(avg_step_length / 100.0, 1.0)  # Max at 100 chars per step
    keyword_factor = complexity_score / step_count if step_count > 0 else 0.0

    return (step_factor + length_factor + keyword_factor) / 3.0

def classify_problem_type(problem: str) -> str:
    """Classify the type of therapeutic problem."""
    problem_lower = problem.lower()

    if any(word in problem_lower for word in ["anxiety", "worry", "panic", "fear"]):
        return "anxiety_related"
    if any(word in problem_lower for word in ["depression", "sad", "hopeless", "mood"]):
        return "depression_related"
    if any(word in problem_lower for word in ["relationship", "partner", "family", "conflict"]):
        return "relationship_issues"
    if any(word in problem_lower for word in ["trauma", "ptsd", "abuse", "flashback"]):
        return "trauma_related"
    if any(word in problem_lower for word in ["addiction", "substance", "alcohol", "drug"]):
        return "addiction_related"
    if any(word in problem_lower for word in ["work", "career", "job", "stress"]):
        return "work_related"
    if any(word in problem_lower for word in ["identity", "self", "confidence", "esteem"]):
        return "identity_related"
    return "general_therapeutic"

def classify_solution_approach(solution: str) -> str:
    """Classify the therapeutic approach used in the solution."""
    solution_lower = solution.lower()

    if any(word in solution_lower for word in ["cognitive", "thought", "thinking", "belief"]):
        return "cognitive_behavioral"
    if any(word in solution_lower for word in ["mindfulness", "meditation", "present", "awareness"]):
        return "mindfulness_based"
    if any(word in solution_lower for word in ["emotion", "feeling", "emotional", "express"]):
        return "emotion_focused"
    if any(word in solution_lower for word in ["behavior", "action", "activity", "practice"]):
        return "behavioral"
    if any(word in solution_lower for word in ["family", "system", "relationship", "interpersonal"]):
        return "systemic"
    if any(word in solution_lower for word in ["solution", "goal", "outcome", "result"]):
        return "solution_focused"
    return "integrative"

def assess_tot_quality(conversation: dict[str, Any]) -> bool:
    """Assess quality of ToT reasoning conversation."""
    try:
        messages = conversation.get("conversation", [])
        reasoning_chain = conversation.get("reasoning_chain", [])

        # Must have exactly 2 messages (client problem, therapist solution)
        if len(messages) != 2:
            return False

        # Must have reasoning chain
        if not reasoning_chain or len(reasoning_chain) < 2:
            return False

        # Check content quality
        for msg in messages:
            content = msg.get("content", "")

            # Minimum content length
            if len(content.strip()) < 50:
                return False

            # Check for therapeutic relevance in therapist response
            if msg["role"] == "therapist":
                therapeutic_indicators = [
                    "therapy", "therapeutic", "treatment", "intervention", "approach",
                    "strategy", "technique", "method", "process", "support", "help",
                    "healing", "recovery", "coping", "management", "assessment"
                ]
                if not any(indicator in content.lower() for indicator in therapeutic_indicators):
                    return False

        return True

    except Exception:
        return False

def generate_tot_report(config: dict, conversations: list, stats: dict) -> dict[str, Any]:
    """Generate comprehensive ToT reasoning processing report."""

    # Analyze reasoning patterns
    reasoning_approaches = {}
    problem_types = {}
    complexity_scores = []

    for conv in conversations:
        metadata = conv.get("metadata", {})

        approach = metadata.get("solution_approach", "unknown")
        reasoning_approaches[approach] = reasoning_approaches.get(approach, 0) + 1

        problem_type = metadata.get("problem_type", "unknown")
        problem_types[problem_type] = problem_types.get(problem_type, 0) + 1

        complexity = metadata.get("reasoning_complexity", 0.0)
        complexity_scores.append(complexity)

    avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.0

    return {
        "task": "5.25: ToT Reasoning Problem Solving Dataset V2 Processing",
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
        "reasoning_analysis": {
            "average_complexity_score": avg_complexity,
            "reasoning_approaches": reasoning_approaches,
            "problem_type_distribution": problem_types,
            "complexity_distribution": stats["reasoning_complexity_distribution"]
        },
        "conversation_analysis": {
            "avg_reasoning_steps": sum(len(conv.get("reasoning_chain", [])) for conv in conversations) / len(conversations) if conversations else 0,
            "total_reasoning_steps": sum(len(conv.get("reasoning_chain", [])) for conv in conversations),
            "conversation_length_stats": {
                "min": min(len(conv.get("conversation", [])) for conv in conversations) if conversations else 0,
                "max": max(len(conv.get("conversation", [])) for conv in conversations) if conversations else 0,
                "average": sum(len(conv.get("conversation", [])) for conv in conversations) / len(conversations) if conversations else 0
            }
        },
        "dataset_info": {
            "tier": config["tier"],
            "expected_quality": config["expected_quality"],
            "description": config["description"],
            "source_file": config["json_path"]
        }
    }

def create_error_report(config: dict, error_message: str) -> dict[str, Any]:
    """Create error report for failed processing."""
    return {
        "task": "5.25: ToT Reasoning Problem Solving Dataset V2 Processing",
        "status": "FAILED",
        "error": error_message,
        "dataset_config": config,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    process_tot_reasoning()
