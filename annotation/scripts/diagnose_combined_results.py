"""
Diagnostic script for combined multi-agent annotation results
"""

import json
from pathlib import Path


def analyze_combined_results(results_file: str):
    results_path = Path(results_file)
    if not results_path.exists():
        return

    disagreements = {"crisis_label": [], "primary_emotion": []}

    total_items = 0

    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue

            total_items += 1
            entry = json.loads(line)
            task_id = entry.get("task_id")
            individual = entry.get("individual_annotations", [])

            if len(individual) < 2:
                continue

            dr_a, dr_b = _resolve_agent_pair(individual)

            if not dr_a or not dr_b:
                continue

            # Crisis Disagreement
            a_crisis = dr_a.get("crisis_label")
            b_crisis = dr_b.get("crisis_label")
            if a_crisis != b_crisis:
                disagreements["crisis_label"].append({"task_id": task_id, "dr_a": a_crisis, "dr_b": b_crisis})

            # Emotion Disagreement
            a_emotion = dr_a.get("primary_emotion")
            b_emotion = dr_b.get("primary_emotion")
            if a_emotion != b_emotion:
                disagreements["primary_emotion"].append({"task_id": task_id, "dr_a": a_emotion, "dr_b": b_emotion})

    for _category, items in disagreements.items():
        count = len(items)
        (count / total_items * 100) if total_items > 0 else 0

        if count > 0:
            for _item in items[:5]:
                pass


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "ai/annotation/results/batch_001_annotated.jsonl"
    analyze_combined_results(input_file)


def _resolve_agent_pair(individual: list[dict]):
    # Identify agents by their known IDs or roles
    # Fallback to positional index only if IDs are not found
    dr_a = next(
        (a for a in individual if "dr_a" in a.get("agent_id", "") or "crisis_expert" in a.get("role", "")),
        None,
    )
    dr_b = next(
        (a for a in individual if "dr_b" in a.get("agent_id", "") or "emotion_analyst" in a.get("role", "")),
        None,
    )

    # If matching failed, use positional fallback only as last resort
    if not dr_a and individual:
        # Ensure we don't pick the same agent if dr_b is already confirmed
        if candidates := [a for a in individual if a != dr_b]:
            dr_a = candidates[0]

    if not dr_b and individual:
        # Ensure we don't pick the same agent if dr_a is already confirmed
        if candidates := [a for a in individual if a != dr_a]:
            dr_b = candidates[0]

    return dr_a, dr_b
