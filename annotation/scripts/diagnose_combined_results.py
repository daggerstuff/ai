"""
Diagnostic script for combined multi-agent annotation results
"""

import json
from pathlib import Path


def analyze_combined_results(results_file: str):
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"Error: {results_file} not found")
        return

    disagreements = {"crisis_label": [], "primary_emotion": []}

    total_items = 0

    with open(results_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            total_items += 1
            entry = json.loads(line)
            task_id = entry.get("task_id")
            individual = entry.get("individual_annotations", [])

            if len(individual) < 2:
                continue

            # Identify agents by their known IDs or roles
            # Fallback to positional index only if IDs are not found
            dr_a = next(
                (
                    a
                    for a in individual
                    if "dr_a" in a.get("agent_id", "") or "crisis_expert" in a.get("role", "")
                ),
                None,
            )
            dr_b = next(
                (
                    a
                    for a in individual
                    if "dr_b" in a.get("agent_id", "") or "emotion_analyst" in a.get("role", "")
                ),
                None,
            )

            # If matching failed, use positional fallback only as last resort
            if not dr_a and len(individual) > 0:
                dr_a = individual[0]
            if not dr_b and len(individual) > 1:
                dr_b = individual[1]

            if not dr_a or not dr_b:
                continue

            # Crisis Disagreement
            a_crisis = dr_a.get("crisis_label")
            b_crisis = dr_b.get("crisis_label")
            if a_crisis != b_crisis:
                disagreements["crisis_label"].append(
                    {"task_id": task_id, "dr_a": a_crisis, "dr_b": b_crisis}
                )

            # Emotion Disagreement
            a_emotion = dr_a.get("primary_emotion")
            b_emotion = dr_b.get("primary_emotion")
            if a_emotion != b_emotion:
                disagreements["primary_emotion"].append(
                    {"task_id": task_id, "dr_a": a_emotion, "dr_b": b_emotion}
                )

    print("=" * 60)
    print(f"DIAGNOSTIC REPORT: {results_file}")
    print(f"Total items analyzed: {total_items}")
    print("=" * 60)

    for category, items in disagreements.items():
        count = len(items)
        pct = (count / total_items * 100) if total_items > 0 else 0
        print(f"\nDisagreements in {category.replace('_', ' ').upper()}: {count} ({pct:.1f}%)")

        if count > 0:
            print("Sample disagreements:")
            for item in items[:5]:
                print(f"  • {item['task_id']}: Dr.A={item['dr_a']}, Dr.B={item['dr_b']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    input_file = (
        sys.argv[1] if len(sys.argv) > 1 else "ai/annotation/results/batch_001_annotated.jsonl"
    )
    analyze_combined_results(input_file)
