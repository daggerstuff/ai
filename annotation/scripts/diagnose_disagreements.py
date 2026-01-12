"""
Diagnostic script to identify disagreements between Dr. A and Dr. B annotations
"""

import json
from pathlib import Path


def load_annotations(filepath):
    """Load annotations from JSONL file"""
    annotations = {}
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            task_id = entry.get("task_id")
            if task_id:
                annotations[task_id] = entry.get("annotations", {})
    return annotations


def compare_annotations(dr_a_file, dr_b_file):
    """Compare annotations and identify disagreements"""
    dr_a = load_annotations(dr_a_file)
    dr_b = load_annotations(dr_b_file)

    # Find common tasks
    common_tasks = set(dr_a.keys()) & set(dr_b.keys())
    print(f"📊 Total common tasks: {len(common_tasks)}\n")

    # Track disagreements
    disagreements = {
        "crisis_label": [],
        "primary_emotion": [],
        "emotion_intensity": [],
        "empathy_score": [],
    }

    agreements = {
        "crisis_label": 0,
        "primary_emotion": 0,
        "emotion_intensity": 0,
        "empathy_score": 0,
    }

    for task_id in sorted(common_tasks):
        a_ann = dr_a[task_id]
        b_ann = dr_b[task_id]

        # Crisis label
        if a_ann.get("crisis_label") == b_ann.get("crisis_label"):
            agreements["crisis_label"] += 1
        else:
            disagreements["crisis_label"].append(
                {
                    "task_id": task_id,
                    "dr_a": a_ann.get("crisis_label"),
                    "dr_b": b_ann.get("crisis_label"),
                }
            )

        # Primary emotion
        if a_ann.get("primary_emotion") == b_ann.get("primary_emotion"):
            agreements["primary_emotion"] += 1
        else:
            disagreements["primary_emotion"].append(
                {
                    "task_id": task_id,
                    "dr_a": a_ann.get("primary_emotion"),
                    "dr_b": b_ann.get("primary_emotion"),
                }
            )

        # Emotion intensity (allow ±1 tolerance)
        a_int = a_ann.get("emotion_intensity", 0)
        b_int = b_ann.get("emotion_intensity", 0)
        if abs(a_int - b_int) <= 1:
            agreements["emotion_intensity"] += 1
        else:
            disagreements["emotion_intensity"].append(
                {
                    "task_id": task_id,
                    "dr_a": a_int,
                    "dr_b": b_int,
                    "diff": abs(a_int - b_int),
                }
            )

        # Empathy score (allow ±1 tolerance)
        a_emp = a_ann.get("empathy_score", 0)
        b_emp = b_ann.get("empathy_score", 0)
        if abs(a_emp - b_emp) <= 1:
            agreements["empathy_score"] += 1
        else:
            disagreements["empathy_score"].append(
                {
                    "task_id": task_id,
                    "dr_a": a_emp,
                    "dr_b": b_emp,
                    "diff": abs(a_emp - b_emp),
                }
            )

    # Print summary
    print("=" * 80)
    print("AGREEMENT SUMMARY")
    print("=" * 80)

    for field in [
        "crisis_label",
        "primary_emotion",
        "emotion_intensity",
        "empathy_score",
    ]:
        total = len(common_tasks)
        agree_count = agreements[field]
        disagree_count = len(disagreements[field])
        agree_pct = (agree_count / total * 100) if total > 0 else 0

        print(f"\n{field.upper().replace('_', ' ')}:")
        print(f"  ✅ Agreements: {agree_count}/{total} ({agree_pct:.1f}%)")
        print(f"  ❌ Disagreements: {disagree_count}/{total} ({100 - agree_pct:.1f}%)")

        if disagree_count > 0 and disagree_count <= 10:
            print("\n  Sample disagreements:")
            for item in disagreements[field][:5]:
                print(
                    f"    • {item['task_id']}: Dr.A={item['dr_a']}, Dr.B={item['dr_b']}"
                )

    print("\n" + "=" * 80)

    # Save detailed report
    report = {
        "summary": {
            "total_tasks": len(common_tasks),
            "agreements": {k: v for k, v in agreements.items()},
            "disagreement_counts": {k: len(v) for k, v in disagreements.items()},
        },
        "disagreements": disagreements,
    }

    output_path = Path("ai/annotation/results/disagreement_analysis.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Detailed report saved to: {output_path}")


if __name__ == "__main__":
    dr_a_file = "ai/annotation/results/dr_a_real_enhanced.jsonl"
    dr_b_file = "ai/annotation/results/dr_b_real_enhanced.jsonl"
    compare_annotations(dr_a_file, dr_b_file)
