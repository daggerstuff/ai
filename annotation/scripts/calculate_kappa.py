import argparse
import json
from pathlib import Path

try:
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def calculate_kappa(results_dir):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Directory {results_dir} does not exist.")
        return

    print(f"Scanning {results_path} for annotated batches...")

    # Check for both result files and batch files in case they are mixed up
    files = list(results_path.glob("*.jsonl"))
    if not files:
        print("No results found.")
        return

    # Organize annotations by task_id
    # Structure: task_annotations[task_id] = {annotator_id: annotation_data}
    task_annotations = {}
    annotator_ids = set()

    for file_path in files:
        try:
            with open(file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        task_id = entry.get("task_id")
                        # Use provided annotator_id or fall back to filename
                        annotator_id = entry.get("annotator_id") or file_path.stem
                        anns = entry.get("annotations")

                        if task_id and anns:
                            if task_id not in task_annotations:
                                task_annotations[task_id] = {}

                            task_annotations[task_id][annotator_id] = anns
                            annotator_ids.add(annotator_id)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # We need at least 2 distinct annotators to calculate Kappa
    if len(annotator_ids) < 2:
        print(
            f"\nFound {len(annotator_ids)} annotators. "
            "Need at least 2 for Kappa calculation."
        )
        print(f"Annotators: {list(annotator_ids)}")
        return

    print(f"\nAnnotators found: {', '.join(sorted(annotator_ids))}")

    # Prepare paired data for comparison
    # We'll take the first two sorted annotators for pairwise comparison
    sorted_annotators = sorted(list(annotator_ids))
    ann1_id = sorted_annotators[0]
    ann2_id = sorted_annotators[1]

    # Crisis Score Arrays
    crisis_y1 = []
    crisis_y2 = []

    # Emotion Arrays
    emotion_y1 = []
    emotion_y2 = []

    common_tasks = 0

    for task_id, anns_by_annotator in task_annotations.items():
        if ann1_id in anns_by_annotator and ann2_id in anns_by_annotator:
            common_tasks += 1

            # Crisis Labels (Ordinal)
            c1 = anns_by_annotator[ann1_id].get("crisis_label")
            c2 = anns_by_annotator[ann2_id].get("crisis_label")

            if c1 is not None and c2 is not None:
                crisis_y1.append(int(c1))
                crisis_y2.append(int(c2))

            # Emotion Labels (Categorical)
            e1 = anns_by_annotator[ann1_id].get("primary_emotion")
            e2 = anns_by_annotator[ann2_id].get("primary_emotion")

            if e1 and e2:
                emotion_y1.append(e1)
                emotion_y2.append(e2)

    print("\nStats:")
    print(f"  Total unique tasks: {len(task_annotations)}")
    print(f"  Common tasks annotated by {ann1_id} and {ann2_id}: {common_tasks}")

    if SKLEARN_AVAILABLE:
        # Calculate Crisis Kappa (Ordinal - Quadratic Weights)
        if len(crisis_y1) > 0:
            crisis_kappa = cohen_kappa_score(crisis_y1, crisis_y2, weights="quadratic")
            crisis_acc = accuracy_score(crisis_y1, crisis_y2)
            print("\nMetric: Crisis Label Agreement")
            print(f"  - Accuracy: {crisis_acc:.2%}")
            print(f"  - Cohen's Kappa (Quadratic): {crisis_kappa:.4f}")
            if crisis_kappa > 0.85:
                print("  -> ✅ PASSED (> 0.85)")
            else:
                print("  -> ⚠️  BELOW TARGET (< 0.85)")

        # Calculate Emotion Kappa (Categorical - No Weights)
        if len(emotion_y1) > 0:
            emotion_kappa = cohen_kappa_score(emotion_y1, emotion_y2)
            emotion_acc = accuracy_score(emotion_y1, emotion_y2)
            print("\nMetric: Primary Emotion Agreement")
            print(f"  - Accuracy: {emotion_acc:.2%}")
            print(f"  - Cohen's Kappa (Unweighted): {emotion_kappa:.4f}")
    else:
        print(
            "  ❌ sklearn not installed. "
            "Please install scikit-learn to calculate Kappa."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Inter-Annotator Agreement (Kappa)"
    )
    parser.add_argument(
        "--results",
        default="../results",
        help="Directory containing annotated JSONL files",
    )
    args = parser.parse_args()

    calculate_kappa(args.results)
