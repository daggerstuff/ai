import argparse
import json
from pathlib import Path

try:
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def parse_file_pairs(
    explicit_a: str | None, explicit_b: str | None, results_path: Path
) -> tuple[list[Path], list[str]]:
    """
    Resolve annotation file inputs.

    If explicit file paths are provided, use only those.
    Otherwise, require exactly two files in the results directory.
    """
    if explicit_a and explicit_b:
        a_path = Path(explicit_a)
        b_path = Path(explicit_b)
        if not a_path.exists():
            raise FileNotFoundError(f"Annotator A file not found: {a_path}")
        if not b_path.exists():
            raise FileNotFoundError(f"Annotator B file not found: {b_path}")
        return [a_path, b_path], [a_path.stem, b_path.stem]

    if explicit_a or explicit_b:
        raise ValueError("Both --annotator-a-file and --annotator-b-file must be provided together.")

    files = sorted(results_path.glob("*.jsonl"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .jsonl files found in {results_path}")
    if len(files) != 2:
        raise ValueError(
            "Directory mode requires exactly 2 JSONL files. "
            "To prevent batch contamination, pass --annotator-a-file and "
            "--annotator-b-file explicitly, or ensure the directory contains "
            "only the two intended annotation files."
        )
    return files, [files[0].stem, files[1].stem]


def load_task_annotations(file_paths: list[Path]) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """Load annotations grouped by task ID and annotator."""
    task_annotations: dict[str, dict[str, dict]] = {}
    annotator_ids: set[str] = set()

    for file_path in file_paths:
        annotator_id = file_path.stem
        with open(file_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                task_id = entry.get("task_id")
                anns = entry.get("annotations")
                if not task_id or not isinstance(anns, dict):
                    continue
                file_annotation_id = entry.get("annotator_id")
                annotator_key = str(file_annotation_id) if file_annotation_id else annotator_id
                task_annotations.setdefault(task_id, {})[annotator_key] = anns
                annotator_ids.add(annotator_key)

    return task_annotations, sorted(annotator_ids)


def calculate_kappa(
    results_dir: str,
    annotator_a_file: str | None = None,
    annotator_b_file: str | None = None,
):
    """Calculate inter-annotator agreement metrics for crisis and primary emotion."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    if not SKLEARN_AVAILABLE:
        raise RuntimeError(
            "scikit-learn is required for kappa and accuracy metrics. Install it in the active environment."
        )

    files, fallback_annotators = parse_file_pairs(
        explicit_a=annotator_a_file,
        explicit_b=annotator_b_file,
        results_path=results_path,
    )
    task_annotations, detected_annotators = load_task_annotations(files)

    if len(detected_annotators) < 2:
        raise ValueError("Need at least 2 distinct annotators to compute agreement.")

    ann1_id = fallback_annotators[0]
    ann2_id = fallback_annotators[1]
    if ann1_id not in detected_annotators or ann2_id not in detected_annotators:
        ann1_id, ann2_id = detected_annotators[0], detected_annotators[1]

    crisis_y1: list[int] = []
    crisis_y2: list[int] = []
    emotion_y1: list[str] = []
    emotion_y2: list[str] = []
    common_tasks = 0

    for _task_id, anns_by_annotator in task_annotations.items():
        if ann1_id in anns_by_annotator and ann2_id in anns_by_annotator:
            common_tasks += 1

            c1 = anns_by_annotator[ann1_id].get("crisis_label")
            c2 = anns_by_annotator[ann2_id].get("crisis_label")
            if c1 is not None and c2 is not None:
                crisis_y1.append(int(c1))
                crisis_y2.append(int(c2))

            e1 = anns_by_annotator[ann1_id].get("primary_emotion")
            e2 = anns_by_annotator[ann2_id].get("primary_emotion")
            if e1 and e2:
                emotion_y1.append(str(e1))
                emotion_y2.append(str(e2))

    if not common_tasks:
        raise ValueError("No overlapping tasks found for the selected annotator pair.")


    if crisis_y1:
        cohen_kappa_score(crisis_y1, crisis_y2, weights="quadratic")
        accuracy_score(crisis_y1, crisis_y2)

    if emotion_y1:
        cohen_kappa_score(emotion_y1, emotion_y2)
        accuracy_score(emotion_y1, emotion_y2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Inter-Annotator Agreement (Kappa)")
    parser.add_argument(
        "--results",
        default="../results",
        help="Directory containing annotated JSONL files",
    )
    parser.add_argument(
        "--annotator-a-file",
        help="Explicit path to annotator A JSONL output file",
    )
    parser.add_argument(
        "--annotator-b-file",
        help="Explicit path to annotator B JSONL output file",
    )
    args = parser.parse_args()

    calculate_kappa(args.results, args.annotator_a_file, args.annotator_b_file)
