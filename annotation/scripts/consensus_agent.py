"""
Consensus Agent - Resolves disagreements between primary annotators

Based on NVIDIA Ambient Healthcare Agents multi-agent orchestration pattern.
Takes annotations from Dr. A and Dr. B and produces consensus labels.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

MAX_SECONDARY_EMOTIONS = 2
PREFERENCE_ORDER = [
    "Fear",
    "Sadness",
    "Anger",
    "Disgust",
    "Surprise",
    "Anticipation",
    "Joy",
    "Trust",
    "Calm",
    "Neutral",
]


def normalize_secondary_emotions(raw_values: Any, primary_emotion: str | None = None) -> list[str]:
    """Normalize secondary emotion annotations into a deterministic ordered list."""
    if raw_values is None:
        return []

    if isinstance(raw_values, str):
        values = [raw_values]
    elif isinstance(raw_values, (list, tuple, set)):
        values = list(raw_values)
    else:
        return []

    ordered = []
    seen = set()
    for raw in values:
        if not isinstance(raw, str):
            continue
        for part in raw.replace(";", ",").split(","):
            candidate = part.strip()
            if not candidate or candidate in seen:
                continue
            if primary_emotion and candidate == primary_emotion:
                continue
            seen.add(candidate)
            ordered.append(candidate)
            if len(ordered) >= MAX_SECONDARY_EMOTIONS:
                return ordered

    return ordered


class ConsensusAgent:
    """
    Consensus agent that resolves disagreements between annotators

    Strategies:
    - Majority voting for categorical labels
    - Averaging for numerical scores
    - Confidence-weighted decisions
    """

    def __init__(self, strategy: str = "weighted"):
        """
        Initialize consensus agent

        Args:
            strategy: "majority", "average", or "weighted"
        """
        self.strategy = strategy
        self.consensus_count = 0
        self.agreement_stats = {
            "crisis_label": [],
            "primary_emotion": [],
            "total_tasks": 0,
            "emotion_tie_breaks": 0,
            "secondary_emotion": [],
        }

    def resolve_crisis_label(self, annotations: list[dict[str, Any]]) -> dict[str, Any]:
        """Resolve crisis label using confidence-weighted voting"""
        if self.strategy == "weighted":
            # Weight by confidence score
            weighted_sum = sum(a["crisis_label"] * a["crisis_confidence"] for a in annotations)
            total_weight = sum(a["crisis_confidence"] for a in annotations)
            consensus_label = round(weighted_sum / total_weight)
            consensus_confidence = round(total_weight / len(annotations))
        else:
            # Simple majority
            labels = [a["crisis_label"] for a in annotations]
            consensus_label = Counter(labels).most_common(1)[0][0]
            consensus_confidence = round(mean([a["crisis_confidence"] for a in annotations]))

        return {
            "crisis_label": consensus_label,
            "crisis_confidence": consensus_confidence,
        }

    def resolve_emotion(self, annotations: list[dict[str, Any]]) -> dict[str, Any]:
        """Resolve primary emotion with explicit tie-break strategy"""
        emotions = [a["primary_emotion"] for a in annotations]
        emotion_counts = Counter(emotions)
        max_count = max(emotion_counts.values())
        top_emotions = [e for e, c in emotion_counts.items() if c == max_count]
        tie_resolved = False

        if len(top_emotions) == 1:
            consensus_emotion = top_emotions[0]
            agreed_intensities = [
                a.get("emotion_intensity", 0) for a in annotations if a.get("primary_emotion") == consensus_emotion
            ]
            consensus_intensity = round(mean(agreed_intensities)) if agreed_intensities else 0
        else:
            tie_resolved = True
            preference_order = [
                "Fear",
                "Sadness",
                "Anger",
                "Disgust",
                "Surprise",
                "Anticipation",
                "Joy",
                "Trust",
                "Calm",
                "Neutral",
            ]
            preference_rank = {emotion: index for index, emotion in enumerate(preference_order)}
            ranked = []
            for emotion in top_emotions:
                supporting = [a for a in annotations if a.get("primary_emotion") == emotion]
                confidence_scores = [float(a.get("confidence_scores", {}).get("emotion", 0.0)) for a in supporting]
                confidence = mean(confidence_scores) if confidence_scores else 0.0
                intensity = mean(a.get("emotion_intensity", 0) for a in supporting)
                rank = preference_rank.get(emotion, len(preference_order))
                ranked.append((emotion, confidence, intensity, -rank))

            ranked.sort(key=lambda item: item[1:], reverse=True)
            consensus_emotion = ranked[0][0]
            consensus_intensity = round(mean(a.get("emotion_intensity", 0) for a in annotations))

        return {
            "primary_emotion": consensus_emotion,
            "emotion_intensity": consensus_intensity,
            "emotion_tie_resolved": tie_resolved,
        }

    def resolve_secondary_emotions(
        self,
        annotations: list[dict[str, Any]],
        primary_emotion: str,
    ) -> list[str]:
        """Resolve secondary emotions from supporting annotations."""
        counts = Counter()
        for annotation in annotations:
            for emotion in normalize_secondary_emotions(
                raw_values=annotation.get("secondary_emotions"),
                primary_emotion=primary_emotion,
            ):
                counts[emotion] += 1

        if not counts:
            return []

        preference_rank = {emotion: index for index, emotion in enumerate(PREFERENCE_ORDER)}
        scored: list[tuple[str, tuple[int, float, float, int]]] = []
        for emotion, count in counts.items():
            if emotion == primary_emotion:
                continue
            supporting = [
                ann
                for ann in annotations
                if emotion
                in normalize_secondary_emotions(ann.get("secondary_emotions"), primary_emotion=primary_emotion)
            ]
            confidence_scores = [float(ann.get("confidence_scores", {}).get("emotion", 0.0)) for ann in supporting]
            confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            intensity = (
                sum(float(ann.get("emotion_intensity", 0)) for ann in supporting) / len(supporting)
                if supporting
                else 0.0
            )
            rank = preference_rank.get(emotion, len(PREFERENCE_ORDER))
            scored.append((emotion, (count, confidence, intensity, -rank)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [emotion for emotion, _ in scored[:MAX_SECONDARY_EMOTIONS]]

    def resolve_valence_arousal(self, annotations: list[dict[str, Any]]) -> dict[str, float]:
        """Resolve valence and arousal by averaging"""
        return {
            "valence": round(mean([a["valence"] for a in annotations]), 2),
            "arousal": round(mean([a["arousal"] for a in annotations]), 2),
        }

    def resolve_empathy_safety(self, annotations: list[dict[str, Any]]) -> dict[str, Any]:
        """Resolve empathy score and safety pass"""
        empathy_scores = [a["empathy_score"] for a in annotations if a["empathy_score"] is not None]
        safety_passes = [a["safety_pass"] for a in annotations if a["safety_pass"] is not None]

        return {
            "empathy_score": round(mean(empathy_scores)) if empathy_scores else None,
            "safety_pass": all(safety_passes) if safety_passes else None,
        }

    def merge_notes(self, annotations: list[dict[str, Any]], annotator_ids: list[str]) -> str:
        """Merge clinical notes from all annotators"""
        merged = "CONSENSUS ANNOTATION\n\n"

        for _i, (ann, annotator_id) in enumerate(zip(annotations, annotator_ids, strict=False)):
            merged += f"[{annotator_id.upper()}]: {ann['notes']}\n\n"

        return merged.strip()

    def create_consensus(
        self, task_id: str, annotations: list[dict[str, Any]], annotator_ids: list[str]
    ) -> dict[str, Any]:
        """
        Create consensus annotation from multiple annotators

        Args:
            task_id: Task identifier
            annotations: List of annotation dicts
            annotator_ids: List of annotator IDs

        Returns:
            Consensus annotation dict
        """
        if len(annotations) < 2:
            raise ValueError("Need at least 2 annotations for consensus")

        # Track agreement
        crisis_labels = [a["crisis_label"] for a in annotations]
        emotions = [a["primary_emotion"] for a in annotations]

        crisis_agreement = len(set(crisis_labels)) == 1
        emotion_agreement = len(set(emotions)) == 1
        emotion_result = self.resolve_emotion(annotations)
        secondary_emotions = self.resolve_secondary_emotions(
            annotations=annotations,
            primary_emotion=emotion_result["primary_emotion"],
        )
        secondary_overlap = any(
            set(secondary_emotions)
            & set(
                normalize_secondary_emotions(
                    annotation.get("secondary_emotions"),
                    primary_emotion=emotion_result["primary_emotion"],
                )
            )
            for annotation in annotations
        )

        self.agreement_stats["crisis_label"].append(crisis_agreement)
        self.agreement_stats["primary_emotion"].append(emotion_agreement)
        self.agreement_stats["secondary_emotion"].append(secondary_overlap)
        self.agreement_stats["total_tasks"] += 1
        if emotion_result.get("emotion_tie_resolved"):
            self.agreement_stats["emotion_tie_breaks"] += 1

        # Resolve each component
        crisis_result = self.resolve_crisis_label(annotations)
        valence_arousal = self.resolve_valence_arousal(annotations)
        empathy_safety = self.resolve_empathy_safety(annotations)

        # Create consensus annotation
        consensus = {
            **crisis_result,
            **{k: v for k, v in emotion_result.items() if k != "emotion_tie_resolved"},
            "secondary_emotions": secondary_emotions,
            **valence_arousal,
            **empathy_safety,
            "notes": self.merge_notes(annotations, annotator_ids),
        }

        if emotion_result.get("emotion_tie_resolved"):
            consensus["notes"] = f"{consensus['notes']} | Emotion tie resolved via confidence/intensity tie-break"

        self.consensus_count += 1

        return {
            "task_id": task_id,
            "annotator_id": "consensus",
            "annotations": consensus,
            "metadata": {
                "strategy": self.strategy,
                "num_annotators": len(annotations),
                "annotators": annotator_ids,
                "crisis_agreement": crisis_agreement,
                "emotion_agreement": emotion_agreement,
                "secondary_emotions": secondary_emotions,
                "emotion_tie_resolved": emotion_result.get("emotion_tie_resolved", False),
            },
        }

    def get_agreement_report(self) -> dict[str, Any]:
        """Generate agreement statistics report"""
        if self.agreement_stats["total_tasks"] == 0:
            return {"error": "No tasks processed"}

        crisis_agreement_rate = sum(self.agreement_stats["crisis_label"]) / self.agreement_stats["total_tasks"]
        emotion_agreement_rate = sum(self.agreement_stats["primary_emotion"]) / self.agreement_stats["total_tasks"]
        secondary_emotion_agreement_rate = (
            sum(self.agreement_stats["secondary_emotion"]) / self.agreement_stats["total_tasks"]
        )

        return {
            "total_consensus_annotations": self.consensus_count,
            "emotion_tie_breaks": self.agreement_stats["emotion_tie_breaks"],
            "crisis_agreement_rate": round(crisis_agreement_rate, 3),
            "emotion_agreement_rate": round(emotion_agreement_rate, 3),
            "secondary_emotion_agreement_rate": round(secondary_emotion_agreement_rate, 3),
            "overall_agreement_rate": round((crisis_agreement_rate + emotion_agreement_rate) / 2, 3),
        }


def load_annotations(file_path: Path) -> dict[str, dict[str, Any]]:
    """Load annotations from JSONL file"""
    annotations = {}

    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            task_id = entry.get("task_id")
            if task_id:
                annotations[task_id] = entry

    return annotations


def create_consensus_annotations(file1: str, file2: str, output_file: str, strategy: str = "weighted"):
    """
    Create consensus annotations from two annotator files

    Args:
        file1: Path to first annotator's results
        file2: Path to second annotator's results
        output_file: Path to save consensus results
        strategy: Consensus strategy ("weighted", "majority", "average")
    """

    # Load annotations
    ann1 = load_annotations(Path(file1))
    ann2 = load_annotations(Path(file2))

    # Find common tasks
    common_tasks = set(ann1.keys()) & set(ann2.keys())

    if not common_tasks:
        return

    # Create consensus agent
    agent = ConsensusAgent(strategy=strategy)

    # Process each task
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f_out:
        for task_id in sorted(common_tasks):
            entry1 = ann1[task_id]
            entry2 = ann2[task_id]

            consensus = agent.create_consensus(
                task_id=task_id,
                annotations=[entry1["annotations"], entry2["annotations"]],
                annotator_ids=[entry1["annotator_id"], entry2["annotator_id"]],
            )

            f_out.write(json.dumps(consensus) + "\n")

    # Print agreement report
    report = agent.get_agreement_report()

    # Save report
    report_path = output_path.parent / f"{output_path.stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consensus Agent - Resolve disagreements between annotators")
    parser.add_argument("--file1", required=True, help="First annotator's results (JSONL)")
    parser.add_argument("--file2", required=True, help="Second annotator's results (JSONL)")
    parser.add_argument("--output", required=True, help="Output file for consensus annotations (JSONL)")
    parser.add_argument(
        "--strategy",
        choices=["weighted", "majority", "average"],
        default="weighted",
        help="Consensus strategy (default: weighted)",
    )

    args = parser.parse_args()

    create_consensus_annotations(
        file1=args.file1,
        file2=args.file2,
        output_file=args.output,
        strategy=args.strategy,
    )
