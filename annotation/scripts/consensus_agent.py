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
from typing import Any, Dict, List


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
        }

    def resolve_crisis_label(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve crisis label using confidence-weighted voting"""
        if self.strategy == "weighted":
            # Weight by confidence score
            weighted_sum = sum(
                a["crisis_label"] * a["crisis_confidence"] for a in annotations
            )
            total_weight = sum(a["crisis_confidence"] for a in annotations)
            consensus_label = round(weighted_sum / total_weight)
            consensus_confidence = round(total_weight / len(annotations))
        else:
            # Simple majority
            labels = [a["crisis_label"] for a in annotations]
            consensus_label = Counter(labels).most_common(1)[0][0]
            consensus_confidence = round(
                mean([a["crisis_confidence"] for a in annotations])
            )

        return {
            "crisis_label": consensus_label,
            "crisis_confidence": consensus_confidence,
        }

    def resolve_emotion(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve primary emotion using majority voting"""
        emotions = [a["primary_emotion"] for a in annotations]
        intensities = [a["emotion_intensity"] for a in annotations]

        # Majority vote for emotion
        emotion_counts = Counter(emotions)
        consensus_emotion = emotion_counts.most_common(1)[0][0]

        # Average intensity for agreed emotion
        agreed_intensities = [
            annotations[i]["emotion_intensity"]
            for i, e in enumerate(emotions)
            if e == consensus_emotion
        ]
        consensus_intensity = (
            round(mean(agreed_intensities))
            if agreed_intensities
            else round(mean(intensities))
        )

        return {
            "primary_emotion": consensus_emotion,
            "emotion_intensity": consensus_intensity,
        }

    def resolve_valence_arousal(
        self, annotations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Resolve valence and arousal by averaging"""
        return {
            "valence": round(mean([a["valence"] for a in annotations]), 2),
            "arousal": round(mean([a["arousal"] for a in annotations]), 2),
        }

    def resolve_empathy_safety(
        self, annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Resolve empathy score and safety pass"""
        empathy_scores = [
            a["empathy_score"] for a in annotations if a["empathy_score"] is not None
        ]
        safety_passes = [
            a["safety_pass"] for a in annotations if a["safety_pass"] is not None
        ]

        return {
            "empathy_score": round(mean(empathy_scores)) if empathy_scores else None,
            "safety_pass": all(safety_passes) if safety_passes else None,
        }

    def merge_notes(
        self, annotations: List[Dict[str, Any]], annotator_ids: List[str]
    ) -> str:
        """Merge clinical notes from all annotators"""
        merged = "CONSENSUS ANNOTATION\n\n"

        for i, (ann, annotator_id) in enumerate(zip(annotations, annotator_ids)):
            merged += f"[{annotator_id.upper()}]: {ann['notes']}\n\n"

        return merged.strip()

    def create_consensus(
        self, task_id: str, annotations: List[Dict[str, Any]], annotator_ids: List[str]
    ) -> Dict[str, Any]:
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

        self.agreement_stats["crisis_label"].append(crisis_agreement)
        self.agreement_stats["primary_emotion"].append(emotion_agreement)
        self.agreement_stats["total_tasks"] += 1

        # Resolve each component
        crisis_result = self.resolve_crisis_label(annotations)
        emotion_result = self.resolve_emotion(annotations)
        valence_arousal = self.resolve_valence_arousal(annotations)
        empathy_safety = self.resolve_empathy_safety(annotations)

        # Create consensus annotation
        consensus = {
            **crisis_result,
            **emotion_result,
            **valence_arousal,
            **empathy_safety,
            "notes": self.merge_notes(annotations, annotator_ids),
        }

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
            },
        }

    def get_agreement_report(self) -> Dict[str, Any]:
        """Generate agreement statistics report"""
        if self.agreement_stats["total_tasks"] == 0:
            return {"error": "No tasks processed"}

        crisis_agreement_rate = (
            sum(self.agreement_stats["crisis_label"])
            / self.agreement_stats["total_tasks"]
        )
        emotion_agreement_rate = (
            sum(self.agreement_stats["primary_emotion"])
            / self.agreement_stats["total_tasks"]
        )

        return {
            "total_consensus_annotations": self.consensus_count,
            "crisis_agreement_rate": round(crisis_agreement_rate, 3),
            "emotion_agreement_rate": round(emotion_agreement_rate, 3),
            "overall_agreement_rate": round(
                (crisis_agreement_rate + emotion_agreement_rate) / 2, 3
            ),
        }


def load_annotations(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load annotations from JSONL file"""
    annotations = {}

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            task_id = entry.get("task_id")
            if task_id:
                annotations[task_id] = entry

    return annotations


def create_consensus_annotations(
    file1: str, file2: str, output_file: str, strategy: str = "weighted"
):
    """
    Create consensus annotations from two annotator files

    Args:
        file1: Path to first annotator's results
        file2: Path to second annotator's results
        output_file: Path to save consensus results
        strategy: Consensus strategy ("weighted", "majority", "average")
    """
    print("Creating consensus annotations...")
    print(f"  Annotator 1: {file1}")
    print(f"  Annotator 2: {file2}")
    print(f"  Strategy: {strategy}")

    # Load annotations
    ann1 = load_annotations(Path(file1))
    ann2 = load_annotations(Path(file2))

    # Find common tasks
    common_tasks = set(ann1.keys()) & set(ann2.keys())
    print(f"\nFound {len(common_tasks)} common tasks")

    if not common_tasks:
        print("ERROR: No common tasks found between annotators")
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

    print(f"\n✅ Consensus annotations saved to {output_path}")

    # Print agreement report
    report = agent.get_agreement_report()
    print("\n📊 Agreement Report:")
    print(f"  Total tasks: {report['total_consensus_annotations']}")
    print(f"  Crisis agreement: {report['crisis_agreement_rate']:.1%}")
    print(f"  Emotion agreement: {report['emotion_agreement_rate']:.1%}")
    print(f"  Overall agreement: {report['overall_agreement_rate']:.1%}")

    # Save report
    report_path = output_path.parent / f"{output_path.stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Agreement report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consensus Agent - Resolve disagreements between annotators"
    )
    parser.add_argument(
        "--file1", required=True, help="First annotator's results (JSONL)"
    )
    parser.add_argument(
        "--file2", required=True, help="Second annotator's results (JSONL)"
    )
    parser.add_argument(
        "--output", required=True, help="Output file for consensus annotations (JSONL)"
    )
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
