"""
Multi-Agent Inter-Annotator Agreement Calculator
Calculate Cohen's Kappa and other agreement metrics for multi-agent annotations
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def calculate_cohens_kappa(annotations_a: list[int], annotations_b: list[int]) -> float:
    """
    Calculate Cohen's Kappa for two annotators

    Args:
        annotations_a: List of annotations from annotator A
        annotations_b: List of annotations from annotator B

    Returns:
        Cohen's Kappa coefficient
    """
    if len(annotations_a) != len(annotations_b):
        raise ValueError("Annotation lists must be same length")

    n = len(annotations_a)
    if n == 0:
        return 0.0

    # Calculate observed agreement
    agreements = sum(1 for a, b in zip(annotations_a, annotations_b, strict=False) if a == b)
    p_o = agreements / n

    # Calculate expected agreement
    unique_labels = set(annotations_a + annotations_b)
    p_e = 0.0

    for label in unique_labels:
        p_a = annotations_a.count(label) / n
        p_b = annotations_b.count(label) / n
        p_e += p_a * p_b

    # Calculate Kappa
    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def calculate_agreement_metrics(results_file: str) -> dict[str, any]:
    """
    Calculate comprehensive agreement metrics from multi-agent results
    """
    results_path = Path(results_file)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    # Load all results
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Extract annotations by agent
    agent_annotations = defaultdict(lambda: defaultdict(list))

    for result in results:
        individual = result.get("individual_annotations", [])

        for i, annotation in enumerate(individual):
            agent_id = f"agent_{i}"

            # Extract key fields
            agent_annotations[agent_id]["crisis_label"].append(annotation.get("crisis_label", 0))
            agent_annotations[agent_id]["primary_emotion"].append(annotation.get("primary_emotion", "Neutral"))
            agent_annotations[agent_id]["emotion_intensity"].append(annotation.get("emotion_intensity", 5))

    # Calculate pairwise Kappa scores
    agent_ids = list(agent_annotations.keys())
    kappa_scores = {}

    if len(agent_ids) >= 2:
        for i, agent_a in enumerate(agent_ids):
            for agent_b in agent_ids[i + 1 :]:
                # Crisis label Kappa
                crisis_kappa = calculate_cohens_kappa(
                    agent_annotations[agent_a]["crisis_label"],
                    agent_annotations[agent_b]["crisis_label"],
                )

                # Emotion Kappa (convert to numeric)
                emotions_a = agent_annotations[agent_a]["primary_emotion"]
                emotions_b = agent_annotations[agent_b]["primary_emotion"]

                # Create emotion mapping
                unique_emotions = set(emotions_a + emotions_b)
                emotion_map = {e: i for i, e in enumerate(unique_emotions)}

                emotions_a_numeric = [emotion_map[e] for e in emotions_a]
                emotions_b_numeric = [emotion_map[e] for e in emotions_b]

                emotion_kappa = calculate_cohens_kappa(emotions_a_numeric, emotions_b_numeric)

                kappa_scores[f"{agent_a}_vs_{agent_b}"] = {
                    "crisis_kappa": round(crisis_kappa, 4),
                    "emotion_kappa": round(emotion_kappa, 4),
                    "average_kappa": round((crisis_kappa + emotion_kappa) / 2, 4),
                }

    # Calculate overall agreement statistics
    agreement_stats = calculate_agreement_statistics(results)

    # Calculate consensus quality
    consensus_quality = calculate_consensus_quality(results)

    return {
        "total_items": len(results),
        "num_agents": len(agent_ids),
        "kappa_scores": kappa_scores,
        "agreement_statistics": agreement_stats,
        "consensus_quality": consensus_quality,
    }


def calculate_agreement_statistics(results: list[dict]) -> dict:
    """Calculate overall agreement statistics"""
    crisis_agreements = []
    emotion_agreements = []
    overall_agreements = []

    for result in results:
        metrics = result.get("agreement_metrics", {})
        crisis_agreements.append(metrics.get("crisis_agreement", 0.0))
        emotion_agreements.append(metrics.get("emotion_agreement", 0.0))
        overall_agreements.append(metrics.get("overall_agreement", 0.0))

    return {
        "crisis_agreement": {
            "mean": round(np.mean(crisis_agreements), 4),
            "std": round(np.std(crisis_agreements), 4),
        },
        "emotion_agreement": {
            "mean": round(np.mean(emotion_agreements), 4),
            "std": round(np.std(emotion_agreements), 4),
        },
        "overall_agreement": {
            "mean": round(np.mean(overall_agreements), 4),
            "std": round(np.std(overall_agreements), 4),
        },
    }


def calculate_consensus_quality(results: list[dict]) -> dict:
    """Calculate quality metrics for consensus annotations"""
    consensus_annotations = []

    for result in results:
        consensus = result.get("consensus_annotation", {})
        if consensus:
            consensus_annotations.append(consensus)

    if not consensus_annotations:
        return {}

    # Calculate distribution statistics
    crisis_labels = [a.get("crisis_label", 0) for a in consensus_annotations]
    intensities = [a.get("emotion_intensity", 5) for a in consensus_annotations]
    valences = [a.get("valence", 0.0) for a in consensus_annotations]

    return {
        "crisis_distribution": {
            "mean": round(np.mean(crisis_labels), 4),
            "crisis_rate": round(sum(1 for c in crisis_labels if c > 0) / len(crisis_labels), 4),
        },
        "intensity_distribution": {
            "mean": round(np.mean(intensities), 4),
            "std": round(np.std(intensities), 4),
        },
        "valence_distribution": {
            "mean": round(np.mean(valences), 4),
            "std": round(np.std(valences), 4),
        },
    }


def print_metrics_report(metrics: dict):
    """Print formatted metrics report"""

    for _pair, scores in metrics["kappa_scores"].items():
        # Interpret Kappa
        avg_kappa = scores["average_kappa"]
        if avg_kappa >= 0.85 or avg_kappa >= 0.70 or avg_kappa >= 0.50:
            pass
        else:
            pass

    metrics["agreement_statistics"]

    metrics["consensus_quality"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate multi-agent inter-annotator agreement")
    parser.add_argument(
        "--input",
        required=True,
        help="Multi-agent annotation results JSONL file",
    )
    parser.add_argument(
        "--output",
        help="Optional: Save metrics to JSON file",
    )

    args = parser.parse_args()

    # Calculate metrics
    metrics = calculate_agreement_metrics(args.input)

    # Print report
    print_metrics_report(metrics)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
