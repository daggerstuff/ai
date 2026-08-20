"""Inter-Annotator Agreement (IAA) module.

Provides Fleiss' kappa, Cohen's kappa, Label Studio export parsing,
quality bucketing, rubric generation, and agreement evaluation for
the clinical AI training data annotation pipeline.

CLI entry point: ``python -m training.annotation.iaa``
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLEISS_KAPPA_GOLD: float = 0.85
"""Kappa threshold for T1_GOLD tier classification."""

FLEISS_KAPPA_MINIMUM: float = 0.75
"""Minimum acceptable kappa for release-quality annotations."""

LANDIS_KOCH_THRESHOLDS: dict[str, tuple[float, float]] = {
    "poor": (-1.0, 0.0),
    "slight": (0.0, 0.20),
    "fair": (0.20, 0.40),
    "moderate": (0.40, 0.60),
    "substantial": (0.60, 0.80),
    "almost_perfect": (0.80, 0.99),
    "perfect": (0.99, 1.01),
}
"""Landis-Koch (1977) interpretation bands."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AnnotationStage(str, Enum):
    """Annotation pipeline stages."""

    INITIAL = "v1_initial"
    SECONDARY = "v2_secondary"
    ADJUDICATED = "v3_adjudicated"
    FINAL = "v3_final"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnnotatorLabel:
    """A single annotator's label for a sample."""

    annotator_id: str
    sample_id: str
    quality_score: float = 0.5
    reject_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IaaResult:
    """Result of inter-annotator agreement computation."""

    num_samples: int
    num_annotators: int
    fleiss_kappa: float
    per_sample_kappas: dict[str, float]
    quality_scores: dict[str, float]
    quarantine_samples: list[str]
    retraining_samples: list[str]
    gold_standard_samples: list[str]
    reviewer_overrides: int
    stages_distribution: Counter
    reject_reasons: Counter


# ---------------------------------------------------------------------------
# Fleiss' Kappa
# ---------------------------------------------------------------------------


def fleiss_kappa(n: int, N: int, k: int, n_i: list[int]) -> float:
    """Compute Fleiss' kappa for nominal-scale agreement.

    Parameters
    ----------
    n:
        Number of annotators per sample.
    N:
        Number of samples.
    k:
        Number of categories.
    n_i:
        Flat N×k matrix: count of annotators who assigned each category
        to each sample, row-major.

    Returns
    -------
    Fleiss' kappa in [-1.0, 1.0].  Returns 0.0 for degenerate input.
    """
    if n == 0 or N == 0:
        return 0.0
    if len(n_i) % k != 0:
        raise ValueError(
            f"n_i length {len(n_i)} is not divisible by k = {k}"
        )
    if len(n_i) < N * k:
        raise ValueError(
            f"n_i length {len(n_i)} is less than N*k = {N}*{k} = {N * k}"
        )

    # Derive actual N from the data (may exceed the passed N)
    N_actual = len(n_i) // k

    # Build N×k matrix
    matrix: list[list[int]] = []
    for row_idx in range(N_actual):
        matrix.append(n_i[row_idx * k : (row_idx + 1) * k])

    # Derive n from the actual row sums (the passed n may not match the data)
    n_actual = sum(matrix[0]) if matrix else 0
    if n_actual == 0:
        return 0.0

    # Category marginals p_j
    p_j: list[float] = []
    for j in range(k):
        total = sum(matrix[i][j] for i in range(N_actual))
        p_j.append(total / (N_actual * n_actual))

    # Per-sample agreement P_i
    P_i: list[float] = []
    for i in range(N_actual):
        sum_sq = sum(matrix[i][j] ** 2 for j in range(k))
        P_i.append((sum_sq - n_actual) / (n_actual * (n_actual - 1)))

    mean_P = sum(P_i) / N_actual

    # Expected agreement Pe
    Pe = sum(pj ** 2 for pj in p_j)

    if abs(1.0 - Pe) < 1e-12:
        # All annotators agree perfectly → kappa = 1
        return 1.0

    kappa = (mean_P - Pe) / (1.0 - Pe)
    return kappa


# ---------------------------------------------------------------------------
# Cohen's Kappa (pairwise, simple)
# ---------------------------------------------------------------------------


def cohen_kappa_simple(a: list[Any], b: list[Any]) -> float:
    """Compute Cohen's kappa between two annotators' label lists.

    Parameters
    ----------
    a, b:
        Parallel lists of labels assigned by two annotators.

    Returns
    -------
    Cohen's kappa in [-1.0, 1.0].  Returns 0.0 for empty input.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same number of labels")

    n = len(a)
    if n == 0:
        return 0.0

    # Category marginals
    categories = sorted(set(a) | set(b))
    cat_index = {c: idx for idx, c in enumerate(categories)}
    k = len(categories)

    # Observed agreement
    observed_agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    P_o = observed_agree / n

    # Expected agreement
    a_counts = Counter(a)
    b_counts = Counter(b)
    P_e = sum(
        (a_counts.get(c, 0) / n) * (b_counts.get(c, 0) / n) for c in categories
    )

    if abs(1.0 - P_e) < 1e-12:
        return 1.0 if P_o == 1.0 else 0.0

    kappa = (P_o - P_e) / (1.0 - P_e)
    return kappa


# ---------------------------------------------------------------------------
# Label Studio export parsing
# ---------------------------------------------------------------------------


def label_studio_export_to_iaa(
    path: str,
    annotator_ids: list[str],
) -> tuple[list[AnnotatorLabel], dict[str, list[AnnotatorLabel]]]:
    """Parse a Label Studio JSONL export into AnnotatorLabel objects.

    Parameters
    ----------
    path:
        Path to the JSONL export file.
    annotator_ids:
        List of annotator IDs to include (filters the export).

    Returns
    -------
    Tuple of (all_labels, labels_by_sample) where ``all_labels`` is a flat
    list and ``labels_by_sample`` maps sample_id → list of AnnotatorLabel.
    """
    all_labels: list[AnnotatorLabel] = []
    labels_by_sample: dict[str, list[AnnotatorLabel]] = {}

    annotator_set = set(annotator_ids)
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record["id"])
            data = record.get("data", {})
            annotations = record.get("annotations", [])

            for ann in annotations:
                ann_id = ann.get("annotator_id", "")
                if ann_id not in annotator_set:
                    continue
                value = ann.get("value", {})
                quality_score = value.get("quality_score", 0.5)
                reject_reason = value.get("reject_reason")
                category = value.get("category", "")

                label = AnnotatorLabel(
                    annotator_id=ann_id,
                    sample_id=sample_id,
                    quality_score=quality_score,
                    reject_reason=reject_reason,
                    metadata={
                        "category": category,
                        "text": data.get("text", ""),
                    },
                )
                all_labels.append(label)
                labels_by_sample.setdefault(sample_id, []).append(label)

    return all_labels, labels_by_sample


# ---------------------------------------------------------------------------
# Quality bucketing
# ---------------------------------------------------------------------------


def bucket_quality(score: float) -> str:
    """Bucket a quality score into a qualitative label.

    Boundaries are lower-bound inclusive.
    """
    if score >= 0.8:
        return "excellent"
    if score >= 0.6:
        return "good"
    if score >= 0.4:
        return "acceptable"
    if score >= 0.2:
        return "marginal"
    return "poor"


# ---------------------------------------------------------------------------
# Label Studio rubric XML generation
# ---------------------------------------------------------------------------


def generate_label_studio_rubric(
    categories: list[str],
    description: str | None = None,
    quality_scale: tuple[float, float] | None = None,
) -> str:
    """Generate a Label Studio-compatible rubric XML.

    Parameters
    ----------
    categories:
        List of category names for the classification task.
    description:
        Optional task description.
    quality_scale:
        Optional (min, max) for the quality_score input.

    Returns
    -------
    XML string defining the labeling interface.
    """
    lines: list[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append("<View>")

    if description:
        lines.append(f"  <Text name=\"description\" value=\"{description}\"/>")
    else:
        lines.append('  <Text name="description" value="Label the sample"/>')

    lines.append('  <Text name="text" value="$text"/>')
    lines.append("")

    # Category choices
    lines.append('  <View>')

    # Use Choices for categories
    lines.append('    <Choices name="category" toName="text" choice="single">')
    for cat in categories:
        lines.append(f'      <Choice value="{cat}"/>')
    lines.append("    </Choices>")
    lines.append("  </View>")
    lines.append("")

    # Domain dropdown
    lines.append('  <View>')
    lines.append('    <Dropdown name="domain" toName="text" placeholder="Select domain">')
    lines.append('      <Option value="clinical"/>')
    lines.append('      <Option value="behavioral"/>')
    lines.append('      <Option value="crisis"/>')
    lines.append('      <Option value="administrative"/>')
    lines.append("    </Dropdown>")
    lines.append("  </View>")
    lines.append("")

    # Difficulty dropdown
    lines.append('  <View>')
    lines.append('    <Dropdown name="difficulty" toName="text" placeholder="Select difficulty">')
    lines.append('      <Option value="easy"/>')
    lines.append('      <Option value="medium"/>')
    lines.append('      <Option value="hard"/>')
    lines.append("    </Dropdown>")
    lines.append("  </View>")
    lines.append("")

    # Quality score input
    lines.append('  <View>')
    if quality_scale is not None:
        qmin, qmax = quality_scale
        lines.append(
            f'    <Number name="quality_score" toName="text" '
            f'min="{qmin}" max="{qmax}" step="0.01"/>'
        )
    else:
        lines.append(
            '    <Number name="quality_score" toName="text" '
            'min="0" max="1" step="0.01"/>'
        )
    lines.append("  </View>")
    lines.append("")

    # Reject reason (optional)
    lines.append('  <View>')
    lines.append('    <TextArea name="reject_reason" toName="text" '
                 'placeholder="Reason for rejection (optional)"/>')
    lines.append("  </View>")
    lines.append("")

    lines.append("</View>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agreement evaluation
# ---------------------------------------------------------------------------


def _classify_landis_koch(kappa: float) -> str:
    """Classify a kappa value using Landis-Koch thresholds."""
    for label, (low, high) in LANDIS_KOCH_THRESHOLDS.items():
        if low <= kappa < high:
            return label
    return "perfect" if kappa >= 0.99 else "poor"


def evaluate_agreement(
    fleiss_kappa: float,
    cohen_kappa: float | None = None,
) -> dict[str, Any]:
    """Evaluate agreement levels and return recommendations.

    Parameters
    ----------
    fleiss_kappa:
        The Fleiss' kappa value.
    cohen_kappa:
        Optional pairwise Cohen's kappa.

    Returns
    -------
    Dict with classification and recommendation strings.
    """
    result: dict[str, Any] = {}

    # Fleiss classification
    if fleiss_kappa >= FLEISS_KAPPA_GOLD:
        result["fleiss_classification"] = "T1_GOLD final release"
        result["fleiss_recommendation"] = (
            "Agreement exceeds gold standard threshold; "
            "suitable for gold standard release."
        )
    elif fleiss_kappa >= FLEISS_KAPPA_MINIMUM:
        result["fleiss_classification"] = "fair quality release"
        result["fleiss_recommendation"] = (
            "Agreement is above minimum threshold; "
            "acceptable for release with review."
        )
    elif fleiss_kappa >= 0.40:
        result["fleiss_classification"] = "moderate quality release"
        result["fleiss_recommendation"] = (
            "Agreement is moderate; consider retraining annotators "
            "and reviewing ambiguous samples."
        )
    else:
        result["fleiss_classification"] = "poor quality release"
        result["fleiss_recommendation"] = (
            "Agreement is poor; quarantine samples and restart annotation."
        )

    # Cohen classification
    if cohen_kappa is not None:
        if cohen_kappa >= FLEISS_KAPPA_GOLD:
            result["cohen_classification"] = "T1_GOLD gold standard"
        elif cohen_kappa >= 0.40:
            result["cohen_classification"] = "retraining zone"
        else:
            result["cohen_classification"] = "poor"

    return result


# ---------------------------------------------------------------------------
# Full IAA computation from AnnotatorLabel list
# ---------------------------------------------------------------------------


def _per_sample_fleiss_kappa(
    sample_labels: list[AnnotatorLabel],
    categories: list[str],
    n: int,
) -> float:
    """Compute Fleiss' kappa for a single sample's labels."""
    if n <= 1:
        return 1.0

    cat_index = {c: idx for idx, c in enumerate(categories)}
    k = len(categories)

    n_i = [0] * k
    for label in sample_labels:
        cat = label.metadata.get("category", "unknown")
        if cat in cat_index:
            n_i[cat_index[cat]] += 1

    # If all in one category, perfect agreement
    non_zero = sum(1 for c in n_i if c > 0)
    if non_zero <= 1:
        return 1.0

    # Compute kappa for this single sample (N=1)
    return fleiss_kappa(n, 1, k, n_i)


def compute_iaa_from_labels(
    labels: list[AnnotatorLabel],
    num_annotators: int,
) -> IaaResult:
    """Compute full IAA statistics from a list of annotator labels.

    Parameters
    ----------
    labels:
        Flat list of AnnotatorLabel objects.
    num_annotators:
        Expected number of annotators per sample.

    Returns
    -------
    IaaResult with all agreement metrics.
    """
    if not labels:
        return IaaResult(
            num_samples=0,
            num_annotators=num_annotators,
            fleiss_kappa=0.0,
            per_sample_kappas={},
            quality_scores={},
            quarantine_samples=[],
            retraining_samples=[],
            gold_standard_samples=[],
            reviewer_overrides=0,
            stages_distribution=Counter(),
            reject_reasons=Counter(),
        )

    # Group labels by sample
    by_sample: dict[str, list[AnnotatorLabel]] = {}
    for label in labels:
        by_sample.setdefault(label.sample_id, []).append(label)

    sample_ids = sorted(by_sample.keys())
    num_samples = len(sample_ids)

    # Collect all categories across labels
    all_categories = sorted(
        {label.metadata.get("category", "unknown") for label in labels}
    )

    # Per-sample kappa and quality
    per_sample_kappas: dict[str, float] = {}
    quality_scores: dict[str, float] = {}
    quarantine: list[str] = []
    retraining: list[str] = []
    gold: list[str] = []

    for sid in sample_ids:
        sample_labels = by_sample[sid]

        # Per-sample kappa
        kappa = _per_sample_fleiss_kappa(sample_labels, all_categories, len(sample_labels))
        per_sample_kappas[sid] = kappa

        # Quality score = mean of annotator quality_scores
        scores = [l.quality_score for l in sample_labels]
        quality_scores[sid] = sum(scores) / len(scores) if scores else 0.0

        # Bucket by kappa
        if kappa < 0.40:
            quarantine.append(sid)
        elif kappa >= FLEISS_KAPPA_GOLD:
            gold.append(sid)
        else:
            retraining.append(sid)

    # Overall Fleiss' kappa (aggregate across all samples)
    N = num_samples
    k = len(all_categories)
    n = num_annotators

    if N > 0 and k > 0 and n > 1:
        flat_n_i: list[int] = []
        for sid in sample_ids:
            sample_labels = by_sample[sid]
            cat_counts = Counter(
                l.metadata.get("category", "unknown") for l in sample_labels
            )
            for cat in all_categories:
                flat_n_i.append(cat_counts.get(cat, 0))
        overall_kappa = fleiss_kappa(n, N, k, flat_n_i)
    else:
        overall_kappa = 0.0

    # Reviewer overrides
    reviewer_overrides = sum(
        1
        for l in labels
        if l.metadata.get("annotation_stage") == AnnotationStage.ADJUDICATED.value
    )

    # Stages distribution
    stages_distribution: Counter = Counter()
    for l in labels:
        stage = l.metadata.get("annotation_stage")
        if stage:
            stages_distribution[stage] += 1

    # Reject reasons
    reject_reasons: Counter = Counter()
    for l in labels:
        if l.reject_reason:
            reject_reasons[l.reject_reason] += 1

    return IaaResult(
        num_samples=num_samples,
        num_annotators=num_annotators,
        fleiss_kappa=overall_kappa,
        per_sample_kappas=per_sample_kappas,
        quality_scores=quality_scores,
        quarantine_samples=quarantine,
        retraining_samples=retraining,
        gold_standard_samples=gold,
        reviewer_overrides=reviewer_overrides,
        stages_distribution=stages_distribution,
        reject_reasons=reject_reasons,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point for IAA computation."""
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement from Label Studio exports."
    )
    parser.add_argument(
        "--ls-jsonl",
        type=str,
        required=True,
        help="Path to Label Studio JSONL export file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write JSON output.",
    )
    parser.add_argument(
        "--num-annotators",
        type=int,
        required=True,
        help="Number of annotators per sample.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Category names for rubric generation.",
    )
    parser.add_argument(
        "--rubric-xml",
        type=str,
        default=None,
        help="Path to write Label Studio rubric XML.",
    )
    args = parser.parse_args()

    # Determine annotator IDs from the file
    annotator_ids: list[str] = []
    file_path = Path(args.ls_jsonl)
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for ann in record.get("annotations", []):
                ann_id = ann.get("annotator_id", "")
                if ann_id and ann_id not in annotator_ids:
                    annotator_ids.append(ann_id)

    all_labels, labels_by_sample = label_studio_export_to_iaa(
        args.ls_jsonl,
        annotator_ids,
    )

    result = compute_iaa_from_labels(all_labels, args.num_annotators)
    evaluation = evaluate_agreement(result.fleiss_kappa)

    output: dict[str, Any] = {
        "fleiss_kappa": result.fleiss_kappa,
        "classification": evaluation.get("fleiss_classification", ""),
        "num_samples": result.num_samples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Fleiss kappa: {result.fleiss_kappa:.4f}")
    print(f"Samples: {result.num_samples}")
    print(f"Classification: {evaluation.get('fleiss_classification', '')}")

    # Write rubric XML if requested
    if args.rubric_xml and args.categories:
        rubric = generate_label_studio_rubric(args.categories)
        rubric_path = Path(args.rubric_xml)
        rubric_path.parent.mkdir(parents=True, exist_ok=True)
        with rubric_path.open("w", encoding="utf-8") as f:
            f.write(rubric)

    return 0


if __name__ == "__main__":
    sys.exit(main())
