"""Calibration metrics aggregator for clinical validity enhancement pipeline.

This module implements the CalibrationMetricsAggregator that reads scoring reports
(data/reports/*.json) and promotion reports (data/closed_loop/reports/*.json) to
emit a calibration snapshot JSON with per-scorer agreement, borderline rate,
expert-disagreement rate, and safety-score variance.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger("calibration_metrics")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScorerMetrics:
    """Metrics for a single scorer."""

    scorer_id: str
    agreement: float  # Agreement rate with expert (0.0 to 1.0)
    borderline_count: int  # Number of borderline items (score in [0.4, 0.6))
    total_items: int  # Total items scored


@dataclass
class CalibrationSnapshot:
    """Calibration snapshot containing all metrics."""

    # Per-scorer agreement metrics
    per_scorer_agreement: dict[str, float]

    # Borderline rate: items with score in [0.4, 0.6) / total items
    borderline_rate: float

    # Expert-disagreement rate: |expert - scorer| > 0.2 / total items
    expert_disagreement_rate: float

    # Variance of safety scores across scorers
    safety_variance: float

    # Metadata
    generated_at: str
    scoring_report_count: int
    promotion_report_count: int

    # Additional counts for debugging
    total_items: int
    borderline_count: int
    disagreement_count: int


class CalibrationSnapshotDict(TypedDict):
    """TypedDict representation of CalibrationSnapshot for JSON serialization."""

    per_scorer_agreement: dict[str, float]
    borderline_rate: float
    expert_disagreement_rate: float
    safety_variance: float
    generated_at: str
    scoring_report_count: int
    promotion_report_count: int
    total_items: int
    borderline_count: int
    disagreement_count: int


# ---------------------------------------------------------------------------
# Threshold constants (from clinical_validity_scorer.py)
# ---------------------------------------------------------------------------

EXCLUDE_THRESHOLD = 0.4
BORDERLINE_MAX = 0.6
ACCEPT_THRESHOLD = 0.6

# Expert disagreement threshold
EXPERT_DISAGREEMENT_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# Tier computation
# ---------------------------------------------------------------------------


def _get_tier(score: float) -> str:
    """Determine the quality tier for a score.

    Args:
        score: The score value (0.0 to 1.0)

    Returns:
        Tier name: "exclude", "borderline", or "accept"
    """
    if score < EXCLUDE_THRESHOLD:
        return "exclude"
    if score < BORDERLINE_MAX:
        return "borderline"
    return "accept"


def _is_borderline(score: float) -> bool:
    """Check if a score falls in the borderline range [0.4, 0.6).

    Args:
        score: The score value

    Returns:
        True if score is in [0.4, 0.6), False otherwise
    """
    return EXCLUDE_THRESHOLD <= score < BORDERLINE_MAX


def _is_expert_disagreement(scorer_score: float, expert_score: float) -> bool:
    """Check if there is expert disagreement: |expert - scorer| > 0.2.

    Args:
        scorer_score: The scorer's score
        expert_score: The expert's score

    Returns:
        True if |expert - scorer| > 0.2, False otherwise
    """
    return abs(expert_score - scorer_score) > EXPERT_DISAGREEMENT_THRESHOLD


def _compute_tier_agreement(scorer_score: float, expert_score: float) -> float:
    """Compute agreement between scorer and expert based on tier matching.

    Args:
        scorer_score: The scorer's score
        expert_score: The expert's score

    Returns:
        1.0 if both scores are in the same tier, 0.0 otherwise
    """
    return 1.0 if _get_tier(scorer_score) == _get_tier(expert_score) else 0.0


# ---------------------------------------------------------------------------
# Scoring report loading
# ---------------------------------------------------------------------------


@dataclass
class ScoredItem:
    """A scored item from a scoring report."""

    item_id: str
    scorer_id: str
    score: float
    safety_score: float | None
    per_dimension_scores: dict[str, float]
    expert_score: float | None  # From associated review/promotion data
    timestamp: str


@dataclass
class ScoringReport:
    """A single scoring report file."""

    file_path: Path
    scorer_id: str
    timestamp: str
    items: list[ScoredItem] = field(default_factory=list)


def _load_scoring_reports(reports_dir: Path | str) -> list[ScoringReport]:
    """Load all scoring reports from the reports directory.

    Args:
        reports_dir: Path to data/reports/

    Returns:
        List of ScoringReport objects
    """
    reports_dir = Path(reports_dir)
    if not reports_dir.exists():
        logger.warning("Scoring reports directory does not exist: %s", reports_dir)
        return []

    reports: list[ScoringReport] = []
    for json_file in sorted(reports_dir.glob("*.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Extract scorer_id from report data or filename
            scorer_id = data.get("scorer_id", json_file.stem)

            # Extract timestamp
            timestamp = data.get("timestamp", datetime.now(UTC).isoformat())

            # Parse items
            items: list[ScoredItem] = []
            raw_items = data.get("items", [])
            for item_data in raw_items:
                safety_score = item_data.get("safety_score")
                per_dim = item_data.get("per_dimension_scores", {})
                items.append(ScoredItem(
                    item_id=str(item_data.get("item_id", "")),
                    scorer_id=scorer_id,
                    score=float(item_data.get("score", 0.0)),
                    safety_score=float(safety_score) if safety_score is not None else None,
                    per_dimension_scores={k: float(v) for k, v in per_dim.items()},
                    expert_score=float(item_data["expert_score"]) if "expert_score" in item_data else None,
                    timestamp=item_data.get("timestamp", timestamp),
                ))

            reports.append(ScoringReport(
                file_path=json_file,
                scorer_id=scorer_id,
                timestamp=timestamp,
                items=items,
            ))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load scoring report %s: %s", json_file, e)
            continue

    return reports


# ---------------------------------------------------------------------------
# Promotion report loading
# ---------------------------------------------------------------------------


@dataclass
class PromotionReport:
    """A promotion report from the closed-loop service."""

    file_path: Path
    timestamp: str
    received: int
    validated: int
    rejected: int
    merged: int
    reasons: dict[str, int]


def _load_promotion_reports(reports_dir: Path | str) -> list[PromotionReport]:
    """Load all promotion reports from the closed_loop reports directory.

    Args:
        reports_dir: Path to data/closed_loop/reports/

    Returns:
        List of PromotionReport objects
    """
    reports_dir = Path(reports_dir)
    if not reports_dir.exists():
        logger.warning("Promotion reports directory does not exist: %s", reports_dir)
        return []

    reports: list[PromotionReport] = []
    for json_file in sorted(reports_dir.glob("*.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Extract timestamp from filename (format: YYYYMMDD-HHMMSS.json)
            timestamp = json_file.stem

            reports.append(PromotionReport(
                file_path=json_file,
                timestamp=timestamp,
                received=int(data.get("received", 0)),
                validated=int(data.get("validated", 0)),
                rejected=int(data.get("rejected", 0)),
                merged=int(data.get("merged", 0)),
                reasons=data.get("reasons", {}),
            ))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load promotion report %s: %s", json_file, e)
            continue

    return reports


# ---------------------------------------------------------------------------
# CalibrationMetricsAggregator
# ---------------------------------------------------------------------------


# Minimum number of safety scores needed for variance computation
_MIN_SAFETY_SCORES_FOR_VARIANCE = 2


class CalibrationMetricsAggregator:
    """Aggregator for computing calibration metrics from scoring and promotion reports.

    Reads scoring reports (data/reports/*.json) and promotion reports
    (data/closed_loop/reports/*.json) to emit a calibration snapshot JSON
    with:
    - per_scorer_agreement: Agreement rate per scorer vs expert
    - borderline_rate: Proportion of items with scores in [0.4, 0.6)
    - expert_disagreement_rate: Proportion of items where |expert - scorer| > 0.2
    - safety_variance: Variance of safety scores across scorers

    The aggregator is deterministic: same input reports always produce
    the same output snapshot.
    """

    def __init__(
        self,
        scoring_reports_dir: Path | str | None = None,
        promotion_reports_dir: Path | str | None = None,
    ) -> None:
        """Initialize the CalibrationMetricsAggregator.

        Args:
            scoring_reports_dir: Directory containing scoring reports.
                Defaults to data/reports/
            promotion_reports_dir: Directory containing promotion reports.
                Defaults to data/closed_loop/reports/
        """
        if scoring_reports_dir is None:
            scoring_reports_dir = Path("data/reports")
        if promotion_reports_dir is None:
            promotion_reports_dir = Path("data/closed_loop/reports")

        self.scoring_reports_dir = Path(scoring_reports_dir)
        self.promotion_reports_dir = Path(promotion_reports_dir)

    def _compute_per_scorer_agreement(
        self,
        items: list[ScoredItem],
    ) -> dict[str, float]:
        """Compute agreement rate per scorer.

        Args:
            items: All scored items with expert scores

        Returns:
            Dictionary mapping scorer_id to agreement rate (0.0 to 1.0)
        """
        scorer_agreements: dict[str, list[float]] = {}

        for item in items:
            if item.expert_score is None:
                continue

            if item.scorer_id not in scorer_agreements:
                scorer_agreements[item.scorer_id] = []

            agreement = _compute_tier_agreement(item.score, item.expert_score)
            scorer_agreements[item.scorer_id].append(agreement)

        # Convert to rates
        per_scorer: dict[str, float] = {}
        for scorer_id, agreements in sorted(scorer_agreements.items()):
            if agreements:
                per_scorer[scorer_id] = sum(agreements) / len(agreements)

        return per_scorer

    def _compute_borderline_rate(self, items: list[ScoredItem]) -> tuple[float, int, int]:
        """Compute the borderline rate.

        Args:
            items: All scored items

        Returns:
            Tuple of (borderline_rate, borderline_count, total_count)
        """
        total = len(items)
        if total == 0:
            return 0.0, 0, 0

        borderline_count = sum(1 for item in items if _is_borderline(item.score))
        rate = borderline_count / total

        return rate, borderline_count, total

    def _compute_expert_disagreement_rate(
        self,
        items: list[ScoredItem],
    ) -> tuple[float, int, int]:
        """Compute the expert-disagreement rate.

        Args:
            items: All scored items with expert scores

        Returns:
            Tuple of (disagreement_rate, disagreement_count, total_with_expert_count)
        """
        items_with_expert = [item for item in items if item.expert_score is not None]
        total = len(items_with_expert)
        if total == 0:
            return 0.0, 0, 0

        disagreement_count = sum(
            1 for item in items_with_expert
            if _is_expert_disagreement(item.score, item.expert_score)
        )
        rate = disagreement_count / total

        return rate, disagreement_count, total

    def _compute_safety_variance(self, items: list[ScoredItem]) -> float:
        """Compute the variance of safety scores across all scorers.

        Args:
            items: All scored items

        Returns:
            Variance of safety scores, or 0.0 if no valid safety scores
        """
        safety_scores: list[float] = []
        for item in items:
            if item.safety_score is not None:
                safety_scores.append(item.safety_score)

        if len(safety_scores) < _MIN_SAFETY_SCORES_FOR_VARIANCE:
            return 0.0

        return float(statistics.variance(safety_scores))

    def aggregate(self) -> CalibrationSnapshot:
        """Aggregate all scoring and promotion reports into a calibration snapshot.

        Returns:
            CalibrationSnapshot with all computed metrics
        """
        # Load reports
        scoring_reports = _load_scoring_reports(self.scoring_reports_dir)
        promotion_reports = _load_promotion_reports(self.promotion_reports_dir)

        # Collect all items
        all_items: list[ScoredItem] = []
        for report in scoring_reports:
            all_items.extend(report.items)

        # Compute metrics
        per_scorer_agreement = self._compute_per_scorer_agreement(all_items)
        borderline_rate, borderline_count, total_items = self._compute_borderline_rate(all_items)
        disagreement_rate, disagreement_count, _ = self._compute_expert_disagreement_rate(all_items)
        safety_variance = self._compute_safety_variance(all_items)

        snapshot = CalibrationSnapshot(
            per_scorer_agreement=per_scorer_agreement,
            borderline_rate=borderline_rate,
            expert_disagreement_rate=disagreement_rate,
            safety_variance=safety_variance,
            generated_at=datetime.now(UTC).isoformat(),
            scoring_report_count=len(scoring_reports),
            promotion_report_count=len(promotion_reports),
            total_items=total_items,
            borderline_count=borderline_count,
            disagreement_count=disagreement_count,
        )

        logger.info(
            "Generated calibration snapshot: borderline_rate=%.4f, "
            "expert_disagreement_rate=%.4f, safety_variance=%.4f, "
            "scoring_reports=%d, promotion_reports=%d",
            borderline_rate,
            disagreement_rate,
            safety_variance,
            len(scoring_reports),
            len(promotion_reports),
        )

        return snapshot

    def emit(self, output_path: Path | str | None = None) -> CalibrationSnapshot:
        """Aggregate and optionally write the snapshot to a file.

        Args:
            output_path: Optional path to write the JSON snapshot

        Returns:
            CalibrationSnapshot with all computed metrics
        """
        snapshot = self.aggregate()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            snapshot_dict: CalibrationSnapshotDict = {
                "per_scorer_agreement": snapshot.per_scorer_agreement,
                "borderline_rate": snapshot.borderline_rate,
                "expert_disagreement_rate": snapshot.expert_disagreement_rate,
                "safety_variance": snapshot.safety_variance,
                "generated_at": snapshot.generated_at,
                "scoring_report_count": snapshot.scoring_report_count,
                "promotion_report_count": snapshot.promotion_report_count,
                "total_items": snapshot.total_items,
                "borderline_count": snapshot.borderline_count,
                "disagreement_count": snapshot.disagreement_count,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(snapshot_dict, f, indent=2)
                f.write("\n")

            logger.info("Written calibration snapshot to %s", output_path)

        return snapshot


def aggregate(
    scoring_reports_dir: Path | str | None = None,
    promotion_reports_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> CalibrationSnapshot:
    """Convenience function to aggregate calibration metrics.

    Args:
        scoring_reports_dir: Directory containing scoring reports.
            Defaults to data/reports/
        promotion_reports_dir: Directory containing promotion reports.
            Defaults to data/closed_loop/reports/
        output_path: Optional path to write the JSON snapshot

    Returns:
        CalibrationSnapshot with all computed metrics
    """
    aggregator = CalibrationMetricsAggregator(
        scoring_reports_dir=scoring_reports_dir,
        promotion_reports_dir=promotion_reports_dir,
    )
    return aggregator.emit(output_path)
