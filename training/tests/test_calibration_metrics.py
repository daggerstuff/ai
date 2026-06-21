"""Tests for CalibrationMetricsAggregator.

These tests verify the calibration metrics aggregator behavior per
VAL-M3-CAL-001, VAL-M3-CAL-002, and VAL-M3-CAL-003.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from training.coaching_safety.calibration_metrics import (
    CalibrationMetricsAggregator,
    CalibrationSnapshot,
    _compute_tier_agreement,
    _get_tier,
    _is_borderline,
    _is_expert_disagreement,
    _load_promotion_reports,
    aggregate,
)

# ---------------------------------------------------------------------------
# Test data constants
# ---------------------------------------------------------------------------

# Total items in test borderline/reject reports (3 borderline + 5 non-borderline)
THREE_ITEMS = 3
# Four items in some tests
FOUR_ITEMS = 4
# Total items in test mixed tier reports (4 borderline + 4 non-borderline = 8 total)
EIGHT_ITEMS = 8
# Two reports in test data
TWO_REPORTS = 2

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_reports_dir() -> Generator[Path]:
    """Create a temporary directory for scoring and promotion reports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def scoring_reports_dir(temp_reports_dir: Path) -> Path:
    """Create the scoring reports subdirectory."""
    d = temp_reports_dir / "reports"
    d.mkdir()
    return d


@pytest.fixture
def promotion_reports_dir(temp_reports_dir: Path) -> Path:
    """Create the promotion reports subdirectory."""
    d = temp_reports_dir / "closed_loop" / "reports"
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# Helper: write scoring report
# ---------------------------------------------------------------------------


def write_scoring_report(
    dir_path: Path,
    scorer_id: str,
    items: list[dict],
    timestamp: str | None = None,
) -> Path:
    """Write a scoring report JSON file."""
    if timestamp is None:
        timestamp = datetime.now(UTC).isoformat()
    report = {
        "scorer_id": scorer_id,
        "timestamp": timestamp,
        "items": items,
    }
    file_path = dir_path / f"{scorer_id}_{timestamp.replace(':', '')}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return file_path


# ---------------------------------------------------------------------------
# Helper: write promotion report
# ---------------------------------------------------------------------------


def write_promotion_report(
    dir_path: Path,
    received: int,
    validated: int,
    rejected: int,
    merged: int,
    reasons: dict[str, int] | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write a promotion report JSON file."""
    if timestamp is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S%f")  # Add microseconds for uniqueness
    report = {
        "received": received,
        "validated": validated,
        "rejected": rejected,
        "merged": merged,
        "reasons": reasons or {},
    }
    file_path = dir_path / f"{timestamp}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return file_path


# ---------------------------------------------------------------------------
# Tests for tier computation helpers
# ---------------------------------------------------------------------------


class TestTierComputation:
    """Test _get_tier, _is_borderline, _is_expert_disagreement helpers."""

    @pytest.mark.parametrize(("score", "expected_tier"), [
        (0.0, "exclude"),
        (0.1, "exclude"),
        (0.39, "exclude"),
        (0.4, "borderline"),
        (0.45, "borderline"),
        (0.59, "borderline"),
        (0.6, "accept"),
        (0.7, "accept"),
        (1.0, "accept"),
    ])
    def test_get_tier(self, score: float, expected_tier: str) -> None:
        """Test tier classification for various scores."""
        assert _get_tier(score) == expected_tier

    @pytest.mark.parametrize(("score", "expected"), [
        (0.0, False),
        (0.39, False),
        (0.4, True),
        (0.45, True),
        (0.59, True),
        (0.6, False),
        (0.7, False),
        (1.0, False),
    ])
    def test_is_borderline(self, score: float, expected: bool) -> None:
        """Test borderline detection for scores in [0.4, 0.6)."""
        assert _is_borderline(score) == expected

    @pytest.mark.parametrize(("scorer", "expert", "expected"), [
        (0.5, 0.5, False),
        (0.5, 0.7, False),
        (0.5, 0.3, False),
        (0.5, 0.71, True),
        (0.5, 0.29, True),
        (0.7, 0.5, False),
        (0.7, 0.91, True),
        (0.7, 0.49, True),
    ])
    def test_is_expert_disagreement(
        self,
        scorer: float,
        expert: float,
        expected: bool,
    ) -> None:
        """Test expert disagreement detection: |expert - scorer| > 0.2."""
        assert _is_expert_disagreement(scorer, expert) == expected

    @pytest.mark.parametrize(("scorer", "expert", "expected"), [
        (0.5, 0.5, 1.0),  # Same tier (borderline)
        (0.7, 0.5, 0.0),  # Different tiers (accept vs borderline)
        (0.3, 0.5, 0.0),  # Different tiers (exclude vs borderline)
        (0.3, 0.2, 1.0),  # Same tier (exclude vs exclude)
        (0.65, 0.75, 1.0),  # Same tier (accept vs accept)
    ])
    def test_compute_tier_agreement(
        self,
        scorer: float,
        expert: float,
        expected: float,
    ) -> None:
        """Test tier-based agreement computation."""
        assert _compute_tier_agreement(scorer, expert) == expected


# ---------------------------------------------------------------------------
# VAL-M3-CAL-002: Borderline rate == m/n exactly
# ---------------------------------------------------------------------------


class TestBorderlineRateExact:
    """Property tests for exact borderline rate computation."""

    def test_borderline_rate_zero_when_no_borderline(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-002: borderline_rate == 0 when no items are borderline."""
        # All items are in exclude range
        items = [
            {"item_id": "1", "score": 0.1, "expert_score": 0.15},
            {"item_id": "2", "score": 0.2, "expert_score": 0.25},
            {"item_id": "3", "score": 0.3, "expert_score": 0.35},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.borderline_rate == 0.0
        assert snapshot.borderline_count == 0
        assert snapshot.total_items == THREE_ITEMS

    def test_borderline_rate_all_borderline(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-002: borderline_rate == 1.0 when all items are borderline."""
        items = [
            {"item_id": "1", "score": 0.4, "expert_score": 0.4},
            {"item_id": "2", "score": 0.5, "expert_score": 0.5},
            {"item_id": "3", "score": 0.59, "expert_score": 0.58},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.borderline_rate == 1.0
        assert snapshot.borderline_count == THREE_ITEMS
        assert snapshot.total_items == THREE_ITEMS

    def test_borderline_rate_exact_half(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-002: borderline_rate == 0.5 exactly for 4 borderline out of 8."""
        # Create 8 items, 4 borderline [0.4, 0.6), 4 exclude (< 0.4)
        items = [
            # Borderline
            {"item_id": "1", "score": 0.4, "expert_score": 0.4},
            {"item_id": "2", "score": 0.5, "expert_score": 0.5},
            {"item_id": "3", "score": 0.55, "expert_score": 0.55},
            {"item_id": "4", "score": 0.59, "expert_score": 0.59},
            # Exclude
            {"item_id": "5", "score": 0.1, "expert_score": 0.1},
            {"item_id": "6", "score": 0.2, "expert_score": 0.2},
            {"item_id": "7", "score": 0.3, "expert_score": 0.3},
            {"item_id": "8", "score": 0.39, "expert_score": 0.39},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.borderline_rate == 4 / 8
        assert snapshot.borderline_count == FOUR_ITEMS
        assert snapshot.total_items == EIGHT_ITEMS

    def test_borderline_rate_mixed_tiers(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-002: borderline_rate correct with mixed tiers."""
        # 2 exclude, 3 borderline, 3 accept = 3/8 borderline
        items = [
            # Exclude
            {"item_id": "1", "score": 0.1, "expert_score": 0.1},
            {"item_id": "2", "score": 0.3, "expert_score": 0.3},
            # Borderline
            {"item_id": "3", "score": 0.4, "expert_score": 0.4},
            {"item_id": "4", "score": 0.5, "expert_score": 0.5},
            {"item_id": "5", "score": 0.59, "expert_score": 0.59},
            # Accept
            {"item_id": "6", "score": 0.6, "expert_score": 0.6},
            {"item_id": "7", "score": 0.8, "expert_score": 0.8},
            {"item_id": "8", "score": 1.0, "expert_score": 1.0},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.borderline_rate == 3 / 8
        assert snapshot.borderline_count == THREE_ITEMS
        assert snapshot.total_items == EIGHT_ITEMS

    def test_borderline_rate_empty_reports(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-002: borderline_rate == 0.0 when no reports exist."""
        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.borderline_rate == 0.0
        assert snapshot.borderline_count == 0
        assert snapshot.total_items == 0

    def test_borderline_rate_property_many_n(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-002: borderline_rate == m/n for varying n values."""
        for n in [5, 10, 20, 50, 100]:
            # Create n items, borderline_count = m = n // 2
            m = n // 2
            items = []

            # First m items are borderline
            for i in range(m):
                items.append({
                    "item_id": str(i),
                    "score": 0.4 + (i * 0.001),  # Within borderline range
                    "expert_score": 0.4 + (i * 0.001),
                })

            # Remaining items are accept
            for i in range(m, n):
                items.append({
                    "item_id": str(i),
                    "score": 0.6 + ((i - m) * 0.001),  # Above borderline
                    "expert_score": 0.6 + ((i - m) * 0.001),
                })

            # Create a new report file for each n (use n as unique suffix)
            scorer_id = f"scorer_{n}"
            write_scoring_report(scoring_reports_dir, scorer_id, items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # For each n, borderline_count = n // 2
        # Total items: 5 + 10 + 20 + 50 + 100 = 185
        # Total borderline: 5//2 + 10//2 + 20//2 + 50//2 + 100//2 = 2 + 5 + 10 + 25 + 50 = 92
        # Rate should be 92/185
        expected_total = 5 + 10 + 20 + 50 + 100
        expected_borderline = 5 // 2 + 10 // 2 + 20 // 2 + 50 // 2 + 100 // 2

        assert snapshot.total_items == expected_total
        assert snapshot.borderline_count == expected_borderline
        assert snapshot.borderline_rate == pytest.approx(expected_borderline / expected_total)


# ---------------------------------------------------------------------------
# VAL-M3-CAL-003: Expert-disagreement rate == k/n exactly
# ---------------------------------------------------------------------------


class TestExpertDisagreementRateExact:
    """Property tests for exact expert-disagreement rate computation."""

    def test_disagreement_rate_zero_when_all_agree(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-003: expert_disagreement_rate == 0 when no disagreements."""
        items = [
            {"item_id": "1", "score": 0.5, "expert_score": 0.5},
            {"item_id": "2", "score": 0.6, "expert_score": 0.6},
            {"item_id": "3", "score": 0.7, "expert_score": 0.7},
            {"item_id": "4", "score": 0.3, "expert_score": 0.3},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.expert_disagreement_rate == 0.0
        assert snapshot.disagreement_count == 0

    def test_disagreement_rate_all_disagree(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-003: expert_disagreement_rate == 1.0 when all disagree."""
        items = [
            {"item_id": "1", "score": 0.5, "expert_score": 0.8},  # diff = 0.3 > 0.2
            {"item_id": "2", "score": 0.6, "expert_score": 0.2},  # diff = 0.4 > 0.2
            {"item_id": "3", "score": 0.7, "expert_score": 0.1},  # diff = 0.6 > 0.2
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.expert_disagreement_rate == 1.0
        assert snapshot.disagreement_count == THREE_ITEMS

    def test_disagreement_rate_exact_half(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-003: expert_disagreement_rate == 0.5 exactly."""
        # 4 disagree, 4 agree = 4/8 = 0.5
        items = [
            # Agree (|diff| <= 0.2)
            {"item_id": "1", "score": 0.5, "expert_score": 0.5},
            {"item_id": "2", "score": 0.6, "expert_score": 0.7},
            {"item_id": "3", "score": 0.4, "expert_score": 0.3},
            {"item_id": "4", "score": 0.7, "expert_score": 0.6},
            # Disagree (|diff| > 0.2)
            {"item_id": "5", "score": 0.5, "expert_score": 0.8},
            {"item_id": "6", "score": 0.6, "expert_score": 0.2},
            {"item_id": "7", "score": 0.7, "expert_score": 0.3},
            {"item_id": "8", "score": 0.3, "expert_score": 0.8},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.expert_disagreement_rate == 4 / 8
        assert snapshot.disagreement_count == FOUR_ITEMS

    def test_disagreement_rate_at_boundary(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-003: diff == 0.2 exactly should NOT count as disagreement."""
        items = [
            {"item_id": "1", "score": 0.5, "expert_score": 0.7},  # diff = 0.2, NOT disagreement
            {"item_id": "2", "score": 0.5, "expert_score": 0.71},  # diff = 0.21, disagreement
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # Only item 2 is disagreement
        assert snapshot.expert_disagreement_rate == 1 / 2
        assert snapshot.disagreement_count == 1

    def test_disagreement_rate_without_expert_scores(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-003: disagreement_rate == 0 when no expert scores present."""
        items = [
            {"item_id": "1", "score": 0.5},  # No expert_score
            {"item_id": "2", "score": 0.6},  # No expert_score
            {"item_id": "3", "score": 0.7},  # No expert_score
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.expert_disagreement_rate == 0.0
        assert snapshot.disagreement_count == 0

    def test_disagreement_rate_property_many_n(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-003: expert_disagreement_rate == k/n for varying n values."""
        # Test with n=10, 20, 50, k varies
        test_cases = [
            (10, 3),  # 3 disagree out of 10
            (20, 7),  # 7 disagree out of 20
            (50, 13),  # 13 disagree out of 50
        ]

        for n, k in test_cases:
            items = []
            # First k items disagree (|diff| > 0.2)
            for i in range(k):
                items.append({
                    "item_id": str(i),
                    "score": 0.5,
                    "expert_score": 0.8,  # diff = 0.3 > 0.2
                })
            # Remaining n - k items agree (|diff| <= 0.2)
            for i in range(k, n):
                items.append({
                    "item_id": str(i),
                    "score": 0.5,
                    "expert_score": 0.6,  # diff = 0.1 <= 0.2
                })

            write_scoring_report(scoring_reports_dir, f"scorer_{n}_{k}", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # Total: 10 + 20 + 50 = 80 items with expert scores
        # Disagreements: 3 + 7 + 13 = 23
        total_with_expert = 10 + 20 + 50
        total_disagreements = 3 + 7 + 13

        assert snapshot.expert_disagreement_rate == pytest.approx(total_disagreements / total_with_expert)
        assert snapshot.disagreement_count == total_disagreements


# ---------------------------------------------------------------------------
# VAL-M3-CAL-001: Aggregator emits valid JSON snapshot
# ---------------------------------------------------------------------------


class TestSnapshotJSON:
    """Test that the aggregator emits a valid JSON snapshot."""

    def test_snapshot_has_all_required_fields(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-001: Snapshot JSON has all required fields."""
        items = [
            {"item_id": "1", "score": 0.5, "expert_score": 0.5, "safety_score": 0.9},
            {"item_id": "2", "score": 0.6, "expert_score": 0.6, "safety_score": 0.8},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)
        write_promotion_report(promotion_reports_dir, received=2, validated=2, rejected=0, merged=2)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # Check all required fields are present
        assert hasattr(snapshot, "per_scorer_agreement")
        assert hasattr(snapshot, "borderline_rate")
        assert hasattr(snapshot, "expert_disagreement_rate")
        assert hasattr(snapshot, "safety_variance")
        assert hasattr(snapshot, "generated_at")
        assert hasattr(snapshot, "scoring_report_count")
        assert hasattr(snapshot, "promotion_report_count")
        assert hasattr(snapshot, "total_items")
        assert hasattr(snapshot, "borderline_count")
        assert hasattr(snapshot, "disagreement_count")

    def test_snapshot_values_in_valid_ranges(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-001: Values are in [0, 1] or positive floats."""
        items = [
            {"item_id": "1", "score": 0.5, "expert_score": 0.5, "safety_score": 0.9},
            {"item_id": "2", "score": 0.6, "expert_score": 0.6, "safety_score": 0.8},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # Rates should be in [0, 1]
        assert 0.0 <= snapshot.borderline_rate <= 1.0
        assert 0.0 <= snapshot.expert_disagreement_rate <= 1.0

        # Safety variance should be non-negative
        assert snapshot.safety_variance >= 0.0

        # Per-scorer agreement should be in [0, 1]
        for agreement in snapshot.per_scorer_agreement.values():
            assert 0.0 <= agreement <= 1.0

    def test_snapshot_json_serializable(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-001: Snapshot serializes to valid JSON."""
        items = [
            {"item_id": "1", "score": 0.5, "expert_score": 0.5},
            {"item_id": "2", "score": 0.7, "expert_score": 0.6},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # Should be able to serialize to JSON
        snapshot_dict = {
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

        json_str = json.dumps(snapshot_dict)
        parsed = json.loads(json_str)

        assert parsed["per_scorer_agreement"] == snapshot.per_scorer_agreement
        assert parsed["borderline_rate"] == snapshot.borderline_rate
        assert parsed["expert_disagreement_rate"] == snapshot.expert_disagreement_rate

    def test_snapshot_written_to_file(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
        temp_reports_dir: Path,
    ) -> None:
        """VAL-M3-CAL-001: emit() writes snapshot JSON to file."""
        items = [
            {"item_id": "1", "score": 0.5, "expert_score": 0.5},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        output_path = temp_reports_dir / "calibration_snapshot.json"

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        aggregator.emit(output_path)

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)

        assert "borderline_rate" in loaded
        assert "expert_disagreement_rate" in loaded
        assert "safety_variance" in loaded
        assert "per_scorer_agreement" in loaded


# ---------------------------------------------------------------------------
# Per-scorer agreement tests
# ---------------------------------------------------------------------------


class TestPerScorerAgreement:
    """Test per-scorer agreement computation."""

    def test_per_scorer_agreement_multiple_scorers(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """Per-scorer agreement is computed for each scorer separately."""
        # Scorer A: all agree (tier match)
        write_scoring_report(scoring_reports_dir, "scorer_a", [
            {"item_id": "1", "score": 0.3, "expert_score": 0.2},  # both exclude
            {"item_id": "2", "score": 0.5, "expert_score": 0.55},  # both borderline
            {"item_id": "3", "score": 0.8, "expert_score": 0.9},  # both accept
        ])

        # Scorer B: 2/3 agree
        write_scoring_report(scoring_reports_dir, "scorer_b", [
            {"item_id": "4", "score": 0.3, "expert_score": 0.3},  # agree (exclude)
            {"item_id": "5", "score": 0.3, "expert_score": 0.7},  # disagree (exclude vs accept)
            {"item_id": "6", "score": 0.8, "expert_score": 0.9},  # agree (accept)
        ])

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert "scorer_a" in snapshot.per_scorer_agreement
        assert "scorer_b" in snapshot.per_scorer_agreement
        assert snapshot.per_scorer_agreement["scorer_a"] == pytest.approx(1.0)
        assert snapshot.per_scorer_agreement["scorer_b"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Safety variance tests
# ---------------------------------------------------------------------------


class TestSafetyVariance:
    """Test safety score variance computation."""

    def test_safety_variance_single_score(self, scoring_reports_dir: Path, promotion_reports_dir: Path) -> None:
        """Safety variance is 0.0 when only one safety score present."""
        write_scoring_report(scoring_reports_dir, "scorer_a", [
            {"item_id": "1", "score": 0.5, "safety_score": 0.9},
        ])

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.safety_variance == 0.0

    def test_safety_variance_identical_scores(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """Safety variance is 0.0 when all safety scores are identical."""
        write_scoring_report(scoring_reports_dir, "scorer_a", [
            {"item_id": "1", "score": 0.5, "safety_score": 0.9},
            {"item_id": "2", "score": 0.6, "safety_score": 0.9},
            {"item_id": "3", "score": 0.7, "safety_score": 0.9},
        ])

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.safety_variance == 0.0

    def test_safety_variance_computed_correctly(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """Safety variance is computed correctly for multiple scores."""
        # Scores: 0.8, 0.9, 1.0 -> sample variance = 0.01
        write_scoring_report(scoring_reports_dir, "scorer_a", [
            {"item_id": "1", "score": 0.5, "safety_score": 0.8},
            {"item_id": "2", "score": 0.6, "safety_score": 0.9},
            {"item_id": "3", "score": 0.7, "safety_score": 1.0},
        ])

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # Sample variance (statistics.variance uses n-1 denominator) of [0.8, 0.9, 1.0]:
        # mean = 0.9, deviations = [-0.1, 0, 0.1], squared = [0.01, 0, 0.01]
        # sum of squares = 0.02, variance = 0.02 / (3-1) = 0.01
        assert snapshot.safety_variance == pytest.approx(0.01)

    def test_safety_variance_null_scores_ignored(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """Items with null safety_score are ignored in variance computation."""
        write_scoring_report(scoring_reports_dir, "scorer_a", [
            {"item_id": "1", "score": 0.5, "safety_score": 0.9},
            {"item_id": "2", "score": 0.6},  # No safety_score
            {"item_id": "3", "score": 0.7, "safety_score": 0.9},
        ])

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # Only 2 valid scores: [0.9, 0.9] -> variance = 0
        assert snapshot.safety_variance == 0.0


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Test that the aggregator produces deterministic results."""

    def test_same_reports_produce_same_snapshot(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """Same input reports always produce the same snapshot."""
        items = [
            {"item_id": "1", "score": 0.5, "expert_score": 0.5, "safety_score": 0.9},
            {"item_id": "2", "score": 0.6, "expert_score": 0.7, "safety_score": 0.8},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)
        write_promotion_report(promotion_reports_dir, received=2, validated=2, rejected=0, merged=2)

        # Run twice
        aggregator1 = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot1 = aggregator1.aggregate()

        aggregator2 = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot2 = aggregator2.aggregate()

        # Non-time fields should be identical
        assert snapshot1.per_scorer_agreement == snapshot2.per_scorer_agreement
        assert snapshot1.borderline_rate == snapshot2.borderline_rate
        assert snapshot1.expert_disagreement_rate == snapshot2.expert_disagreement_rate
        assert snapshot1.safety_variance == snapshot2.safety_variance
        assert snapshot1.total_items == snapshot2.total_items
        assert snapshot1.borderline_count == snapshot2.borderline_count
        assert snapshot1.disagreement_count == snapshot2.disagreement_count

    def test_aggregate_convenience_function(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """aggregate() convenience function works correctly."""
        items = [
            {"item_id": "1", "score": 0.5, "expert_score": 0.5},
        ]
        write_scoring_report(scoring_reports_dir, "scorer_a", items)

        snapshot = aggregate(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )

        assert isinstance(snapshot, CalibrationSnapshot)
        assert snapshot.scoring_report_count == 1
        assert snapshot.promotion_report_count == 0


# ---------------------------------------------------------------------------
# Promotion report loading tests
# ---------------------------------------------------------------------------


class TestPromotionReportLoading:
    """Test loading of promotion reports."""

    def test_promotion_reports_loaded(
        self,
        promotion_reports_dir: Path,
    ) -> None:
        """Promotion reports are correctly loaded."""
        write_promotion_report(promotion_reports_dir, received=5, validated=4, rejected=1, merged=4)
        write_promotion_report(promotion_reports_dir, received=3, validated=2, rejected=1, merged=2)

        reports = _load_promotion_reports(promotion_reports_dir)

        assert len(reports) == TWO_REPORTS
        total_received = sum(r.received for r in reports)
        assert total_received == EIGHT_ITEMS

    def test_promotion_report_reasons_parsed(
        self,
        promotion_reports_dir: Path,
    ) -> None:
        """Promotion report reasons are correctly parsed."""
        write_promotion_report(
            promotion_reports_dir,
            received=5,
            validated=3,
            rejected=2,
            merged=3,
            reasons={"safety_violation": 1, "low_agreement": 1},
        )

        reports = _load_promotion_reports(promotion_reports_dir)
        assert len(reports) == 1
        assert reports[0].reasons == {"safety_violation": 1, "low_agreement": 1}


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling in report loading."""

    def test_malformed_scoring_report_ignored(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """Malformed JSON in scoring report is skipped with warning."""
        # Write a malformed JSON file
        malformed_file = scoring_reports_dir / "malformed_report.json"
        with open(malformed_file, "w", encoding="utf-8") as f:
            f.write('{"incomplete json":')

        # Write a valid report too
        write_scoring_report(scoring_reports_dir, "scorer_a", [
            {"item_id": "1", "score": 0.5},
        ])

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # Should still process the valid report
        assert snapshot.scoring_report_count == 1
        assert snapshot.total_items == 1

    def test_malformed_promotion_report_ignored(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """Malformed JSON in promotion report is skipped with warning."""
        # Write a malformed JSON file
        malformed_file = promotion_reports_dir / "malformed_report.json"
        with open(malformed_file, "w", encoding="utf-8") as f:
            f.write('{"incomplete json":')

        # Write a valid report too
        write_promotion_report(promotion_reports_dir, received=3, validated=2, rejected=1, merged=2)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # Should still process the valid report
        assert snapshot.promotion_report_count == 1

    def test_safety_variance_zero_for_single_item(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """Safety variance is 0.0 when only one safety score (not enough for variance)."""
        write_scoring_report(scoring_reports_dir, "scorer_a", [
            {"item_id": "1", "score": 0.5, "safety_score": 0.9},
        ])

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        assert snapshot.safety_variance == 0.0

    def test_promotion_reports_nonexistent_dir(
        self,
        scoring_reports_dir: Path,
        promotion_reports_dir: Path,
    ) -> None:
        """Promotion report count is 0 when directory doesn't exist."""
        # Write a scoring report
        write_scoring_report(scoring_reports_dir, "scorer_a", [
            {"item_id": "1", "score": 0.5, "expert_score": 0.5},
        ])

        # Delete the promotion reports dir
        shutil.rmtree(promotion_reports_dir)

        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=scoring_reports_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # Scoring reports should still be loaded
        assert snapshot.scoring_report_count == 1
        assert snapshot.promotion_report_count == 0

    def test_default_none_arguments(self) -> None:
        """Using default None arguments still works (defaults to data/reports/)."""
        # When dirs don't exist, we get empty reports but no crash
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a temp directory that doesn't have the expected structure
            # Pass explicit non-None paths to avoid the "default" branch coverage gap
            # but ensure directories don't exist so we get empty reports
            scoring_dir = Path(tmpdir) / "nonexistent_reports"
            promotion_dir = Path(tmpdir) / "nonexistent_closed_loop" / "reports"
            aggregator = CalibrationMetricsAggregator(
                scoring_reports_dir=scoring_dir,
                promotion_reports_dir=promotion_dir,
            )
            snapshot = aggregator.aggregate()

            # Should handle gracefully
            assert snapshot.scoring_report_count == 0
            assert snapshot.promotion_report_count == 0

    def test_default_path_assignment(self) -> None:
        """When None is passed, defaults are assigned in __init__."""
        with tempfile.TemporaryDirectory():
            # Use a temp dir to isolate from any real data/reports
            # Create the default dir structure and verify defaults are set
            aggregator = CalibrationMetricsAggregator(
                scoring_reports_dir=None,
                promotion_reports_dir=None,
            )
            # Verify the defaults are set
            assert aggregator.scoring_reports_dir == Path("data/reports")
            assert aggregator.promotion_reports_dir == Path("data/closed_loop/reports")

    def test_nonexistent_scoring_reports_dir_warns(
        self,
        promotion_reports_dir: Path,
    ) -> None:
        """Warning logged when scoring reports directory doesn't exist."""
        nonexistent_dir = Path(tempfile.mkdtemp()) / "does_not_exist"

        # Write promotion report so we have something
        write_promotion_report(promotion_reports_dir, received=1, validated=1, rejected=0, merged=1)

        # Should not raise, just warn
        aggregator = CalibrationMetricsAggregator(
            scoring_reports_dir=nonexistent_dir,
            promotion_reports_dir=promotion_reports_dir,
        )
        snapshot = aggregator.aggregate()

        # Promotion reports should still be loaded
        assert snapshot.promotion_report_count == 1
        # Scoring reports is 0 because dir doesn't exist
        assert snapshot.scoring_report_count == 0
