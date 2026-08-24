"""Closed-loop promotion service for clinical validity enhancement pipeline.

This module implements the ClosedLoopPromotionService that reads reviewed queue
items from the annotation database, validates them against safety, schema,
duplicate, and agreement criteria, then writes validated items to JSONL files
and a promotion report to JSON.

Promotion flow:
1. Read items with status=REVIEWED from the database
2. Apply safety check: reject items with crisis keywords
3. Apply schema validation: reject items with per_dimension_scores outside [0, 1]
4. Apply duplicate detection: reject items whose text hash is already seen
5. Apply agreement check: reject items where original and reviewer scores
   fall in different quality tiers
6. Write validated items to JSONL in validated_output_dir
7. Write promotion report to JSON in report_output_dir
8. Emit structured JSON log lines for every decision
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from annotation.api.models import QueueItem, QueueItemStatus

logger = logging.getLogger("closed_loop_promotion")


# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

# Crisis keywords that trigger safety violation rejection
CRISIS_KEYWORDS = ("kill myself", "end my life")

# Per-dimension score boundaries
SCORE_MIN = 0.0
SCORE_MAX = 1.0

# Quality tier boundaries (derived from gold_agreement_threshold and clinical validity conventions)
EXCLUDE_THRESHOLD = 0.4
BORDERLINE_MAX = 0.6

# Default gold agreement threshold
DEFAULT_GOLD_AGREEMENT_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ValidatedRecord:
    """A validated record written to the JSONL output."""

    item_id: int
    sample_text: str
    original_score: float
    per_dimension_scores: dict[str, float]

    def to_dict(self) -> dict:
        """Serialize validated record to dictionary."""
        return {
            "item_id": self.item_id,
            "sample_text": self.sample_text,
            "original_score": self.original_score,
            "per_dimension_scores": self.per_dimension_scores,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ValidatedRecord:
        """Deserialize validated record from dictionary."""
        return cls(
            item_id=int(data["item_id"]),
            sample_text=str(data["sample_text"]),
            original_score=float(data["original_score"]),
            per_dimension_scores=dict(data.get("per_dimension_scores", {})),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> ValidatedRecord:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class ValidatedRecordDict(TypedDict):
    """TypedDict representation of ValidatedRecord for JSON serialization."""

    item_id: int
    sample_text: str
    original_score: float
    per_dimension_scores: dict[str, float]


@dataclass
class PromotionReport:
    """Promotion report containing counts and rejection reasons."""

    received: int = 0
    validated: int = 0
    rejected: int = 0
    merged: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize promotion report to dictionary."""
        return {
            "received": self.received,
            "validated": self.validated,
            "rejected": self.rejected,
            "merged": self.merged,
            "reasons": dict(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PromotionReport:
        """Deserialize promotion report from dictionary."""
        return cls(
            received=int(data.get("received", 0)),
            validated=int(data.get("validated", 0)),
            rejected=int(data.get("rejected", 0)),
            merged=int(data.get("merged", 0)),
            reasons=dict(data.get("reasons", {})),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> PromotionReport:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class PromotionReportDict(TypedDict):
    """TypedDict representation of PromotionReport for JSON serialization."""

    received: int
    validated: int
    rejected: int
    merged: int
    reasons: dict[str, int]


# ---------------------------------------------------------------------------
# Tier computation
# ---------------------------------------------------------------------------


def _get_tier(score: float, gold_agreement_threshold: float) -> str:
    """Determine the quality tier for a score.

    Args:
        score: The score value (0.0 to 1.0)
        gold_agreement_threshold: Threshold for the accept tier boundary.

    Returns:
        Tier name: "exclude", "borderline", or "accept"
    """
    if score < EXCLUDE_THRESHOLD:
        return "exclude"
    if score < gold_agreement_threshold:
        return "borderline"
    return "accept"


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def _check_safety(text: str) -> str | None:
    """Check if text contains crisis keywords.

    Args:
        text: The sample text to check.

    Returns:
        "safety_violation" if crisis keywords found, None otherwise.
    """
    text_lower = text.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return "safety_violation"
    return None


def _check_schema(per_dimension_scores: dict[str, float]) -> str | None:
    """Check if per-dimension scores are within [0, 1] range.

    Args:
        per_dimension_scores: Dictionary of dimension name to score.

    Returns:
        "schema_violation" if any score is outside [0, 1], None otherwise.
    """
    for _dim, score in per_dimension_scores.items():
        if score < SCORE_MIN or score > SCORE_MAX:
            return "schema_violation"
    return None


def _check_duplicate(text: str, seen_hashes: set[str]) -> str | None:
    """Check if text hash has already been seen.

    Args:
        text: The sample text to check.
        seen_hashes: Set of SHA-256 hex digest strings for previously seen texts.

    Returns:
        "duplicate_text" if the hash is already in seen_hashes, None otherwise.
    """
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_hash in seen_hashes:
        return "duplicate_text"
    return None


def _check_agreement(
    original_score: float,
    reviewer_score: float,
    gold_agreement_threshold: float,
) -> str | None:
    """Check if original and reviewer scores are in the same quality tier.

    Args:
        original_score: The original scorer's score.
        reviewer_score: The expert reviewer's score.
        gold_agreement_threshold: Threshold for tier boundaries.

    Returns:
        "low_agreement" if scores are in different tiers, None otherwise.
    """
    original_tier = _get_tier(original_score, gold_agreement_threshold)
    reviewer_tier = _get_tier(reviewer_score, gold_agreement_threshold)
    if original_tier != reviewer_tier:
        return "low_agreement"
    return None


# ---------------------------------------------------------------------------
# ClosedLoopPromotionService
# ---------------------------------------------------------------------------


class ClosedLoopPromotionService:
    """Service for closed-loop promotion of reviewed queue items.

    Reads items with status=REVIEWED from the annotation database, validates
    them against safety, schema, duplicate, and agreement criteria, writes
    validated items to JSONL, and produces a promotion report.

    All validation is deterministic: same input always produces the same output.
    """

    def __init__(
        self,
        validated_output_dir: Path,
        report_output_dir: Path,
        gold_agreement_threshold: float = DEFAULT_GOLD_AGREEMENT_THRESHOLD,
    ) -> None:
        """Initialize the ClosedLoopPromotionService.

        Args:
            validated_output_dir: Directory for validated JSONL output files.
            report_output_dir: Directory for promotion report JSON files.
            gold_agreement_threshold: Threshold for agreement tier boundaries.
                Default is 0.6.
        """
        self.validated_output_dir = Path(validated_output_dir)
        self.report_output_dir = Path(report_output_dir)
        self.gold_agreement_threshold = gold_agreement_threshold
        self._seen_text_hashes: set[str] = set()

    def _get_db_engine(self) -> object:
        """Get the SQLAlchemy database engine.

        Returns:
            A SQLAlchemy engine instance for querying the annotation database.

        By default, creates an engine from the DATABASE_URL environment variable.
        Tests patch this method to provide an in-memory database.
        """
        import os

        database_url = os.getenv("DATABASE_URL", "sqlite:///data/annotation.db")
        return create_engine(database_url)

    def _emit_log(self, item_id: int, decision: str, reason: str) -> None:
        """Emit a structured JSON log line for a promotion decision.

        Args:
            item_id: The queue item ID.
            decision: "validated" or "rejected".
            reason: The reason for the decision (e.g., "safety_violation",
                "validated" for successful promotion).
        """
        log_entry = json.dumps(
            {
                "item_id": item_id,
                "decision": decision,
                "reason": reason,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        logger.info(log_entry)

    def run_once(self) -> dict:
        """Run a single promotion cycle.

        Reads all items with status=REVIEWED from the database, validates each
        against safety, schema, duplicate, and agreement criteria, writes
        validated items to JSONL, and produces a promotion report.

        Returns:
            Dictionary with keys: received, validated, rejected, merged, reasons.
        """
        engine = self._get_db_engine()

        # Read reviewed items from the database
        reviewed_items: list[QueueItem] = []
        with Session(engine) as session:
            reviewed_items = list(session.query(QueueItem).filter(QueueItem.status == QueueItemStatus.REVIEWED).all())

        report = PromotionReport(
            received=len(reviewed_items),
            validated=0,
            rejected=0,
            merged=0,
            reasons={},
        )

        validated_records: list[ValidatedRecord] = []

        for item in reviewed_items:
            # Get the latest reviewer score from reviews
            reviewer_score = 0.0
            with Session(engine) as session:
                fresh_item = session.query(QueueItem).filter(QueueItem.id == item.id).first()
                if fresh_item and fresh_item.reviews:
                    reviewer_score = fresh_item.reviews[-1].reviewer_score

            # Parse per_dimension_scores from JSON string
            try:
                per_dimension_scores = json.loads(item.per_dimension_scores)
            except (json.JSONDecodeError, TypeError):
                per_dimension_scores = {}

            rejection_reason = None

            # Check 1: Safety
            if rejection_reason is None:
                rejection_reason = _check_safety(item.sample_text)

            # Check 2: Schema validation
            if rejection_reason is None:
                rejection_reason = _check_schema(per_dimension_scores)

            # Check 3: Duplicate detection
            if rejection_reason is None:
                rejection_reason = _check_duplicate(item.sample_text, self._seen_text_hashes)

            # Check 4: Agreement check
            if rejection_reason is None:
                rejection_reason = _check_agreement(item.original_score, reviewer_score, self.gold_agreement_threshold)

            if rejection_reason is not None:
                # Item is rejected
                report.rejected += 1
                report.reasons[rejection_reason] = report.reasons.get(rejection_reason, 0) + 1
                self._emit_log(item.id, "rejected", rejection_reason)
            else:
                # Item is validated
                report.validated += 1
                text_hash = hashlib.sha256(item.sample_text.encode("utf-8")).hexdigest()
                self._seen_text_hashes.add(text_hash)
                validated_records.append(
                    ValidatedRecord(
                        item_id=item.id,
                        sample_text=item.sample_text,
                        original_score=item.original_score,
                        per_dimension_scores=per_dimension_scores,
                    )
                )
                self._emit_log(item.id, "validated", "validated")

        # Write validated items to JSONL
        self.validated_output_dir.mkdir(parents=True, exist_ok=True)
        if validated_records:
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            jsonl_path = self.validated_output_dir / f"{timestamp}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for record in validated_records:
                    f.write(record.to_json() + "\n")

        # Write promotion report
        self.report_output_dir.mkdir(parents=True, exist_ok=True)
        report_timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        report_path = self.report_output_dir / f"{report_timestamp}.json"

        # merged = validated (validated items are immediately merged)
        report.merged = report.validated

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
            f.write("\n")

        logger.info(
            "Promotion cycle complete: received=%d, validated=%d, rejected=%d, merged=%d",
            report.received,
            report.validated,
            report.rejected,
            report.merged,
        )

        return report.to_dict()

    def merge_into_dataset(self, validated_jsonl_dir: Path, final_dataset_dir: Path) -> int:
        """Merge validated JSONL records into the final dataset.

        Reads all JSONL files from validated_jsonl_dir and appends their
        records to final_dataset_dir/dataset.jsonl.

        Args:
            validated_jsonl_dir: Directory containing validated JSONL files.
            final_dataset_dir: Directory for the final merged dataset.

        Returns:
            Number of records merged.
        """
        validated_jsonl_dir = Path(validated_jsonl_dir)
        final_dataset_dir = Path(final_dataset_dir)

        if not validated_jsonl_dir.exists():
            return 0

        # Collect all records from validated JSONL files
        records: list[str] = []
        for jsonl_file in sorted(validated_jsonl_dir.glob("*.jsonl")):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        records.append(stripped)

        if not records:
            return 0

        # Append to final dataset
        final_dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = final_dataset_dir / "dataset.jsonl"

        with open(dataset_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(record + "\n")

        logger.info("Merged %d records into %s", len(records), dataset_path)
        return len(records)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def run_once(
    validated_output_dir: Path | str | None = None,
    report_output_dir: Path | str | None = None,
    gold_agreement_threshold: float = DEFAULT_GOLD_AGREEMENT_THRESHOLD,
) -> dict:
    """Convenience function to run a single promotion cycle.

    Creates a ClosedLoopPromotionService instance and runs run_once().

    Args:
        validated_output_dir: Directory for validated JSONL output.
            Defaults to data/closed_loop/validated/
        report_output_dir: Directory for promotion report JSON.
            Defaults to data/closed_loop/reports/
        gold_agreement_threshold: Threshold for agreement tier boundaries.

    Returns:
        Dictionary with keys: received, validated, rejected, merged, reasons.
    """
    if validated_output_dir is None:
        validated_output_dir = Path("data/closed_loop/validated")
    if report_output_dir is None:
        report_output_dir = Path("data/closed_loop/reports")

    service = ClosedLoopPromotionService(
        validated_output_dir=Path(validated_output_dir),
        report_output_dir=Path(report_output_dir),
        gold_agreement_threshold=gold_agreement_threshold,
    )
    return service.run_once()
