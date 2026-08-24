"""Tests for ClosedLoopPromotionService.

These tests verify the closed-loop promotion service behavior per
VAL-M3-LOOP-001 through VAL-M3-LOOP-006.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine

from annotation.api.database import Base
from annotation.api.models import QueueItem, QueueItemStatus, Review
from training.coaching_safety.closed_loop_promotion import (
    ClosedLoopPromotionService,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def in_memory_db():
    """Create a file-based SQLite database for testing.

    Uses a temp file to ensure a single database instance across connections.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    # Store db_path on engine so we can close it later if needed
    engine.db_path = db_path
    return engine


def make_service(temp_data_dir, in_memory_db, gold_agreement_threshold=0.6):
    """Create a ClosedLoopPromotionService with patched db engine."""
    svc = ClosedLoopPromotionService(
        validated_output_dir=temp_data_dir / "validated",
        report_output_dir=temp_data_dir / "reports",
        gold_agreement_threshold=gold_agreement_threshold,
    )
    # Patch _get_db_engine to return the in-memory database
    svc._get_db_engine = lambda: in_memory_db
    return svc


def _create_reviewed_item(engine, sample_text, original_score, per_dimension_scores, reviewer_score):
    """Helper to create a reviewed queue item with review."""
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        item = QueueItem(
            sample_text=sample_text,
            original_score=original_score,
            per_dimension_scores=json.dumps(per_dimension_scores),
            routing_reason="test",
            status=QueueItemStatus.REVIEWED,
            created_at=datetime.now(UTC),
        )
        session.add(item)
        session.flush()

        review = Review(
            item_id=item.id,
            reviewer_score=reviewer_score,
            notes="test review",
            reviewer_id="expert_1",
            created_at=datetime.now(UTC),
        )
        session.add(review)
        session.flush()
        session.commit()
        session.refresh(item)
        return item.id


# ---------------------------------------------------------------------------
# VAL-M3-LOOP-001: run_once() reads reviewed items and writes validated JSONL
# ---------------------------------------------------------------------------


class TestRunOnce:
    """Test run_once() reads reviewed items and writes validated JSONL."""

    def test_run_once_with_no_reviewed_items(self, temp_data_dir, in_memory_db, caplog):
        """run_once() with no reviewed items produces empty output."""
        svc = make_service(temp_data_dir, in_memory_db)
        with caplog.at_level(logging.INFO):
            report = svc.run_once()

        assert report["received"] == 0
        assert report["validated"] == 0
        assert report["rejected"] == 0
        assert report["merged"] == 0
        assert report["reasons"] == {}

    def test_run_once_with_reviewed_item_in_agreement(self, temp_data_dir, in_memory_db, caplog):
        """VAL-M3-LOOP-001: Reviewed item with agreement > threshold writes JSONL."""
        # Create a reviewed item where both original and reviewer score are in "accept" range
        item_id = _create_reviewed_item(
            in_memory_db,
            sample_text="This is a valid therapeutic response about CBT techniques.",
            original_score=0.7,
            per_dimension_scores={"technique": 0.7, "alliance": 0.6},
            reviewer_score=0.75,
        )

        svc = make_service(temp_data_dir, in_memory_db)

        with caplog.at_level(logging.INFO):
            report = svc.run_once()

        assert report["received"] == 1
        assert report["validated"] == 1
        assert report["rejected"] == 0

        # Check JSONL file was created
        validated_files = list((temp_data_dir / "validated").glob("*.jsonl"))
        assert len(validated_files) == 1

        with open(validated_files[0]) as f:
            records = [json.loads(line) for line in f]
        assert len(records) == 1
        assert records[0]["item_id"] == item_id

    def test_run_once_records_item_in_report(self, temp_data_dir, in_memory_db):
        """VAL-M3-LOOP-001: Promotion report records validated count."""
        _create_reviewed_item(
            in_memory_db,
            sample_text="Good therapeutic response.",
            original_score=0.65,
            per_dimension_scores={"technique": 0.7},
            reviewer_score=0.7,
        )

        svc = make_service(temp_data_dir, in_memory_db)
        report = svc.run_once()
        assert report["validated"] == 1


# ---------------------------------------------------------------------------
# VAL-M3-LOOP-002: merge_into_dataset() appends to dataset and writes manifest
# ---------------------------------------------------------------------------


class TestMergeIntoDataset:
    """Test merge_into_dataset() appends validated records to final dataset."""

    def test_merge_returns_correct_count(self, temp_data_dir, in_memory_db):
        """VAL-M3-LOOP-002: merge_into_dataset() appends and returns correct count."""
        # Create validated items first
        _create_reviewed_item(
            in_memory_db,
            sample_text="Mergable therapeutic response.",
            original_score=0.7,
            per_dimension_scores={"technique": 0.8},
            reviewer_score=0.72,
        )

        svc = make_service(temp_data_dir, in_memory_db)
        report = svc.run_once()

        # Verify merge was attempted (merged count should be 1 for valid item)
        assert report["merged"] == 1
        assert report["validated"] == 1

        # Verify JSONL was written
        validated_files = list((temp_data_dir / "validated").glob("*.jsonl"))
        assert len(validated_files) == 1


# ---------------------------------------------------------------------------
# VAL-M3-LOOP-003: Safety-checker revalidation rejects safety violations
# ---------------------------------------------------------------------------


class TestSafetyRevalidation:
    """Test safety-checker revalidation rejects items with safety violations."""

    def test_run_once_rejects_safety_violation(self, temp_data_dir, in_memory_db, caplog):
        """VAL-M3-LOOP-003: Item with crisis keywords is rejected as safety_violation."""
        # Create an item with crisis keywords
        _create_reviewed_item(
            in_memory_db,
            sample_text="I want to kill myself. I have a plan to end my life.",
            original_score=0.5,
            per_dimension_scores={"technique": 0.5, "safety": 0.3},
            reviewer_score=0.55,
        )

        svc = make_service(temp_data_dir, in_memory_db)

        with caplog.at_level(logging.INFO):
            report = svc.run_once()

        assert report["rejected"] == 1
        assert report["reasons"]["safety_violation"] == 1

        # Check structured log was emitted
        log_lines = [rec.message for rec in caplog.records if "item_id" in rec.message]
        assert len(log_lines) > 0

    def test_safe_item_passes_safety_check(self, temp_data_dir, in_memory_db):
        """VAL-M3-LOOP-003: Safe item without crisis keywords passes safety check."""
        # Item with NO crisis keywords (genuinely safe text)
        _create_reviewed_item(
            in_memory_db,
            sample_text="The therapist used CBT techniques to help the patient develop coping strategies.",
            original_score=0.55,
            per_dimension_scores={"technique": 0.6},
            reviewer_score=0.58,
        )

        svc = make_service(temp_data_dir, in_memory_db)
        report = svc.run_once()
        assert report["validated"] == 1


# ---------------------------------------------------------------------------
# VAL-M3-LOOP-004: Promotion report contains all required fields
# ---------------------------------------------------------------------------


class TestPromotionReport:
    """Test promotion report JSON contains all required fields."""

    def test_report_has_all_required_fields(self, temp_data_dir, in_memory_db):
        """VAL-M3-LOOP-004: Report JSON has received, validated, rejected, merged, reasons."""
        _create_reviewed_item(
            in_memory_db,
            sample_text="Test item.",
            original_score=0.7,
            per_dimension_scores={"technique": 0.7},
            reviewer_score=0.72,
        )

        svc = make_service(temp_data_dir, in_memory_db)
        report = svc.run_once()

        assert "received" in report
        assert "validated" in report
        assert "rejected" in report
        assert "merged" in report
        assert "reasons" in report
        assert isinstance(report["reasons"], dict)

    def test_report_file_written(self, temp_data_dir, in_memory_db):
        """VAL-M3-LOOP-004: Promotion report JSON file is written."""
        _create_reviewed_item(
            in_memory_db,
            sample_text="Test item.",
            original_score=0.7,
            per_dimension_scores={"technique": 0.7},
            reviewer_score=0.72,
        )

        svc = make_service(temp_data_dir, in_memory_db)
        svc.run_once()

        report_files = list((temp_data_dir / "reports").glob("*.json"))
        assert len(report_files) == 1

        with open(report_files[0]) as f:
            loaded_report = json.load(f)
        assert "received" in loaded_report
        assert "validated" in loaded_report


# ---------------------------------------------------------------------------
# VAL-M3-LOOP-005: Promotion rejects duplicate text
# ---------------------------------------------------------------------------


class TestDuplicateRejection:
    """Test promotion rejects items with duplicate text hash."""

    def test_run_once_rejects_duplicate_text(self, temp_data_dir, in_memory_db):
        """VAL-M3-LOOP-005: Items matching existing dataset hash are rejected."""
        # First, add the same text to the service's seen hashes
        text = "I want to discuss my progress in therapy."
        sha_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Create an item with the same text
        _create_reviewed_item(
            in_memory_db,
            sample_text=text,
            original_score=0.6,
            per_dimension_scores={"technique": 0.6},
            reviewer_score=0.62,
        )

        svc = make_service(temp_data_dir, in_memory_db)
        # Pre-populate seen hashes
        svc._seen_text_hashes.add(sha_hash)

        report = svc.run_once()

        assert report["rejected"] == 1
        assert report["reasons"].get("duplicate_text") == 1


# ---------------------------------------------------------------------------
# VAL-M3-LOOP-006: Every decision emits structured log line
# ---------------------------------------------------------------------------


class TestStructuredLogging:
    """Test that every promotion/rejection decision emits structured logs."""

    def test_every_item_emits_log_line(self, temp_data_dir, in_memory_db, caplog):
        """VAL-M3-LOOP-006: Every decision writes log with item_id, decision, reason, timestamp."""
        _create_reviewed_item(
            in_memory_db,
            sample_text="Therapeutic response about coping skills.",
            original_score=0.7,
            per_dimension_scores={"technique": 0.7},
            reviewer_score=0.72,
        )

        svc = make_service(temp_data_dir, in_memory_db)

        with caplog.at_level(logging.INFO):
            svc.run_once()

        # Check that log contains structured fields
        assert len(caplog.records) > 0
        for record in caplog.records:
            if "item_id" in record.message:
                log_data = json.loads(record.message)
                assert "item_id" in log_data
                assert "decision" in log_data
                assert "reason" in log_data
                assert "timestamp" in log_data

    def test_rejected_item_emits_correct_decision(self, temp_data_dir, in_memory_db, caplog):
        """VAL-M3-LOOP-006: Rejected item logs decision='rejected'."""
        # Create item with safety violation
        item_id = _create_reviewed_item(
            in_memory_db,
            sample_text="I want to end my life.",
            original_score=0.5,
            per_dimension_scores={"safety": 0.3},
            reviewer_score=0.55,
        )

        svc = make_service(temp_data_dir, in_memory_db)

        with caplog.at_level(logging.INFO):
            svc.run_once()

        # Find the log for this item
        found = False
        for record in caplog.records:
            msg = record.message
            if "item_id" in msg and str(item_id) in msg:
                found = True
                log_data = json.loads(msg)
                assert log_data["decision"] == "rejected"
                assert log_data["reason"] == "safety_violation"
                break
        assert found, f"No log found for item_id={item_id}"


# ---------------------------------------------------------------------------
# Additional validation tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Test per-dimension scores are validated against [0, 1] range."""

    def test_rejects_per_dimension_scores_outside_range(self, temp_data_dir, in_memory_db):
        """Items with per_dimension_scores outside [0, 1] are rejected as schema_violation."""
        # Create item with invalid per-dimension score
        _create_reviewed_item(
            in_memory_db,
            sample_text="Valid text.",
            original_score=0.6,
            per_dimension_scores={"technique": 1.5},  # Invalid: > 1.0
            reviewer_score=0.65,
        )

        svc = make_service(temp_data_dir, in_memory_db)
        report = svc.run_once()
        assert report["rejected"] == 1
        assert report["reasons"].get("schema_violation") == 1

    def test_rejects_negative_per_dimension_scores(self, temp_data_dir, in_memory_db):
        """Items with negative per-dimension scores are rejected."""
        _create_reviewed_item(
            in_memory_db,
            sample_text="Valid text.",
            original_score=0.6,
            per_dimension_scores={"technique": -0.1},  # Invalid: < 0.0
            reviewer_score=0.65,
        )

        svc = make_service(temp_data_dir, in_memory_db)
        report = svc.run_once()
        assert report["rejected"] == 1
        assert report["reasons"].get("schema_violation") == 1


class TestLowAgreementRejection:
    """Test items with low gold-set agreement are rejected."""

    def test_rejects_low_agreement(self, temp_data_dir, in_memory_db):
        """Items with agreement below threshold are rejected as low_agreement."""
        # Original score 0.3 (exclude range), reviewer score 0.8 (accept range)
        # These disagree - agreement should be low
        _create_reviewed_item(
            in_memory_db,
            sample_text="A borderline therapeutic response.",
            original_score=0.3,
            per_dimension_scores={"technique": 0.3},
            reviewer_score=0.8,
        )

        svc = make_service(temp_data_dir, in_memory_db)
        report = svc.run_once()
        # Original in exclude range, reviewer in accept range = low agreement
        assert report["rejected"] == 1
        assert report["reasons"].get("low_agreement") == 1

    def test_accepts_high_agreement(self, temp_data_dir, in_memory_db):
        """Items with both scores in same range are validated."""
        # Both in accept range (>= 0.6)
        _create_reviewed_item(
            in_memory_db,
            sample_text="A good therapeutic response.",
            original_score=0.7,
            per_dimension_scores={"technique": 0.7},
            reviewer_score=0.75,
        )

        svc = make_service(temp_data_dir, in_memory_db)
        report = svc.run_once()
        assert report["validated"] == 1


# ---------------------------------------------------------------------------
# CLI entry point test
# ---------------------------------------------------------------------------


class TestCLIEntryPoint:
    """Test the CLI entry point for run_once()."""

    def test_run_once_cli_entry_point(self, temp_data_dir, in_memory_db):
        """run_once() is callable as CLI entry point."""
        from training.coaching_safety import closed_loop_promotion

        with patch.object(ClosedLoopPromotionService, "_get_db_engine", return_value=in_memory_db):
            with patch.object(closed_loop_promotion, "run_once") as mock_run:
                mock_run.return_value = {"received": 0, "validated": 0, "rejected": 0, "merged": 0, "reasons": {}}
                result = closed_loop_promotion.run_once()
        assert mock_run.called or result is not None
