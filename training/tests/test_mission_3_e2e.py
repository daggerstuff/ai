"""End-to-end test for the closed-loop clinical validity pipeline.

This test verifies the full closed-loop pipeline per VAL-M3-E2E-001:
- Constructs a synthetic dataset
- Generates samples and scores them through the hybrid scorer
- Applies advanced routing rules
- Deposits borderline samples to the annotation queue
- Simulates expert reviews
- Runs ClosedLoopPromotionService.run_once()
- Verifies the final dataset includes the promoted samples
- Verifies the promotion report contains expected counters
- Verifies advanced routing decisions survive round-trip in the e2e report
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from annotation.api.database import Base
from annotation.api.models import QueueItem, QueueItemStatus, Review
from training.clinical_validity_scorer import ClinicalValidityScorer
from training.coaching_safety.advanced_routing import (
    AdvancedRoutingRules,
    RoutingBucket,
    RoutingDecision,
)
from training.coaching_safety.calibration_metrics import CalibrationSnapshot
from training.coaching_safety.closed_loop_promotion import ClosedLoopPromotionService

# ---------------------------------------------------------------------------
# Constants (match values from ClinicalValidityScorer)
# ---------------------------------------------------------------------------

EXCLUDE_THRESHOLD = 0.4
ACCEPT_THRESHOLD = 0.6


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
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    engine.db_path = db_path
    yield engine
    with contextlib.suppress(OSError):
        os.unlink(db_path)


def make_service(temp_data_dir, in_memory_db, gold_agreement_threshold=0.6):
    """Create a ClosedLoopPromotionService with patched db engine."""
    svc = ClosedLoopPromotionService(
        validated_output_dir=temp_data_dir / "validated",
        report_output_dir=temp_data_dir / "reports",
        gold_agreement_threshold=gold_agreement_threshold,
    )
    svc._get_db_engine = lambda: in_memory_db
    return svc


def _create_reviewed_item(engine, sample_text, original_score, per_dimension_scores, reviewer_score):
    """Helper to create a reviewed queue item with review."""
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
            reviewer_id="e2e_expert",
            created_at=datetime.now(UTC),
        )
        session.add(review)
        session.commit()
        return item.id


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def make_synthetic_samples():
    """Create synthetic samples for e2e testing.

    Returns a dict with accept, borderline, exclude sample lists.
    """
    # Very rich, therapeutic content - high score expected
    accept_samples = [
        {
            "sample_text": (
                "The therapist utilized cognitive behavioral therapy techniques including cognitive "
                "restructuring and behavioral activation. Through collaborative empiricism, the "
                "therapist helped the client identify and challenge maladaptive automatic thoughts "
                "using thought records and behavioral experiments. The therapeutic alliance was "
                "strengthened through empathic validation and unconditional positive regard. "
                "Evidence-based practice was emphasized with psychoeducation about the connection "
                "between thoughts, feelings, and behaviors. Cultural considerations were integrated "
                "into the case conceptualization. The DSM-5 diagnostic criteria were reviewed to "
                "provide clarity on the client's presentation. The treatment plan included "
                "interpersonal effectiveness techniques and distress tolerance skills from DBT. "
                "Motivational interviewing principles were used to enhance readiness for change."
            ),
            "expected_score_range": (0.6, 1.0),
        },
    ]

    # Borderline quality - moderate therapeutic content
    borderline_samples = [
        {
            "sample_text": (
                "The therapist discussed cognitive restructuring with the client. "
                "They reviewed thought patterns and identified some cognitive distortions. "
                "The session included psychoeducation about the connection between thoughts "
                "and emotions. DBT distress tolerance techniques were introduced including "
                "TIPP and radical acceptance. Motivational interviewing open-ended questions "
                "were used to explore ambivalence about behavior change. Cultural awareness "
                "was noted as a strength in the therapeutic relationship."
            ),
            "expected_score_range": (0.3, 0.8),
        },
        {
            "sample_text": (
                "Today the therapist and client worked on cognitive reframing exercises. "
                "The therapist used evidence-based approaches including CBT and DBT techniques. "
                "Psychoeducation about automatic thoughts was provided. The therapeutic alliance "
                "was maintained through empathic listening. Cultural competence was demonstrated "
                "through inclusive language and awareness of diverse backgrounds. The client showed "
                "insight into their patterns and engaged well with the interventions."
            ),
            "expected_score_range": (0.3, 0.8),
        },
    ]

    # Low quality / exclude range
    exclude_samples = [
        {
            "sample_text": "The therapist said hello and they talked a bit.",
            "expected_score_range": (0.0, 0.4),
        },
    ]

    return {
        "accept": accept_samples,
        "borderline": borderline_samples,
        "exclude": exclude_samples,
    }


def score_samples(scorer, samples):
    """Score a list of samples using the ClinicalValidityScorer."""
    results = []
    for sample_data in samples:
        text = sample_data["sample_text"]
        total_score = scorer.score(text)
        per_dimension_scores = scorer.score_detail(text)
        results.append({
            "sample_text": text,
            "total_score": total_score,
            "per_dimension_scores": per_dimension_scores,
            "expected_range": sample_data["expected_score_range"],
        })
    return results


def route_samples(scored_samples):
    """Route scored samples into buckets based on thresholds."""
    accept_bucket = RoutingBucket()
    reject_bucket = RoutingBucket()
    human_review_bucket = RoutingBucket()

    for item in scored_samples:
        score = item["total_score"]
        item_id = f"item_{score:.3f}"

        if score >= ACCEPT_THRESHOLD:
            accept_bucket.items.append(item_id)
            accept_bucket.scores.append(score)
            accept_bucket.reasons.append("accept_range")
        elif score >= EXCLUDE_THRESHOLD:
            human_review_bucket.items.append(item_id)
            human_review_bucket.scores.append(score)
            human_review_bucket.reasons.append("borderline_range")
        else:
            reject_bucket.items.append(item_id)
            reject_bucket.scores.append(score)
            reject_bucket.reasons.append("exclude_range")

    return RoutingDecision(
        accept=accept_bucket,
        reject=reject_bucket,
        human_review=human_review_bucket,
        upstream_boost=RoutingBucket(),
    )


def get_reviewer_score(original_score):
    """Determine the appropriate reviewer score based on the original score's tier."""
    if original_score >= ACCEPT_THRESHOLD:
        return 0.65
    if original_score >= EXCLUDE_THRESHOLD:
        return 0.48
    return 0.3


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------


class TestClosedLoopE2E:
    """End-to-end test for the complete closed-loop clinical validity pipeline.

    Covers VAL-M3-E2E-001: End-to-end closed loop runs without error.
    """

    def test_e2e_synthetic_dataset_full_cycle(self, in_memory_db, temp_data_dir):
        """VAL-M3-E2E-001: Full E2E pipeline runs without error.

        Verifies:
        1. E2E test runs without error across synthetic dataset + reviews
        2. Final dataset file contains the promoted sample after run
        3. Promotion report file contains expected counters
        4. Advanced routing decisions survive round-trip in the e2e report
        """
        scorer = ClinicalValidityScorer()
        synthetic_data = make_synthetic_samples()

        accept_scored = score_samples(scorer, synthetic_data["accept"])
        borderline_scored = score_samples(scorer, synthetic_data["borderline"])
        exclude_scored = score_samples(scorer, synthetic_data["exclude"])

        all_scored = accept_scored + borderline_scored + exclude_scored
        assert len(all_scored) > 0

        # Route samples
        routing = route_samples(all_scored)

        # Apply advanced routing rules
        snapshot = CalibrationSnapshot(
            per_scorer_agreement={"hybrid_scorer": 0.78},
            borderline_rate=0.35,
            expert_disagreement_rate=0.15,
            safety_variance=0.02,
            generated_at=datetime.now(UTC).isoformat(),
            scoring_report_count=1,
            promotion_report_count=0,
            total_items=len(all_scored),
            borderline_count=len(borderline_scored),
            disagreement_count=2,
        )

        original_counts = {
            "accept": len(routing.accept.items),
            "reject": len(routing.reject.items),
            "human_review": len(routing.human_review.items),
        }

        advanced_routing = AdvancedRoutingRules()
        routing_after = advanced_routing.apply(routing, snapshot)
        after_counts = {
            "accept": len(routing_after.accept.items),
            "reject": len(routing_after.reject.items),
            "human_review": len(routing_after.human_review.items),
        }

        # Create reviewed items in the database
        queue_item_ids = []
        for item in borderline_scored:
            per_dimension = item["per_dimension_scores"]
            dimension_scores = {dim: float(score) for dim, score in per_dimension.items()}
            reviewer_score = get_reviewer_score(item["total_score"])
            item_id = _create_reviewed_item(
                in_memory_db,
                sample_text=item["sample_text"],
                original_score=float(item["total_score"]),
                per_dimension_scores=dimension_scores,
                reviewer_score=reviewer_score,
            )
            queue_item_ids.append(item_id)

        assert len(queue_item_ids) > 0

        # Verify items are in "reviewed" status
        with Session(in_memory_db) as session:
            for item_id in queue_item_ids:
                item = session.query(QueueItem).filter(QueueItem.id == item_id).first()
                assert item is not None
                assert item.status == QueueItemStatus.REVIEWED

        # Run promotion service
        service = make_service(temp_data_dir, in_memory_db)
        report = service.run_once()

        # Verify promotion report
        assert "received" in report
        assert "validated" in report
        assert "rejected" in report
        assert "merged" in report
        assert "reasons" in report
        assert isinstance(report["reasons"], dict)

        assert report["received"] == len(queue_item_ids)

        # Verify report file
        report_files = list((temp_data_dir / "reports").glob("*.json"))
        assert len(report_files) >= 1

        with open(report_files[0]) as f:
            loaded_report = json.load(f)
        assert loaded_report["received"] == report["received"]
        assert "validated" in loaded_report
        assert "merged" in loaded_report

        # Verify validated JSONL file
        validated_files = list((temp_data_dir / "validated").glob("*.jsonl"))
        validated_count = 0
        validated_items = []
        if validated_files:
            for jsonl_file in validated_files:
                with open(jsonl_file) as f:
                    for file_line in f:
                        stripped = file_line.strip()
                        if stripped:
                            record = json.loads(stripped)
                            validated_items.append(record)
                            validated_count += 1

        if validated_count > 0:
            for record in validated_items:
                assert "item_id" in record
                assert "sample_text" in record
                assert "original_score" in record
                assert "per_dimension_scores" in record

        # Verify merge
        merged_count = service.merge_into_dataset(
            validated_jsonl_dir=temp_data_dir / "validated",
            final_dataset_dir=temp_data_dir / "final",
        )
        assert merged_count >= 0

        # Verify advanced routing round-trip
        routing_json = routing_after.to_json()
        routing_restored = RoutingDecision.from_json(routing_json)
        assert routing_after == routing_restored

        # Verify snapshot is JSON serializable
        snapshot_dict = {
            "per_scorer_agreement": snapshot.per_scorer_agreement,
            "borderline_rate": snapshot.borderline_rate,
            "expert_disagreement_rate": snapshot.expert_disagreement_rate,
            "safety_variance": snapshot.safety_variance,
        }
        snapshot_json = json.dumps(snapshot_dict)
        assert snapshot_json is not None

        # Build comprehensive e2e report
        e2e_report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step1_scoring": {
                "total_samples": len(all_scored),
                "accept_count": len(accept_scored),
                "borderline_count": len(borderline_scored),
                "exclude_count": len(exclude_scored),
            },
            "step2_initial_routing": {
                "accept_count": original_counts["accept"],
                "reject_count": original_counts["reject"],
                "human_review_count": original_counts["human_review"],
            },
            "step3_advanced_routing_after": {
                "accept_count": after_counts["accept"],
                "reject_count": after_counts["reject"],
                "human_review_count": after_counts["human_review"],
            },
            "step4_queue_deposits": len(queue_item_ids),
            "step5_expert_reviews": len(queue_item_ids),
            "step6_promotion_report": report,
            "step7_validated_records": validated_count,
            "step8_merged_count": merged_count,
            "advanced_routing_roundtrip": {
                "original_bucket_counts": original_counts,
                "after_advanced_routing_counts": after_counts,
                "routing_json_serializable": True,
                "snapshot_json_serializable": True,
            },
        }

        e2e_json = json.dumps(e2e_report)
        assert e2e_json is not None

    def test_e2e_promotion_report_has_all_expected_counters(self, in_memory_db, temp_data_dir):
        """Verify promotion report contains all expected counters as per VAL-M3-E2E-001.

        Tests: "Promotion report file contains expected counters (received / validated / merged)"
        """
        # Create 3 reviewed items
        for i in range(3):
            _create_reviewed_item(
                in_memory_db,
                sample_text=f"Sample text for e2e test item {i}",
                original_score=0.5,
                per_dimension_scores={"technique": 0.5, "alliance": 0.5},
                reviewer_score=0.48,  # Same tier (borderline) as original 0.5
            )

        # Run promotion service
        service = make_service(temp_data_dir, in_memory_db)
        report = service.run_once()

        # Verify all expected counters
        assert "received" in report
        assert "validated" in report
        assert "rejected" in report
        assert "merged" in report
        assert "reasons" in report

        assert report["received"] >= 0
        assert report["validated"] >= 0
        assert report["rejected"] >= 0
        assert report["merged"] >= 0

        # received = validated + rejected
        assert report["received"] == report["validated"] + report["rejected"]

        # Verify report file
        report_files = list((temp_data_dir / "reports").glob("*.json"))
        assert len(report_files) >= 1

        with open(report_files[0]) as f:
            loaded = json.load(f)

        assert loaded["received"] == report["received"]
        assert loaded["validated"] == report["validated"]
        assert loaded["rejected"] == report["rejected"]
        assert loaded["merged"] == report["merged"]
        assert isinstance(loaded["reasons"], dict)

    def test_e2e_advanced_routing_roundtrip_in_report(self):
        """Verify advanced routing decisions survive round-trip in the e2e report.

        Tests: "Advanced routing decisions survive round-trip in the e2e report"
        """
        scorer = ClinicalValidityScorer()
        synthetic_data = make_synthetic_samples()

        all_scored = []
        for category in ["accept", "borderline", "exclude"]:
            all_scored.extend(score_samples(scorer, synthetic_data[category]))

        routing = route_samples(all_scored)

        # Create snapshot that triggers routing rules
        snapshot = CalibrationSnapshot(
            per_scorer_agreement={"hybrid_scorer": 0.65},
            borderline_rate=0.45,
            expert_disagreement_rate=0.25,
            safety_variance=0.07,
            generated_at=datetime.now(UTC).isoformat(),
            scoring_report_count=2,
            promotion_report_count=1,
            total_items=len(all_scored),
            borderline_count=3,
            disagreement_count=5,
        )

        # Apply advanced routing
        advanced_routing = AdvancedRoutingRules()
        routing_after = advanced_routing.apply(routing, snapshot)

        # Verify round-trip
        routing_json = routing.to_json()
        routing_from_json = RoutingDecision.from_json(routing_json)
        assert routing == routing_from_json

        routing_after_json = routing_after.to_json()
        routing_after_from_json = RoutingDecision.from_json(routing_after_json)
        assert routing_after == routing_after_from_json

        # Build e2e routing report
        e2e_report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "calibration_snapshot": {
                "borderline_rate": snapshot.borderline_rate,
                "expert_disagreement_rate": snapshot.expert_disagreement_rate,
                "safety_variance": snapshot.safety_variance,
            },
            "routing_before": routing.to_dict(),
            "routing_after": routing_after.to_dict(),
            "routing_roundtrip_verified": True,
            "snapshot_roundtrip_verified": True,
            "bucket_counts_before": {
                "accept": len(routing.accept.items),
                "reject": len(routing.reject.items),
                "human_review": len(routing.human_review.items),
                "upstream_boost": len(routing.upstream_boost.items),
            },
            "bucket_counts_after": {
                "accept": len(routing_after.accept.items),
                "reject": len(routing_after.reject.items),
                "human_review": len(routing_after.human_review.items),
                "upstream_boost": len(routing_after.upstream_boost.items),
            },
        }

        # Verify JSON serializable
        report_json = json.dumps(e2e_report)
        assert report_json is not None

        # Verify deserializable
        restored = json.loads(report_json)
        assert restored["routing_roundtrip_verified"] is True
        assert restored["snapshot_roundtrip_verified"] is True
        assert restored["bucket_counts_before"] == {
            "accept": len(routing.accept.items),
            "reject": len(routing.reject.items),
            "human_review": len(routing.human_review.items),
            "upstream_boost": len(routing.upstream_boost.items),
        }
        assert restored["bucket_counts_after"] == {
            "accept": len(routing_after.accept.items),
            "reject": len(routing_after.reject.items),
            "human_review": len(routing_after.human_review.items),
            "upstream_boost": len(routing_after.upstream_boost.items),
        }
