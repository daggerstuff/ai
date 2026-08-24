"""Tests for annotation_api models."""

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from annotation.api.models import Base, QueueItem, QueueItemStatus, Review


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def session(engine):
    """Create a database session for testing."""
    with Session(engine) as session:
        yield session
        session.rollback()


class TestQueueItemModel:
    """Test QueueItem model."""

    def test_create_queue_item(self, session):
        """Test creating a queue item."""
        item = QueueItem(
            sample_text="Sample text for testing",
            original_score=0.5,
            per_dimension_scores=json.dumps({"technique": 0.6, "alliance": 0.4}),
            routing_reason="Borderline score",
            status=QueueItemStatus.PENDING,
            created_at=datetime.now(UTC),
        )
        session.add(item)
        session.flush()

        assert item.id is not None
        assert item.sample_text == "Sample text for testing"
        assert item.original_score == 0.5
        assert item.status == QueueItemStatus.PENDING
        assert isinstance(item.created_at, datetime)

    def test_queue_item_repr(self):
        """Test QueueItem __repr__ method."""
        item = QueueItem(
            id=1,
            sample_text="test",
            original_score=0.5,
            per_dimension_scores="{}",
            routing_reason="test",
            status=QueueItemStatus.PENDING,
            created_at=datetime.now(UTC),
        )
        repr_str = repr(item)
        assert "QueueItem" in repr_str
        assert "id=1" in repr_str
        assert "status=pending" in repr_str

    def test_queue_item_default_status(self, session):
        """Test that QueueItem defaults to PENDING status when persisted."""
        item = QueueItem(
            sample_text="test",
            original_score=0.5,
            per_dimension_scores="{}",
            routing_reason="test",
        )
        session.add(item)
        session.flush()
        assert item.status == QueueItemStatus.PENDING

    def test_queue_item_status_enum(self):
        """Test QueueItemStatus enum values."""
        assert QueueItemStatus.PENDING == "pending"
        assert QueueItemStatus.REVIEWED == "reviewed"
        assert QueueItemStatus.VALIDATED == "validated"
        assert QueueItemStatus.MERGED == "merged"
        assert len(list(QueueItemStatus)) == 4


class TestReviewModel:
    """Test Review model."""

    def test_create_review(self, session):
        """Test creating a review."""
        # First create a queue item
        item = QueueItem(
            sample_text="Sample text",
            original_score=0.5,
            per_dimension_scores=json.dumps({"technique": 0.6}),
            routing_reason="Test",
        )
        session.add(item)
        session.flush()

        # Create review for the item
        review = Review(
            item_id=item.id,
            reviewer_score=0.8,
            notes="Good clinical response",
            reviewer_id="expert_1",
            created_at=datetime.now(UTC),
        )
        session.add(review)
        session.flush()

        assert review.id is not None
        assert review.item_id == item.id
        assert review.reviewer_score == 0.8
        assert review.notes == "Good clinical response"
        assert review.reviewer_id == "expert_1"

    def test_review_repr(self):
        """Test Review __repr__ method."""
        review = Review(
            id=1,
            item_id=2,
            reviewer_score=0.7,
            notes="test",
            reviewer_id="expert_1",
            created_at=datetime.now(UTC),
        )
        repr_str = repr(review)
        assert "Review" in repr_str
        assert "id=1" in repr_str
        assert "item_id=2" in repr_str
        assert "reviewer_score=0.7" in repr_str

    def test_review_relationship(self, session):
        """Test relationship between Review and QueueItem."""
        # Create queue item
        item = QueueItem(
            sample_text="Sample",
            original_score=0.5,
            per_dimension_scores=json.dumps({"technique": 0.6}),
            routing_reason="Test",
        )
        session.add(item)
        session.flush()

        # Create review
        review = Review(
            item_id=item.id,
            reviewer_score=0.8,
            notes="Test review",
            reviewer_id="expert_1",
        )
        session.add(review)
        session.flush()

        # Test relationship
        assert review.queue_item.id == item.id
        assert review in item.reviews


class TestDatabaseSchema:
    """Test database schema constraints."""

    def test_queue_item_required_fields(self, engine):
        """Test that QueueItem requires all fields."""
        with Session(engine) as session:
            # Missing sample_text should raise error
            item = QueueItem(
                # Missing sample_text
                original_score=0.5,
                per_dimension_scores="{}",
                routing_reason="test",
            )
            session.add(item)
            with pytest.raises(Exception):
                session.flush()

    def test_review_required_fields(self, engine):
        """Test that Review requires all fields."""
        with Session(engine) as session:
            # Create a queue item first
            item = QueueItem(
                sample_text="test",
                original_score=0.5,
                per_dimension_scores="{}",
                routing_reason="test",
            )
            session.add(item)
            session.flush()

            # Missing reviewer_score should raise error
            review = Review(
                item_id=item.id,
                # Missing reviewer_score
                notes="test",
                reviewer_id="expert_1",
            )
            session.add(review)
            with pytest.raises(Exception):
                session.flush()
