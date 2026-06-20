"""SQLAlchemy models for the annotation queue."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class QueueItemStatus(StrEnum):
    """Status of a queue item."""
    PENDING = "pending"
    REVIEWED = "reviewed"


class QueueItem(Base):
    """Queue item representing a sample that needs expert review."""

    __tablename__ = "queue_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_text: Mapped[str] = mapped_column(Text, nullable=False)
    original_score: Mapped[float] = mapped_column(Float, nullable=False)
    per_dimension_scores: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string
    routing_reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[QueueItemStatus] = mapped_column(String, default=QueueItemStatus.PENDING.value, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship to reviews
    reviews: Mapped[list[Review]] = relationship("Review", back_populates="queue_item", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_queue_items_status", "status"),
        Index("idx_queue_items_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"QueueItem(id={self.id}, status={self.status}, created_at={self.created_at})"


class Review(Base):
    """Review submitted by an expert for a queue item."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("queue_items.id"), nullable=False)
    reviewer_score: Mapped[float] = mapped_column(Float, nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship to queue item
    queue_item: Mapped[QueueItem] = relationship("QueueItem", back_populates="reviews")

    __table_args__ = (
        Index("idx_reviews_item_id", "item_id"),
        Index("idx_reviews_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"Review(id={self.id}, item_id={self.item_id}, reviewer_score={self.reviewer_score})"
