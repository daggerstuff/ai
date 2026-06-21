"""FastAPI application for the annotation queue."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from .database import DATABASE_URL, get_db_session, init_db
from .models import QueueItem, QueueItemStatus, Review
from .schemas import (
    HealthResponse,
    QueueItemCreate,
    QueueItemResponse,
    QueueStats,
    ReviewCreate,
    ReviewResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Annotation Queue API",
    description="API for managing clinical validity annotation queue",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    logger.info("Starting annotation API")
    init_db()
    logger.info(f"Database initialized at {DATABASE_URL}")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/queue", response_model=QueueItemResponse, status_code=status.HTTP_201_CREATED)
async def add_to_queue(item: QueueItemCreate) -> QueueItemResponse:
    """Add a new item to the annotation queue."""
    with get_db_session() as session:
        db_item = QueueItem(
            sample_text=item.sample_text,
            original_score=item.original_score,
            per_dimension_scores=json.dumps(item.per_dimension_scores),
            routing_reason=item.routing_reason,
            status=QueueItemStatus.PENDING,
            created_at=datetime.now(UTC),
        )
        session.add(db_item)
        session.flush()
        session.refresh(db_item)

        # Convert per_dimension_scores back from JSON
        per_dimension_scores = json.loads(db_item.per_dimension_scores)

        return QueueItemResponse(
            id=db_item.id,
            sample_text=db_item.sample_text,
            original_score=db_item.original_score,
            per_dimension_scores=per_dimension_scores,
            routing_reason=db_item.routing_reason,
            status=db_item.status,
            created_at=db_item.created_at,
        )


@app.get("/queue/pending", response_model=list[QueueItemResponse])
@app.post("/queue/pending", response_model=list[QueueItemResponse])
async def list_pending_items() -> list[QueueItemResponse]:
    """List all pending items in the queue."""
    with get_db_session() as session:
        items = (
            session.query(QueueItem)
            .filter(QueueItem.status == QueueItemStatus.PENDING)
            .order_by(QueueItem.created_at.asc())
            .all()
        )

        result = []
        for item in items:
            per_dimension_scores = json.loads(item.per_dimension_scores)
            result.append(
                QueueItemResponse(
                    id=item.id,
                    sample_text=item.sample_text,
                    original_score=item.original_score,
                    per_dimension_scores=per_dimension_scores,
                    routing_reason=item.routing_reason,
                    status=item.status,
                    created_at=item.created_at,
                )
            )
        return result


@app.post("/queue/submit", response_model=ReviewResponse)
async def submit_review(item_id: int, review: ReviewCreate) -> ReviewResponse:
    """Submit a review for a queue item."""
    with get_db_session() as session:
        # Check if item exists
        item = session.query(QueueItem).filter(QueueItem.id == item_id).first()
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Queue item with ID {item_id} not found",
            )

        # Create review
        db_review = Review(
            item_id=item_id,
            reviewer_score=review.reviewer_score,
            notes=review.notes,
            reviewer_id=review.reviewer_id,
            created_at=datetime.now(UTC),
        )
        session.add(db_review)

        # Update item status
        item.status = QueueItemStatus.REVIEWED
        session.flush()
        session.refresh(db_review)

        return ReviewResponse(
            id=db_review.id,
            item_id=db_review.item_id,
            reviewer_score=db_review.reviewer_score,
            notes=db_review.notes,
            reviewer_id=db_review.reviewer_id,
            created_at=db_review.created_at,
        )


@app.patch("/queue/{item_id}/review", response_model=ReviewResponse)
async def submit_review_by_path(item_id: int, review: ReviewCreate) -> ReviewResponse:
    """Submit a review for a queue item via PATCH path parameter.

    This endpoint is idempotent: duplicate reviews are allowed and both are logged.
    The latest review's score is authoritative for the item.
    """
    logger.info(
        "Review submission: item_id=%s, reviewer_id=%s, timestamp=%s",
        item_id,
        review.reviewer_id,
        datetime.now(UTC).isoformat(),
    )
    with get_db_session() as session:
        # Check if item exists
        item = session.query(QueueItem).filter(QueueItem.id == item_id).first()
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Queue item with ID {item_id} not found",
            )

        # Create review (idempotent - always creates a new row)
        db_review = Review(
            item_id=item_id,
            reviewer_score=review.reviewer_score,
            notes=review.notes,
            reviewer_id=review.reviewer_id,
            created_at=datetime.now(UTC),
        )
        session.add(db_review)

        # Update item status to reviewed
        item.status = QueueItemStatus.REVIEWED
        session.flush()
        session.refresh(db_review)

        logger.info(
            "Review persisted: item_id=%s, review_id=%s, reviewer_id=%s",
            item_id,
            db_review.id,
            review.reviewer_id,
        )

        return ReviewResponse(
            id=db_review.id,
            item_id=db_review.item_id,
            reviewer_score=db_review.reviewer_score,
            notes=db_review.notes,
            reviewer_id=db_review.reviewer_id,
            created_at=db_review.created_at,
        )


@app.get("/queue/stats", response_model=QueueStats)
async def get_queue_stats() -> QueueStats:
    """Get queue statistics."""
    with get_db_session() as session:
        total = session.query(QueueItem).count()
        pending = session.query(QueueItem).filter(QueueItem.status == QueueItemStatus.PENDING).count()
        reviewed = session.query(QueueItem).filter(QueueItem.status == QueueItemStatus.REVIEWED).count()

        return QueueStats(pending=pending, reviewed=reviewed, total=total)


@app.get("/queue/detail", response_model=QueueItemResponse)
async def get_item_detail(item_id: int = Query(..., alias="item_id")) -> QueueItemResponse:
    """Get detailed information about a specific queue item."""
    with get_db_session() as session:
        item = session.query(QueueItem).filter(QueueItem.id == item_id).first()
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Queue item with ID {item_id} not found",
            )

        per_dimension_scores = json.loads(item.per_dimension_scores)
        return QueueItemResponse(
            id=item.id,
            sample_text=item.sample_text,
            original_score=item.original_score,
            per_dimension_scores=per_dimension_scores,
            routing_reason=item.routing_reason,
            status=item.status,
            created_at=item.created_at,
        )


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(app, host="127.0.0.1", port=3102)


if __name__ == "__main__":
    run_server()
