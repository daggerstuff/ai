"""FastAPI application for the annotation queue."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from .database import DATABASE_URL, get_db_session, init_db
from .models import PromotionAudit, QueueItem, QueueItemStatus, Review
from .schemas import (
    HealthResponse,
    PromotionRequest,
    PromotionResponse,
    QueueItemCreate,
    QueueItemDetailResponse,
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


# Stage progression order for promotion workflow
STAGE_ORDER = [
    QueueItemStatus.PENDING,
    QueueItemStatus.REVIEWED,
    QueueItemStatus.VALIDATED,
    QueueItemStatus.MERGED,
]


def _get_next_stage(current_status: QueueItemStatus) -> QueueItemStatus | None:
    """Get the next stage in the promotion workflow.

    Args:
        current_status: The current status of the queue item.

    Returns:
        The next stage, or None if already at the final stage.
    """
    try:
        current_index = STAGE_ORDER.index(current_status)
        if current_index < len(STAGE_ORDER) - 1:
            return STAGE_ORDER[current_index + 1]
        return None
    except ValueError:
        return None


@app.post("/queue/{item_id}/promote", response_model=PromotionResponse)
async def promote_queue_item(
    item_id: int,
    promotion: PromotionRequest,
    x_promoter_role: str | None = Header(None, alias="X-Promoter-Role"),
) -> PromotionResponse:
    """Promote a queue item to the next stage in the workflow.

    Enforces staged workflow: pending -> reviewed -> validated -> merged.
    Skipping a stage returns 409 Conflict.

    Authentication is via X-Promoter-Role header (dev default: 'promoter').
    Missing header returns 401. Wrong role returns 403.

    Records promoter_id, timestamp, before_score, after_score, and optional notes.
    """
    # Authentication check
    if x_promoter_role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Promoter-Role header",
        )
    if x_promoter_role != "promoter":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Invalid role '{x_promoter_role}'. Required role: 'promoter'",
        )

    logger.info(
        "Promotion request: item_id=%s, promoter_id=%s, timestamp=%s",
        item_id,
        promotion.promoter_id,
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

        current_status = QueueItemStatus(item.status)
        before_score = item.original_score

        # Determine next stage
        expected_next = _get_next_stage(current_status)
        if expected_next is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Item is already at the final stage '{current_status.value}'. Cannot promote further.",
            )

        # Update item status
        item.status = expected_next
        after_score = before_score  # Score stays the same during promotion

        # Create promotion audit record
        db_audit = PromotionAudit(
            item_id=item_id,
            promoter_id=promotion.promoter_id,
            before_score=before_score,
            after_score=after_score,
            target_stage=expected_next.value,
            notes=promotion.notes,
            timestamp=datetime.now(UTC),
        )
        session.add(db_audit)
        session.flush()
        session.refresh(db_audit)

        logger.info(
            "Promotion persisted: item_id=%s, audit_id=%s, promoter_id=%s, "
            "from=%s to=%s, before_score=%s, after_score=%s",
            item_id,
            db_audit.id,
            promotion.promoter_id,
            current_status.value,
            expected_next.value,
            before_score,
            after_score,
        )

        return PromotionResponse(
            id=db_audit.id,
            item_id=db_audit.item_id,
            promoter_id=db_audit.promoter_id,
            before_score=db_audit.before_score,
            after_score=db_audit.after_score,
            target_stage=db_audit.target_stage,
            notes=db_audit.notes,
            timestamp=db_audit.timestamp,
        )


@app.get("/queue/stats", response_model=QueueStats)
async def get_queue_stats() -> QueueStats:
    """Get queue statistics."""
    with get_db_session() as session:
        total = session.query(QueueItem).count()
        pending = session.query(QueueItem).filter(QueueItem.status == QueueItemStatus.PENDING).count()
        reviewed = session.query(QueueItem).filter(QueueItem.status == QueueItemStatus.REVIEWED).count()

        return QueueStats(pending=pending, reviewed=reviewed, total=total)


@app.get("/queue/detail", response_model=QueueItemDetailResponse)
async def get_item_detail(item_id: int = Query(..., alias="item_id")) -> QueueItemDetailResponse:
    """Get detailed information about a specific queue item, including review and promotion history."""
    with get_db_session() as session:
        item = session.query(QueueItem).filter(QueueItem.id == item_id).first()
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Queue item with ID {item_id} not found",
            )

        per_dimension_scores = json.loads(item.per_dimension_scores)

        # Build review responses
        review_responses = [
            ReviewResponse(
                id=r.id,
                item_id=r.item_id,
                reviewer_score=r.reviewer_score,
                notes=r.notes,
                reviewer_id=r.reviewer_id,
                created_at=r.created_at,
            )
            for r in item.reviews
        ]

        # Build promotion history responses
        promotion_responses = [
            PromotionResponse(
                id=a.id,
                item_id=a.item_id,
                promoter_id=a.promoter_id,
                before_score=a.before_score,
                after_score=a.after_score,
                target_stage=a.target_stage,
                notes=a.notes,
                timestamp=a.timestamp,
            )
            for a in item.promotion_audits
        ]

        return QueueItemDetailResponse(
            id=item.id,
            sample_text=item.sample_text,
            original_score=item.original_score,
            per_dimension_scores=per_dimension_scores,
            routing_reason=item.routing_reason,
            status=item.status,
            created_at=item.created_at,
            reviews=review_responses,
            promotion_history=promotion_responses,
        )


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(app, host="127.0.0.1", port=3102)


if __name__ == "__main__":
    run_server()
