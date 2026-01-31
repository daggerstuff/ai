"""Quarantine module for dataset pipeline Phase 02.

Handles storage of validation-failed records for later review/reprocessing.
Uses MongoDB for persistence (aligned with project stack). Provides models,
store operations, and basic operator review workflow (list, review, reprocess).

QuarantineRecord captures raw input, validation errors, metadata for audit.
Operator workflow: simple functions to query, approve/reject, trigger revalidation.
Integrates with validation.py by catching errors and storing instead of raising.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from .ingestion_interface import IngestRecord
from .monitoring import log_quarantine_insert, log_validation_fail


class QuarantineStatus(Enum):
    """Status of quarantined records."""
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REPROCESSED = "reprocessed"
    ERROR = "error"


class QuarantineRecord(BaseModel):
    """Model for quarantined failed validation records."""
    quarantine_id: str = Field(..., description="Auto-generated unique ID")
    original_record_id: str = Field(..., description="ID from IngestRecord")
    raw_payload: dict[str, Any] = Field(..., description="Raw payload from ingestion")
    validation_errors: list[str] = Field(default_factory=list, description="List of validation error messages")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Ingestion metadata + provenance")
    status: QuarantineStatus = Field(default=QuarantineStatus.PENDING_REVIEW, description="Current status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Quarantine timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")
    review_notes: str | None = Field(None, max_length=1000, description="Operator review comments")
    reprocess_attempts: int = Field(default=0, ge=0, description="Number of reprocess tries")

    class Config:
        arbitrary_types_allowed = True  # For raw_payload flexibility

    def update_status(self, status: QuarantineStatus, notes: str | None = None) -> None:
        """Update status and notes, set updated_at."""
        self.status = status
        if notes:
            self.review_notes = notes
        self.updated_at = datetime.utcnow()
        if status == QuarantineStatus.REPROCESSED:
            self.reprocess_attempts += 1


class QuarantineStore:
    """MongoDB-backed store for quarantined records.

    Assumes MONGO_URI env var or default connection. Collections: 'quarantine_records'.
    Provides CRUD for quarantine ops and basic review workflow.
    """

    def __init__(self, mongo_uri: str | None = None, db_name: str = "pixelated_pipeline", collection_name: str = "quarantine_records"):
        self.mongo_uri = mongo_uri or "mongodb://localhost:27017"  # Default; use env in prod
        self.db_name = db_name
        self.collection_name = collection_name
        self.client: MongoClient | None = None
        self.collection = None
        self.connect()

    def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.client.admin.command("ping")  # Test connection
            db = self.client[self.db_name]
            self.collection = db[self.collection_name]
            # Ensure indexes for performance
            self.collection.create_index("original_record_id", unique=True)
            self.collection.create_index("status")
            self.collection.create_index("created_at")
        except ConnectionFailure as e:
            raise ConnectionError(f"Failed to connect to MongoDB at {self.mongo_uri}: {e}")

    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()

    def quarantine_record(self, record: IngestRecord, errors: list[str]) -> str:
        """Store a failed record in quarantine.

        Returns quarantine_id. Logs validation fail and quarantine insert.
        """
        qr = QuarantineRecord(
            quarantine_id=str(datetime.utcnow().timestamp()) + "_" + record.id,  # Simple unique ID
            original_record_id=record.id,
            raw_payload=record.payload,
            validation_errors=errors,
            metadata=record.metadata,
        )
        # Convert to dict for Mongo insert
        doc = qr.dict()
        result = self.collection.insert_one(doc)
        if not result.inserted_id:
            raise ValueError("Failed to insert quarantine record")
        log_validation_fail()
        log_quarantine_insert()
        return qr.quarantine_id

    def get_quarantined(self, status: QuarantineStatus | None = None, limit: int = 50, skip: int = 0) -> Iterable[QuarantineRecord]:
        """Fetch quarantined records, optionally filtered by status.

        Pagination via limit/skip.
        """
        query = {"status": status.value} if status else {}
        cursor = self.collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
        for doc in cursor:
            yield QuarantineRecord(**doc)

    def update_status(self, quarantine_id: str, status: QuarantineStatus, notes: str | None = None) -> bool:
        """Update status and notes for a record.

        Returns True if updated.
        """
        qr = self.collection.find_one({"quarantine_id": quarantine_id})
        if not qr:
            return False
        qr_obj = QuarantineRecord(**qr)
        qr_obj.update_status(status, notes)
        result = self.collection.replace_one(
            {"quarantine_id": quarantine_id},
            qr_obj.dict()
        )
        return result.modified_count > 0

    def reprocess_record(self, quarantine_id: str) -> IngestRecord | None:
        """Attempt to reprocess a quarantined record (re-run validation).

        If successful, update status to REPROCESSED and return validated record.
        If fails again, increment attempts and return None.
        """
        qr = self.collection.find_one({"quarantine_id": quarantine_id})
        if not qr:
            return None
        qr_obj = QuarantineRecord(**qr)
        if qr_obj.reprocess_attempts >= 3:
            qr_obj.update_status(QuarantineStatus.ERROR, "Max reprocess attempts exceeded")
            self.collection.replace_one({"quarantine_id": quarantine_id}, qr_obj.dict())
            return None

        # Reconstruct IngestRecord
        rec = IngestRecord(
            id=qr_obj.original_record_id,
            payload=qr_obj.raw_payload,
            metadata=qr_obj.metadata,
        )
        try:
            from .validation import validate_record
            validated = validate_record(rec)
            qr_obj.update_status(QuarantineStatus.REPROCESSED)
            self.collection.replace_one({"quarantine_id": quarantine_id}, qr_obj.dict())
            return validated  # Or yield to downstream
        except Exception as e:
            qr_obj.reprocess_attempts += 1
            qr_obj.validation_errors.append(str(e))
            if qr_obj.reprocess_attempts >= 3:
                qr_obj.update_status(QuarantineStatus.ERROR, str(e))
            else:
                qr_obj.updated_at = datetime.utcnow()
            self.collection.replace_one({"quarantine_id": quarantine_id}, qr_obj.dict())
            return None

    def delete_record(self, quarantine_id: str) -> bool:
        """Permanently delete a quarantined record (e.g., after rejection)."""
        result = self.collection.delete_one({"quarantine_id": quarantine_id})
        return result.deleted_count > 0

    def stats(self) -> dict[str, int]:
        """Get quarantine stats: counts by status."""
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
            {"$group": {"_id": None, "stats": {"$push": {"status": "$_id", "count": "$count"}}}},
            {"$replaceRoot": {"newRoot": "$stats"}}
        ]
        result = list(self.collection.aggregate(pipeline))
        if result:
            return {item["status"]: item["count"] for item in result[0]}
        return {}


# Global store instance (thread-safe in production with proper config)
_store: QuarantineStore | None = None


def get_quarantine_store() -> QuarantineStore:
    """Get or create global quarantine store."""
    global _store
    if _store is None:
        _store = QuarantineStore()
    return _store


# Operator review workflow helpers
def review_quarantined(limit: int = 10) -> Iterable[QuarantineRecord]:
    """Simple CLI-friendly review: yield pending records."""
    store = get_quarantine_store()
    return store.get_quarantined(QuarantineStatus.PENDING_REVIEW, limit=limit)


def approve_record(quarantine_id: str, notes: str = "") -> bool:
    """Approve for reprocessing."""
    store = get_quarantine_store()
    return store.update_status(quarantine_id, QuarantineStatus.APPROVED, notes)


def reject_record(quarantine_id: str, notes: str = "") -> bool:
    """Reject and optionally delete."""
    store = get_quarantine_store()
    if store.update_status(quarantine_id, QuarantineStatus.REJECTED, notes):
        return store.delete_record(quarantine_id)
    return False


__all__ = [
    "QuarantineRecord",
    "QuarantineStatus",
    "QuarantineStore",
    "approve_record",
    "get_quarantine_store",
    "reject_record",
    "review_quarantined",
]
