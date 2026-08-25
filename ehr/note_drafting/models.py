"""Pydantic models for the note drafting request/response contract."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class NoteFormat(StrEnum):
    """Supported clinical note formats."""

    SOAP = "SOAP"
    DAP = "DAP"


class DraftRequest(BaseModel):
    """Request body for POST /draft.

    Contains the telehealth transcript and metadata needed to generate
    a clinical note draft. No PHI is persisted — this is an in-memory request only.
    """

    transcript: str = Field(
        ...,
        min_length=10,
        max_length=50000,
        description="Raw telehealth session transcript text.",
    )
    patient_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Identifier of the patient (used for reference only, not persisted).",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Identifier of the telehealth session.",
    )
    note_format: NoteFormat = Field(
        default=NoteFormat.SOAP,
        description="Desired clinical note format: SOAP or DAP.",
    )

    @field_validator("transcript")
    @classmethod
    def transcript_not_blank(cls, v: str) -> str:
        """Ensure transcript is not just whitespace."""
        if not v.strip():
            raise ValueError("transcript must not be blank or whitespace-only.")
        return v

    @field_validator("patient_id", "session_id")
    @classmethod
    def ids_not_blank(cls, v: str) -> str:
        """Ensure IDs are not blank."""
        if not v.strip():
            raise ValueError("ID must not be blank or whitespace-only.")
        return v


class NoteSections(BaseModel):
    """Structured sections of a clinical note draft.

    For SOAP format: subjective, objective, assessment, plan are populated.
    For DAP format: data, assessment, plan are populated.
    Unused fields remain None.
    """

    # SOAP sections
    subjective: str | None = Field(default=None, description="SOAP Subjective section.")
    objective: str | None = Field(default=None, description="SOAP Objective section.")
    # DAP sections
    data: str | None = Field(default=None, description="DAP Data section.")
    # Shared sections
    assessment: str | None = Field(default=None, description="SOAP/DAP Assessment section.")
    plan: str | None = Field(default=None, description="SOAP/DAP Plan section.")


class DraftResponse(BaseModel):
    """Response body for POST /draft."""

    draft_note: str = Field(
        ...,
        description="Full clinical note draft as formatted text.",
    )
    sections: NoteSections = Field(
        ...,
        description="Structured sections of the clinical note.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score for the draft (0.0-1.0).",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings (e.g. low confidence, partial transcript).",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Human-readable error message.")
