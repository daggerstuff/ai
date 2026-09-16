"""Quadit data models — content-source-agnostic audit records."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

# Severity ladder: a critical finding flips the audit to FAIL; warnings
# accumulate without flipping on their own; info is observational.
SEVERITIES = ("info", "warning", "critical")


class AuditItem(BaseModel):
    """One piece of content submitted to the quadit judges."""

    id: str
    kind: str = Field(
        default="text",
        description="Content kind, e.g. 'ai_response', 'dataset_record', 'email', 'chat_burst'.",
    )
    author_role: str = Field(default="", description="Who produced the item, e.g. 'pe-service' or a persona name.")
    content: str = Field(min_length=1)
    context: dict[str, str] = Field(
        default_factory=dict,
        description="Free-form adapter metadata (user_id hash, month, session_id, ...).",
    )


class Finding(BaseModel):
    """A single auditor finding pinned to one item."""

    item_id: str
    severity: str = Field(description="One of: info, warning, critical.")

    @field_validator("severity")
    @classmethod
    def _severity_must_be_known(cls, value: str) -> str:
        if value not in SEVERITIES:
            msg = f"severity must be one of {SEVERITIES}, got {value!r}"
            raise ValueError(msg)
        return value

    signature: str = Field(default="", description="Anti-signal label, e.g. 'clinical_abstraction_over_warmth'.")
    rationale: str = ""
    example_excerpt: str = Field(default="", max_length=400)


class PersonaVerdict(BaseModel):
    """One persona's judgment over the full item set."""

    persona: str
    role: str = Field(default="", description="'judge' or 'adversarial_auditor'.")
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    notes: str = ""
    flagged_ids: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)


class QuadAuditReport(BaseModel):
    """Full quad-audit result — all judges plus adversarial auditors."""

    passed: bool
    mode: str = Field(description="'llm' or 'deterministic'.")
    model: str = Field(default="deterministic")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    item_count: int = 0
    verdicts: list[PersonaVerdict] = Field(default_factory=list)
    critical_count: int = 0
    warning_count: int = 0
    summary: str = ""
