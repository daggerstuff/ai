"""Consent-gated retrieval for PIX-511 memory ingestion.

This gate keeps a local, auditable consent record per user and maps consent
state to the shared memory-ingestion gate interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from ai.research.gates import GateDecision, GateResult
from ai.research.schema import ConsentGate, MemoryGating

ConsentAuditAction = Literal["check", "grant", "revoke", "expire"]
GATE_NAME = "gate3_consent"
DEFAULT_CONSENT_SCOPE = "memory_ingestion"


@dataclass
class ConsentRecord:
    user_id: str
    consent_type: ConsentGate
    granted_at: str
    expires_at: str | None
    scope: str
    revoked: bool
    revoked_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "consent_type": self.consent_type.value,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "scope": self.scope,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
        }


@dataclass
class ConsentAuditEntry:
    timestamp: str
    user_id: str
    action: ConsentAuditAction
    memory_id: str | None
    result: str
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "action": self.action,
            "memory_id": self.memory_id,
            "result": self.result,
            "details": self.details,
        }


@dataclass
class ConsentGateResult:
    allowed: bool
    consent_tier: ConsentGate
    reason: str
    expired: bool
    audit_entry: ConsentAuditEntry

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "consent_tier": self.consent_tier.value,
            "reason": self.reason,
            "expired": self.expired,
            "audit_entry": self.audit_entry.to_dict(),
        }


class ConsentGateChecker:
    """In-memory consent manager for retrieval-time memory access decisions."""

    def __init__(self, default_consent: ConsentGate = ConsentGate.BLOCKED) -> None:
        self._consent_store: dict[str, ConsentRecord] = {}
        self._audit_log: list[ConsentAuditEntry] = []
        self._default_consent = default_consent

    def grant_consent(
        self,
        user_id: str,
        consent_type: ConsentGate,
        scope: str = DEFAULT_CONSENT_SCOPE,
        expires_in_days: int | None = None,
    ) -> ConsentRecord:
        """Grant consent to a user with an optional UTC expiration."""
        now = datetime.now(UTC)
        expires_at = None
        if expires_in_days is not None:
            expires_at = (now + timedelta(days=expires_in_days)).isoformat()

        record = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type,
            granted_at=now.isoformat(),
            expires_at=expires_at,
            scope=scope,
            revoked=False,
            revoked_at=None,
        )
        self._consent_store[user_id] = record
        self._record_audit(
            user_id=user_id,
            action="grant",
            memory_id=None,
            result=consent_type.value if isinstance(consent_type, ConsentGate) else consent_type,
            details=f"Consent granted for scope '{scope}'",
        )
        return record

    def revoke_consent(self, user_id: str) -> None:
        """Revoke an existing consent record or log that no record exists."""
        record = self._consent_store.get(user_id)
        if record is None:
            self._record_audit(
                user_id=user_id,
                action="revoke",
                memory_id=None,
                result="blocked",
                details="No consent record found to revoke",
            )
            return

        record.revoked = True
        record.revoked_at = datetime.now(UTC).isoformat()
        self._record_audit(
            user_id=user_id,
            action="revoke",
            memory_id=None,
            result="revoked",
            details=f"Consent revoked for scope '{record.scope}'",
        )

    def check_consent(self, user_id: str, memory_id: str | None = None) -> ConsentGateResult:
        """Evaluate a user's consent state and append an audit entry for the check."""
        record = self._consent_store.get(user_id)

        # We'll set the parameters for _build_result
        build_user_id_and_memory_id = (user_id, memory_id)
        build_allowed = False
        build_consent_tier = ConsentGate.BLOCKED
        build_reason = ""
        build_expired = False

        if record is None:
            if self._default_consent == ConsentGate.BLOCKED:
                build_allowed = False
                build_consent_tier = ConsentGate.BLOCKED
                build_reason = "No consent record found"
            else:
                build_allowed = True
                build_consent_tier = self._default_consent
                build_reason = "Default consent applied"
        elif record.revoked:
            build_allowed = False
            build_consent_tier = record.consent_type
            build_reason = "Consent has been revoked"
        else:
            build_expired = self.is_expired(record)
            if build_expired:
                build_allowed = False
                build_consent_tier = record.consent_type
                build_reason = "Consent has expired"
            elif record.consent_type == ConsentGate.OPEN:
                build_allowed = True
                build_consent_tier = record.consent_type
                build_reason = "Consent granted"
            elif record.consent_type == ConsentGate.RESTRICTED:
                build_allowed = True
                build_consent_tier = record.consent_type
                build_reason = "Consent granted with restrictions"
            else:  # BLOCKED
                build_allowed = False
                build_consent_tier = ConsentGate.BLOCKED
                build_reason = "Consent tier is blocked"

        # Now, call _build_result to get the base result
        result = self._build_result(
            build_user_id_and_memory_id,
            build_allowed,
            build_consent_tier,
            build_reason,
            build_expired,
        )

        # If expired, we need to record an expire audit and append its timestamp to the check audit details
        if build_expired:
            expire_entry = self._record_audit(
                user_id=user_id,
                action="expire",
                memory_id=memory_id,
                result="blocked",
                details="Consent expiration observed during check",
            )
            result.audit_entry.details = f"{result.audit_entry.details}; expiration_audit={expire_entry.timestamp}"

        return result

    def evaluate(self, user_id: str, memory_id: str | None = None) -> GateResult:
        """Map consent state to the shared gate result interface."""
        result = self.check_consent(user_id=user_id, memory_id=memory_id)
        gating = MemoryGating(consentGate=result.consent_tier)
        details = [
            f"consent_tier: {result.consent_tier.value}",
            f"consent_gate: {gating.consentGate.value}",
            f"expired: {result.expired}",
        ]
        if memory_id is not None:
            details.append(f"memory_id: {memory_id}")

        if not result.allowed:
            return GateResult(
                gate=GATE_NAME,
                decision=GateDecision.BLOCK,
                reason=result.reason,
                confidence=1.0,
                details=details,
            )

        if result.consent_tier == ConsentGate.RESTRICTED:
            return GateResult(
                gate=GATE_NAME,
                decision=GateDecision.PASS,
                reason="Consent granted with restrictions",
                confidence=1.0,
                details=details,
            )

        return GateResult(
            gate=GATE_NAME,
            decision=GateDecision.PASS,
            reason="Consent granted",
            confidence=1.0,
            details=details,
        )

    def get_audit_log(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return serializable audit entries, optionally scoped to a user."""
        entries = self._audit_log
        if user_id is not None:
            entries = [entry for entry in entries if entry.user_id == user_id]
        return [entry.to_dict() for entry in entries]

    def is_expired(self, record: ConsentRecord) -> bool:
        """Return True when the record has an expiration in the past."""
        if record.expires_at is None:
            return False

        expires_at = datetime.fromisoformat(record.expires_at)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        return expires_at <= datetime.now(UTC)

    def to_dict(self, result: ConsentGateResult) -> dict[str, Any]:
        """Return a JSON-serializable representation of a consent check."""
        return result.to_dict()

    def _build_result(
        self,
        user_id_and_memory_id: tuple[str, str | None],
        allowed: bool,
        consent_tier: ConsentGate,
        reason: str,
        expired: bool,
    ) -> ConsentGateResult:
        user_id, memory_id = user_id_and_memory_id
        audit_entry = self._record_audit(
            user_id=user_id,
            action="check",
            memory_id=memory_id,
            result="pass" if allowed else "blocked",
            details=reason,
        )
        return ConsentGateResult(
            allowed=allowed,
            consent_tier=consent_tier,
            reason=reason,
            expired=expired,
            audit_entry=audit_entry,
        )

    def _record_audit(
        self,
        user_id: str,
        action: ConsentAuditAction,
        memory_id: str | None,
        result: str,
        details: str,
    ) -> ConsentAuditEntry:
        entry = ConsentAuditEntry(
            timestamp=datetime.now(UTC).isoformat(),
            user_id=user_id,
            action=action,
            memory_id=memory_id,
            result=result,
            details=details,
        )
        self._audit_log.append(entry)
        return entry


__all__ = [
    "ConsentAuditEntry",
    "ConsentGateChecker",
    "ConsentGateResult",
    "ConsentRecord",
]
