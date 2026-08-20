"""Privacy and content-handling gates for the modern dataset pipeline.

This module defines the policy layer that determines how acquired data is handled
before promotion into training-ready datasets. It makes three categories of
decisions explicit and auditable:

  Retention   — what can be retained, for how long, and under what conditions
  Treatment   — what must be transformed (scrubbed, normalized) before use
  Escalation  — what must be routed to human review or rejected outright

Design principles
-----------------
* Every decision is a named gate with a documented reason, not a boolean toggle.
* Gates compose: a source that fails Gate 1 never reaches Gate 2.
* Human-review lanes are explicit, not implied fallback paths.
* Privacy classification is independent of content sensitivity; a source can be
  HIGH privacy risk (contains PII) but NORMAL content sensitivity (therapy
  session) — both dimensions drive different handling requirements.

Gate overview
-------------
  Gate 0 — Intake classification
              Assigns privacy tier and content sensitivity tier.
              Required before any other gate evaluation.

  Gate 1 — PII detection and treatment
              Detects personally identifiable or health-related information.
              Determines whether scrubbing is sufficient or rejection is required.

  Gate 2 — Content safety
              Detects crisis indicators and unsafe content.
              Determines whether automated filtering or escalation is required.

  Gate 3 — License and consent validation
              Checks that source licensing permits inclusion and that any
              required consent records exist or are implied.

  Gate 4 — Human-review lane
              Items that are borderline on any prior gate are routed here
              rather than silently promoted or dropped. Explicit review
              decision is required before promotion.

Promotion is blocked if any of Gates 0–3 returns BLOCK. Items that pass all
applicable gates (or receive an explicit human-review override) are promoted
to the curation stage.

Downstream consumers
---------------------
  PIX-250 (human-review formalization) — consumes Gate 4 decision records
  PIX-506 (training-readiness validation) — consumes promoted items with
            their gate audit trail attached
  PIX-507 (pipeline observability) — consumes gate outcome events for
            reporting and alerting
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .crisis_intervention_detector import CrisisInterventionDetector
from .processing.pii_scrubber import PiiScrubber, PiiScrubberConfig

# Optional import for review queue (PIX-250) - only used if available
try:
    from .human_review_queue import HumanReviewQueue, Reviewer, ReviewerRole  # type: ignore

    REVIEW_QUEUE_ENABLED = True
except ImportError:
    REVIEW_QUEUE_ENABLED = False
    HumanReviewQueue = None  # type: ignore
    Reviewer = None  # type: ignore
    ReviewerRole = None  # type: ignore

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------


class PrivacyTier(StrEnum):
    """How much identifiable information a source contains."""

    NONE = "none"  # No identifiable information present
    LOW = "low"  # Minimal risk; redaction is straightforward
    MEDIUM = "medium"  # Identifiable; requires scrubbing before use
    HIGH = "high"  # Contains PHI or sensitive identifiers; manual review required
    PROHIBITED = "prohibited"  # Cannot be included under any circumstances


class ContentSensitivity(StrEnum):
    """Risk level of the content itself, independent of privacy classification."""

    NORMAL = "normal"  # General therapeutic or educational content
    SENSITIVE = "sensitive"  # Topics that require additional care (trauma, addiction)
    RESTRICTED = "restricted"  # Requires clinical oversight or specific expertise
    PROHIBITED = "prohibited"  # Cannot be included in training data


class RetentionPolicy(StrEnum):
    """Data-retention rules tied to privacy tier."""

    USE_IMMEDIATELY = "use_immediately"  # Tier NONE only
    SCRUB_AND_USE = "scrub_and_use"  # Tier LOW, MEDIUM after scrubbing
    REVIEW_THEN_USE = "review_then_use"  # Tier HIGH; human review required
    DO_NOT_USE = "do_not_use"  # Tier PROHIBITED
    REJECT = "reject"  # Explicit rejection; do not process


class GateDecision(StrEnum):
    PASS = "pass"
    BLOCK = "block"
    ESCALATE = "escalate"  # Blocked but recoverable via human review


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PiiFinding:
    pii_type: str
    count: int
    treatment: str  # e.g. "scrubbed", "rejected", "none_required"


@dataclass
class CrisisFinding:
    crisis_type: str
    severity: str  # critical, high, elevated, moderate, low
    score: float
    requires_escalation: bool


@dataclass
class LicenseCheck:
    license_id: str
    status: str  # approved, exception, blocked
    requires_consent: bool
    consent_recorded: bool


@dataclass
class GateResult:
    """Outcome of a single gate evaluation."""

    gate: str
    decision: GateDecision
    reason: str
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "decision": self.decision.value,
            "reason": self.reason,
            "details": self.details,
        }


@dataclass
class PrivacyContentReport:
    """Full gate report for a single source or item."""

    source_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    gate0_result: GateResult | None = None
    gate1_result: GateResult | None = None
    gate2_result: GateResult | None = None
    gate3_result: GateResult | None = None
    gate4_result: GateResult | None = None  # Human review — set externally

    privacy_tier: PrivacyTier = PrivacyTier.NONE
    content_sensitivity: ContentSensitivity = ContentSensitivity.NORMAL
    retention_policy: RetentionPolicy = RetentionPolicy.USE_IMMEDIATELY
    pii_findings: list[PiiFinding] = field(default_factory=list)
    crisis_findings: list[CrisisFinding] = field(default_factory=list)
    license_check: LicenseCheck | None = None

    @property
    def passed(self) -> bool:
        """True when the item is cleared for promotion.

        If Gate 4 (human review) has been set, it overrides any prior ESCALATE.
        If Gate 4 is not set, all Gates 0-3 must be PASS.
        """
        if self.gate4_result is not None:
            # Human review has been applied; PASS overrides prior ESCALATE
            return self.gate4_result.decision == GateDecision.PASS
        results = [g for g in self._gate_results if g is not None]
        return all(r.decision == GateDecision.PASS for r in results)

    @property
    def blocked(self) -> bool:
        """True when any mandatory gate blocked the item."""
        results = [g for g in self._gate_results if g is not None]
        return any(r.decision == GateDecision.BLOCK for r in results)

    @property
    def needs_review(self) -> bool:
        """True when human review is required before promotion."""
        results = [g for g in self._gate_results if g is not None]
        return any(r.decision == GateDecision.ESCALATE for r in results)

    @property
    def promotion_ready(self) -> bool:
        """True when item can be promoted to curation without further gates."""
        return self.passed and self.gate4_result is not None

    @property
    def _gate_results(self) -> list[GateResult | None]:
        return [
            self.gate0_result,
            self.gate1_result,
            self.gate2_result,
            self.gate3_result,
            self.gate4_result,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "privacy_tier": self.privacy_tier.value,
            "content_sensitivity": self.content_sensitivity.value,
            "retention_policy": self.retention_policy.value,
            "passed": self.passed,
            "blocked": self.blocked,
            "needs_review": self.needs_review,
            "promotion_ready": self.promotion_ready,
            "gates": {
                f"gate{i}": (r.to_dict() if r else None)
                for i, r in enumerate(
                    [self.gate0_result, self.gate1_result, self.gate2_result, self.gate3_result, self.gate4_result]
                )
            },
            "pii_findings": [
                {"pii_type": f.pii_type, "count": f.count, "treatment": f.treatment} for f in self.pii_findings
            ],
            "crisis_findings": [
                {
                    "crisis_type": f.crisis_type,
                    "severity": f.severity,
                    "score": f.score,
                    "requires_escalation": f.requires_escalation,
                }
                for f in self.crisis_findings
            ],
            "license_check": (
                {
                    "license_id": self.license_check.license_id,
                    "status": self.license_check.status,
                    "requires_consent": self.license_check.requires_consent,
                    "consent_recorded": self.license_check.consent_recorded,
                }
                if self.license_check
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Approved-license set (mirrors acquisition_rubric.py)
# ---------------------------------------------------------------------------

APPROVED_LICENSES: set[str] = {
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "unlicense",
    "odc-by",
    "odbl-1.0",
}

EXCEPTION_LICENSES: set[str] = {
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
}

# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------

PII_HIGH_THRESHOLD: int = 5
CRISIS_BLOCK_THRESHOLD: float = 0.8
CRISIS_ESCALATE_THRESHOLD: float = 0.4

_APPROVED_LICENSES: frozenset[str] = frozenset(APPROVED_LICENSES)
_EXCEPTION_LICENSES: frozenset[str] = frozenset(EXCEPTION_LICENSES)

UNSAFE_PATTERNS: list[tuple[str, str]] = [
    ("medical_misuse", r"\bofficial\s+diagnosis\b"),
    ("dangerous_instructions", r"\bhow\s+to\s+(kill|harm|attack)\b"),
    ("non_consent", r"\b(dox|steal|manipulat)\b"),
]

SENSITIVE_PATTERNS: list[str] = [
    r"\b(anxiety|anxious|anxiet|panic|social anxiety|generalized anxiety)\b",
    r"\b(trauma|abuse|addict|crisis|self.?harm|suicid|overdose)\b",
    r"\b(anxiety|anxious|panic|social anxiety|generalized anxiety)\b",
    r"\b(trauma|abuse|addict|crisis|self.?harm|suicid|overdose)",
    r"\b(personality\s+disorder|bpd|narciss|histrionic)\b",
    r"\b(cptsd|complex\s+trauma|dissociat)\b",
]

RESTRICTED_PATTERNS: list[str] = [
    r"\b(inpatient|psychiatric\s+hold|involuntary\s+commitment)\b",
    r"\b(prescription.*\bdosage\b|prescribe.*without.*doctor)\b",
]


# ---------------------------------------------------------------------------
# PrivacyContentGates
# ---------------------------------------------------------------------------


class PrivacyContentGates:
    """Evaluates data against the privacy and content-handling policy.

    Gates are evaluated sequentially (0 → 1 → 2 → 3 → 4). Evaluation stops
    on the first BLOCK. ESCALATE does not stop evaluation but surfaces a flag
    in the report.

    Usage::

        gates = PrivacyContentGates()
        report = gates.evaluate(
            source_id="src-001",
            text="Hello, I am John and I want to die.",
            license_id="cc-by-4.0",
        )
        if report.retention_policy == RetentionPolicy.SCRUB_AND_USE:
            clean_text, _ = gates.apply_scrub(text)
    """

    def __init__(
        self,
        pii_config: PiiScrubberConfig | None = None,
        review_queue: HumanReviewQueue | None = None,
    ) -> None:
        self._pii_scrubber = PiiScrubber(pii_config or PiiScrubberConfig())
        self._crisis_detector = CrisisInterventionDetector()
        # Review queue integration (PIX-250)
        if REVIEW_QUEUE_ENABLED and review_queue is None:
            try:
                from .human_review_queue import HumanReviewQueue  # type: ignore

                review_queue = HumanReviewQueue()
            except Exception:
                # Queue initialization failed; debug logging recommended
                pass
        self._review_queue = review_queue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        source_id: str,
        text: str,
        license_id: str,
        consent_recorded: bool = False,
    ) -> PrivacyContentReport:
        """Evaluate a single text item against all applicable gates.

        Args:
            source_id:        Identifier for the source or record being evaluated.
            text:             The content to evaluate.
            license_id:       SPDX license identifier (e.g. "cc-by-4.0").
            consent_recorded: Whether a valid consent record exists for this data.
        """
        report = PrivacyContentReport(source_id=source_id)

        # Gate 0 — Classification
        gate0, pii_findings = self._gate0_classify(text)
        report.gate0_result = gate0
        report.pii_findings = pii_findings
        report.privacy_tier = self._tier_from_findings(pii_findings)
        report.content_sensitivity = self._sensitivity_from_text(text)
        report.retention_policy = self._retention_for_tier(report.privacy_tier)

        if gate0.decision == GateDecision.BLOCK:
            return report

        # Gate 1 — PII treatment
        gate1 = self._gate1_pii(report.privacy_tier)
        report.gate1_result = gate1
        if gate1.decision == GateDecision.BLOCK:
            return report
        # Enqueue for review if ESCALATE
        if gate1.decision == GateDecision.ESCALATE:
            self._enqueue_for_review(report, text)

        # Gate 2 — Content safety
        gate2, crisis_findings = self._gate2_safety(text)
        report.gate2_result = gate2
        report.crisis_findings = crisis_findings
        if gate2.decision == GateDecision.BLOCK:
            return report
        # Enqueue for review if ESCALATE
        if gate2.decision == GateDecision.ESCALATE:
            self._enqueue_for_review(report, text)

        # Gate 3 — License and consent
        gate3 = self._gate3_license(license_id, report.content_sensitivity, consent_recorded)
        report.gate3_result = gate3
        report.license_check = LicenseCheck(
            license_id=license_id,
            status=gate3.details[0] if gate3.details else "unknown",
            requires_consent=(gate3.details[1] == "consent_required" if len(gate3.details) > 1 else False),
            consent_recorded=consent_recorded,
        )
        # Enqueue for review if ESCALATE
        if gate3.decision == GateDecision.ESCALATE:
            self._enqueue_for_review(report, text)

        return report

    def evaluate_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[PrivacyContentReport]:
        """Evaluate a batch of items."""
        reports: list[PrivacyContentReport] = []
        for idx, item in enumerate(items):
            try:
                report = self.evaluate(
                    source_id=str(item.get("id", idx)),
                    text=item.get("text", item.get("content", "")),
                    license_id=item.get("license_id", "unknown"),
                    consent_recorded=item.get("consent_recorded", False),
                )
                reports.append(report)
            except Exception as exc:
                # Graceful degradation: report failure rather than breaking the batch
                report = PrivacyContentReport(source_id=str(item.get("id", idx)))
                report.gate0_result = GateResult(
                    gate="gate0",
                    decision=GateDecision.BLOCK,
                    reason=f"evaluation failed: {exc}",
                    details=[],
                )
                reports.append(report)
        return reports

    def override_with_review(
        self,
        report: PrivacyContentReport,
        review_decision: GateDecision,
        reviewer: str,
        reason: str,
    ) -> PrivacyContentReport:
        """Apply a human-review override (Gate 4).

        review_decision must be PASS (promotion approved) or BLOCK (rejected).
        """
        if review_decision not in (GateDecision.PASS, GateDecision.BLOCK):
            raise ValueError(f"review_decision must be PASS or BLOCK, got {review_decision.value}")
        report.gate4_result = GateResult(
            gate="gate4",
            decision=review_decision,
            reason=reason,
            details=[
                f"reviewer: {reviewer}",
                f"timestamp: {datetime.now(UTC).isoformat()}",
            ],
        )
        return report

    def apply_scrub(self, text: str) -> tuple[str, list[PiiFinding]]:
        """Scrub text and return the sanitized result with findings.

        Use this after evaluate() when retention_policy is SCRUB_AND_USE or
        REVIEW_THEN_USE.
        """
        result = self._pii_scrubber.scrub(text)
        findings = [PiiFinding(pii_type=k, count=c, treatment="scrubbed") for k, c in result.pii_counts.items()]
        return result.scrubbed_text, findings

    # ------------------------------------------------------------------
    # Gate 0 — Intake classification
    # ------------------------------------------------------------------

    def _gate0_classify(self, text: str) -> tuple[GateResult, list[PiiFinding]]:
        if not text or not isinstance(text, str) or not text.strip():
            return GateResult(
                gate="gate0",
                decision=GateDecision.BLOCK,
                reason="empty or malformed input",
                details=["text is empty or wrong type"],
            ), []

        try:
            pii_result = self._pii_scrubber.scrub(text)
        except Exception as exc:
            return GateResult(
                gate="gate0",
                decision=GateDecision.BLOCK,
                reason=f"PII scrubber raised: {exc}",
                details=[],
            ), []

        pii_types = list(pii_result.pii_counts.keys())
        total_pii = pii_result.total_pii_count
        findings = [PiiFinding(pii_type=k, count=c, treatment="scrubbed") for k, c in pii_result.pii_counts.items()]

        return GateResult(
            gate="gate0",
            decision=GateDecision.PASS,
            reason="classified for privacy tier and content sensitivity",
            details=[
                f"pii_types: {', '.join(pii_types) or 'none'}",
                f"total_pii: {total_pii}",
            ],
        ), findings

    # ------------------------------------------------------------------
    # Gate 1 — PII treatment
    # ------------------------------------------------------------------

    def _gate1_pii(self, tier: PrivacyTier) -> GateResult:
        if tier == PrivacyTier.PROHIBITED:
            return GateResult(
                gate="gate1",
                decision=GateDecision.BLOCK,
                reason="PII tier is prohibited — no acceptable treatment",
                details=["source contains prohibited PHI"],
            )
        if tier in (PrivacyTier.NONE, PrivacyTier.LOW):
            return GateResult(
                gate="gate1",
                decision=GateDecision.PASS,
                reason=f"PII tier {tier.value} requires no treatment",
                details=["no treatment required"],
            )
        if tier == PrivacyTier.MEDIUM:
            return GateResult(
                gate="gate1",
                decision=GateDecision.PASS,
                reason="PII will be scrubbed with standard redaction",
                details=["scrubbing required before use"],
            )
        # Tier HIGH — human review required
        return GateResult(
            gate="gate1",
            decision=GateDecision.ESCALATE,
            reason="PII tier HIGH requires human review before promotion",
            details=["manual scrub verification required"],
        )

    # ------------------------------------------------------------------
    # Gate 2 — Content safety
    # ------------------------------------------------------------------

    def _gate2_safety(self, text: str) -> tuple[GateResult, list[CrisisFinding]]:
        crisis = self._crisis_detector.process(text)
        crisis_findings = self._collect_crisis_findings(crisis)

        if crisis.score >= CRISIS_BLOCK_THRESHOLD:
            return GateResult(
                gate="gate2",
                decision=GateDecision.BLOCK,
                reason=f"crisis score {crisis.score:.2f} exceeds block threshold",
                details=[
                    f"crisis_type: {crisis.crisis_type}",
                    f"severity: {crisis.severity}",
                    f"matches: {', '.join(crisis.matches) or 'none'}",
                    "Escalation contacts: 988 (Suicide & Crisis Lifeline), 911",
                ],
            ), crisis_findings

        if crisis.score >= CRISIS_ESCALATE_THRESHOLD:
            return GateResult(
                gate="gate2",
                decision=GateDecision.ESCALATE,
                reason=f"crisis score {crisis.score:.2f} requires clinical review",
                details=[
                    f"crisis_type: {crisis.crisis_type}",
                    f"severity: {crisis.severity}",
                    "queued for human review",
                ],
            ), crisis_findings

        # Additional unsafe pattern checks
        unsafe = self._detect_unsafe_patterns(text)
        if unsafe:
            return GateResult(
                gate="gate2",
                decision=GateDecision.BLOCK,
                reason="unsafe content detected",
                details=[f"pattern: {p}" for p in unsafe],
            ), crisis_findings

        return GateResult(
            gate="gate2",
            decision=GateDecision.PASS,
            reason="no crisis indicators or unsafe content detected",
            details=[f"crisis_score: {crisis.score:.3f}"],
        ), crisis_findings

    # ------------------------------------------------------------------
    # Gate 3 — License and consent
    # ------------------------------------------------------------------

    def _gate3_license(
        self,
        license_id: str,
        sensitivity: ContentSensitivity,
        consent_recorded: bool,
    ) -> GateResult:
        requires_consent = sensitivity in (
            ContentSensitivity.SENSITIVE,
            ContentSensitivity.RESTRICTED,
        )

        if license_id in _APPROVED_LICENSES:
            if requires_consent and not consent_recorded:
                return GateResult(
                    gate="gate3",
                    decision=GateDecision.ESCALATE,
                    reason=f"license approved but consent required for {sensitivity.value} content",
                    details=["approved", "consent_required", "consent_not_recorded"],
                )
            return GateResult(
                gate="gate3",
                decision=GateDecision.PASS,
                reason=f"license {license_id} is approved",
                details=["approved", "no_consent_required"],
            )

        if license_id in _EXCEPTION_LICENSES:
            if requires_consent and not consent_recorded:
                return GateResult(
                    gate="gate3",
                    decision=GateDecision.ESCALATE,
                    reason="NC license requires documented consent for sensitive content",
                    details=["exception", "consent_required", "consent_not_recorded"],
                )
            return GateResult(
                gate="gate3",
                decision=GateDecision.PASS,
                reason=f"license {license_id} is eligible for exception",
                details=[
                    "exception",
                    "consent_required" if requires_consent else "no_consent_required",
                ],
            )

        return GateResult(
            gate="gate3",
            decision=GateDecision.BLOCK,
            reason=f"license {license_id} is not approved and not exception-eligible",
            details=["blocked: unknown license"],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tier_from_findings(self, findings: list[PiiFinding]) -> PrivacyTier:
        if not findings:
            return PrivacyTier.NONE
        total = sum(f.count for f in findings)
        has_phi = any(f.pii_type in ("ssn", "medical_record_number", "dob") for f in findings)
        has_name = any(f.pii_type == "name" for f in findings)
        if has_phi or total >= PII_HIGH_THRESHOLD:
            return PrivacyTier.HIGH
        if total >= 2 or has_name:
            return PrivacyTier.MEDIUM
        return PrivacyTier.LOW

    def _sensitivity_from_text(self, text: str) -> ContentSensitivity:
        text_lower = text.lower()
        if any(re.search(p, text_lower) for p in RESTRICTED_PATTERNS):
            return ContentSensitivity.RESTRICTED
        if any(re.search(p, text_lower) for p in SENSITIVE_PATTERNS):
            return ContentSensitivity.SENSITIVE
        return ContentSensitivity.NORMAL

    def _retention_for_tier(self, tier: PrivacyTier) -> RetentionPolicy:
        return {
            PrivacyTier.NONE: RetentionPolicy.USE_IMMEDIATELY,
            PrivacyTier.LOW: RetentionPolicy.SCRUB_AND_USE,
            PrivacyTier.MEDIUM: RetentionPolicy.SCRUB_AND_USE,
            PrivacyTier.HIGH: RetentionPolicy.REVIEW_THEN_USE,
            PrivacyTier.PROHIBITED: RetentionPolicy.REJECT,
        }[tier]

    def _collect_crisis_findings(
        self,
        crisis_result: CrisisInterventionResult,  # type: ignore[name-defined]
    ) -> list[CrisisFinding]:
        if crisis_result.score == 0.0:
            return []
        return [
            CrisisFinding(
                crisis_type=crisis_result.crisis_type,
                severity=crisis_result.severity,
                score=crisis_result.score,
                requires_escalation=crisis_result.score >= CRISIS_ESCALATE_THRESHOLD,
            )
        ]

    def _detect_unsafe_patterns(self, text: str) -> list[str]:
        found: list[str] = []
        for label, pattern in UNSAFE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(label)
        return found

    def _enqueue_for_review(self, report: PrivacyContentReport, text: str) -> None:
        """Enqueue item for human review when ESCALATE occurs (PIX-250).

        Args:
            report: The PrivacyContentReport with ESCALATE decision
            text: Original text content for preview
        """
        if not REVIEW_QUEUE_ENABLED or self._review_queue is None:
            return

        try:
            # Create review item from gate report
            review_item = self._review_queue.create_item_from_report(
                source_id=report.source_id,
                gate_result=report.to_dict(),
                content_preview=text[:500] if text else None,
                content_length=len(text) if text else 0,
                priority="high" if report.privacy_tier == PrivacyTier.HIGH else "normal",
            )
            # Enqueue for review
            self._review_queue.enqueue(review_item)
        except Exception:
            # Queue operation failed; item will still be marked as needs_review
            # Debug logging recommended but don't block the pipeline
            pass


__all__ = [
    "APPROVED_LICENSES",
    "EXCEPTION_LICENSES",
    "ContentSensitivity",
    "CrisisFinding",
    "GateDecision",
    "GateResult",
    "LicenseCheck",
    "PiiFinding",
    "PrivacyContentGates",
    "PrivacyContentReport",
    "PrivacyTier",
    "RetentionPolicy",
]
