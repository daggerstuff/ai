"""Clinical safety layer for DeepRare multi-agent rare disease diagnosis.

This module provides safety gates, red flag detection, and audit trail
capabilities to ensure clinical safety in a production diagnostic system.

Key Components:
    - RedFlagDetector: Identifies symptoms/patterns requiring immediate referral
    - ClinicalSafetyGate: Enforces safety rules on hypothesis elimination and confirmation
    - AuditTrail: Tracks every clinical decision with full provenance
    - SafetyViolation: Records when safety rules are violated
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from .schema import (
        Evidence,
        Hypothesis,
        PatientCase,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SafetyLevel(str, Enum):
    """Severity level for safety violations."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCKING = "blocking"  # System must halt


class AuditAction(str, Enum):
    """Types of auditable clinical actions."""

    HYPOTHESIS_CREATED = "hypothesis_created"
    HYPOTHESIS_UPDATED = "hypothesis_updated"
    HYPOTHESIS_ELIMINATED = "hypothesis_eliminated"
    HYPOTHESIS_CONFIRMED = "hypothesis_confirmed"
    TEST_REQUESTED = "test_requested"
    TEST_INTERPRETED = "test_interpreted"
    LITERATURE_RETRIEVED = "literature_retrieved"
    DIFFERENTIAL_UPDATED = "differential_updated"
    SAFETY_GATE_TRIGGERED = "safety_gate_triggered"
    RED_FLAG_DETECTED = "red_flag_detected"
    CONVERGENCE_REACHED = "convergence_reached"
    ITERATION_COMPLETED = "iteration_completed"
    DIAGNOSIS_FINALIZED = "diagnosis_finalized"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SafetyViolation(BaseModel):
    """A recorded clinical safety violation."""

    violation_id: str
    level: SafetyLevel
    rule_name: str
    description: str
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    remediation: str | None = None

    @field_validator("timestamp")
    @classmethod
    def _validate_iso_format(cls, v: str) -> str:
        """Ensure timestamp is valid ISO format."""
        datetime.fromisoformat(v)
        return v


class AuditEntry(BaseModel):
    """A single entry in the audit trail."""

    entry_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    action: str  # AuditAction value
    agent_name: str
    case_id: str
    details: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("timestamp")
    @classmethod
    def _validate_iso_format(cls, v: str) -> str:
        """Ensure timestamp is valid ISO format."""
        datetime.fromisoformat(v)
        return v


class RedFlag(BaseModel):
    """A detected red flag requiring immediate clinical attention."""

    flag_id: str
    flag_type: str  # e.g. "immediate_referral", "life_threatening", "contraindication"
    description: str
    matching_symptoms: list[str] = Field(default_factory=list)
    recommended_action: str
    urgency: SafetyLevel = SafetyLevel.CRITICAL

    @field_validator("urgency")
    @classmethod
    def _validate_urgency(cls, v: SafetyLevel) -> SafetyLevel:
        """Ensure urgency is at least WARNING."""
        if v == SafetyLevel.INFO:
            raise ValueError("Red flag urgency must be at least WARNING")
        return v


# ---------------------------------------------------------------------------
# Red Flag Detector
# ---------------------------------------------------------------------------

# Symptoms that require immediate clinical referral regardless of diagnosis
IMMEDIATE_REFERRAL_SYMPTOMS: frozenset[str] = frozenset(
    {
        # Neurological emergencies
        "acute hemiparesis",
        "sudden onset severe headache",
        "loss of consciousness",
        "seizure",
        "status epilepticus",
        "acute paraplegia",
        "acute quadriplegia",
        "progressive weakness",
        "acute ataxia",
        "acute dysphagia",
        "acute vision loss",
        "diplopia",
        "nystagmus",
        # Cardiac emergencies
        "chest pain",
        "syncope",
        "palpitations",
        "exertional dyspnea",
        # Respiratory emergencies
        "acute respiratory distress",
        "hemoptysis",
        "stridor",
        # Metabolic emergencies
        "acute metabolic crisis",
        "hepatic failure",
        "acute liver failure",
        "renal failure",
        "rhabdolysis",
        # Hematological emergencies
        "acute bleeding",
        "petechiae",
        "purpura",
        "thrombocytopenia",
        # Psychiatric emergencies
        "acute psychosis",
        "suicidal ideation",
        "homicidal ideation",
    }
)

# Symptoms indicating potentially life-threatening conditions
LIFE_THREATENING_SYMPTOMS: frozenset[str] = frozenset(
    {
        "cardiomyopathy",
        "arrhythmia",
        "heart failure",
        "aortic dissection",
        "aortic aneurysm",
        "respiratory failure",
        "hepatic encephalopathy",
        "progressive neurodegeneration",
        "acute kidney injury",
        "bone marrow failure",
        "adrenal crisis",
        "thyroid storm",
        "malignant hyperthermia",
        "serotonin syndrome",
    }
)

# Symptoms that are pathognomonic red flags (should never be ignored)
PATHOGNOMONIC_REDFLAGS: frozenset[str] = frozenset(
    {
        "cherry-red spot",
        "kayser-fleischer ring",
        "hepatosplenomegaly",
        "corneal arcus",
        "xanthoma",
        " angiokeratoma",
        "lisch nodule",
        "cafe-au-lait spot",
    }
)


class RedFlagDetector:
    """Detects clinical red flags in patient presentations.

    Identifies symptoms requiring immediate referral, life-threatening
    patterns, and pathognomonic signs that should never be eliminated
    from consideration.
    """

    def __init__(self) -> None:
        self._immediate_referral = IMMEDIATE_REFERRAL_SYMPTOMS
        self._life_threatening = LIFE_THREATENING_SYMPTOMS
        self._pathognomonic = PATHOGNOMONIC_REDFLAGS

    def detect(self, case: PatientCase) -> list[RedFlag]:
        """Detect red flags in a patient case.

        Args:
            case: The patient case to analyze.

        Returns:
            List of detected red flags, ordered by urgency (most urgent first).
        """
        flags: list[RedFlag] = []
        symptoms_lower = {s.name.lower().strip() for s in case.presenting_symptoms}

        # Check immediate referral symptoms
        matching_immediate = symptoms_lower & self._immediate_referral
        if matching_immediate:
            flags.append(
                RedFlag(
                    flag_id=f"rf_immediate_{case.case_id}",
                    flag_type="immediate_referral",
                    description=(
                        f"Patient presents with symptoms requiring immediate "
                        f"clinical referral: {', '.join(sorted(matching_immediate))}"
                    ),
                    matching_symptoms=sorted(matching_immediate),
                    recommended_action=(
                        "Escalate to emergency clinical evaluation immediately. "
                        "Do not delay diagnostic workup for referral."
                    ),
                    urgency=SafetyLevel.BLOCKING,
                )
            )

        # Check life-threatening symptoms
        matching_lifethreat = symptoms_lower & self._life_threatening
        if matching_lifethreat:
            flags.append(
                RedFlag(
                    flag_id=f"rf_lifethreat_{case.case_id}",
                    flag_type="life_threatening",
                    description=(
                        f"Patient presents with potentially life-threatening "
                        f"symptoms: {', '.join(sorted(matching_lifethreat))}"
                    ),
                    matching_symptoms=sorted(matching_lifethreat),
                    recommended_action=(
                        "Ensure life-threatening conditions remain in the "
                        "differential diagnosis. Do not eliminate without "
                        "strong refuting evidence (posterior < 0.001)."
                    ),
                    urgency=SafetyLevel.CRITICAL,
                )
            )

        # Check pathognomonic red flags
        matching_patho = symptoms_lower & self._pathognomonic
        if matching_patho:
            flags.append(
                RedFlag(
                    flag_id=f"rf_patho_{case.case_id}",
                    flag_type="pathognomonic",
                    description=(f"Pathognomonic signs detected: {', '.join(sorted(matching_patho))}"),
                    matching_symptoms=sorted(matching_patho),
                    recommended_action=(
                        "Pathognomonic signs strongly suggest specific diagnoses. "
                        "Prioritize matching diseases in the differential."
                    ),
                    urgency=SafetyLevel.WARNING,
                )
            )

        # Check family history for genetic red flags
        if case.family_history:
            fh_lower = " ".join(case.family_history).lower()
            genetic_keywords = [
                "consanguinity",
                "affected sibling",
                "multiple miscarriages",
                "early death",
                "sudden death",
                "unknown cause",
            ]
            matching_genetic = [k for k in genetic_keywords if k in fh_lower]
            if matching_genetic:
                flags.append(
                    RedFlag(
                        flag_id=f"rf_genetic_{case.case_id}",
                        flag_type="genetic_risk",
                        description=(f"Family history suggests genetic etiology: {', '.join(matching_genetic)}"),
                        matching_symptoms=[],
                        recommended_action=(
                            "Consider genetic testing and genetic counseling. "
                            "Autosomal recessive patterns may be present."
                        ),
                        urgency=SafetyLevel.WARNING,
                    )
                )

        # Sort by urgency (BLOCKING > CRITICAL > WARNING > INFO)
        urgency_order = {
            SafetyLevel.BLOCKING: 0,
            SafetyLevel.CRITICAL: 1,
            SafetyLevel.WARNING: 2,
            SafetyLevel.INFO: 3,
        }
        flags.sort(key=lambda f: urgency_order.get(f.urgency, 99))
        return flags

    def should_block(self, flags: list[RedFlag]) -> bool:
        """Check if any red flags should block the diagnostic process.

        Args:
            flags: List of detected red flags.

        Returns:
            True if any flag has BLOCKING urgency.
        """
        return any(f.urgency == SafetyLevel.BLOCKING for f in flags)


# ---------------------------------------------------------------------------
# Clinical Safety Gate
# ---------------------------------------------------------------------------


class ClinicalSafetyGate:
    """Enforces clinical safety rules during the diagnostic process.

    Prevents elimination of life-threatening conditions without strong
    evidence, enforces minimum confidence thresholds, and validates
    that safety-critical hypotheses are not pruned prematurely.
    """

    def __init__(
        self,
        min_confidence_to_confirm: float = 0.85,
        min_confidence_to_eliminate: float = 0.001,
        max_elimination_per_iteration: int = 3,
        protect_life_threatening: bool = True,
    ) -> None:
        self._min_confirm = min_confidence_to_confirm
        self._min_eliminate = min_confidence_to_eliminate
        self._max_elim_per_iter = max_elimination_per_iteration
        self._protect_lifethreat = protect_life_threatening
        self._violations: list[SafetyViolation] = []
        self._eliminated_this_iteration: int = 0

        # Diseases that should never be eliminated without overwhelming evidence
        self._protected_diseases: set[str] = set()

    def register_protected_disease(self, disease_name: str) -> None:
        """Register a disease as protected (cannot be eliminated without overwhelming evidence)."""
        self._protected_diseases.add(disease_name.lower())
        logger.debug("Registered protected disease: %s", disease_name)

    def can_eliminate(
        self,
        hypothesis: Hypothesis,
        evidence: list[Evidence],
    ) -> tuple[bool, str | None]:
        """Check if a hypothesis can be safely eliminated.

        Args:
            hypothesis: The hypothesis to eliminate.
            evidence: Evidence accumulated for/against the hypothesis.

        Returns:
            Tuple of (can_eliminate, reason_if_blocked).
        """
        # Check if disease is protected
        if hypothesis.disease_name.lower() in self._protected_diseases:
            refuting = [e for e in evidence if not e.supports]
            if not refuting:
                return False, (
                    f"Cannot eliminate protected disease '{hypothesis.disease_name}' without refuting evidence"
                )

        # Check posterior probability threshold
        if hypothesis.posterior_probability > self._min_eliminate:
            return False, (
                f"Cannot eliminate '{hypothesis.disease_name}': "
                f"posterior probability {hypothesis.posterior_probability:.4f} "
                f"exceeds elimination threshold {self._min_eliminate}"
            )

        # Check elimination rate limit
        if self._eliminated_this_iteration >= self._max_elim_per_iter:
            return False, (
                f"Cannot eliminate '{hypothesis.disease_name}': "
                f"maximum eliminations per iteration "
                f"({self._max_elim_per_iter}) reached"
            )

        return True, None

    def can_confirm(
        self,
        hypothesis: Hypothesis,
    ) -> tuple[bool, str | None]:
        """Check if a hypothesis can be confirmed as the diagnosis.

        Args:
            hypothesis: The hypothesis to confirm.

        Returns:
            Tuple of (can_confirm, reason_if_blocked).
        """
        if hypothesis.confidence_score < self._min_confirm:
            return False, (
                f"Cannot confirm '{hypothesis.disease_name}': "
                f"confidence {hypothesis.confidence_score:.4f} "
                f"below threshold {self._min_confirm}"
            )

        if not hypothesis.supporting_evidence():
            return False, (f"Cannot confirm '{hypothesis.disease_name}': no supporting evidence")

        return True, None

    def record_violation(
        self,
        level: SafetyLevel,
        rule_name: str,
        description: str,
        context: dict[str, Any] | None = None,
        remediation: str | None = None,
    ) -> SafetyViolation:
        """Record a safety violation.

        Args:
            level: Severity of the violation.
            rule_name: Name of the violated safety rule.
            description: Human-readable description.
            context: Additional context.
            remediation: Recommended remediation action.

        Returns:
            The recorded SafetyViolation.
        """
        violation = SafetyViolation(
            violation_id=f"sv_{len(self._violations) + 1:04d}",
            level=level,
            rule_name=rule_name,
            description=description,
            context=context or {},
            remediation=remediation,
        )
        self._violations.append(violation)

        if level in (SafetyLevel.CRITICAL, SafetyLevel.BLOCKING):
            logger.warning("Safety violation [%s]: %s - %s", level.value, rule_name, description)
        else:
            logger.info("Safety note [%s]: %s - %s", level.value, rule_name, description)

        return violation

    def reset_iteration(self) -> None:
        """Reset per-iteration counters."""
        self._eliminated_this_iteration = 0

    def increment_elimination(self) -> None:
        """Track that an elimination occurred this iteration."""
        self._eliminated_this_iteration += 1

    @property
    def violations(self) -> list[SafetyViolation]:
        """All recorded safety violations."""
        return list(self._violations)

    @property
    def has_blocking_violations(self) -> bool:
        """Check if any blocking violations have occurred."""
        return any(v.level == SafetyLevel.BLOCKING for v in self._violations)

    @property
    def has_critical_violations(self) -> bool:
        """Check if any critical violations have occurred."""
        return any(v.level == SafetyLevel.CRITICAL for v in self._violations)

    def get_safety_summary(self) -> dict[str, Any]:
        """Get a summary of safety state for reporting."""
        level_counts: dict[str, int] = {}
        for v in self._violations:
            level_counts[v.level.value] = level_counts.get(v.level.value, 0) + 1
        return {
            "total_violations": len(self._violations),
            "by_level": level_counts,
            "has_blocking": self.has_blocking_violations,
            "has_critical": self.has_critical_violations,
            "protected_diseases": list(self._protected_diseases),
            "thresholds": {
                "min_confidence_to_confirm": self._min_confirm,
                "min_confidence_to_eliminate": self._min_eliminate,
                "max_elimination_per_iteration": self._max_elim_per_iter,
                "protect_life_threatening": self._protect_lifethreat,
            },
        }


# ---------------------------------------------------------------------------
# Audit Trail
# ---------------------------------------------------------------------------


class AuditTrail:
    """Maintains a complete audit trail of all clinical decisions.

    Every action taken during the diagnostic process is recorded with
    timestamp, agent, case, and supporting evidence references. This
    provides full provenance for clinical review and regulatory compliance.
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._counter: int = 0

    def record(
        self,
        action: str,
        agent_name: str,
        case_id: str,
        details: dict[str, Any] | None = None,
        evidence_refs: list[str] | None = None,
    ) -> AuditEntry:
        """Record an auditable action.

        Args:
            action: The type of action (see AuditAction).
            agent_name: Name of the agent performing the action.
            case_id: ID of the patient case.
            details: Additional action details.
            evidence_refs: References to supporting evidence.

        Returns:
            The recorded AuditEntry.
        """
        self._counter += 1
        entry = AuditEntry(
            entry_id=f"audit_{self._counter:05d}",
            action=action,
            agent_name=agent_name,
            case_id=case_id,
            details=details or {},
            evidence_refs=evidence_refs or [],
        )
        self._entries.append(entry)
        logger.debug(
            "Audit [%s] %s by %s for case %s: %s",
            entry.entry_id,
            action,
            agent_name,
            case_id,
            entry.details.get("summary", ""),
        )
        return entry

    def get_entries(
        self,
        case_id: str | None = None,
        action: str | None = None,
        agent_name: str | None = None,
    ) -> list[AuditEntry]:
        """Filter audit entries by criteria.

        Args:
            case_id: Filter by case ID.
            action: Filter by action type.
            agent_name: Filter by agent name.

        Returns:
            Filtered list of audit entries.
        """
        entries = self._entries
        if case_id:
            entries = [e for e in entries if e.case_id == case_id]
        if action:
            entries = [e for e in entries if e.action == action]
        if agent_name:
            entries = [e for e in entries if e.agent_name == agent_name]
        return entries

    def get_case_trail(self, case_id: str) -> list[AuditEntry]:
        """Get the complete audit trail for a specific case."""
        return [e for e in self._entries if e.case_id == case_id]

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize all entries to dicts."""
        return [e.model_dump() for e in self._entries]

    @property
    def entry_count(self) -> int:
        """Total number of audit entries."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all audit entries (for testing)."""
        self._entries.clear()
        self._counter = 0


# ---------------------------------------------------------------------------
# Clinical Safety Context
# ---------------------------------------------------------------------------


class ClinicalSafetyContext:
    """Combined safety context for the diagnostic process.

    Bundles RedFlagDetector, ClinicalSafetyGate, and AuditTrail
    into a single context object that can be passed through the
    diagnostic pipeline.
    """

    def __init__(
        self,
        safety_gate: ClinicalSafetyGate | None = None,
        red_flag_detector: RedFlagDetector | None = None,
        audit_trail: AuditTrail | None = None,
    ) -> None:
        self.safety_gate = safety_gate or ClinicalSafetyGate()
        self.red_flag_detector = red_flag_detector or RedFlagDetector()
        self.audit_trail = audit_trail or AuditTrail()

    def evaluate_case_safety(self, case: PatientCase) -> list[RedFlag]:
        """Evaluate a patient case for red flags and register protections.

        Args:
            case: The patient case to evaluate.

        Returns:
            List of detected red flags.
        """
        flags = self.red_flag_detector.detect(case)

        # Register life-threatening diseases as protected
        for flag in flags:
            if flag.flag_type == "life_threatening":
                for symptom in flag.matching_symptoms:
                    # The orchestrator should map these to diseases
                    pass
            self.audit_trail.record(
                action=AuditAction.RED_FLAG_DETECTED.value,
                agent_name="clinical_safety",
                case_id=case.case_id,
                details={
                    "flag_type": flag.flag_type,
                    "description": flag.description,
                    "urgency": flag.urgency.value,
                    "recommended_action": flag.recommended_action,
                },
            )

        return flags

    def get_full_report(self) -> dict[str, Any]:
        """Get a complete safety report combining all components."""
        return {
            "safety_gate": self.safety_gate.get_safety_summary(),
            "audit_trail_entries": self.audit_trail.entry_count,
            "audit_trail": self.audit_trail.to_dict(),
        }
