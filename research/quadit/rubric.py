"""Quadit rubric — thresholds, severity weights, and shared signal tables."""

from __future__ import annotations

# A persona passes at >= 0.7 (ported from the hackathon corpus quadit).
PASS_THRESHOLD = 0.70

# Severity ladder, ordered. A single critical flips the audit verdict to FAIL
# regardless of judge scores; warnings pollute the noise without flipping.
SEVERITY_ORDER: tuple[str, ...] = ("info", "warning", "critical")

SEVERITY_WEIGHTS: dict[str, float] = {"info": 0.0, "warning": 0.25, "critical": 1.0}


def severity_weight(severity: str) -> float:
    """Weight for a severity label; unknown labels count as critical."""
    return SEVERITY_WEIGHTS.get(severity, SEVERITY_WEIGHTS["critical"])


# Hard voice constraints shared by the Voice Fidelity judge (deterministic
# fallback scans these verbatim; the LLM judge gets the full prompt).
BANNED_PHRASES: tuple[str, ...] = (
    "circle back",
    "double-click",
    "synergize",
    "I hope this finds you well",
)

DIRECTOR_SUPERLATIVES: tuple[str, ...] = (
    "amazing",
    "incredible",
    "game-changing",
)

# Anti-signal taxonomy — each label maps onto the canonical body-defect
# buckets used for the cross-reference fields in adapter reports.
ANTI_SIGNAL_TAXONOMY: dict[str, list[str]] = {
    "armoring-as-strong-leadership": ["stacked_salutation", "repeated_signoff"],
    "platitude-without-cost": ["tout_empathy_without_cost"],
    "clinical-abstraction-over-warmth": ["clinical_abstraction_over_warmth"],
    "premature-let-it-go-closure": ["premature_let_it_go_closure"],
    "dismissive-strength-talk": ["performative_toughness_as_armor"],
    "shame-guilt-blur": ["shame_guilt_blur"],
    "missing-emotion-naming": ["missing_emotion_naming"],
}

# Deterministic platitude patterns used by the adversarial fallback scorer.
PLATITUDE_PATTERNS: tuple[str, ...] = (
    "treat yourself kindly",
    "be the change",
    "it's ok to not be ok",
    "i'm here for you",
    "i understand this is a challenging time",
)
