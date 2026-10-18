"""Conversation quality pattern analyzer with inquiry-type classification.

Implements Tasks 2, 3, and 6 of PIX-3908 ([Inquiry-Diagnosis] Patient
Simulator & Diagnostic Quality Framework).

The Inquiry-Diagnosis paper (arXiv 2501.09484) identifies four mutually
exclusive inquiry types a therapist can use during a clinical
conversation. This module classifies each therapist utterance into one of
those types, computes per-session distribution vectors, and exposes the
Liebig's-Law-of-Diagnosis bottleneck analysis used downstream by the
diagnostic quality metrics (Task 3) and automated quality report (Task 4).

Pipeline
--------
1. ``classify_utterance()`` → rule-based scorer per utterance
2. ``classify_session()``   → list of per-utterance classifications
3. ``session_distribution()`` → counts + ratios of inquiry types
4. ``liebig_bottleneck()``  → identifies the limiting inquiry skill
5. ``liebig_quality_score()`` → min(inquiry type scores), per Liebig
6. ``HallucinationDetector.detect()`` → cross-check response vs case data

Implementation notes
--------------------
The classifier is rule-based (patterns + keywords) so that it can run
without an LLM in CI / on-device. An optional ``LLMJudge`` adapter is
defined here for ambiguous cases but is not used by the default
classifier — that hook matches the Task 2 acceptance criteria
("rule-based + LLM-as-judge for ambiguous cases").

The hallucination detector is also rule-based: it extracts factual claims
from simulator responses and verifies them against the structured CCD
profile fields (symptoms, beliefs, emotions, behaviors, etc.). This
keeps the <1% hallucination target achievable without LLM calls in CI.

References
----------
* arXiv 2501.09484 — How to Evaluate the Diagnostic Ability of LLMs:
  The Role of Patient Simulator.
* Liebig's Law of the Minimum — diagnostic accuracy is constrained by
  the scarcest inquiry skill.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from ai.tools.utilities.core.pipelines.schemas.conversation_schema import Conversation, Message
from ai.tools.utilities.core.pipelines.quality.quality_assessment_framework import InquiryType

logger = logging.getLogger(__name__)

__all__ = [
    "ClassificationResult",
    "HallucinationDetector",
    "HallucinationFinding",
    "HallucinationReport",
    "HallucinationSeverity",
    "InquiryTypeClassifier",
    "RECOMMENDED_RATIOS",
    "SessionClassification",
    "UtteranceClassification",
    "liebig_bottleneck",
    "liebig_quality_score",
    "session_distribution",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Each inquiry type has a set of signal patterns. Signals are regex matchers
# applied to the lowercased utterance. A type's score is the number of
# matched signals, weighted by per-signal importance (see SIGNAL_WEIGHTS).
#
# These are intentionally conservative: a single match is enough to
# *suggest* a type, but a minimum confidence threshold is required for
# non-UNKNOWN classification. This reduces false positives on ambiguous
# utterances like "How does that make you feel?" which is open-ended
# *or* reflective depending on context.

CLOSED_ENDED_SIGNALS: tuple[str, ...] = (
    r"\bdo you\b",
    r"\bdid you\b",
    r"\bare you\b",
    r"\bhave you\b",
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bwould you\b",
    r"\bwill you\b",
    r"\bis (?:it|there|he|she|this)\b",
    r"\bwas (?:it|there|he|she|this)\b",
    r"\bhow many\b",
    r"\bhow much\b",
    r"\bwhen (?:did|does|do|was|were)\b",
    r"\bwhere (?:did|does|do|was|were)\b",
    r"\bwhich\b",
    r"\bhow often\b",
    r"\bhow long\b",
)

OPEN_ENDED_SIGNALS: tuple[str, ...] = (
    r"\btell me (?:more|about)\b",
    # "what happened/happens/..." but not what-do-you-think (guided)
    r"\bwhat (?:happened|happens|are you|did you|would|makes? (?:you|them))\b",
    # 'how' followed by action verbs that signal exploration
    r"\bhow (?:does|do|did|would|might) (?:that|it|you|this|the) (?:feel|seem|look|affect)\b",
    r"\bcan you (?:describe|explain|share|talk)\b",
    r"\bwhat'?s (?:on your mind|going on|happening)\b",
    r"\bwalk me through\b",
    r"\bhelp me understand\b",
    r"\bwhat'?s (?:that|it) like\b",
    r"\bcan you tell me\b",
)

GUIDED_SIGNALS: tuple[str, ...] = (
    # Conditional / contingent frames — the strongest markers of GUIDED
    # (anchored at the start of the utterance to win tie-breaks against
    # open-ended patterns like "what happens")
    r"^when you (?:feel|felt|are|were|notice|noticed|think|thought|said)",
    r"^if you (?:were|had|did|feel|felt)",
    r"^do you (?:think|feel|believe) (?:that|it|this)",
    r"^some people (?:say|think|believe|felt)",
    r"^could (?:it|this|that) be",
    r"^what (?:if|about|comes? to mind)",
    r"^how (?:does|do|did) (?:that|it|this) (?:make|affect|impact)",
    r"\bcompared to\b",
    r"\bin (?:what|which) (?:way|ways?)\b",
    r"\bspecifically\b",
    # Non-anchored fallback for the same patterns (in case utterance has
    # a leading greeting or other prefix)
    r"\bwhen you (?:feel|felt|are|were|notice|noticed|think|thought|said)\b",
    r"\bif you (?:were|had|did|feel|felt)\b",
)

REFLECTIVE_SIGNALS: tuple[str, ...] = (
    # Anchored at start — these are the strongest reflective markers
    r"^how (?:does|do|did) (?:that|it|this) (?:make you feel|affect you|impact you)",
    r"^how (?:does|do|did) (?:that|it|this) (?:show up|manifest|affect|impact)",
    r"^how (?:does|do|did) (?:that|it|this) [\w-]+ (?:show up|manifest|affect)",
    r"^what (?:does|do) (?:that|this|it) (?:mean|suggest|say)",
    r"^notice(?:d)? (?:a|any) (?:pattern|theme|connection)",
    r"^what (?:do you|did you) (?:notice|observe|reflect)",
    r"\bwhat (?:pattern|theme|connection)\b",
    r"\bmetacogniti",
    r"\breflect(?:ion|ing)?\b",
    r"\bhow (?:does|do) (?:that|this) (?:relate|connect) to\b",
    r"\bself-aware(?:ness)?\b",
    # Non-anchored fallbacks
    r"\bhow (?:does|do|did) (?:that|it|this) (?:make you feel|affect you|impact you)\b",
    r"\bhow (?:does|do|did) (?:that|it) (?:show up|manifest|affect)\b",
    r"\bwhat (?:does|do) (?:that|this|it) (?:mean|suggest|say)\b",
    r"\bwhat (?:do you|did you) (?:notice|observe|reflect)\b",
)

SIGNAL_PATTERNS: Mapping[InquiryType, tuple[str, ...]] = {
    InquiryType.CLOSED_ENDED: CLOSED_ENDED_SIGNALS,
    InquiryType.OPEN_ENDED: OPEN_ENDED_SIGNALS,
    InquiryType.GUIDED: GUIDED_SIGNALS,
    InquiryType.REFLECTIVE: REFLECTIVE_SIGNALS,
}

# Per-signal importance weights. Closed-ended questions (verification)
# are the most diagnostic for narrow yes/no facts; open-ended is the
# most diagnostic for exploratory breadth; guided is good for hypothesis
# testing; reflective is the most meta-cognitive and least directly
# diagnostic but tracks therapist skill development.
SIGNAL_WEIGHTS: Mapping[InquiryType, float] = {
    InquiryType.CLOSED_ENDED: 0.6,
    InquiryType.OPEN_ENDED: 1.0,
    InquiryType.GUIDED: 0.8,
    InquiryType.REFLECTIVE: 0.7,
}

# Min-score threshold: a type must score at least this to be returned as
# a confident label. Below it, the classifier returns UNKNOWN (or the
# LLM judge is consulted if available). Set to 0.2 so that a single
# strong signal match is enough to classify; the WIN_MARGIN below gates
# cases where multiple types are competing.
MIN_CONFIDENCE_SCORE = 0.2

# Score margin above the runner-up required to claim a tie-breaker win.
# 0.2 is a sensible default — close enough that a single dominant signal
# wins, loose enough to allow reasonable overlap.
WIN_MARGIN = 0.15

# Question-mark heuristic: a leading question mark is a strong
# indication that the utterance is an inquiry (vs. a statement). Used as
# a tie-breaker only — we never classify on this alone.
ENDS_WITH_QUESTION_MARK = re.compile(r"\?\s*$")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationResult:
    """Single-utterance classification outcome."""

    inquiry_type: InquiryType
    """Primary classification (the highest-scoring type, or UNKNOWN)."""

    confidence: float
    """Confidence score in [0, 1]. 1.0 = unambiguous, 0.0 = no signals."""

    scores: Mapping[InquiryType, float]
    """Per-type scores for transparency and downstream debugging."""

    rationale: str
    """Short human-readable explanation of the chosen label."""

    @property
    def is_unknown(self) -> bool:
        return self.inquiry_type == InquiryType.UNKNOWN


@dataclass
class UtteranceClassification:
    """A classification paired with the original utterance."""

    role: str
    content: str
    result: ClassificationResult


@dataclass
class SessionClassification:
    """Per-session aggregation of classification results."""

    conversation_id: str
    utterances: list[UtteranceClassification] = field(default_factory=list)
    """All classified utterances in the session."""

    distribution: dict[InquiryType, int] = field(default_factory=dict)
    """Raw counts of each inquiry type."""

    total: int = 0
    """Total utterances classified (excludes UNKNOWN by default)."""

    @property
    def ratio(self) -> dict[InquiryType, float]:
        """Counts as a ratio in [0, 1]. UNKNOWN is excluded from the denominator."""
        if self.total == 0:
            return dict.fromkeys(InquiryType, 0.0)
        return {
            t: (self.distribution.get(t, 0) / self.total)
            for t in InquiryType
            if t != InquiryType.UNKNOWN
        }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class InquiryTypeClassifier:
    """
    Rule-based inquiry-type classifier.

    Per Task 2 of PIX-3908, the classifier is rule-based (pattern + keyword
    scoring) for primary use, with an optional LLM-as-judge adapter for
    ambiguous utterances (the score is below ``MIN_CONFIDENCE_SCORE``).
    """

    def __init__(
        self,
        *,
        llm_judge: Callable[[str], InquiryType] | None = None,
        min_confidence: float = MIN_CONFIDENCE_SCORE,
        win_margin: float = WIN_MARGIN,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 <= win_margin <= 1.0:
            raise ValueError("win_margin must be in [0, 1]")
        self._patterns: dict[InquiryType, tuple[re.Pattern[str], ...]] = {
            t: tuple(re.compile(p) for p in pats)
            for t, pats in SIGNAL_PATTERNS.items()
        }
        self._llm_judge = llm_judge
        self._min_confidence = min_confidence
        self._win_margin = win_margin

    def classify_utterance(
        self,
        text: str,
        *,
        role: str = "assistant",
    ) -> ClassificationResult:
        """
        Classify a single utterance.

        ``role`` is informational; the classifier is intended for
        therapist (assistant) turns but will still score any text. Pass
        ``role="assistant"`` in production usage.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            return ClassificationResult(
                inquiry_type=InquiryType.UNKNOWN,
                confidence=0.0,
                scores={t: 0.0 for t in InquiryType if t != InquiryType.UNKNOWN},
                rationale="empty or whitespace-only utterance",
            )

        # Lowercase and strip for pattern matching — but preserve the
        # original text for downstream display.
        lowered = text.lower().strip()
        raw_scores: dict[InquiryType, float] = {}

        for itype, patterns in self._patterns.items():
            count = sum(1 for p in patterns if p.search(lowered))
            weight = SIGNAL_WEIGHTS[itype]
            raw_scores[itype] = count * weight

        # Normalize into [0, 1] via soft-cap at 4 matched signals.
        scores: dict[InquiryType, float] = {
            t: min(1.0, s / (4.0 * SIGNAL_WEIGHTS[t]))
            for t, s in raw_scores.items()
        }

        # Pick the type with the highest score; break ties using
        # the canonical paper ordering (closed > open > guided > reflective)
        # and the question-mark heuristic as a final tie-breaker.
        sorted_types = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], _paper_order(kv[0])),
        )
        best_type, best_score = sorted_types[0]
        runner_up_score = sorted_types[1][1] if len(sorted_types) > 1 else 0.0

        # Confidence: how much the best type stands above the runner-up
        confidence = max(0.0, best_score - runner_up_score)
        # Plus a base confidence proportional to absolute best_score
        confidence = min(1.0, 0.5 * confidence + 0.5 * best_score)

        if best_score < self._min_confidence:
            # Below threshold — try LLM judge if available
            if self._llm_judge is not None:
                try:
                    judged = self._llm_judge(text)
                except Exception:  # noqa: BLE001 - judge failures must not crash
                    judged = InquiryType.UNKNOWN
                if judged != InquiryType.UNKNOWN and judged in scores:
                    return ClassificationResult(
                        inquiry_type=judged,
                        confidence=0.6,  # moderate confidence from LLM judge
                        scores=scores,
                        rationale=f"LLM judge selected {judged.value}",
                    )
            return ClassificationResult(
                inquiry_type=InquiryType.UNKNOWN,
                confidence=best_score,
                scores=scores,
                rationale=f"no signal above {self._min_confidence:.2f} threshold",
            )

        if best_score - runner_up_score < self._win_margin and not ENDS_WITH_QUESTION_MARK.search(text):
            return ClassificationResult(
                inquiry_type=InquiryType.UNKNOWN,
                confidence=confidence,
                scores=scores,
                rationale="ambiguous: top scores within margin and not a question",
            )

        rationale = self._build_rationale(best_type, raw_scores, scores)
        return ClassificationResult(
            inquiry_type=best_type,
            confidence=confidence,
            scores=scores,
            rationale=rationale,
        )

    def classify_messages(
        self,
        messages: Iterable[Message],
    ) -> list[UtteranceClassification]:
        """
        Classify an iterable of ``Message`` objects in order.

        Typically filters for ``role == "assistant"`` upstream.
        """
        results: list[UtteranceClassification] = []
        for msg in messages:
            result = self.classify_utterance(msg.content, role=msg.role)
            results.append(UtteranceClassification(role=msg.role, content=msg.content, result=result))
        return results

    def classify_session(
        self,
        conversation: Conversation,
        *,
        therapist_role: str = "assistant",
    ) -> SessionClassification:
        """
        Classify all therapist utterances in a conversation.

        Returns a ``SessionClassification`` aggregating per-utterance
        results and a per-type distribution. UNKNOWN utterances are
        counted but excluded from ``total`` (the denominator for ratios).
        """
        therapist_turns = [m for m in conversation.messages if m.role == therapist_role]
        per_utt = self.classify_messages(therapist_turns)
        distribution: dict[InquiryType, int] = Counter()
        for uc in per_utt:
            distribution[uc.result.inquiry_type] += 1

        total = sum(
            distribution.get(t, 0)
            for t in InquiryType
            if t != InquiryType.UNKNOWN
        )

        return SessionClassification(
            conversation_id=conversation.conversation_id,
            utterances=per_utt,
            distribution=dict(distribution),
            total=total,
        )

    @staticmethod
    def _build_rationale(
        chosen: InquiryType,
        raw: Mapping[InquiryType, float],
        scores: Mapping[InquiryType, float],
    ) -> str:
        chosen_signals = int(raw.get(chosen, 0.0) / max(SIGNAL_WEIGHTS[chosen], 1e-9))
        if chosen_signals == 0:
            return f"{chosen.value} selected (zero direct signals, used LLM/tie-breaker)"
        return f"{chosen.value} selected on {chosen_signals} signal match(es)"


# ---------------------------------------------------------------------------
# Liebig's Law: bottleneck analysis
# ---------------------------------------------------------------------------


# Recommended ratio bands per inquiry type, derived from the
# Inquiry-Diagnosis paper's findings. Numbers are illustrative and
# intentionally wide — the bottleneck is computed from the actual
# distribution, not from these targets.
RECOMMENDED_RATIOS: Mapping[InquiryType, tuple[float, float]] = {
    InquiryType.OPEN_ENDED: (0.20, 0.50),     # breadth
    InquiryType.GUIDED: (0.20, 0.40),         # hypothesis
    InquiryType.REFLECTIVE: (0.10, 0.30),     # meta-cognition
    InquiryType.CLOSED_ENDED: (0.10, 0.30),   # verification
}


def session_distribution(
    classification: SessionClassification,
) -> dict[InquiryType, float]:
    """Convenience accessor for the inquiry-type distribution as ratios."""
    return classification.ratio


def liebig_bottleneck(
    classification: SessionClassification,
    *,
    recommended: Mapping[InquiryType, tuple[float, float]] = RECOMMENDED_RATIOS,
) -> tuple[InquiryType, float]:
    """
    Identify the limiting inquiry skill (Liebig's Law of the Minimum).

    The "bottleneck" is the inquiry type whose actual ratio is the
    greatest distance below the bottom of its recommended range. Returns
    ``(inquiry_type, deficit)`` where ``deficit = max(0, min - actual)``.

    If the session is well-balanced and no type is below its recommended
    range, the bottleneck is the type with the lowest actual ratio
    (zero deficit).
    """
    if classification.total == 0:
        return InquiryType.UNKNOWN, 0.0

    ratio = classification.ratio
    deficits: list[tuple[InquiryType, float]] = []
    for itype, (lo, _hi) in recommended.items():
        actual = ratio.get(itype, 0.0)
        deficit = max(0.0, lo - actual)
        deficits.append((itype, deficit))

    deficits.sort(
        key=lambda kv: (-kv[1], -ratio.get(kv[0], 0.0))
    )
    return deficits[0][0], deficits[0][1]


def liebig_quality_score(
    classification: SessionClassification,
    *,
    ideal_min: float = 0.10,
) -> float:
    """
    Compute the Liebig quality score for a session.

    The score is the minimum actual inquiry-type ratio (excluding
    UNKNOWN). Per Liebig's Law, the weakest skill determines the
    overall diagnostic-utility of the conversation. The result is
    floored at ``ideal_min`` so a session that completely lacks one
    type is not zeroed out (callers can interpret low values as a
    warning sign).
    """
    if classification.total == 0:
        return 0.0
    ratios = [
        classification.ratio.get(t, 0.0)
        for t in InquiryType
        if t != InquiryType.UNKNOWN
    ]
    if not ratios:
        return 0.0
    return max(ideal_min, min(ratios))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _paper_order(t: InquiryType) -> int:
    """
    Canonical Inquiry-Diagnosis paper ordering for tie-breaking.

    Lower number = higher priority. Reflective is intentionally last
    because it is the most meta-cognitive and least directly
    diagnostic; closed-ended is first because it is the most specific
    and easiest to disambiguate.
    """
    order = {
        InquiryType.CLOSED_ENDED: 0,
        InquiryType.OPEN_ENDED: 1,
        InquiryType.GUIDED: 2,
        InquiryType.REFLECTIVE: 3,
        InquiryType.UNKNOWN: 4,
    }
    return order.get(t, 5)


# ---------------------------------------------------------------------------
# Task 6: Hallucination Detection System
# ---------------------------------------------------------------------------
#
# The Inquiry-Diagnosis paper achieves 0.31% hallucination rate via a
# structured patient knowledge base + post-hoc verification. This module
# implements the verification layer: ``HallucinationDetector`` cross-checks
# each simulator response against the source CCD case data using four
# detection methods:
#
#   1. Factual consistency  — claims about symptoms, beliefs, emotions
#      must match the CCD profile.
#   2. Temporal consistency — timeline references must not contradict
#      the session's turn history.
#   3. Numerical accuracy   — intensity, conviction, and other numeric
#      values must match the profile's declared ranges.
#   4. Scope compliance     — the response must not introduce entities
#      or topics outside the case's defined scope.
#
# Each finding is severity-graded (LOW / MEDIUM / HIGH / CRITICAL) and
# logged for analysis. The detector is rule-based so it can run in CI
# without LLM calls, keeping the <1% hallucination target achievable.


class HallucinationSeverity(str, Enum):
    """Severity levels for hallucination findings.

    - LOW:    Minor factual drift that does not affect diagnostic utility.
    - MEDIUM: Noticeable inconsistency that could mislead a therapist.
    - HIGH:   Significant factual error that undermines the case.
    - CRITICAL: Direct contradiction of core case data (e.g. wrong diagnosis).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def numeric(self) -> float:
        """Numeric weight for aggregation (0.0 – 1.0)."""
        return {
            HallucinationSeverity.LOW: 0.25,
            HallucinationSeverity.MEDIUM: 0.5,
            HallucinationSeverity.HIGH: 0.75,
            HallucinationSeverity.CRITICAL: 1.0,
        }[self]


@dataclass(frozen=True)
class HallucinationFinding:
    """A single hallucination detection result.

    Attributes:
        detection_type: Which of the four detection methods flagged this.
        severity: How serious the hallucination is.
        description: Human-readable explanation of what was inconsistent.
        evidence: The specific text snippet from the response that triggered
            the finding (or the expected value for comparison).
        expected: The value the response *should* have had, if applicable.
    """

    detection_type: str
    """One of: 'factual_consistency', 'temporal_consistency',
    'numerical_accuracy', 'scope_compliance'."""

    severity: HallucinationSeverity
    description: str
    evidence: str = ""
    expected: str = ""

    @property
    def is_critical(self) -> bool:
        return self.severity == HallucinationSeverity.CRITICAL


@dataclass
class HallucinationReport:
    """Aggregated hallucination detection results for a single response.

    Attributes:
        response: The simulator response that was checked.
        findings: All findings from the four detection methods.
        overall_severity: The highest-severity finding, or None if clean.
        hallucination_rate: Fraction of findings that are hallucinations
            (0.0 = no findings, 1.0 = all checks failed).
    """

    response: str
    findings: list[HallucinationFinding] = field(default_factory=list)

    @property
    def overall_severity(self) -> HallucinationSeverity | None:
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: f.severity.numeric).severity

    @property
    def hallucination_rate(self) -> float:
        """Fraction of findings classified as hallucinations (0.0 – 1.0).

        A finding is a hallucination if its severity is at least MEDIUM.
        Returns 0.0 when there are no findings.
        """
        if not self.findings:
            return 0.0
        hallucinations = sum(
            1 for f in self.findings if f.severity.numeric >= 0.5
        )
        return hallucinations / len(self.findings)

    @property
    def is_hallucinated(self) -> bool:
        """True if any finding is at least MEDIUM severity."""
        return any(f.severity.numeric >= 0.5 for f in self.findings)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON logging."""
        return {
            "response": self.response,
            "findings": [
                {
                    "detection_type": f.detection_type,
                    "severity": f.severity.value,
                    "description": f.description,
                    "evidence": f.evidence,
                    "expected": f.expected,
                }
                for f in self.findings
            ],
            "overall_severity": (
                self.overall_severity.value if self.overall_severity else None
            ),
            "hallucination_rate": round(self.hallucination_rate, 4),
                        "is_hallucinated": self.is_hallucinated,
        }


# ---------------------------------------------------------------------------
# Case data extraction helpers
# ---------------------------------------------------------------------------

# Regex patterns for extracting structured facts from a CCD profile dict.
# These are used to build a searchable "ground truth" index that the
# detector cross-references against simulator responses.

# Extract symptom mentions from typical_symptoms list.
# Each keyword can match the base word and common morphological variants
# (e.g. "tired" → "fatigue", "anxious" → "anxiety").
_SYMPTOM_KEYWORDS: tuple[str, ...] = (
    "sad", "depressed", "anxious", "worry", "empty", "worthless",
    "guilty", "fatigue", "sleep", "appetite", "concentrat",
    "hopeless", "irritable", "panic", "avoid", "withdraw",
    "ruminat", "obsess", "compuls", "flashback", "numb",
)

# Synonym groups: if the profile contains a word in this group, the
# corresponding synonyms in the response are considered consistent.
# This prevents false positives like "tired" vs "fatigue".
_SYMPTOM_SYNONYMS: dict[str, set[str]] = {
    "fatigue": {"tired", "exhausted", "weary", "drowsy", "lethargic"},
    "sad": {"unhappy", "down", "blue", "low", "dejected"},
    "depressed": {"depression", "down", "low", "hopeless"},
    "anxious": {"anxiety", "worried", "nervous", "tense", "uneasy"},
    "worry": {"worried", "concerned", "anxious"},
    "empty": {"hollow", "numb", "void"},
    "worthless": {"inadequate", "useless", "pathetic", "inferior"},
    "guilty": {"ashamed", "remorseful", "blame"},
    "hopeless": {"despair", "helpless", "defeated", "hopelessness"},
    "irritable": {"angry", "frustrated", "agitated", "annoyed"},
    "panic": {"panicked", "terrified", "frightened"},
    "avoid": {"avoidance", "evade", "escape"},
    "withdraw": {"isolated", "isolating", "withdrawn", "withdrawal",
                 "withdrawing", "withdraws"},
    "flashback": {"flashbacks", "intrusive memory", "intrusive memories"},
    "numb": {"numbness", "detached", "disconnected"},
    "obsess": {"obsession", "obsessive", "intrusive", "obsessing",
               "obsessed", "obsesses"},
    "compuls": {"compulsion", "compulsive"},
}


def _extract_case_facts(case_data: dict) -> dict[str, set[str]]:
    """Build a ground-truth index from a CCD profile dict.

    Returns a dict mapping category names to sets of normalised keywords
    and phrases that the detector can search for in simulator responses.
    """
    facts: dict[str, set[str]] = {
        "symptoms": set(),
        "beliefs": set(),
        "emotions": set(),
        "behaviors": set(),
        "diagnoses": set(),
        "situations": set(),
    }

    # Symptoms from typical_symptoms
    for symptom in case_data.get("typical_symptoms", []):
        facts["symptoms"].add(symptom.lower().strip())

    # Core beliefs
    for belief in case_data.get("core_beliefs", []):
        content = belief.get("content", "")
        if content:
            facts["beliefs"].add(content.lower().strip())

    # Intermediate beliefs
    for belief in case_data.get("intermediate_beliefs", []):
        content = belief.get("content", "")
        if content:
            facts["beliefs"].add(content.lower().strip())

    # Emotional responses
    for emo in case_data.get("emotional_responses", []):
        emotion = emo.get("emotion", "")
        if emotion:
            facts["emotions"].add(emotion.lower().strip())

    # Behavioral responses
    for beh in case_data.get("behavioral_responses", []):
        behavior = beh.get("behavior", "")
        if behavior:
            facts["behaviors"].add(behavior.lower().strip())

    # Diagnoses
    for diag in case_data.get("diagnoses", []):
        facts["diagnoses"].add(diag.lower().strip())

    # Situation interpretations
    for sit in case_data.get("situation_interpretations", []):
        situation = sit.get("situation", "")
        if situation:
            facts["situations"].add(situation.lower().strip())

    # Also extract symptom keywords from descriptions
    desc = case_data.get("description", "").lower()
    for kw in _SYMPTOM_KEYWORDS:
        if kw in desc:
            facts["symptoms"].add(kw)

    return facts


def _extract_numerical_facts(case_data: dict) -> dict[str, float]:
    """Extract numeric values from a CCD profile for accuracy checking.

    Returns a dict mapping descriptive keys to their declared numeric values.
    """
    nums: dict[str, float] = {}

    # Cognitive triad values
    triads = case_data.get("triads")
    if triads and isinstance(triads, dict):
        for key in ("self_views", "world_views", "future_views"):
            if key in triads:
                nums[f"triad_{key}"] = float(triads[key])

    # Conviction values for beliefs
    for i, belief in enumerate(case_data.get("core_beliefs", [])):
        if "conviction" in belief:
            nums[f"core_belief_{i}_conviction"] = float(belief["conviction"])

    for i, belief in enumerate(case_data.get("intermediate_beliefs", [])):
        if "conviction" in belief:
            nums[f"intermediate_belief_{i}_conviction"] = float(belief["conviction"])

    # Emotional response intensities
    for i, emo in enumerate(case_data.get("emotional_responses", [])):
        if "intensity" in emo:
            nums[f"emotion_{i}_intensity"] = float(emo["intensity"])

    # Coping strategy effectiveness
    for i, strat in enumerate(case_data.get("coping_strategies", [])):
        if "effectiveness" in strat:
            nums[f"coping_{i}_effectiveness"] = float(strat["effectiveness"])

    return nums


def _extract_timeline(case_data: dict) -> list[str]:
    """Extract temporal markers from case data for consistency checking.

    Returns a list of normalised temporal phrases found in the case data.
    """
    timeline: list[str] = []
    for sit in case_data.get("situation_interpretations", []):
        situation = sit.get("situation", "")
        if situation:
            timeline.append(situation.lower().strip())
    return timeline


def _expand_with_synonyms(terms: set[str]) -> set[str]:
    """Expand a set of known terms with their synonyms.

    For each known symptom/keyword, adds the synonyms from
    ``_SYMPTOM_SYNONYMS`` so that downstream checks (e.g. scope
    compliance) don't flag morphologically related words as novel.
    Also generates common morphological variants (plural/singular,
    -ness, -ed, -ing) so that "hopelessness" matches "hopeless",
    "flashback" matches "flashbacks", etc.
    """
    expanded = set(terms)
    # Use exact match only for synonym expansion to avoid false positives
    # from short common substrings (e.g. "back" being a substring of
    # "flashback" would incorrectly add flashback synonyms to a profile
    # that just contains the word "back" in "feedback").
    for term in list(terms):
        for key, synonyms in _SYMPTOM_SYNONYMS.items():
            if term == key:
                expanded.update(synonyms)
            elif term in synonyms:
                expanded.add(key)
                expanded.update(synonyms)

    # Generate morphological variants for each term.
    morph_variants: set[str] = set()
    for term in list(expanded):
        if len(term) <= 2:
            continue
        # Plural -> singular
        if term.endswith("ies") and len(term) > 4:
            morph_variants.add(term[:-3] + "y")
        elif term.endswith("es") and len(term) > 4:
            morph_variants.add(term[:-2])
            morph_variants.add(term[:-1])
        elif term.endswith("s") and len(term) > 3:
            morph_variants.add(term[:-1])
        # Singular -> plural
        if term.endswith("y") and len(term) > 2:
            morph_variants.add(term[:-1] + "ies")
        elif not term.endswith("s"):
            morph_variants.add(term + "s")
            morph_variants.add(term + "es")
        # -ness variants: hopeless -> hopelessness
        if not term.endswith("ness"):
            morph_variants.add(term + "ness")
        if term.endswith("ness") and len(term) > 5:
            morph_variants.add(term[:-4])  # hopelessness -> hopeless
        # -ed, -ing variants
        if not term.endswith("ed"):
            morph_variants.add(term + "ed")
        if not term.endswith("ing"):
            morph_variants.add(term + "ing")
        # Strip -ing to recover base form: concentrating -> concentrate
        if term.endswith("ing") and len(term) > 5:
            base = term[:-3]
            morph_variants.add(base)
            morph_variants.add(base + "e")
        # Strip -ed to recover base form: worried -> worry
        if term.endswith("ed") and len(term) > 4:
            morph_variants.add(term[:-2])
            morph_variants.add(term[:-2] + "e")
    expanded.update(morph_variants)
    return expanded


# ---------------------------------------------------------------------------
# HallucinationDetector
# ---------------------------------------------------------------------------


class HallucinationDetector:
    """Cross-checks patient-simulator responses against source case data.

    Implements Task 6 of PIX-3908. The detector is rule-based so it can
    run in CI / on-device without LLM calls, keeping the <1% hallucination
    target achievable.

    Four detection methods are applied to each response:

    1. **Factual consistency** — extracts key claims from the response and
       verifies they appear in the CCD profile's symptoms, beliefs,
       emotions, behaviors, or diagnoses.
    2. **Temporal consistency** — checks that temporal references in the
       response do not contradict the session's turn history.
    3. **Numerical accuracy** — verifies that any numeric values mentioned
       in the response match the profile's declared conviction, intensity,
       or effectiveness values.
    4. **Scope compliance** — ensures the response does not introduce
       entities or topics outside the case's defined scope.

    Usage::

        detector = HallucinationDetector(case_data)
        report = detector.detect("I've been feeling very sad lately.")
        if report.is_hallucinated:
            logger.warning("Hallucination detected: %s", report.findings)
    """

    # Minimum overlap threshold: if a response contains fewer than this
    # fraction of its key terms from the case data, it may be hallucinated.
    _SCOPE_MIN_OVERLAP = 0.3

    # Maximum number of novel entities (words not in case data) before
    # scope compliance is flagged.
    _SCOPE_MAX_NOVEL_ENTITIES = 5

    # Number words that the numerical checker recognises.
    _NUMBER_WORDS: dict[str, int] = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10,
    }

    def __init__(self, case_data: dict | None = None) -> None:
        """Initialise the detector with source case data.

        Args:
            case_data: A CCD profile dict (as produced by
                ``ClinicalProfile.ccd_config`` or ``PatientCCD.to_dict()``).
                If None, the detector will only perform scope compliance
                checks (which will flag everything as out-of-scope).
        """
        self._case_data = case_data or {}
        self._facts = _extract_case_facts(self._case_data)
        self._numerical_facts = _extract_numerical_facts(self._case_data)
        self._timeline = _extract_timeline(self._case_data)
        self._all_terms: set[str] = set()
        for terms in self._facts.values():
            self._all_terms.update(terms)
        # Also index individual words from case data for scope checking
        for fact_set in self._facts.values():
            for phrase in fact_set:
                self._all_terms.update(phrase.split())
        # Expand with synonyms to prevent false positives on morphologically
        # related words (e.g. "tired" should not be flagged if the profile
        # contains "fatigue").
        self._all_terms = _expand_with_synonyms(self._all_terms)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        response: str,
        *,
        session_history: list[str] | None = None,
    ) -> HallucinationReport:
        """Run all four detection methods on a simulator response.

        Args:
            response: The patient simulator's response text.
            session_history: Previous therapist utterances (for temporal
                consistency checking). If None, temporal checking is
                skipped.

        Returns:
            A ``HallucinationReport`` with all findings.
        """
        findings: list[HallucinationFinding] = []

        findings.extend(self.check_factual_consistency(response))
        if session_history is not None:
            findings.extend(
                self.check_temporal_consistency(response, session_history)
            )
        findings.extend(self.check_numerical_accuracy(response))
        findings.extend(self.check_scope_compliance(response))

        report = HallucinationReport(response=response, findings=findings)

        if report.is_hallucinated:
            logger.warning(
                "Hallucination detected (severity=%s, rate=%.2f): %s",
                report.overall_severity.value
                if report.overall_severity
                else "unknown",
                report.hallucination_rate,
                findings[0].description,
            )

        return report

    def verify_session(
        self,
        turns: list[tuple[str, str]],
    ) -> dict:
        """Post-hoc verification of all responses in a session.

        Args:
            turns: List of (therapist_utterance, patient_response) pairs
                in chronological order.

        Returns:
            A dict with:
                - ``hallucination_rate``: fraction of responses that were
                  hallucinated.
                - ``total_responses``: number of responses checked.
                - ``hallucinated_count``: number of hallucinated responses.
                - ``reports``: list of ``HallucinationReport.to_dict()``
                  for each response.
        """
        reports: list[dict] = []
        hallucinated_count = 0
        session_history: list[str] = []

        for therapist_utt, patient_response in turns:
            report = self.detect(
                patient_response, session_history=session_history
            )
            reports.append(report.to_dict())
            if report.is_hallucinated:
                hallucinated_count += 1
            session_history.append(therapist_utt)

        total = len(turns)
        rate = hallucinated_count / total if total > 0 else 0.0

        return {
            "hallucination_rate": round(rate, 4),
            "total_responses": total,
            "hallucinated_count": hallucinated_count,
                        "reports": reports,
        }

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def check_factual_consistency(
        self, response: str
    ) -> list[HallucinationFinding]:
        """Verify that factual claims in the response match the case data.

        Extracts key noun phrases and symptom/emotion mentions from the
        response and checks whether they appear in the CCD profile.
        """
        findings: list[HallucinationFinding] = []
        if not self._case_data:
            return findings

        lowered = response.lower()
        response_words = set(re.findall(r"[a-z]+", lowered))

        # Check for claims that contradict the profile
        # Look for negation patterns around profile facts.
        # For each fact, we extract key individual words from the fact
        # and check if they appear with negation in the response.
        negation_words = {
            "not", "no", "never", "don't", "doesn't", "didn't",
            "isn't", "wasn't", "aren't", "weren't", "cannot", "can't",
            "won't", "wouldn't", "shouldn't", "couldn't",
        }
        for fact_set_name, fact_set in self._facts.items():
            if fact_set_name == "diagnoses":
                continue  # Diagnoses are checked separately
            for fact in fact_set:
                # Get individual content words from the fact
                fact_words = set(re.findall(r"[a-z]+", fact.lower()))
                # Filter out common stop words and short words
                stop_words = {"i", "am", "is", "are", "was", "were", "be", "been",
                              "being", "the", "a", "an", "and", "or", "of", "to",
                              "in", "on", "at", "for", "with", "by", "this", "that",
                              "it", "its", "as", "if", "no", "not"}
                meaningful_words = {
                    w for w in fact_words if w not in stop_words and len(w) > 2
                }
                if not meaningful_words:
                    continue
                # Check each meaningful word for negation
                for word in meaningful_words:
                    negation_pattern = re.compile(
                        rf"\b(?:not|no|never|don't|doesn't|didn't|isn't|wasn't|"
                        rf"aren't|weren't|cannot|can't|won't|wouldn't|shouldn't|"
                        rf"couldn't)\s+\w*\s*{re.escape(word)}\b",
                        re.IGNORECASE,
                    )
                    if negation_pattern.search(response):
                        findings.append(HallucinationFinding(
                            detection_type="factual_consistency",
                            severity=HallucinationSeverity.HIGH,
                            description=(
                                f"Response negates a profile fact: '{word}' "
                                f"from '{fact}' is part of the CCD profile "
                                f"but is negated in the response."
                            ),
                            evidence=word,
                            expected=f"Positive mention of '{word}'",
                        ))

        # Check for claims about symptoms NOT in the profile.
        # Use the expanded symptom keywords to handle morphological variants
        # (e.g. "flashbacks" matches "flashback", "obsessing" matches "obsess").
        expanded_symptom_keywords = _expand_with_synonyms(set(_SYMPTOM_KEYWORDS))
        for word in response_words:
            if word in expanded_symptom_keywords and word not in self._all_terms:
                # The response mentions a symptom that's not in the case data
                findings.append(HallucinationFinding(
                    detection_type="factual_consistency",
                    severity=HallucinationSeverity.MEDIUM,
                    description=(
                        f"Response mentions symptom '{word}' which is not "
                        f"part of the CCD profile's declared symptoms."
                    ),
                    evidence=word,
                    expected="Symptom from profile's typical_symptoms list",
                ))

        return findings

    def check_temporal_consistency(
        self,
        response: str,
        session_history: list[str],
    ) -> list[HallucinationFinding]:
        """Verify that temporal references in the response are consistent
        with the session's turn history.

        Checks for:
        - References to events that haven't been mentioned yet.
        - Contradictions with previously established timeline facts.
        - Anachronistic references (e.g., mentioning future events as past).
        """
        findings: list[HallucinationFinding] = []
        if not session_history:
            return findings

        lowered = response.lower()
        history_text = " ".join(session_history).lower()

        # Check for references to events not yet mentioned
        retrospective_patterns = [
            r"as i said before",
            r"like i mentioned",
            r"as we discussed",
            r"earlier i told you",
            r"i already told you",
        ]
        for pattern in retrospective_patterns:
            if re.search(pattern, lowered):
                if not history_text.strip():
                    findings.append(HallucinationFinding(
                        detection_type="temporal_consistency",
                        severity=HallucinationSeverity.MEDIUM,
                        description=(
                            f"Response references prior discussion ('{pattern}') "
                            f"but session history is empty or lacks context."
                        ),
                        evidence=pattern,
                        expected="Prior session context matching the reference",
                    ))

        # Check for temporal contradiction patterns
        contradiction_patterns = [
            (r"i never said", r"i said|i told|i mentioned"),
            (r"i didn't say", r"i said|i told|i mentioned"),
            (r"that's not what i said", r"i said|i told|i mentioned"),
        ]
        for neg_pattern, pos_pattern in contradiction_patterns:
            if re.search(neg_pattern, lowered):
                if re.search(pos_pattern, history_text):
                    findings.append(HallucinationFinding(
                        detection_type="temporal_consistency",
                        severity=HallucinationSeverity.HIGH,
                        description=(
                            f"Response contradicts session history: claims "
                            f"'{neg_pattern}' but history contains prior "
                            f"statement."
                        ),
                        evidence=neg_pattern,
                        expected="Consistent with prior session statements",
                    ))

        return findings

    def check_numerical_accuracy(
        self, response: str
    ) -> list[HallucinationFinding]:
        """Verify that numeric values in the response match the case data.

        Checks for:
        - Intensity ratings that don't match the profile's declared values.
        - Conviction levels that contradict the CCD beliefs.
        - Effectiveness ratings that differ from the profile.
        """
        findings: list[HallucinationFinding] = []
        if not self._numerical_facts:
            return findings

        lowered = response.lower()

        # Extract numbers from the response
        number_patterns = [
            (r"intensity\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?)", "intensity"),
            (r"conviction\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?)", "conviction"),
            (r"effectiveness\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?)", "effectiveness"),
            (r"severity\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?)", "severity"),
        ]

        for pattern, label in number_patterns:
            matches = re.findall(pattern, lowered)
            for match in matches:
                try:
                    value = float(match)
                except ValueError:
                    continue

                # Check against all numerical facts in the profile
                for fact_key, fact_value in self._numerical_facts.items():
                    if label in fact_key or label in fact_key.replace("_", " "):
                        # Allow tolerance of ±0.15 for floating point
                        if abs(value - fact_value) > 0.15:
                            findings.append(HallucinationFinding(
                                detection_type="numerical_accuracy",
                                severity=HallucinationSeverity.MEDIUM,
                                description=(
                                    f"Response states {label}={value} but "
                                    f"the CCD profile declares "
                                    f"{fact_key}={fact_value}."
                                ),
                                evidence=f"{label}={value}",
                                expected=f"{fact_key}={fact_value}",
                            ))

        # Check for percentage claims that might be hallucinated
        pct_pattern = re.compile(r"(\d+)%")
        for match in pct_pattern.finditer(lowered):
            pct = int(match.group(1))
            if pct in (50, 75, 90, 95, 100) and "percent" not in lowered:
                if "accuracy" not in lowered and "rate" not in lowered:
                    findings.append(HallucinationFinding(
                        detection_type="numerical_accuracy",
                        severity=HallucinationSeverity.LOW,
                        description=(
                            f"Response claims {pct}% without a corresponding "
                            f"metric in the CCD profile."
                        ),
                        evidence=f"{pct}%",
                        expected="Percentage backed by profile data",
                    ))

        return findings

    def check_scope_compliance(
        self, response: str
    ) -> list[HallucinationFinding]:
        """Verify that the response stays within the case's defined scope.

        Checks for:
        - Introduction of entities (names, places, topics) not in the case.
        - Discussion of symptoms or conditions not in the profile.
        - Claims about treatment or history not in the profile.
        """
        findings: list[HallucinationFinding] = []
        if not self._case_data:
            # Without case data, everything is out of scope
            findings.append(HallucinationFinding(
                detection_type="scope_compliance",
                severity=HallucinationSeverity.CRITICAL,
                description=(
                    "No case data provided — all response content is "
                    "unverifiable and potentially hallucinated."
                ),
                evidence=response[:100],
                expected="CCD profile data for verification",
            ))
            return findings

        lowered = response.lower()
        response_words = set(re.findall(r"[a-z]+", lowered))

        # Check for novel symptom mentions not in the profile.
        # Use the expanded symptom keywords to handle morphological variants.
        expanded_symptom_keywords = _expand_with_synonyms(set(_SYMPTOM_KEYWORDS))
        novel_symptoms = []
        for word in response_words:
            if word in expanded_symptom_keywords and word not in self._all_terms:
                novel_symptoms.append(word)

        if novel_symptoms:
            findings.append(HallucinationFinding(
                detection_type="scope_compliance",
                severity=HallucinationSeverity.HIGH,
                description=(
                    f"Response introduces symptoms not in the CCD profile: "
                    f"{', '.join(novel_symptoms)}."
                ),
                evidence=", ".join(novel_symptoms),
                expected="Symptoms from profile's typical_symptoms list",
            ))

        # Check for novel emotional states not in the profile.
        # The check also considers known symptoms (expanded with synonyms)
        # because physical symptoms like "fatigue" can manifest as emotional
        # states like "tired" or "exhausted".
        known_emotions = self._facts["emotions"]
        common_emotions = {
            "sad", "happy", "angry", "anxious", "depressed", "numb",
            "empty", "hopeful", "despair", "panic", "calm", "stressed",
            "overwhelmed", "lonely", "isolated", "guilty", "ashamed",
            "irritable", "agitated", "restless", "exhausted", "tired",
        }
        # Expand known emotions AND symptoms with synonyms to avoid false
        # positives where a profile symptom (e.g. "fatigue") matches a
        # response emotion (e.g. "tired", "exhausted").
        expanded_known_emotions = _expand_with_synonyms(
            known_emotions | self._facts["symptoms"]
        )
        novel_emotions = []
        for emo in common_emotions:
            if emo in response_words and emo not in expanded_known_emotions:
                is_known = any(emo in known_emo for known_emo in expanded_known_emotions)
                if not is_known:
                    novel_emotions.append(emo)

        if novel_emotions:
            findings.append(HallucinationFinding(
                detection_type="scope_compliance",
                severity=HallucinationSeverity.MEDIUM,
                description=(
                    f"Response mentions emotional states not in the CCD "
                    f"profile: {', '.join(novel_emotions)}."
                ),
                evidence=", ".join(novel_emotions),
                expected="Emotions from profile's emotional_responses list",
            ))

        # Check for novel behavioral claims not in the profile
        known_behaviors = self._facts["behaviors"]
        # Expand known behaviors with synonyms to avoid false positives.
        expanded_known_behaviors = _expand_with_synonyms(known_behaviors)
        behavior_keywords = {
            "avoid", "withdraw", "ruminate", "obsess", "overcompensate",
            "isolate", "escape", "distract", "procrastinate", "overwork",
        }
        novel_behaviors = []
        for beh in behavior_keywords:
            if beh in response_words and beh not in expanded_known_behaviors:
                is_known = any(beh in known_b for known_b in expanded_known_behaviors)
                if not is_known:
                    novel_behaviors.append(beh)

        if novel_behaviors:
            findings.append(HallucinationFinding(
                detection_type="scope_compliance",
                severity=HallucinationSeverity.MEDIUM,
                description=(
                    f"Response describes behaviors not in the CCD profile: "
                    f"{', '.join(novel_behaviors)}."
                ),
                evidence=", ".join(novel_behaviors),
                expected="Behaviors from profile's behavioral_responses list",
            ))

        # Check for novel diagnoses or conditions
        known_diagnoses = self._facts["diagnoses"]
        diagnosis_keywords = {
            "depression", "anxiety", "bipolar", "ptsd", "ocd", "panic",
            "schizophrenia", "borderline", "eating", "substance",
            "psychosis", "psychotic", "voices", "hallucination",
            "hallucinations", "delusion", "delusions",
        }
        for diag in diagnosis_keywords:
            if diag in lowered and not any(
                diag in known_diag for known_diag in known_diagnoses
            ):
                findings.append(HallucinationFinding(
                    detection_type="scope_compliance",
                    severity=HallucinationSeverity.HIGH,
                    description=(
                        f"Response references diagnosis/condition '{diag}' "
                        f"which is not in the CCD profile's diagnoses."
                    ),
                    evidence=diag,
                    expected="Diagnoses from profile's diagnoses list",
                ))

        # ------------------------------------------------------------------
        # Overall term-overlap and novel-entity enforcement
        # ------------------------------------------------------------------
        _STOP_WORDS = {
            "i", "am", "is", "are", "was", "were", "be", "been", "being",
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "at",
            "for", "with", "by", "this", "that", "it", "its", "as", "if",
            "no", "not", "my", "me", "we", "they", "he", "she", "you",
            "have", "has", "had", "do", "does", "did", "so", "but",
        }
        meaningful_response_words = {
            w for w in response_words
            if w not in _STOP_WORDS and len(w) > 2
        }
        if len(meaningful_response_words) >= 5:
            overlap_words = meaningful_response_words & self._all_terms
            overlap_ratio = len(overlap_words) / len(meaningful_response_words)
            if overlap_ratio < self._SCOPE_MIN_OVERLAP:
                findings.append(HallucinationFinding(
                    detection_type="scope_compliance",
                    severity=HallucinationSeverity.MEDIUM,
                    description=(
                        f"Response has low term overlap with case data "
                        f"({overlap_ratio:.0%} < {self._SCOPE_MIN_OVERLAP:.0%} "
                        f"threshold), suggesting potential hallucination."
                    ),
                    evidence=response[:100],
                    expected=(
                        f"At least {self._SCOPE_MIN_OVERLAP:.0%} of response "
                        f"terms should appear in the CCD profile"
                    ),
                ))
            novel_entities = meaningful_response_words - self._all_terms
            if len(novel_entities) > self._SCOPE_MAX_NOVEL_ENTITIES:
                findings.append(HallucinationFinding(
                    detection_type="scope_compliance",
                    severity=HallucinationSeverity.MEDIUM,
                    description=(
                        f"Response introduces {len(novel_entities)} novel "
                        f"entities not in case data (max "
                        f"{self._SCOPE_MAX_NOVEL_ENTITIES} allowed): "
                        f"{', '.join(sorted(novel_entities)[:10])}."
                    ),
                    evidence=", ".join(sorted(novel_entities)[:10]),
                    expected=(
                        f"At most {self._SCOPE_MAX_NOVEL_ENTITIES} novel "
                        f"entities outside the CCD profile"
                    ),
                ))

        return findings



