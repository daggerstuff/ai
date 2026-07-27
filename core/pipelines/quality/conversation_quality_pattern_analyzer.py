"""Conversation quality pattern analyzer with inquiry-type classification.

Implements Task 2 of PIX-3908 ([Inquiry-Diagnosis] Patient Simulator &
Diagnostic Quality Framework).

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

Implementation notes
--------------------
The classifier is rule-based (patterns + keywords) so that it can run
without an LLM in CI / on-device. An optional ``LLMJudge`` adapter is
defined here for ambiguous cases but is not used by the default
classifier — that hook matches the Task 2 acceptance criteria
("rule-based + LLM-as-judge for ambiguous cases").

References
----------
* arXiv 2501.09484 — How to Evaluate the Diagnostic Ability of LLMs:
  The Role of Patient Simulator.
* Liebig's Law of the Minimum — diagnostic accuracy is constrained by
  the scarcest inquiry skill.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from ai.core.pipelines.quality.conversation_schema import Conversation, Message
from ai.core.pipelines.quality.quality_assessment_framework import InquiryType

__all__ = [
    "ClassificationResult",
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
            return {t: 0.0 for t in InquiryType}
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


