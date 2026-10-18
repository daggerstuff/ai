"""Heuristic toxicity scoring for ChatML records.

Uses keyword-based heuristics to flag potentially toxic content in
imported hackathon data. Designed for batch processing without ML model
dependencies. Edge-case training records bypass toxicity filtering to
preserve deliberate clinical edge-case signal.

Toxicity categories:
  - Hate speech / slurs
  - Profanity (severity levels)
  - Self-harm ideation
  - Violence / threats
  - Sexual content (non-clinical)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Toxicity keyword sets
# ---------------------------------------------------------------------------

_HATE_SPEECH = [
    # Derogatory slurs and hate-group references
    "kike", "spic", "chink", "wetback", "towelhead", "raghead",
    "tranny", "shemale", "lynch", "genocide", "ethnic cleansing",
    "go back to your country",
]

_PROFANITY_HIGH = [
    "fuck", "fucking", "fucked", "motherfucker", "cunt",
    "asshole", "bastard", "dickhead", "piss off", "dipshit",
]

_PROFANITY_MEDIUM = [
    "shit", "bullshit", "bitch", "damn", "crap", "piss",
    "jackass", "douche", "screw you",
]

_SELF_HARM = [
    "kill myself", "end my life", "want to die", "suicide",
    "cut myself", "hurt myself", "harm myself",
    "overdose", "pills to end it", "no reason to live",
    "better off dead", "can't go on",
]

_VIOLENCE = [
    "kill you", "murder", "stab", "shoot you", "beat you",
    "strangle", "torture", "mutilate", "decapitate",
]

_SEXUAL_NONCLINICAL = [
    "porn", "xxx", "hookup", "nudes", "dick pic", "sexting",
]

# Severity weights per category
_SEVERITY_WEIGHTS: dict[str, float] = {
    "hate_speech": 1.0,
    "self_harm": 0.8,
    "violence": 0.7,
    "sexual_nonclinical": 0.5,
    "profanity_high": 0.6,
    "profanity_medium": 0.3,
}

# Toxicity threshold: records above this are flagged
TOXICITY_THRESHOLD = 0.5


@dataclass
class ToxicityHit:
    """A single toxicity detection in text."""

    category: str
    keyword: str
    context: str  # surrounding text snippet
    message_index: int
    role: str


@dataclass
class ToxicityResult:
    """Result of toxicity scoring on a single record."""

    score: float
    hits: list[ToxicityHit] = field(default_factory=list)

    @property
    def is_toxic(self) -> bool:
        return self.score >= TOXICITY_THRESHOLD


@dataclass
class ToxicityReport:
    """Aggregate report for a batch of records."""

    total_records: int = 0
    flagged_records: int = 0
    total_hits: int = 0
    by_category: dict[str, int] = field(default_factory=dict)

    def add(self, result: ToxicityResult) -> None:
        self.total_records += 1
        if result.is_toxic:
            self.flagged_records += 1
        self.total_hits += len(result.hits)
        for hit in result.hits:
            self.by_category[hit.category] = (
                self.by_category.get(hit.category, 0) + 1
            )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

_KEYWORD_SETS: list[tuple[str, list[str]]] = [
    ("hate_speech", _HATE_SPEECH),
    ("self_harm", _SELF_HARM),
    ("violence", _VIOLENCE),
    ("sexual_nonclinical", _SEXUAL_NONCLINICAL),
    ("profanity_high", _PROFANITY_HIGH),
    ("profanity_medium", _PROFANITY_MEDIUM),
]


def _find_keyword_hits(
    text: str,
    keywords: list[str],
    category: str,
    message_index: int,
    role: str,
) -> list[ToxicityHit]:
    """Find all keyword matches in text, returning ToxicityHit objects."""
    hits: list[ToxicityHit] = []
    text_lower = text.lower()
    for kw in keywords:
        start = 0
        while True:
            idx = text_lower.find(kw, start)
            if idx == -1:
                break
            # Extract surrounding context (50 chars each side)
            ctx_start = max(0, idx - 50)
            ctx_end = min(len(text), idx + len(kw) + 50)
            context = text[ctx_start:ctx_end]
            hits.append(ToxicityHit(
                category=category,
                keyword=kw,
                context=context,
                message_index=message_index,
                role=role,
            ))
            start = idx + len(kw)
    return hits


def _compute_score(hits: list[ToxicityHit]) -> float:
    """Compute an aggregate toxicity score from hits.

    Score is a weighted sum capped at 1.0.
    Multiple hits in the same category increase the score but with
    diminishing returns (sqrt weighting).
    """
    if not hits:
        return 0.0

    by_cat: dict[str, int] = {}
    for hit in hits:
        by_cat[hit.category] = by_cat.get(hit.category, 0) + 1

    score = 0.0
    for cat, count in by_cat.items():
        weight = _SEVERITY_WEIGHTS.get(cat, 0.3)
        # Diminishing returns: sqrt(count) * weight
        score += weight * (count ** 0.5)

    return min(1.0, score)


def score_record(record: dict[str, Any]) -> ToxicityResult:
    """Score a single ChatML record for toxicity.

    Scans all message contents for toxic keywords.
    Edge-case training records are NOT bypassed here — the caller
    (safety_processor) decides whether to act on the score.
    """
    all_hits: list[ToxicityHit] = []
    messages = record.get("messages", [])

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        role = msg.get("role", "unknown")
        for category, keywords in _KEYWORD_SETS:
            hits = _find_keyword_hits(content, keywords, category, i, role)
            all_hits.extend(hits)

    score = _compute_score(all_hits)
    return ToxicityResult(score=score, hits=all_hits)


def score_batch(records: list[dict[str, Any]]) -> tuple[list[ToxicityResult], ToxicityReport]:
    """Score a batch of ChatML records for toxicity.

    Returns per-record results and an aggregate report.
    """
    report = ToxicityReport()
    results: list[ToxicityResult] = []

    for record in records:
        result = score_record(record)
        report.add(result)
        results.append(result)

    return results, report
