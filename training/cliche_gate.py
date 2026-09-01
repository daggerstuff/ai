"""Zero-tolerance cliché / sycophancy rejection gate for generated assistant turns.

This is the write-time gate fired by the generation + consolidation paths on every
record. It is deliberately a plain, dependency-free string check so it can run in
the hot generation loop without a model round-trip.

The LLM judge (`dual_judge.py`) carries its own canonical copies of
``BANNED_OPENERS`` / ``CAVING_PHRASES`` for *scoring*; this module is the
*rejection* gate. Keep the two lists in sync when the policy changes.
"""

from __future__ import annotations

from typing import Any

# Handoff constraint #2 — anti-sycophancy openers.
BANNED_OPENERS: tuple[str, ...] = (
    "i hear how",
    "it makes sense that you feel",
    "i understand your frustration",
    "i can hear",
    "that sounds really",
    "i'm so sorry to hear",
    "thank you for sharing",
    "it sounds like you",
    "i want you to know",
    "i can imagine how",
    "i hear your",
    "it sounds like",
)

# Caving / backtracking phrases that undercut the clinician's own position.
CAVING_PHRASES: tuple[str, ...] = (
    "you're right",
    "i apologize",
    "i stand corrected",
    "sorry for",
    "my mistake",
    "if you don't want to talk about it",
    "we don't have to",
    "we don't have to talk about",
    "i'll stop",
    "fair enough",
)

# Robotic liability-style crisis interrogations (handoff constraint #2). These read
# as checklist items rather than holding the weight of a disclosure. Kept narrow to
# the textbook questionnaire forms so legitimate safety assessment is not caught.
ROBOTIC_CRISIS_QUESTIONS: tuple[str, ...] = (
    "are you safe right now",
    "do you have a plan to hurt yourself",
    "do you have a plan to kill yourself",
    "are you thinking about hurting yourself",
    "are you thinking about killing yourself",
    "have you thought about hurting yourself",
    "have you thought about killing yourself",
)

# Roles whose turns are treated as clinician output for the purposes of this gate.
ASSISTANT_ROLES: frozenset[str] = frozenset(
    {"assistant", "therapist", "clinician", "counselor"}
)


def is_sycophantic(text: str) -> tuple[bool, str]:
    """Return ``(True, reason)`` if *text* triggers a banned opener, caving phrase,
    or robotic crisis questionnaire; otherwise ``(False, "")``."""
    if not text or not isinstance(text, str):
        return False, ""
    t_lower = text.strip().lower()
    opener = next(
        (b for b in BANNED_OPENERS if t_lower.startswith(b) or f"\n{b}" in t_lower),
        None,
    )
    if opener is not None:
        return True, f"banned_sycophantic_opener: '{opener}'"
    caving = next((c for c in CAVING_PHRASES if c in t_lower), None)
    if caving is not None:
        return True, f"caving_phrase_detected: '{caving}'"
    robotic = next((q for q in ROBOTIC_CRISIS_QUESTIONS if q in t_lower), None)
    if robotic is not None:
        return True, f"robotic_crisis_questionnaire: '{robotic}'"
    return False, ""


def reject_reason_for_record(record: Any, *, family: str = "") -> str | None:
    """Scan every clinician turn in *record*; return the first rejection reason
    (annotated with *family* when provided), or ``None`` when the record passes."""
    messages = record.get("messages") if isinstance(record, dict) else None
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower().strip()
        if role not in ASSISTANT_ROLES:
            continue
        is_bad, reason = is_sycophantic(str(message.get("content", "")))
        if is_bad:
            return f"{reason} (family={family})" if family else reason
    return None
