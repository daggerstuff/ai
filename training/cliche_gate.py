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

# Mechanical quote-parroting and echoing openers that formulaically repeat client words.
PARROTING_OPENERS: tuple[str, ...] = (
    "you used the phrase",
    "you said",
    "you mentioned",
    "you named",
    "you just said",
    "when you say",
    "you're saying that",
    "so you're saying",
    "hearing you say",
    "as you said",
)

# Robotic somatic deflection formulas used as canned evasions.
ROBOTIC_SOMATIC_PHRASES: tuple[str, ...] = (
    "what happens inside your body",
    "what is happening inside your body",
    "what is happening in your body",
    "what happens in your body",
    "where do you feel that in your body",
    "notice what happens in your body",
    "what does your body feel",
    "what are you feeling in your body",
    "what is your body telling you",
)

# Roles whose turns are treated as clinician output for the purposes of this gate.
ASSISTANT_ROLES: frozenset[str] = frozenset(
    {"assistant", "therapist", "clinician", "counselor"}
)


def is_sycophantic(text: str) -> tuple[bool, str]:
    """Return ``(True, reason)`` if *text* triggers a banned opener, parroting opener,
    caving phrase, robotic crisis questionnaire, or robotic somatic cliché;
    otherwise ``(False, "")``."""
    if not text or not isinstance(text, str):
        return False, ""
    t_lower = text.strip().lower()

    # Prefix checks (at string start or right after a newline)
    for prefix in BANNED_OPENERS:
        if t_lower.startswith(prefix) or f"\n{prefix}" in t_lower:
            return True, f"banned_sycophantic_opener: '{prefix}'"
    for prefix in PARROTING_OPENERS:
        if t_lower.startswith(prefix) or f"\n{prefix}" in t_lower:
            return True, f"banned_parroting_opener: '{prefix}'"

    # Substring checks anywhere in the utterance
    checks: tuple[tuple[tuple[str, ...], str], ...] = (
        (CAVING_PHRASES, "caving_phrase_detected"),
        (ROBOTIC_CRISIS_QUESTIONS, "robotic_crisis_questionnaire"),
        (ROBOTIC_SOMATIC_PHRASES, "robotic_somatic_cliche"),
    )
    for phrases, reason_tag in checks:
        matched = next((p for p in phrases if p in t_lower), None)
        if matched is not None:
            return True, f"{reason_tag}: '{matched}'"

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
