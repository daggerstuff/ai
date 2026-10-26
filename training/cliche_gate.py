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
    "i'm glad you told me",
    "it makes sense that",
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

# Stock acknowledgment fragments are common mid-sentence as well as at turn
# openings, so these are rejected anywhere in an assistant response.
CANNED_REFLECTIONS: tuple[str, ...] = (
    "i'm glad",
    "i understand wanting",
    "it makes sense that",
    "it sounds like",
    "i hear that",
    "i hear you",
    "it's helpful to know",
    "i understand you're feeling",
    "holding space",
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
    t_lower = (
        text.strip()
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .lower()
    )

    # Prefix checks (at string start or right after a newline)
    for prefix in BANNED_OPENERS:
        if t_lower.startswith(prefix) or f"\n{prefix}" in t_lower:
            return True, f"banned_sycophantic_opener: '{prefix}'"
    for prefix in PARROTING_OPENERS:
        if t_lower.startswith(prefix) or f"\n{prefix}" in t_lower:
            return True, f"banned_parroting_opener: '{prefix}'"

    for phrase in CANNED_REFLECTIONS:
        if phrase in t_lower:
            return True, f"canned_reflection: '{phrase}'"

    # Substring checks anywhere in the utterance
    checks: tuple[tuple[tuple[str, ...], str], ...] = (
        (CAVING_PHRASES, "caving_phrase_detected"),
        (ROBOTIC_CRISIS_QUESTIONS, "robotic_crisis_questionnaire"),
        (ROBOTIC_SOMATIC_PHRASES, "robotic_somatic_cliche"),
    )
    for phrases, reason_tag in checks:
        matched = next(
            (
                phrase
                for phrase in phrases
                if phrase in t_lower
                and not (
                    phrase == "i'll stop"
                    and any(
                        boundary in t_lower
                        for boundary in (
                            "i'll stop the exercise",
                            "i'll stop asking you to reassure me",
                            "i'll stop asking you to reassure",
                        )
                    )
                )
            ),
            None,
        )
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
        content = str(message.get("content", ""))
        if content.startswith("{") and "\n" in content:
            content = content.split("\n", 1)[1]
        is_bad, reason = is_sycophantic(content)
        if is_bad:
            return f"{reason} (family={family})" if family else reason
    return None


def correction_for_reason(reason: str, *, draft: str = "") -> str:
    """Map a gate rejection reason to targeted rewrite guidance for the retry prompt.

    ``reason`` is a string produced by :func:`is_sycophantic` (``tag: 'matched'``)
    or :func:`reject_reason_for_record` (same, optionally suffixed with
    ``(family=...)``). Unknown tags fall back to generic guidance.
    """
    # Strip family suffix and matched-phrase tail to recover the tag.
    tag = reason.split(":", 1)[0].split("(", 1)[0].strip()
    text = f"CORRECTION: The draft was rejected ({reason})."
    if draft:
        text += f"\nRejected draft: {draft!r}\n"
    guidance = {
        "caving_phrase_detected": (
            " The draft reflexively agreed, apologized, or retreated. Don't agree just to appease or retract a "
            "justified boundary. Respond to the concern directly and correct any actual mistake plainly."
        ),
        "banned_sycophantic_opener": (
            " The draft opened with praise or a reflective cliché. Start with direct engagement in your own words."
        ),
        "banned_parroting_opener": (
            " The draft quoted or echoed the user's words. Respond to the meaning instead of repeating the phrasing."
        ),
        "canned_reflection": (
            " The draft relied on a stock empathy phrase. Remove it and address the specific situation or request "
            "in direct, natural language. Do not repeat the user's wording."
        ),
        "robotic_crisis_questionnaire": (
            " The draft used a checklist-style crisis question. Ask one natural, direct question that clarifies "
            "the immediate risk."
        ),
        "robotic_somatic_cliche": (
            " The draft deflected with a canned body-sensation question. Engage with the user's actual situation."
        ),
    }
    generic = (
        " Avoid parroting or quoting the user's words, stock reflections, and reflexive agreement. Respond directly "
        "to the actual situation and request."
    )
    return text + guidance.get(tag, generic)
