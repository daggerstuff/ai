"""
Therapeutic Accuracy Validation for Mental Health Conversations.

This module provides a function to validate the therapeutic accuracy of a conversation,
using heuristics such as the presence of evidence-based responses, avoidance of harmful
advice, and adherence to basic therapeutic principles.

The score ranges from 0.0 (inaccurate/harmful) to 1.0 (highly accurate/therapeutic).
"""

from typing import Any

from .conversation_schema import Conversation

# Example sets of evidence-based and harmful phrases for demonstration.
EVIDENCE_BASED_PHRASES = {
    "cognitive behavioral therapy",
    "cbt",
    "mindfulness",
    "deep breathing",
    "it's okay to feel",
    "let's explore",
    "can you tell me more",
    "support system",
    "self-care",
    "grounding technique",
    "validate your feelings",
}

HARMFUL_PHRASES = {
    "just get over it",
    "you're overreacting",
    "stop being sad",
    "it's your fault",
    "ignore your feelings",
    "you should be ashamed",
    "why can't you be normal",
}


def validate_therapeutic_accuracy(conversation: Conversation) -> dict[str, Any]:
    """
    Validates the therapeutic accuracy of a conversation.

    Heuristics:
    - Presence of evidence-based therapeutic phrases
    - Absence of harmful or stigmatizing language
    - Encouragement of self-exploration and validation

    Returns:
        dict: {
            "score": float (0.0 to 1.0),
            "issues": list of str (explanations for low scores)
        }
    """
    evidence_count = 0
    harmful_count = 0
    issues = []

    for msg in conversation.messages:
        content = msg.content.lower()
        if any(phrase in content for phrase in EVIDENCE_BASED_PHRASES):
            evidence_count += 1
        if any(phrase in content for phrase in HARMFUL_PHRASES):
            harmful_count += 1

    total_msgs = len(conversation.messages)
    if total_msgs == 0:
        return {"score": 0.0, "issues": ["No messages in conversation"]}

    evidence_ratio = evidence_count / total_msgs
    harmful_ratio = harmful_count / total_msgs

    score = max(0.0, evidence_ratio - harmful_ratio)

    if harmful_count > 0:
        issues.append("Contains harmful or stigmatizing language")
    if evidence_count == 0:
        issues.append("No evidence-based therapeutic content detected")
    if score < 0.5:
        issues.append("Therapeutic accuracy is below recommended threshold")

    return {"score": round(score, 3), "issues": issues}
