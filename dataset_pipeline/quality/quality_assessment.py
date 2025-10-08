"""
Quality scoring and validation system for the dataset pipeline.
Implements conversation coherence assessment and related quality checks.
"""

from typing import Any

from ai.dataset_pipeline.conversation_schema import Conversation


def assess_coherence(conversation: Conversation) -> dict[str, Any]:
    """
    Assess the coherence of a conversation using simple heuristics.
    Returns a dict with a 'score' (0-1), 'issues' (list), and 'details'.
    """
    issues = []
    score = 1.0
    details = {}

    # Heuristic 1: Alternating roles (e.g., user/assistant)
    roles = [msg.role for msg in conversation.messages]
    if len(roles) < 2:
        issues.append("Too few messages for coherence assessment.")
        score = 0.0
    else:
        for i in range(1, len(roles)):
            if roles[i] == roles[i - 1]:
                issues.append(f"Non-alternating roles at position {i}: {roles[i]}")
                score -= 0.2

    # Heuristic 2: No empty messages
    for i, msg in enumerate(conversation.messages):
        if not msg.content or not msg.content.strip():
            issues.append(f"Empty message at position {i}")
            score -= 0.1

    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))

    details["role_sequence"] = roles
    details["num_messages"] = len(conversation.messages)

    return {"score": score, "issues": issues, "details": details}
