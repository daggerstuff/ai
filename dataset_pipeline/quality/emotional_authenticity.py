"""
Emotional Authenticity Scoring for Conversations.

This module provides a function to score the emotional authenticity of a conversation,
based on the presence and diversity of emotion words, context-appropriate affect, and
variation in emotional expression.

The score ranges from 0.0 (inauthentic) to 1.0 (highly authentic).
"""

from typing import Any

from .conversation_schema import Conversation

# A simple set of emotion words for demonstration purposes.
EMOTION_WORDS = {
    "happy",
    "sad",
    "angry",
    "afraid",
    "excited",
    "frustrated",
    "joy",
    "love",
    "hate",
    "disappointed",
    "proud",
    "ashamed",
    "guilty",
    "relieved",
    "anxious",
    "calm",
    "hopeful",
}


def score_emotional_authenticity(conversation: Conversation) -> dict[str, Any]:
    """
    Scores the emotional authenticity of a conversation.

    Heuristics:
    - Presence of emotion words in messages
    - Diversity of emotion words
    - Variation in emotional expression between user and assistant
    - Penalize if all messages are emotionally flat or identical

    Returns:
        dict: {
            "score": float (0.0 to 1.0),
            "issues": list of str (explanations for low scores)
        }
    """
    emotion_counts = set()
    flat_count = 0
    last_emotion = None
    issues = []

    for msg in conversation.messages:
        words = set(msg.content.lower().split())
        found = words & EMOTION_WORDS
        if found:
            emotion_counts.update(found)
            if last_emotion and found == last_emotion:
                flat_count += 1
            last_emotion = found
        else:
            flat_count += 1

    total_msgs = len(conversation.messages)
    if total_msgs == 0:
        return {"score": 0.0, "issues": ["No messages in conversation"]}

    diversity = len(emotion_counts) / max(1, total_msgs)
    flatness = flat_count / total_msgs

    score = 0.5 * diversity + 0.5 * (1 - flatness)

    if diversity < 0.2:
        issues.append("Low diversity of emotional expression")
    if flatness > 0.7:
        issues.append("Most messages lack emotional content")
    if score < 0.5:
        issues.append("Overall emotional authenticity is low")

    return {"score": round(score, 3), "issues": issues}
