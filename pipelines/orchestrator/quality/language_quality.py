"""
Language Quality Assessment for Conversations.

This module provides a function to assess the language quality of a conversation,
using simple linguistic metrics such as spelling, grammar, sentence complexity,
and vocabulary diversity.

The score ranges from 0.0 (poor quality) to 1.0 (high quality).
"""

import re
from typing import Any

from .conversation_schema import Conversation


def _count_spelling_errors(text: str) -> int:
    # Placeholder: In production, use a spellchecker library.
    # Here, count words with obvious typos (e.g., repeated letters, non-alpha).
    words = text.split()
    errors = 0
    for word in words:
        if re.search(r"(.)\\1{2,}", word):  # e.g., "soooo"
            errors += 1
        if not re.match(r"^[a-zA-Z'-]+$", word):
            errors += 1
    return errors


def _sentence_complexity(text: str) -> float:
    # Simple proxy: average sentence length in words.
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    total_words = sum(len(s.split()) for s in sentences)
    return total_words / len(sentences)


def _vocab_diversity(text: str) -> float:
    words = [w.lower() for w in re.findall(r"\\b\\w+\\b", text)]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def assess_language_quality(conversation: Conversation) -> dict[str, Any]:
    """
    Assesses the language quality of a conversation.

    Heuristics:
    - Spelling errors (fewer is better)
    - Sentence complexity (not too simple, not too complex)
    - Vocabulary diversity (higher is better)

    Returns:
        dict: {
            "score": float (0.0 to 1.0),
            "issues": list of str (explanations for low scores)
        }
    """
    total_spelling = 0
    total_complexity = 0.0
    total_diversity = 0.0
    total_msgs = len(conversation.messages)
    issues = []

    if total_msgs == 0:
        return {"score": 0.0, "issues": ["No messages in conversation"]}

    for msg in conversation.messages:
        text = msg.content
        total_spelling += _count_spelling_errors(text)
        total_complexity += _sentence_complexity(text)
        total_diversity += _vocab_diversity(text)

    avg_spelling = total_spelling / total_msgs
    avg_complexity = total_complexity / total_msgs
    avg_diversity = total_diversity / total_msgs

    # Heuristic scoring
    score = 1.0
    if avg_spelling > 1:
        score -= 0.3
        issues.append("Frequent spelling errors detected")
    if avg_complexity < 5:
        score -= 0.2
        issues.append("Sentences are too simple")
    if avg_diversity < 0.3:
        score -= 0.2
        issues.append("Low vocabulary diversity")
    if score < 0.5:
        issues.append("Overall language quality is low")

    return {"score": round(max(score, 0.0), 3), "issues": issues}
