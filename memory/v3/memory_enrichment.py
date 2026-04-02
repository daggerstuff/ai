"""Shared memory enrichment helpers for subconscious v3."""

from __future__ import annotations

from typing import Any, List

from .conversation_manager import ConversationManager


def enrich_user_message(
    *,
    conversation_manager: ConversationManager,
    message: str,
    memories: List[Any],
    max_memories: int,
) -> str:
    """Inject already recalled memories into a user message."""
    if not message or not memories:
        return message

    enriched_messages = [{"role": "user", "content": message}]
    conversation_manager.inject_memories(
        messages=enriched_messages,
        last_user_idx=0,
        memories=memories,
        max_memories=max_memories,
    )
    return enriched_messages[0]["content"]
