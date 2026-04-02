"""Shared conversation history and enrichment helpers for subconscious v3."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, List

from .reflection import dedupe_conversation_suffix, format_memories_xml, trim_conversation_history


@dataclass
class ConversationManager:
    """Track a bounded conversation history and format memory-enriched prompts."""

    _conversation: List[dict[str, str]] = field(default_factory=list)

    @property
    def conversation(self) -> List[dict[str, str]]:
        """Expose the live conversation history."""
        return self._conversation

    def record_messages(self, messages: List[dict]) -> None:
        """Append the unseen suffix of outbound messages."""
        recordable_messages = copy.deepcopy(messages)
        self._conversation.extend(
            dedupe_conversation_suffix(self._conversation, recordable_messages)
        )
        self._conversation = trim_conversation_history(self._conversation)

    def record_message(self, role: str, content: str) -> None:
        """Record a single message and enforce the conversation bound."""
        if not role or not content:
            return
        self._conversation.append({"role": role, "content": copy.deepcopy(content)})
        self._conversation = trim_conversation_history(self._conversation)

    def inject_memories(
        self,
        *,
        messages: List[dict],
        last_user_idx: int,
        memories: List[Any],
        max_memories: int,
    ) -> None:
        """Inject recalled memories into the last user message in place."""
        if not memories:
            return
        if "content" not in messages[last_user_idx]:
            return
        original = messages[last_user_idx].get("content", "")
        memory_block = format_memories_xml(memories, limit=max_memories)
        messages[last_user_idx]["content"] = f"{memory_block}\\n\\n{original}"

    def clear(self) -> None:
        """Clear the recorded conversation history."""
        self._conversation.clear()
