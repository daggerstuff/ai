"""Shared reflection helpers for subconscious v3."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, List, Optional

from .constants import MAX_CONVERSATION_LENGTH, MAX_CONVERSATION_MESSAGES, MAX_TOKENS
from .llm_client_factory import create_default_llm_client
from .llm_transport import call_llm, extract_response_content

logger = logging.getLogger(__name__)


async def extract_learnings_from_llm(
    *,
    api_key: str,
    base_url: str,
    model: str,
    conversation: str,
    focus_prompt: str,
    llm_client: Optional[Any] = None,
    client_factory: Optional[Callable[..., Any]] = None,
) -> List[str]:
    """Extract durable learnings from a conversation using an OpenAI-compatible client."""
    if not api_key:
        logger.warning("No API key configured, skipping reflection")
        return []

    try:
        client = llm_client or (client_factory or create_default_llm_client)(
            api_key=api_key,
            base_url=base_url,
        )
        prompt = (
            f"{focus_prompt}\n\nConversation:\n"
            f"{conversation[:MAX_CONVERSATION_LENGTH]}\n\n"
            "Respond with a JSON array of learnings. "
            'Each learning should be a single string. Example: ["User prefers pnpm"]'
        )
        response = await call_llm(
            client,
            [{"role": "user", "content": prompt}],
            model=model,
            max_tokens=MAX_TOKENS,
        )
        return _parse_learning_response(extract_response_content(response) or "[]")
    except Exception as exc:
        logger.error("LLM reflection failed: %s", exc, exc_info=True)
        return []


def conversation_to_text(conversation: List[dict[str, str]]) -> str:
    """Convert recorded conversation messages into a stable reflection transcript."""
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in conversation
    )


def format_memories_xml(memories: List[Any], *, limit: int) -> str:
    """Format recalled memories for subconscious injection."""
    lines = ["<subconscious_context>", "  <relevant_memories>"]
    for memory in memories[:limit]:
        content = memory.content[:200]
        if len(memory.content) > 200:
            content += "..."
        lines.append(f"    - {content}")
    lines.append("  </relevant_memories>")
    lines.append("</subconscious_context>")
    return "\n".join(lines)


def dedupe_conversation_suffix(
    existing: List[dict[str, str]],
    candidate: List[dict[str, str]],
) -> List[dict[str, str]]:
    """Return only the unseen suffix of a candidate conversation sequence."""
    max_overlap = min(len(existing), len(candidate))
    for overlap in range(max_overlap, -1, -1):
        if overlap == 0:
            return candidate
        if existing[-overlap:] == candidate[:overlap]:
            return candidate[overlap:]
    return candidate


def trim_conversation_history(
    conversation: List[dict[str, str]],
    *,
    max_messages: int = MAX_CONVERSATION_MESSAGES,
) -> List[dict[str, str]]:
    """Keep only the most recent bounded slice of conversation history."""
    if max_messages < 1:
        raise ValueError("max_messages must be >= 1")
    if len(conversation) <= max_messages:
        return conversation
    return conversation[-max_messages:]


async def store_reflection_learnings(
    *,
    provider: Any,
    user_id: str,
    learnings: List[str],
) -> int:
    """Store validated reflection learnings through the configured provider."""
    stored_count = 0
    for learning in learnings:
        if not learning or not isinstance(learning, str):
            logger.warning("Skipping invalid learning: %s", type(learning))
            continue
        if len(learning) < 10:
            logger.debug("Skipping short learning: '%s'", learning[:50])
            continue
        try:
            await provider.store(
                content=learning[:1000],
                user_id=user_id,
                metadata={"source": "reflection"},
            )
            stored_count += 1
        except Exception as exc:
            logger.error("Failed to store memory: %s", exc, exc_info=True)
    return stored_count


async def reflect_conversation(
    *,
    provider: Any,
    user_id: str,
    conversation: List[dict[str, str]],
    api_key: str,
    base_url: str,
    model: str,
    focus_prompt: str,
    llm_client: Optional[Any] = None,
    client_factory: Optional[Callable[..., Any]] = None,
) -> int:
    """Extract and persist durable learnings from a recorded conversation."""
    if not conversation:
        logger.debug("No conversation to reflect on")
        return 0

    conv_text = conversation_to_text(conversation)
    learnings = await extract_learnings_from_llm(
        api_key=api_key,
        base_url=base_url,
        model=model,
        conversation=conv_text,
        focus_prompt=focus_prompt,
        llm_client=llm_client,
        client_factory=client_factory,
    )
    return await store_reflection_learnings(
        provider=provider,
        user_id=user_id,
        learnings=learnings,
    )


def _parse_learning_response(content: str) -> List[str]:
    """Parse the model response into a list of string learnings."""
    try:
        parsed = json.loads(content)
        return _normalize_learning_list(parsed)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*?\]", content)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("JSON parse failed: %s", match.group(0)[:50])
            return []
        return _normalize_learning_list(parsed)


def _normalize_learning_list(payload: Any) -> List[str]:
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, str) and item.strip()]
