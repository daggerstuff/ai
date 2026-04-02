"""Shared LLM transport helpers for subconscious v3."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def call_llm(client: Any, messages: list[dict[str, Any]], **kwargs) -> Any:
    """Call an OpenAI-compatible, Anthropic-compatible, or callable async client."""
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return await client.chat.completions.create(messages=messages, **kwargs)

    if hasattr(client, "messages"):
        return await client.messages.create(messages=messages, **kwargs)

    if callable(client):
        result = client(messages=messages, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    raise ValueError(f"Unsupported client type: {type(client)}")


def extract_response_content(response: Any) -> Optional[str]:
    """Extract assistant text from known LLM response formats."""
    try:
        if hasattr(response, "choices"):
            return response.choices[0].message.content

        if hasattr(response, "content"):
            return response.content[0].text

        return str(response)
    except Exception as exc:
        logger.debug("Could not extract content from response: %s", exc)
        return None
