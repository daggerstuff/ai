"""Factory helpers for OpenAI-compatible subconscious clients."""

from __future__ import annotations


def create_default_llm_client(*, api_key: str, base_url: str):
    """Create the default async LLM client for subconscious operations."""
    import openai

    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
