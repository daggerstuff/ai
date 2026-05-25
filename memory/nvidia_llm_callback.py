"""
Nvidia NIM LLM Callback - OpenAI-compatible callback for reflection subagent.

This module provides an LLM callback function that uses Nvidia NIM's OpenAI-compatible
API to power the reflection subagent with qwen/qwen3.5-397b-a17b.

Uses NVIDIA_API_KEY from environment (already present in .env).
"""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


class NvidiaNIMCallback:
    """
    LLM callback using Nvidia NIM's OpenAI-compatible API.

    This callback can be used with ReflectionSubagent to provide
    crisis-aware memory analysis and consolidation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "qwen/qwen3.5-397b-a17b",
    ):
        """
        Initialize Nvidia NIM callback.

        Args:
            api_key: Nvidia API key. If None, reads from NVIDIA_API_KEY env var.
            base_url: Nvidia NIM base URL.
            model: Model to use. Defaults to qwen/qwen3.5-397b-a17b.
        """
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.base_url = base_url
        self.model = model

        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found. Set NVIDIA_API_KEY environment variable.")

        # Lazy init
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    async def __call__(self, prompt: str) -> str:
        """
        Invoke LLM with given prompt.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            LLM response text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4096,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Nvidia NIM API call failed: {e}")
            raise


def create_nvidia_callback(
    model: str | None = None,
    base_url: str | None = None,
) -> NvidiaNIMCallback:
    """
    Create Nvidia NIM callback for reflection subagent.

    Reads from environment if not specified:
    - SUBCONSCIOUS_MODEL (default: qwen/qwen3.5-397b-a17b)
    - SUBCONSCIOUS_BASE_URL (default: https://integrate.api.nvidia.com/v1)
    """
    model = model or os.environ.get("SUBCONSCIOUS_MODEL", "qwen/qwen3.5-397b-a17b")
    base_url = base_url or os.environ.get("SUBCONSCIOUS_BASE_URL", "https://integrate.api.nvidia.com/v1")
    """
    Create Nvidia NIM callback for reflection subagent.

    Args:
        model: Model to use. Defaults to qwen/qwen3.5-397b-a17b.
        base_url: Nvidia NIM base URL.

    Returns:
        NvidiaNIMCallback instance.
    """
    return NvidiaNIMCallback(model=model, base_url=base_url)


async def nvidia_llm_callback(prompt: str) -> str:
    """
    Simple function-based callback for quick usage.

    Usage:
        callback = nvidia_llm_callback
        subagent = ReflectionSubagent(memory_provider, llm_callback=callback)

    Or with custom model:
        from functools import partial
        callback = partial(nvidia_llm_callback_with_model, model="qwen/qwen3.5-397b-a17b")
    """
    # Use default instance
    callback = create_nvidia_callback()
    return await callback(prompt)
