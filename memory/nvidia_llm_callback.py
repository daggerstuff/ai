"""
Nvidia NIM LLM Callback - OpenAI-compatible callback for reflection subagent.

This module provides an LLM callback function that uses Nvidia NIM's OpenAI-compatible
API to power the reflection subagent with qwen/qwen3.5-397b-a17b.

Env-configurable: reads LLM_API_KEY (fallback NVIDIA_API_KEY), LLM_BASE_URL
(fallback NVIDIA NIM), LLM_MODEL (fallback qwen/qwen3.5-397b-a17b). When
LLM_BASE_URL points to Neon AI Gateway, uses Neon's free open-weight models.
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
        base_url: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize LLM callback.

        Args:
            api_key: API key. If None, reads LLM_API_KEY (fallback NVIDIA_API_KEY) from env.
            base_url: Base URL. If None, reads LLM_BASE_URL from env, defaults to NVIDIA NIM.
            model: Model to use. If None, reads LLM_MODEL from env, defaults to qwen/qwen3.5-397b-a17b.
        """
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model = model or os.environ.get("LLM_MODEL", "qwen/qwen3.5-397b-a17b")

        if not self.api_key:
            raise ValueError("No API key found. Set LLM_API_KEY or NVIDIA_API_KEY environment variable.")

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
            content = response.choices[0].message.content
            if isinstance(content, list):
                return "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ) or ""
            return content or ""
        except Exception as e:
            logger.error(f"Nvidia NIM API call failed: {e}")
            raise


def create_nvidia_callback(
    model: str | None = None,
    base_url: str | None = None,
) -> NvidiaNIMCallback:
    """
    Create LLM callback for reflection subagent.

    Reads from environment if not specified:
    - LLM_MODEL (fallback SUBCONSCIOUS_MODEL, then qwen/qwen3.5-397b-a17b)
    - LLM_BASE_URL (fallback SUBCONSCIOUS_BASE_URL, then NVIDIA NIM)
    """
    model = model or os.environ.get("LLM_MODEL") or os.environ.get("SUBCONSCIOUS_MODEL", "qwen/qwen3.5-397b-a17b")
    base_url = base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("SUBCONSCIOUS_BASE_URL", "https://integrate.api.nvidia.com/v1")
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
