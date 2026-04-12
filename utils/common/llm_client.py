import abc
import json
import logging
import os
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMDriver(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        pass

    @abc.abstractmethod
    def generate_structured(
        self, prompt: str, schema: dict[str, Any], system_prompt: str | None = None
    ) -> dict[str, Any]:
        pass


class MockDriver(LLMDriver):
    """
    Mock driver for testing without API keys.
    Returns deterministic or random responses.
    """

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        logger.info(f"MOCK GENERATE: {prompt[:50]}... (System: {bool(system_prompt)})")
        return "This is a simulated LLM response for testing purposes."

    def generate_structured(
        self, prompt: str, schema: dict[str, Any], system_prompt: str | None = None
    ) -> dict[str, Any]:
        logger.info(
            f"MOCK GENERATE STRUCTURED: {prompt[:50]}... "
            f"(Schema keys: {list(schema.keys())}, System: {bool(system_prompt)})"
        )
        # Simulate a response based on expected keys if possible, or generic
        return {"simulated_key": "simulated_value", "note": "This is mock data"}


class OpenAIDriver(LLMDriver):
    """
    OpenAI-compatible Driver (works with Nvidia NIM, Gemini, vLLM).
    """

    def __init__(self):
        pass


        # Load config from env or defaults
        self.api_key = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY"))
        self.base_url = os.environ.get(
            "LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.model = os.environ.get("LLM_MODEL", "meta/llama-3.1-405b-instruct")

        if not self.api_key:
            logger.warning("No LLM_API_KEY found. OpenAIDriver may fail.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(
        self, prompt: str, system_prompt: str | None = None, max_tokens: int = 8192
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return f"[ERROR: {e!s}]"

    def generate_structured(
        self, prompt: str, schema: dict[str, Any], system_prompt: str | None = None
    ) -> dict[str, Any]:
        """
        Generate structured JSON output.
        Note: Actual JSON mode depends on provider support.
        """

        # Append schema instruction
        schema_prompt = (
            f"\nOutput strictly valid JSON matching this schema: {json.dumps(schema)}"
        )
        full_prompt = prompt + schema_prompt

        try:
            # Force JSON format if supported (Nvidia/OpenAI usually support
            #   response_format={"type": "json_object"})
            # But for broad compatibility, we just ask for it in the prompt.
            content = self.generate(full_prompt, system_prompt)

            # Simple cleanup for markdown code blocks
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Structured Generation failed: {e}")
            return {"error": str(e)}


class LLMClient:
    """
    Client for interacting with LLMs.
    Abstraction layer to switch between providers (Mock, OpenAI, Anthropic, vLLM).
    """

    def __init__(self, driver: str = "mock", config: dict | None = None):
        self.config = config or {}
        self.driver = OpenAIDriver() if driver.lower() == "openai" else MockDriver()

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        return self.driver.generate(prompt, system_prompt, **kwargs)

    def generate_structured(
        self, prompt: str, schema: dict[str, Any], system_prompt: str | None = None
    ) -> dict[str, Any]:
        return self.driver.generate_structured(prompt, schema, system_prompt)
