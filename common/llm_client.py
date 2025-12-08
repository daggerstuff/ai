import abc
import logging
from typing import Any

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
    Placeholder for OpenAI Driver.
    Requires `openai` package and API key.
    """

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        raise NotImplementedError("OpenAI Driver not configured yet.")

    def generate_structured(
        self, prompt: str, schema: dict[str, Any], system_prompt: str | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError("OpenAI Driver not configured yet.")


class LLMClient:
    """
    Client for interacting with LLMs.
    Abstraction layer to switch between providers (Mock, OpenAI, Anthropic, vLLM).
    """

    def __init__(self, driver: str = "mock", config: dict | None = None):
        self.config = config or {}
        if driver.lower() == "openai":
            self.driver = OpenAIDriver()
        else:
            self.driver = MockDriver()

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.driver.generate(prompt, system_prompt)

    def generate_structured(
        self, prompt: str, schema: dict[str, Any], system_prompt: str | None = None
    ) -> dict[str, Any]:
        return self.driver.generate_structured(prompt, schema, system_prompt)
