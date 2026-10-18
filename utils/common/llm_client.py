import abc
import json
import logging
import os
from typing import Any

from openai import OpenAI

from ai.utils.common.rate_limiter import (
    TierAwareRateLimiter,
    default_rate_limiter,
)

logger = logging.getLogger(__name__)

PROVIDER_FIREWORKS = "fireworks"
PROVIDER_OPENAI = "openai"
PROVIDER_NVIDIA = "nvidia"


def _provider_for_driver(driver: str) -> str:
    """Map a driver name to its provider key for tier-aware rate limiting.

    Today the dispatcher maps ``nvidia`` → ``OpenAIDriver`` because the NVIDIA
    NIM endpoint is OpenAI-compatible. The provider key is what the rate
    limiter keys on, not the driver class.
    """

    name = driver.lower()
    if name in ("mock", "anthropic"):
        return name
    if name in ("openai",):
        return PROVIDER_OPENAI
    if name in ("fireworks",):
        return PROVIDER_FIREWORKS
    if name in ("nvidia", "nim"):
        return PROVIDER_NVIDIA
    return name


class LLMDriver(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        pass


class MockDriver(LLMDriver):
    """
    Mock driver for testing without API keys.
    Returns deterministic or random responses.
    """

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        logger.info(f"MOCK GENERATE: {prompt[:50]}... (System: {bool(system_prompt)})")
        return "This is a simulated LLM response for testing purposes."

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        **kwargs,
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

    def __init__(self, model: str | None = None):

        # Load config from env or defaults
        self.api_key = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY"))
        self.base_url = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model = model or os.environ.get("LLM_MODEL", "meta/llama-3.1-405b-instruct")

        if not self.api_key:
            logger.warning("No LLM_API_KEY found. OpenAIDriver may fail.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 8192, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = kwargs.pop("temperature", None)
        if temp is None:
            temp = 0.7

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            # Some providers (Neon AI Gateway, Gemini) return content as a
            # list of typed content blocks, e.g.
            #   [{"type": "reasoning", ...}, {"type": "text", "text": "..."}]
            # Extract text from text blocks for downstream JSON parsing.
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text" and "text" in block:
                            parts.append(str(block["text"]))
                        elif "text" in block:
                            parts.append(str(block["text"]))
                return "\n".join(parts) if parts else ""
            return str(content)
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return f"[ERROR: {e!s}]"

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate structured JSON output.
        Note: Actual JSON mode depends on provider support.
        """

        # Append schema instruction
        schema_prompt = f"\nOutput strictly valid JSON matching this schema: {json.dumps(schema)}"
        full_prompt = prompt + schema_prompt

        try:
            kwargs.setdefault("max_tokens", 8192)
            content = self.generate(full_prompt, system_prompt, **kwargs)

            # Simple cleanup for markdown code blocks
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Structured Generation failed: {e}")
            return {"error": str(e)}


class FireworksDriver(OpenAIDriver):
    """Fireworks-specific driver.

    Fireworks exposes an OpenAI-compatible endpoint at ``/inference/v1`` so we
    reuse :class:`OpenAIDriver`'s call shape and only override credentials.
    Authentication is ``FIREWORKS_API_KEY`` and the model is
    ``accounts/fireworks/models/<model>``; we strip the ``accounts/fireworks/models/``
    prefix when callers pass a fully-qualified name.
    """

    _DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    _DEFAULT_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"

    def __init__(self, model: str | None = None):
        self.api_key = os.environ.get(
            "FIREWORKS_API_KEY", os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY"))
        )
        self.base_url = os.environ.get("FIREWORKS_BASE_URL", self._DEFAULT_BASE_URL)
        env_model = os.environ.get("FIREWORKS_MODEL") or os.environ.get("LLM_MODEL", self._DEFAULT_MODEL)
        resolved = model or env_model
        # Strip the accounts/fireworks/models/ prefix when callers pass
        # a fully-qualified Fireworks name (cubic #1).
        prefix = "accounts/fireworks/models/"
        if resolved.startswith(prefix):
            resolved = resolved[len(prefix) :]
        self.model = resolved
        if not self.api_key:
            logger.warning("No FIREWORKS_API_KEY found. FireworksDriver may fail.")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)


class LLMClient:
    """Client for interacting with LLMs.

    Abstraction layer across providers. The ``driver`` argument selects which
    :class:`LLMDriver` subclass binds; for back-compat, ``"openai"`` continues
    to mean the OpenAI-compatible endpoint and ``"nvidia"`` aliases to the
    same driver (NVIDIA NIM exposes an OpenAI-compatible API). ``"fireworks"``
    routes to :class:`FireworksDriver`.

    All non-mock drivers are gated by :class:`TierAwareRateLimiter` so the
    process stays under Fireworks starter-tier TPM caps (see
    ``rate_limiter.py``). Set ``RATELIMIT_DISABLED=1`` to bypass.
    """

    def __init__(
        self,
        driver: str = "mock",
        config: dict | None = None,
        rate_limiter: TierAwareRateLimiter | None = None,
        model: str | None = None,
    ):
        self.config = config or {}
        self.driver_name = driver.lower()
        self.model = model
        self.driver = self._build_driver(self.driver_name, model)
        self.rate_limiter = rate_limiter or default_rate_limiter()
        # Derive provider from the ACTUAL driver class so an unknown name
        # that fell back to MockDriver is never counted as a real provider
        # upstream of the limiter (cubic #2).
        self.provider = _provider_for_driver(type(self.driver).__name__.lower())
        self._resolved_model = getattr(self.driver, "model", "default")

    def _build_driver(self, name: str, model: str | None = None) -> LLMDriver:
        if name == "mock":
            return MockDriver()
        if name == PROVIDER_FIREWORKS:
            return FireworksDriver(model=model)
        if name in ("openai", "nvidia", "nim"):
            return OpenAIDriver(model=model)
        return MockDriver()

    def _estimated_tokens(self, prompt: str, system_prompt: str | None, kwargs: dict) -> int:
        """Rough heuristic so the limiter has a cost estimate.

        Tolerates ``None``/non-int ``max_tokens`` instead of raising
        ``TypeError`` (cubic #3).
        """

        extra = 0
        if isinstance(kwargs, dict):
            value = kwargs.get("max_tokens")
            if isinstance(value, int) and value > 0:
                extra = value
        text_len = len(prompt) + (len(system_prompt) if system_prompt else 0)
        return max(text_len // 4 + extra, 256)

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        if self.provider not in ("mock",):
            acquired = self.rate_limiter.acquire(
                provider=self.provider,
                model=self._resolved_model,
                estimated_tokens=self._estimated_tokens(prompt, system_prompt, kwargs),
            )
            if not acquired:
                logger.warning(
                    "Rate limiter rejected acquire for provider=%s model=%s; returning empty",
                    self.provider,
                    self._resolved_model,
                )
                return ""
        try:
            return self.driver.generate(prompt, system_prompt, **kwargs)
        finally:
            if self.provider not in ("mock",):
                self.rate_limiter.release_in_flight(self.provider, self._resolved_model)

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if self.provider not in ("mock",):
            base_estimate = self._estimated_tokens(prompt, system_prompt, {})
            try:
                schema_budget = max(len(json.dumps(schema)) // 4, 256)
            except (TypeError, ValueError):
                schema_budget = 0
            acquired = self.rate_limiter.acquire(
                provider=self.provider,
                model=self._resolved_model,
                estimated_tokens=base_estimate + schema_budget,
            )
            if not acquired:
                logger.warning(
                    "Rate limiter rejected acquire for provider=%s model=%s; returning empty",
                    self.provider,
                    self._resolved_model,
                )
                return {}
        try:
            return self.driver.generate_structured(prompt, schema, system_prompt, **kwargs)
        finally:
            if self.provider not in ("mock",):
                self.rate_limiter.release_in_flight(self.provider, self._resolved_model)
