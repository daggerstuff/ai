from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ai.prompts.system.reflection_prompt_provider import ReflectionPromptProvider
from .reflection_response_parser import ReflectionResponseParser
from .reflection_types import ReflectionResult

logger = logging.getLogger(__name__)

LLMCallback = Callable[[str], Awaitable[str]]


class ReflectionAnalysisService:
    """LLM-facing reflection analysis helpers kept out of the subagent shell."""

    def __init__(self, llm_callback: LLMCallback | None) -> None:
        self.llm_callback = llm_callback
        self.prompt_provider = ReflectionPromptProvider()
        self.response_parser = ReflectionResponseParser()

    async def detect_crisis(self, conversation_text: str) -> dict[str, Any]:
        if not self.llm_callback:
            return {"crisis_detected": False, "indicators": []}

        try:
            prompt = self.prompt_provider.crisis_detection_prompt(conversation_text)
            response = await self.llm_callback(prompt)
            result = json.loads(response)
            return {
                "crisis_detected": result.get("severity") in ["high", "critical"],
                "indicators": result.get("indicators", []),
                "severity": result.get("severity", "none"),
            }
        except Exception as exc:
            logger.error("Crisis detection failed: %s", exc)
            return {"crisis_detected": False, "indicators": []}

    async def run_reflection(
        self,
        *,
        conversation_text: str,
        existing_memories: str,
        include_crisis_context: bool,
        crisis_detected: bool | None,
    ) -> str:
        if not self.llm_callback:
            logger.warning("No LLM callback - returning empty analysis")
            return "{}"

        try:
            return await self.llm_callback(
                self.prompt_provider.reflection_prompt(
                    conversation_text=conversation_text,
                    existing_memories=existing_memories,
                    include_crisis_context=include_crisis_context,
                    crisis_detected=crisis_detected,
                )
            )
        except Exception as exc:
            logger.error("LLM reflection invocation failed: %s", exc)
            return "{}"

    def parse_analysis(self, analysis_text: str, *, crisis_detected: bool) -> ReflectionResult:
        return self.response_parser.parse_analysis(
            analysis_text,
            crisis_detected=crisis_detected,
        )
