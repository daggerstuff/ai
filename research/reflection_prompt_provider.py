from __future__ import annotations

from .reflection_prompts import CRISIS_DETECTION_PROMPT, get_reflection_prompt


class ReflectionPromptProvider:
    """Prompt construction for reflection analysis."""

    @staticmethod
    def crisis_detection_prompt(conversation_text: str) -> str:
        return CRISIS_DETECTION_PROMPT.template.format(conversation_text=conversation_text[:5000])

    @staticmethod
    def reflection_prompt(
        *,
        conversation_text: str,
        existing_memories: str,
        include_crisis_context: bool,
        crisis_detected: bool | None,
    ) -> str:
        prompt = get_reflection_prompt(
            crisis_detected=crisis_detected,
            include_crisis_context=include_crisis_context,
        )
        return prompt.template.format(
            conversation_text=conversation_text,
            existing_memories=existing_memories,
        )
