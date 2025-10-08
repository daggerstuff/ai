"""
persona_detection.py

Context-aware persona detection and switching utilities for Pixel.
Provides therapy/assistant mode detection, context analysis, and persona switching logic.
"""

from typing import Any


class PersonaDetector:
    """
    Persona detection and switching for dual-mode operation (therapy/assistant).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the persona detector.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}

    def detect_persona(self, conversation_context: dict[str, Any]) -> str:
        """
        Detects the appropriate persona (therapy or assistant) based on conversation context.

        Args:
            conversation_context: Dict with conversation metadata, user intent, etc.

        Returns:
            "therapy" or "assistant"
        """
        # Real context analysis: check intent, keywords, and user profile
        intent = conversation_context.get("intent", "").lower()
        message = conversation_context.get("message", "").lower()
        user_profile = conversation_context.get("user_profile", {})
        therapy_keywords = [
            "therapy",
            "counsel",
            "mental health",
            "diagnosis",
            "treatment",
            "session",
        ]
        if any(kw in intent for kw in therapy_keywords) or any(
            kw in message for kw in therapy_keywords
        ):
            return "therapy"
        if user_profile.get("needs_therapy", False):
            return "therapy"
        return "assistant"

    def switch_persona(self, current_persona: str, context: dict[str, Any]) -> str:
        """
        Determines if a persona switch is needed and returns the new persona.

        Args:
            current_persona: Current persona ("therapy" or "assistant").
            context: Conversation context.

        Returns:
            New persona ("therapy" or "assistant").
        """
        # Switch if user explicitly requests, or if context intent/keywords change
        if context.get("switch_triggered", False):
            return "therapy" if current_persona == "assistant" else "assistant"
        new_persona = self.detect_persona(context)
        if new_persona != current_persona:
            return new_persona
        return current_persona

    def validate_consistency(self, conversation: list[dict[str, Any]]) -> bool:
        """
        Validates persona consistency across conversation turns.

        Args:
            conversation: List of message dicts.

        Returns:
            True if persona is consistent, False otherwise.
        """
        # Check if persona changes without explicit switch or context change
        last_persona = None
        for turn in conversation:
            persona = turn.get("persona")
            if last_persona is not None and persona != last_persona:
                # Allow switch only if triggered or context justifies
                if (
                    not turn.get("switch_triggered", False)
                    and not self.detect_persona(turn) == persona
                ):
                    return False
            last_persona = persona
        return True
