"""Generate structured therapeutic assistant responses with safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .crisis_intervention_detector import CrisisInterventionDetector


@dataclass
class ResponsePackage:
    response: str
    tone: str
    crisis_action: str
    metadata: dict[str, Any]


class TherapeuticResponseGenerator:
    """Compose safe, empathetic responses for client messages."""

    def __init__(self, default_tone: str = "empathetic") -> None:
        self.default_tone = default_tone
        self.crisis_detector = CrisisInterventionDetector()

    def process(self, data: dict[str, Any] | str) -> ResponsePackage:
        if data is None:
            raise ValueError("Input payload cannot be None")

        text = self._extract_text(data)
        crisis = self.crisis_detector.process(data)

        if crisis.flagged and crisis.score >= 0.8:
            response = (
                "Thank you for sharing this. Your safety is the priority. "
                "I want to make sure you are not alone right now. Please contact "
                f"{', '.join(self.crisis_detector.escalation_contacts['critical'])} "
                "or local emergency services immediately."
            )
            action = "escalate_emergency"
            tone = "crisis"
        elif crisis.flagged:
            response = (
                "I can hear this is hard right now. Let's take this one step at a time. "
                "If the feelings are getting intense, please use your crisis plan and reach out "
                "to a trusted person or a support line right away."
            )
            action = "monitor_and_support"
            tone = "supportive"
        else:
            response = self._compose_standard_response(text, tone=self.default_tone)
            action = "standard_support"
            tone = self.default_tone

        return ResponsePackage(
            response=response,
            tone=tone,
            crisis_action=action,
            metadata={
                "input_length": len(text),
                "crisis_flagged": crisis.flagged,
                "crisis_severity": crisis.severity,
                "crisis_score": crisis.score,
                "recommendations": list(crisis.recommendations),
            },
        )

    def _compose_standard_response(self, text: str, tone: str = "empathetic") -> str:
        if not text.strip():
            return "I am here to help. Could you share a bit more about how you are feeling right now?"

        if tone == "brief":
            return "Thank you for sharing. That sounds important. Let's explore this further when you are ready."

        return (
            "Thank you for sharing that with me. It sounds like you are working through a difficult"
            " situation, and your experience matters. Let's pause for a moment and focus on what feels"
            " most important to you right now."
        )

    def _extract_text(self, data: dict[str, Any] | str) -> str:
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            raise TypeError("TherapeuticResponseGenerator expects mapping or text")
        for key in ("text", "content", "message", "input"):
            value = data.get(key)
            if isinstance(value, str):
                return value
        raise ValueError("No text content found")


__all__ = ["ResponsePackage", "TherapeuticResponseGenerator"]
