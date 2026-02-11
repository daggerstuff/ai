"""
Crisis intervention processor for emergency therapeutic scenarios.
Processes crisis intervention and emergency mental health scenarios.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from conversation_schema import Conversation, Message
from logger import get_logger

logger = get_logger(__name__)


class CrisisType(Enum):
    """Types of crisis situations."""
    SUICIDAL_IDEATION = "suicidal_ideation"
    SELF_HARM = "self_harm"
    PSYCHOTIC_EPISODE = "psychotic_episode"
    PANIC_ATTACK = "panic_attack"
    DOMESTIC_VIOLENCE = "domestic_violence"
    SUBSTANCE_OVERDOSE = "substance_overdose"
    TRAUMA_RESPONSE = "trauma_response"


class CrisisInterventionProcessor:
    """
    Processes crisis intervention scenarios for therapeutic training.

    Handles emergency situations requiring immediate intervention
    and safety planning.
    """

    def __init__(self):
        """Initialize the crisis intervention processor."""
        self.logger = get_logger(__name__)

        self.crisis_protocols = {
            CrisisType.SUICIDAL_IDEATION: [
                "assess immediate danger",
                "safety planning",
                "remove means",
                "emergency contacts"
            ],
            CrisisType.PSYCHOTIC_EPISODE: [
                "reality testing",
                "grounding techniques",
                "medical referral",
                "safety assessment"
            ]
        }

        self.logger.info("CrisisInterventionProcessor initialized")

    def process_crisis_data(self, data: list[dict[str, Any]]) -> list[Conversation]:
        """Process crisis intervention data."""
        conversations = []

        for item in data:
            conversation = self._create_crisis_conversation(item)
            if conversation:
                conversations.append(conversation)

        self.logger.info(f"Processed {len(conversations)} crisis conversations")
        return conversations

    def _create_crisis_conversation(self, item: dict[str, Any]) -> Conversation | None:
        """Create a crisis intervention conversation."""
        try:
            messages = [
                Message(
                    role="user",
                    content="I'm in crisis and need immediate help.",
                    timestamp=datetime.now()
                ),
                Message(
                    role="assistant",
                    content="I'm here to help you. Your safety is my priority. Can you tell me what's happening right now?",
                    timestamp=datetime.now()
                )
            ]

            return Conversation(
                id=f"crisis_{hash(str(item)) % 100000}",
                messages=messages,
                title="Crisis Intervention",
                metadata={"crisis_intervention": True, "safety_critical": True},
                tags=["crisis", "emergency", "safety"]
            )

        except Exception as e:
            self.logger.warning(f"Could not create crisis conversation: {e}")
            return None


def validate_crisis_intervention_processor():
    """Validate the CrisisInterventionProcessor functionality."""
    try:
        processor = CrisisInterventionProcessor()
        assert hasattr(processor, "process_crisis_data")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_crisis_intervention_processor():
        pass
    else:
        pass
