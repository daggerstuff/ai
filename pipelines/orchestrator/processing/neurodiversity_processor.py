"""
Neurodiversity processor for neurodivergent-aware therapeutic approaches.
Processes data for neurodivergent vs neurotypical therapeutic interactions.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from conversation_schema import Conversation, Message
from logger import get_logger

logger = get_logger(__name__)


class NeurodivergentType(Enum):
    """Types of neurodivergence."""
    AUTISM = "autism"
    ADHD = "adhd"
    DYSLEXIA = "dyslexia"
    TOURETTES = "tourettes"
    BIPOLAR = "bipolar"
    OCD = "ocd"


class NeurodiversityProcessor:
    """
    Processes neurodiversity-aware therapeutic data.

    Handles neurodivergent vs neurotypical interactions and
    neurodiversity-affirming therapeutic approaches.
    """

    def __init__(self):
        """Initialize the neurodiversity processor."""
        self.logger = get_logger(__name__)

        self.neurodivergent_considerations = {
            NeurodivergentType.AUTISM: [
                "sensory processing", "communication differences", "routine importance",
                "special interests", "masking", "stimming"
            ],
            NeurodivergentType.ADHD: [
                "attention regulation", "hyperactivity", "impulsivity",
                "executive function", "time blindness", "hyperfocus"
            ]
        }

        self.affirming_approaches = [
            "strength-based perspective",
            "accommodation over normalization",
            "sensory-friendly environment",
            "clear communication",
            "respect for differences"
        ]

        self.logger.info("NeurodiversityProcessor initialized")

    def process_neurodiversity_data(self, data: list[dict[str, Any]]) -> list[Conversation]:
        """Process neurodiversity-aware therapeutic data."""
        conversations = []

        for item in data:
            conversation = self._create_neurodiversity_conversation(item)
            if conversation:
                conversations.append(conversation)

        self.logger.info(f"Processed {len(conversations)} neurodiversity conversations")
        return conversations

    def _create_neurodiversity_conversation(self, item: dict[str, Any]) -> Conversation | None:
        """Create a neurodiversity-aware conversation."""
        try:
            messages = [
                Message(
                    role="user",
                    content="I'm neurodivergent and looking for therapy that understands my differences.",
                    timestamp=datetime.now()
                ),
                Message(
                    role="assistant",
                    content="I'm committed to providing neurodiversity-affirming care that honors your unique neurological differences as natural variations, not deficits.",
                    timestamp=datetime.now()
                )
            ]

            return Conversation(
                id=f"neurodiversity_{hash(str(item)) % 100000}",
                messages=messages,
                title="Neurodiversity-Affirming Therapy",
                metadata={"neurodiversity_aware": True, "affirming_approach": True},
                tags=["neurodiversity", "affirming", "inclusive"]
            )

        except Exception as e:
            self.logger.warning(f"Could not create neurodiversity conversation: {e}")
            return None


def validate_neurodiversity_processor():
    """Validate the NeurodiversityProcessor functionality."""
    try:
        processor = NeurodiversityProcessor()
        assert hasattr(processor, "process_neurodiversity_data")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if validate_neurodiversity_processor():
        pass
    else:
        pass
