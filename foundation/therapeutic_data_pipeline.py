"""
Therapeutic Conversation Data Pipeline.

Core module for:
- Loading and validating therapeutic dialogue datasets
- Normalizing conversation formats
- Metadata enrichment (emotion, technique, therapeutic goal)
- Dataset splitting and balancing
- Deduplication and quality validation
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class TherapeuticTechnique(str, Enum):
    """Standard therapeutic techniques represented in conversations."""

    CBT = "Cognitive Behavioral Therapy"
    DBT = "Dialectical Behavior Therapy"
    MI = "Motivational Interviewing"
    ACT = "Acceptance and Commitment Therapy"
    PSYCHODYNAMIC = "Psychodynamic Therapy"
    HUMANISTIC = "Humanistic Therapy"
    TRAUMA_FOCUSED = "Trauma-Focused Therapy"
    UNKNOWN = "Unknown"


class ConversationRole(str, Enum):
    """Participant roles in therapeutic conversations."""

    THERAPIST = "therapist"
    PATIENT = "patient"
    SYSTEM = "system"


@dataclass
class ConversationTurn:
    """A single turn in a therapeutic conversation."""

    speaker: ConversationRole
    text: str
    timestamp: Optional[float] = None
    emotion_score: Optional[float] = None  # 0-1, neutral to intense
    detected_technique: Optional[TherapeuticTechnique] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TherapeuticConversation:
    """A complete therapeutic conversation session."""

    session_id: str
    turns: List[ConversationTurn]
    technique: TherapeuticTechnique = TherapeuticTechnique.UNKNOWN
    cultural_context: Optional[str] = None
    mental_health_focus: Optional[str] = None  # e.g., "anxiety", "depression", "trauma"
    quality_score: Optional[float] = None  # 0-1 after evaluation
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage and analysis."""
        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "technique": self.technique.value,
            "cultural_context": self.cultural_context,
            "mental_health_focus": self.mental_health_focus,
            "quality_score": self.quality_score,
            "turns": [
                {
                    "speaker": turn.speaker.value,
                    "text": turn.text,
                    "emotion_score": turn.emotion_score,
                    "detected_technique": (
                        turn.detected_technique.value
                        if turn.detected_technique
                        else None
                    ),
                }
                for turn in self.turns
            ],
        }


class TherapeuticDataPipeline:
    """Pipeline for loading, validating, and preparing therapeutic conversation data."""

    def __init__(self, data_dir: Path = Path("/home/vivi/pixelated/data/therapeutic")):
        self.data_dir = data_dir
        self.conversations: List[TherapeuticConversation] = []
        self.metadata_df: Optional[pd.DataFrame] = None

    def initialize(self):
        """Initialize data directory structure."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organization
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "synthetic").mkdir(exist_ok=True)
        (self.data_dir / "metadata").mkdir(exist_ok=True)

    def load_conversations(self, source_path: Path) -> int:
        """Load therapeutic conversations from source (JSON, CSV, etc)."""
        # Placeholder: actual implementation will parse formats
        # Returns count of loaded conversations
        return len(self.conversations)

    def validate_conversation(self, conv: TherapeuticConversation) -> bool:
        """Validate conversation meets quality standards."""
        return (
            bool(conv.session_id)
            and len(conv.turns) >= 2
            and all(turn.text for turn in conv.turns)
        )

    def deduplicate(self) -> int:
        """Remove duplicate or near-duplicate conversations."""
        # Placeholder: will use sentence-transformers for semantic dedup
        return len(self.conversations)

    def balance_by_technique(self) -> Dict[str, int]:
        """Ensure balanced representation across therapeutic techniques."""
        distribution = {}
        for conv in self.conversations:
            technique = conv.technique.value
            distribution[technique] = distribution.get(technique, 0) + 1
        return distribution

    def export_to_dataframe(self) -> pd.DataFrame:
        """Convert conversations to pandas DataFrame for analysis."""
        records = [conv.to_dict() for conv in self.conversations]
        return pd.DataFrame(records)

    def status(self) -> str:
        """Pipeline status report."""
        lines = [
            "=== Therapeutic Data Pipeline Status ===",
            f"Data Directory: {self.data_dir}",
            f"Conversations Loaded: {len(self.conversations)}",
        ]
        if self.conversations:
            balance = self.balance_by_technique()
            lines.append("\nTechnique Distribution:")
            lines.extend(
                f"  {technique}: {count}" for technique, count in balance.items()
            )
        return "\n".join(lines)


if __name__ == "__main__":
    pipeline = TherapeuticDataPipeline()
    pipeline.initialize()
    print(pipeline.status())
