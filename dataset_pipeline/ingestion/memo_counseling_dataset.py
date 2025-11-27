"""
MEMO Counseling Summarization Dataset Integration

Integration module for the MEMO dataset from LCS2-IIITD (KDD 2022):
"A Novel Counseling Summarization Dataset"

MEMO provides:
- Counseling session dialogues
- Expert-written summaries
- Structured summarization annotations

ACCESS REQUIREMENTS:
The MEMO dataset requires academic agreement. To request access:
1. Visit: https://github.com/LCS2-IIITD/MEMO
2. Fill out the data access agreement form
3. Contact the authors via the provided email
4. Wait for approval (typically 1-2 weeks)

Once access is granted, place the dataset files in:
ai/datasets/memo/

This module provides:
- Dataset loading and conversion
- Counseling summarization model integration
- Summary generation utilities

Part of the Pixelated Empathy AI dataset pipeline.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Handle imports
ingestion_path = Path(__file__).parent
pipeline_root = ingestion_path.parent.parent
sys.path.insert(0, str(pipeline_root))

try:
    from schemas.conversation_schema import Conversation, Message
except ImportError:
    try:
        from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message
    except ImportError:
        from conversation_schema import Conversation, Message

try:
    from logger import get_logger

    logger = get_logger("dataset_pipeline.memo_counseling")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class SummaryType(Enum):
    """Types of counseling summaries."""

    EXTRACTIVE = "extractive"  # Key excerpts from dialogue
    ABSTRACTIVE = "abstractive"  # Paraphrased summary
    HYBRID = "hybrid"  # Combination of both
    CLINICAL = "clinical"  # Clinical note style
    PROGRESS = "progress"  # Session progress summary


@dataclass
class CounselingSession:
    """A counseling session with dialogue and summary."""

    session_id: str
    dialogue: list[dict[str, str]]  # List of {speaker, utterance}
    summary: str
    summary_type: SummaryType = SummaryType.ABSTRACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_conversation(self) -> Conversation:
        """Convert to standard Conversation format."""
        messages = []
        for turn in self.dialogue:
            speaker = turn.get("speaker", "unknown").lower()
            content = turn.get("utterance", turn.get("text", ""))

            # Map speaker to standard roles
            if speaker in ["client", "patient", "seeker"]:
                role = "user"
            elif speaker in ["counselor", "therapist", "helper"]:
                role = "assistant"
            else:
                role = speaker

            if content:
                messages.append(Message(role=role, content=content))

        return Conversation(
            conversation_id=self.session_id,
            source="memo_counseling",
            messages=messages,
            metadata={
                "has_summary": True,
                "summary": self.summary,
                "summary_type": self.summary_type.value,
                **self.metadata,
            },
        )


@dataclass
class SummarizationConfig:
    """Configuration for counseling summarization."""

    model_name: str = "facebook/bart-large-cnn"
    max_input_length: int = 1024
    max_summary_length: int = 256
    min_summary_length: int = 50
    num_beams: int = 4
    length_penalty: float = 2.0
    early_stopping: bool = True
    summary_type: SummaryType = SummaryType.ABSTRACTIVE
    include_clinical_notes: bool = True


class MEMODatasetLoader:
    """
    Loader for the MEMO Counseling Summarization Dataset.

    MEMO dataset structure (expected after access approval):
    - train.json: Training dialogues with summaries
    - val.json: Validation set
    - test.json: Test set

    Each entry contains:
    - dialogue: List of speaker/utterance pairs
    - summary: Expert-written summary
    - metadata: Session metadata
    """

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        quality_threshold: float = 0.90,
    ):
        """
        Initialize MEMO dataset loader.

        Args:
            dataset_path: Path to MEMO dataset directory
            quality_threshold: Quality threshold for loaded data
        """
        self.dataset_path = Path(dataset_path) if dataset_path else Path("ai/datasets/memo")
        self.quality_threshold = quality_threshold

        logger.info(f"MEMODatasetLoader initialized: path={self.dataset_path}")

    def check_access(self) -> dict[str, Any]:
        """
        Check if MEMO dataset is accessible.

        Returns:
            Dictionary with access status and available files
        """
        result = {
            "has_access": False,
            "dataset_path": str(self.dataset_path),
            "available_files": [],
            "missing_files": [],
            "access_instructions": self._get_access_instructions(),
        }

        expected_files = ["train.json", "val.json", "test.json"]

        if not self.dataset_path.exists():
            result["missing_files"] = expected_files
            logger.warning(f"MEMO dataset directory not found: {self.dataset_path}")
            return result

        for filename in expected_files:
            filepath = self.dataset_path / filename
            if filepath.exists():
                result["available_files"].append(filename)
            else:
                result["missing_files"].append(filename)

        result["has_access"] = len(result["available_files"]) > 0

        return result

    def _get_access_instructions(self) -> str:
        """Get instructions for requesting MEMO dataset access."""
        return """
MEMO Dataset Access Instructions
================================

The MEMO dataset requires academic agreement for access.

Steps to request access:
1. Visit the GitHub repository: https://github.com/LCS2-IIITD/MEMO
2. Read the README for dataset description
3. Fill out the Data Access Agreement form (linked in repo)
4. Email the completed form to the authors
5. Wait for approval (typically 1-2 weeks)

Once approved:
1. Download the dataset files
2. Place them in: ai/datasets/memo/
3. Expected files: train.json, val.json, test.json

For more information, see the KDD 2022 paper:
"MEMO: A Novel Counseling Summarization Dataset"
"""

    def load_dataset(
        self, split: str = "train"
    ) -> list[CounselingSession]:
        """
        Load MEMO dataset split.

        Args:
            split: Dataset split (train, val, test)

        Returns:
            List of CounselingSession objects
        """
        access_status = self.check_access()
        if not access_status["has_access"]:
            raise FileNotFoundError(
                f"MEMO dataset not found. {access_status['access_instructions']}"
            )

        filepath = self.dataset_path / f"{split}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"MEMO {split} split not found: {filepath}")

        sessions = []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both list and dict formats
            if isinstance(data, dict):
                items = data.get("sessions", data.get("data", [data]))
            else:
                items = data

            for idx, item in enumerate(items):
                session = self._parse_session(item, idx, split)
                if session:
                    sessions.append(session)

            logger.info(f"Loaded {len(sessions)} sessions from MEMO {split} split")

        except Exception as e:
            logger.error(f"Error loading MEMO dataset: {e}")
            raise

        return sessions

    def _parse_session(
        self, item: dict[str, Any], idx: int, split: str
    ) -> Optional[CounselingSession]:
        """Parse a single session from MEMO data."""
        try:
            # Extract dialogue
            dialogue = item.get("dialogue", item.get("conversation", []))
            if isinstance(dialogue, str):
                # Handle string format
                dialogue = self._parse_dialogue_string(dialogue)

            # Extract summary
            summary = item.get("summary", item.get("abstractive_summary", ""))

            if not dialogue or not summary:
                return None

            session_id = item.get("id", item.get("session_id", f"memo_{split}_{idx}"))

            return CounselingSession(
                session_id=session_id,
                dialogue=dialogue,
                summary=summary,
                summary_type=SummaryType.ABSTRACTIVE,
                metadata={
                    "split": split,
                    "original_index": idx,
                    "source": "memo_kdd2022",
                },
            )

        except Exception as e:
            logger.warning(f"Error parsing session {idx}: {e}")
            return None

    def _parse_dialogue_string(self, dialogue_str: str) -> list[dict[str, str]]:
        """Parse dialogue from string format."""
        dialogue = []
        lines = dialogue_str.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to extract speaker: utterance format
            if ":" in line:
                parts = line.split(":", 1)
                speaker = parts[0].strip().lower()
                utterance = parts[1].strip()
                dialogue.append({"speaker": speaker, "utterance": utterance})
            else:
                # Default to alternating speakers
                speaker = "client" if len(dialogue) % 2 == 0 else "counselor"
                dialogue.append({"speaker": speaker, "utterance": line})

        return dialogue

    def to_conversations(
        self, sessions: list[CounselingSession]
    ) -> list[Conversation]:
        """Convert sessions to standard Conversation format."""
        return [session.to_conversation() for session in sessions]


class CounselingSummarizer:
    """
    Counseling session summarization module.

    Generates summaries of therapeutic conversations using:
    - Extractive summarization (key utterances)
    - Abstractive summarization (generated summaries)
    - Clinical note generation
    """

    def __init__(self, config: Optional[SummarizationConfig] = None):
        """
        Initialize counseling summarizer.

        Args:
            config: Summarization configuration
        """
        self.config = config or SummarizationConfig()
        self.model = None
        self.tokenizer = None

        logger.info(f"CounselingSummarizer initialized: model={self.config.model_name}")

    def load_model(self) -> None:
        """Load the summarization model."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

            logger.info(f"Loaded summarization model: {self.config.model_name}")

        except ImportError:
            logger.warning(
                "transformers not installed. Install with: uv pip install transformers"
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def summarize_conversation(
        self, conversation: Conversation, summary_type: Optional[SummaryType] = None
    ) -> dict[str, Any]:
        """
        Generate summary for a conversation.

        Args:
            conversation: Conversation to summarize
            summary_type: Type of summary to generate

        Returns:
            Dictionary with summary and metadata
        """
        summary_type = summary_type or self.config.summary_type

        # Convert conversation to text
        dialogue_text = self._conversation_to_text(conversation)

        if summary_type == SummaryType.EXTRACTIVE:
            return self._extractive_summary(dialogue_text, conversation)
        elif summary_type == SummaryType.CLINICAL:
            return self._clinical_summary(conversation)
        else:
            return self._abstractive_summary(dialogue_text, conversation)

    def _conversation_to_text(self, conversation: Conversation) -> str:
        """Convert conversation to text format."""
        lines = []
        for msg in conversation.messages:
            role_name = "Client" if msg.role in ["user", "client"] else "Counselor"
            lines.append(f"{role_name}: {msg.content}")
        return "\n".join(lines)

    def _abstractive_summary(
        self, dialogue_text: str, conversation: Conversation
    ) -> dict[str, Any]:
        """Generate abstractive summary using model."""
        if self.model is None:
            # Fallback to rule-based summary
            return self._rule_based_summary(dialogue_text, conversation)

        try:
            inputs = self.tokenizer(
                dialogue_text,
                max_length=self.config.max_input_length,
                truncation=True,
                return_tensors="pt",
            )

            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=self.config.max_summary_length,
                min_length=self.config.min_summary_length,
                num_beams=self.config.num_beams,
                length_penalty=self.config.length_penalty,
                early_stopping=self.config.early_stopping,
            )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            return {
                "summary": summary,
                "summary_type": SummaryType.ABSTRACTIVE.value,
                "model": self.config.model_name,
                "conversation_id": conversation.conversation_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.warning(f"Model summarization failed: {e}, using fallback")
            return self._rule_based_summary(dialogue_text, conversation)

    def _extractive_summary(
        self, dialogue_text: str, conversation: Conversation
    ) -> dict[str, Any]:
        """Generate extractive summary by selecting key utterances."""
        key_utterances = []

        # Select important utterances based on heuristics
        for msg in conversation.messages:
            content = msg.content.lower()

            # Key indicators for important utterances
            if any(
                indicator in content
                for indicator in [
                    "i feel",
                    "i think",
                    "problem",
                    "struggle",
                    "help",
                    "difficult",
                    "important",
                    "realize",
                    "understand",
                    "goal",
                ]
            ):
                role = "Client" if msg.role in ["user", "client"] else "Counselor"
                key_utterances.append(f"{role}: {msg.content}")

        # Limit to top 5 utterances
        key_utterances = key_utterances[:5]

        return {
            "summary": "\n".join(key_utterances),
            "summary_type": SummaryType.EXTRACTIVE.value,
            "num_key_utterances": len(key_utterances),
            "conversation_id": conversation.conversation_id,
        }

    def _clinical_summary(self, conversation: Conversation) -> dict[str, Any]:
        """Generate clinical note style summary."""
        # Extract clinical elements
        client_concerns = []
        therapeutic_interventions = []
        session_progress = []

        for msg in conversation.messages:
            content = msg.content.lower()

            if msg.role in ["user", "client"]:
                # Extract concerns
                if any(word in content for word in ["anxious", "depressed", "worried", "scared"]):
                    client_concerns.append(msg.content[:100])
            else:
                # Extract interventions
                if any(
                    word in content
                    for word in ["explore", "consider", "technique", "strategy", "practice"]
                ):
                    therapeutic_interventions.append(msg.content[:100])

        clinical_note = f"""
SESSION SUMMARY
---------------
Date: {datetime.now().strftime("%Y-%m-%d")}
Session ID: {conversation.conversation_id}

PRESENTING CONCERNS:
{chr(10).join(f"- {c}..." for c in client_concerns[:3]) or "- Not explicitly stated"}

THERAPEUTIC INTERVENTIONS:
{chr(10).join(f"- {i}..." for i in therapeutic_interventions[:3]) or "- General supportive counseling"}

SESSION NOTES:
- Total exchanges: {len(conversation.messages)}
- Session duration: Estimated {len(conversation.messages) * 2} minutes

PLAN:
- Continue therapeutic relationship building
- Monitor progress on identified concerns
"""

        return {
            "summary": clinical_note.strip(),
            "summary_type": SummaryType.CLINICAL.value,
            "conversation_id": conversation.conversation_id,
            "clinical_elements": {
                "concerns_identified": len(client_concerns),
                "interventions_noted": len(therapeutic_interventions),
            },
        }

    def _rule_based_summary(
        self, dialogue_text: str, conversation: Conversation
    ) -> dict[str, Any]:
        """Fallback rule-based summary when model is unavailable."""
        num_turns = len(conversation.messages)
        client_turns = sum(1 for m in conversation.messages if m.role in ["user", "client"])

        # Extract key themes
        themes = []
        theme_keywords = {
            "anxiety": ["anxious", "anxiety", "worry", "nervous"],
            "depression": ["depressed", "sad", "hopeless", "down"],
            "relationships": ["relationship", "partner", "family", "friend"],
            "work": ["work", "job", "career", "boss"],
            "stress": ["stress", "overwhelmed", "pressure"],
        }

        combined_text = dialogue_text.lower()
        for theme, keywords in theme_keywords.items():
            if any(kw in combined_text for kw in keywords):
                themes.append(theme)

        summary = (
            f"Counseling session with {num_turns} exchanges ({client_turns} client, "
            f"{num_turns - client_turns} counselor). "
        )

        if themes:
            summary += f"Key themes discussed: {', '.join(themes)}. "

        summary += "The session involved supportive counseling and exploration of the client's concerns."

        return {
            "summary": summary,
            "summary_type": SummaryType.HYBRID.value,
            "method": "rule_based",
            "themes": themes,
            "conversation_id": conversation.conversation_id,
        }

    def batch_summarize(
        self, conversations: list[Conversation]
    ) -> list[dict[str, Any]]:
        """
        Generate summaries for multiple conversations.

        Args:
            conversations: List of conversations to summarize

        Returns:
            List of summary dictionaries
        """
        summaries = []
        for conv in conversations:
            summary = self.summarize_conversation(conv)
            summaries.append(summary)

        logger.info(f"Generated {len(summaries)} summaries")
        return summaries


# Access request documentation
MEMO_ACCESS_REQUEST_TEMPLATE = """
Subject: MEMO Dataset Access Request - [Your Institution]

Dear MEMO Dataset Authors,

I am writing to request access to the MEMO Counseling Summarization Dataset
for use in [describe your project/research].

Researcher Information:
- Name: [Your Name]
- Institution: [Your Institution]
- Position: [Your Position]
- Email: [Your Email]

Intended Use:
[Describe how you plan to use the dataset]

I have read and agree to the terms of the Data Access Agreement.
[Attach completed agreement form]

Thank you for your consideration.

Best regards,
[Your Name]
"""


def get_access_request_template() -> str:
    """Get template for MEMO dataset access request."""
    return MEMO_ACCESS_REQUEST_TEMPLATE


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("MEMO Counseling Dataset Integration")
    print("=" * 50)

    # Check access
    loader = MEMODatasetLoader()
    access = loader.check_access()

    print(f"\nDataset Access Status:")
    print(f"  Has Access: {access['has_access']}")
    print(f"  Path: {access['dataset_path']}")
    print(f"  Available Files: {access['available_files']}")
    print(f"  Missing Files: {access['missing_files']}")

    if not access["has_access"]:
        print("\n" + access["access_instructions"])
        print("\nAccess Request Template:")
        print(get_access_request_template())

    # Demo summarizer
    print("\n" + "=" * 50)
    print("Counseling Summarizer Demo")
    print("=" * 50)

    summarizer = CounselingSummarizer()

    # Create test conversation
    test_conv = Conversation(
        conversation_id="demo_001",
        source="demo",
        messages=[
            Message(role="user", content="I've been feeling really anxious about work lately."),
            Message(
                role="assistant",
                content="I hear that work has been causing you significant anxiety. "
                "Can you tell me more about what specifically at work is triggering these feelings?",
            ),
            Message(
                role="user",
                content="My boss keeps giving me more projects and I'm worried I can't handle it all.",
            ),
            Message(
                role="assistant",
                content="It sounds like you're feeling overwhelmed by the increasing workload. "
                "That's a very understandable concern. Let's explore some strategies for managing this.",
            ),
        ],
    )

    print("\nTest Conversation:")
    for msg in test_conv.messages:
        print(f"  [{msg.role}]: {msg.content[:80]}...")

    print("\nGenerated Summaries:")

    # Rule-based summary (no model needed)
    summary = summarizer.summarize_conversation(test_conv)
    print(f"\n  Abstractive (rule-based):")
    print(f"    {summary['summary']}")

    # Clinical summary
    clinical = summarizer._clinical_summary(test_conv)
    print(f"\n  Clinical Note:")
    print(clinical["summary"][:500] + "...")

