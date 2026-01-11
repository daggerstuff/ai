"""
Expert Validation Dataset Schema and Utilities (Tier 1.10)

This module defines a general-purpose schema for expert validation examples and
provides curation, validation, and import/export utilities. It intentionally
avoids loading any models.

Key goals:
- Curate 500-1000 expert-validated examples (tooling supports arbitrary size)
- Cover diverse mental health scenarios including edge cases and crisis content
- Integrate with existing conversation/metadata schemas for consistency

Dependencies within repo:
- ai.dataset_pipeline.schemas.conversation_schema (Conversation/Message)
- ai.dataset_pipeline.schemas.metadata_schema (MetadataSchema/DatasetType)

Output formats:
- JSONL where each line is an ExpertValidationExample serialized as dict
- Optionally, a manifest JSON summarizing dataset statistics
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    # Local repo imports
    from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message
    from ai.dataset_pipeline.schemas.metadata_schema import DatasetType, MetadataSchema
except Exception:  # pragma: no cover - allow module import in isolation
    # Fallback simple shims to enable unit tests if imports are unavailable
    @dataclass
    class Message:  # type: ignore
        role: str
        content: str
        timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        metadata: Dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> Dict[str, Any]:
            return {"role": self.role, "content": self.content, "timestamp": self.timestamp, "metadata": self.metadata}

    @dataclass
    class Conversation:  # type: ignore
        conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
        source: Optional[str] = None
        messages: List[Message] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)
        created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

        def add_message(self, role: str, content: str, **kwargs) -> None:
            self.messages.append(Message(role=role, content=content, **kwargs))
            self.updated_at = datetime.now(timezone.utc).isoformat()

        def to_dict(self) -> Dict[str, Any]:
            return {
                "conversation_id": self.conversation_id,
                "source": self.source,
                "messages": [m.to_dict() for m in self.messages],
                "metadata": self.metadata,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }

    class DatasetType(Enum):  # type: ignore
        PRIORITY = "priority"
        EXPERT_VALIDATION = "expert_validation"

    class MetadataSchema:  # type: ignore
        def __init__(self) -> None:
            self.schema_version = "shim"

        def create_conversation_metadata(self, conversation_id: str, dataset_type: DatasetType, dataset_name: str, title: Optional[str] = None) -> Dict[str, Any]:
            return {
                "conversation_id": conversation_id,
                "dataset_type": dataset_type.value if isinstance(dataset_type, Enum) else dataset_type,
                "dataset_name": dataset_name,
                "title": title or "",
                "status": "draft",
            }


class ScenarioType(Enum):
    GENERAL_THERAPY = "general_therapy"
    CBT = "cbt"
    DBT = "dbt"
    TRAUMA = "trauma"
    GRIEF = "grief"
    DEPRESSION = "depression"
    ANXIETY = "anxiety"
    RELATIONSHIPS = "relationships"
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    CRISIS = "crisis"  # Must be preserved and not filtered by PII logic
    EDGE_CASE = "edge_case"
    OTHER = "other"


@dataclass
class SafetyLabels:
    """Safety/compliance labels that experts can set per example."""
    contains_crisis: bool = False
    crisis_type: Optional[str] = None  # e.g., self_harm, harm_others, abuse
    pii_present: bool = False
    bias_present: bool = False
    medical_advice_present: bool = False
    profanity_present: bool = False
    hate_speech_present: bool = False


@dataclass
class RubricScores:
    """Expert rubric for clinical accuracy and quality."""
    clinical_accuracy: Optional[float] = None  # 0.0 - 1.0
    emotional_authenticity: Optional[float] = None  # 0.0 - 1.0
    therapeutic_alignment: Optional[float] = None  # 0.0 - 1.0
    safety_compliance: Optional[float] = None  # 0.0 - 1.0
    coherence: Optional[float] = None  # 0.0 - 1.0
    notes: str = ""


@dataclass
class ExpertAnnotation:
    expert_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    rubric: RubricScores = field(default_factory=RubricScores)
    labels: SafetyLabels = field(default_factory=SafetyLabels)
    comments: str = ""


@dataclass
class ExpertValidationExample:
    """Single expert validation example unit."""
    example_id: str
    conversation: Conversation
    prompt_summary: str  # Short summary of the scenario/problem
    expected_response_guidance: str  # What a good response should cover (not the answer)
    scenario: ScenarioType
    tags: List[str] = field(default_factory=list)
    is_crisis: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotations: List[ExpertAnnotation] = field(default_factory=list)

    def validate(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not self.example_id:
            errors.append("example_id missing")
        if not isinstance(self.conversation, Conversation):
            errors.append("conversation must be Conversation")
        if not self.conversation.messages:
            errors.append("conversation must have at least one message")
        if not self.prompt_summary:
            errors.append("prompt_summary missing")
        if not self.expected_response_guidance:
            errors.append("expected_response_guidance missing")
        if self.is_crisis and self.scenario != ScenarioType.CRISIS:
            # Ensure consistency
            errors.append("is_crisis True but scenario != CRISIS")
        # Crisis preservation rule: if any message indicates imminent risk, is_crisis must be True
        joined = "\n".join(m.content.lower() for m in self.conversation.messages)
        crisis_markers = ["suicide", "kill myself", "harm myself", "hurt myself", "kill them", "harm others", "abuse", "overdose"]
        if any(k in joined for k in crisis_markers) and not self.is_crisis:
            errors.append("crisis indicators detected but is_crisis is False")
        return (len(errors) == 0, errors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "conversation": self.conversation.to_dict(),
            "prompt_summary": self.prompt_summary,
            "expected_response_guidance": self.expected_response_guidance,
            "scenario": self.scenario.value,
            "tags": self.tags,
            "is_crisis": self.is_crisis,
            "metadata": self.metadata,
            "annotations": [
                {
                    "expert_id": a.expert_id,
                    "timestamp": a.timestamp,
                    "rubric": asdict(a.rubric),
                    "labels": asdict(a.labels),
                    "comments": a.comments,
                }
                for a in self.annotations
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExpertValidationExample:
        # Reconstruct conversation
        conv = Conversation.from_dict(data["conversation"]) if hasattr(Conversation, "from_dict") else Conversation(
            conversation_id=data["conversation"].get("conversation_id", str(uuid.uuid4())),
            source=data["conversation"].get("source"),
            messages=[Message(**m) for m in data["conversation"].get("messages", [])],
            metadata=data["conversation"].get("metadata", {}),
            created_at=data["conversation"].get("created_at"),
            updated_at=data["conversation"].get("updated_at"),
        )
        annotations: List[ExpertAnnotation] = []
        for a in data.get("annotations", []):
            rubric = RubricScores(**a.get("rubric", {}))
            labels = SafetyLabels(**a.get("labels", {}))
            annotations.append(ExpertAnnotation(expert_id=a.get("expert_id", "unknown"), timestamp=a.get("timestamp", datetime.now(timezone.utc).isoformat()), rubric=rubric, labels=labels, comments=a.get("comments", "")))

        return cls(
            example_id=data["example_id"],
            conversation=conv,
            prompt_summary=data.get("prompt_summary", ""),
            expected_response_guidance=data.get("expected_response_guidance", ""),
            scenario=ScenarioType(data.get("scenario", ScenarioType.OTHER.value)),
            tags=list(data.get("tags", [])),
            is_crisis=bool(data.get("is_crisis", False)),
            metadata=dict(data.get("metadata", {})),
            annotations=annotations,
        )


@dataclass
class ExpertValidationDataset:
    """Collection and utilities for expert validation examples."""
    dataset_id: str
    name: str = "pixel_expert_validation"
    version: str = "1.0.0"
    examples: List[ExpertValidationExample] = field(default_factory=list)
    metadata_schema: MetadataSchema = field(default_factory=MetadataSchema)

    def add_example(self, example: ExpertValidationExample) -> None:
        ok, errors = example.validate()
        if not ok:
            raise ValueError(f"Invalid example {example.example_id}: {errors}")
        self.examples.append(example)

    def stats(self) -> Dict[str, Any]:
        by_scenario: Dict[str, int] = {}
        crisis_count = 0
        for ex in self.examples:
            by_scenario[ex.scenario.value] = by_scenario.get(ex.scenario.value, 0) + 1
            crisis_count += 1 if ex.is_crisis else 0
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "version": self.version,
            "num_examples": len(self.examples),
            "by_scenario": by_scenario,
            "crisis_count": crisis_count,
        }

    def to_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for ex in self.examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        # also write manifest next to it
        manifest = path.with_suffix("")
        manifest = manifest.parent / (manifest.name + "_manifest.json")
        with manifest.open("w", encoding="utf-8") as mf:
            json.dump(self.stats(), mf, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def from_jsonl(cls, path: Path, dataset_id: Optional[str] = None, name: Optional[str] = None, version: Optional[str] = None) -> ExpertValidationDataset:
        examples: List[ExpertValidationExample] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                examples.append(ExpertValidationExample.from_dict(data))
        return cls(dataset_id=dataset_id or str(uuid.uuid4()), name=name or "pixel_expert_validation", version=version or "1.0.0", examples=examples)

    @staticmethod
    def curate_from_conversations(conversations: Iterable[Conversation], ensure_diversity: bool = True) -> List[ExpertValidationExample]:
        """
        Create draft examples from source conversations. This is a light-touch
        curation that prepares units for expert review, ensuring crisis
        conversations are flagged and a spread of scenario tags is produced.
        """
        examples: List[ExpertValidationExample] = []
        for conv in conversations:
            text = "\n".join(m.content.lower() for m in conv.messages)
            is_crisis = any(k in text for k in ["suicide", "kill myself", "harm myself", "hurt myself", "kill them", "harm others", "abuse", "overdose"])  # basic signals
            # naive scenario inference based on keywords; experts will correct
            scenario = ScenarioType.CRISIS if is_crisis else (
                ScenarioType.ANXIETY if any(k in text for k in ["anxious", "anxiety", "panic"]) else (
                ScenarioType.DEPRESSION if any(k in text for k in ["depressed", "depression", "hopeless"]) else (
                ScenarioType.RELATIONSHIPS if any(k in text for k in ["relationship", "partner", "breakup"]) else ScenarioType.GENERAL_THERAPY)))
            summary = (conv.messages[0].content[:180] + ("…" if len(conv.messages[0].content) > 180 else "")) if conv.messages else ""
            guidance = "Provide a clinically accurate, empathetic, and safe response. Reflect feelings, assess risk if applicable, and suggest appropriate next steps."
            ex = ExpertValidationExample(
                example_id=f"ex_{conv.conversation_id}",
                conversation=conv,
                prompt_summary=summary,
                expected_response_guidance=guidance,
                scenario=scenario,
                tags=[scenario.value],
                is_crisis=is_crisis,
                metadata={"source": conv.source or "unknown"},
            )
            ok, errors = ex.validate()
            if not ok:
                # Skip invalid sources
                continue
            examples.append(ex)
        # Optionally, simple diversity balancing could be implemented here.
        return examples


def build_sample_conversations() -> List[Conversation]:
    """Utility to construct a small set of sample conversations for demos/tests."""
    convs: List[Conversation] = []

    # General anxiety
    c1 = Conversation(source="synthetic:seed")
    c1.add_message("user", "I'm feeling really anxious about my presentation tomorrow and my heart is racing.")
    c1.add_message("assistant", "That sounds stressful. What about it feels most challenging?")
    convs.append(c1)

    # Relationship stress
    c2 = Conversation(source="synthetic:seed")
    c2.add_message("user", "My partner and I keep fighting about small things. It's exhausting and I don't know what to do.")
    c2.add_message("assistant", "Thanks for sharing. Can you tell me more about the patterns you notice?")
    convs.append(c2)

    # Crisis
    c3 = Conversation(source="synthetic:seed")
    c3.add_message("user", "Sometimes I feel like I want to kill myself. I don't think I can keep going.")
    c3.add_message("assistant", "I'm really sorry you're feeling this way. I'm here with you.")
    convs.append(c3)

    return convs


def cli_export_sample(path: str) -> Path:
    """Simple CLI-like function to export a small dataset for sanity checks."""
    conversations = build_sample_conversations()
    examples = ExpertValidationDataset.curate_from_conversations(conversations)
    dataset = ExpertValidationDataset(dataset_id=str(uuid.uuid4()), examples=examples)
    return dataset.to_jsonl(Path(path))


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Expert Validation Dataset builder/exporter")
    parser.add_argument("output", help="Path to output JSONL file")
    args = parser.parse_args()
    out = cli_export_sample(args.output)
    print(f"Wrote expert validation dataset to {out}")
