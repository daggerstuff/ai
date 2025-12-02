#!/usr/bin/env python3
"""
Edge Dataset Builder for Mental Health Training Datasets
Builds structured edge datasets from raw/synthetic inputs with EdgeProfile metadata.
Aligned with nightmare categories and intensity levels from the expanded project brief.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone

from ..types.edge_categories import (
    EdgeCategory,
    IntensityLevel,
    EdgeProfile,
    validate_edge_profile,
    get_categories_by_intensity,
)
from ..utils.logger import get_logger

logger = get_logger("dataset_pipeline.edge.edge_builder")


@dataclass
class RawEdgeExample:
    """Raw input data for edge example construction"""
    conversation: Union[str, List[Dict[str, str]]]  # Raw conversation text or message list
    category: Union[str, EdgeCategory]  # Category name or enum
    intensity: Union[str, IntensityLevel]  # Intensity level name or enum
    tone: str = "CRISIS_DIRECT"  # Default tone for edge cases
    stage: int = 3  # Default to Stage 3 (edge stress test)
    scenario_type: Optional[str] = None
    challenge_type: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeExample:
    """Complete edge example with EdgeProfile metadata"""
    example_id: str
    conversation: List[Dict[str, str]]  # Normalized conversation format
    edge_profile: EdgeProfile
    raw_source: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge example to dictionary"""
        return {
            "example_id": self.example_id,
            "conversation": self.conversation,
            "edge_profile": self.edge_profile.to_dict(),
            "raw_source": self.raw_source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeExample":
        """Deserialize edge example from dictionary"""
        return cls(
            example_id=data["example_id"],
            conversation=data["conversation"],
            edge_profile=EdgeProfile.from_dict(data["edge_profile"]),
            raw_source=data.get("raw_source"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


class EdgeDatasetBuilder:
    """
    Builder for edge datasets with EdgeProfile metadata.
    Takes raw/synthetic conversation material and attaches structured edge profiles.
    """

    def __init__(self):
        """Initialize the edge dataset builder"""
        self._category_mapping = self._build_category_mapping()
        self._intensity_mapping = self._build_intensity_mapping()

    def _build_enum_mapping(self, enum_class) -> Dict[str, Any]:
        """Build mapping from string names to enum values"""
        mapping = {}
        for enum_value in enum_class:
            # Map both enum name and value
            mapping[enum_value.name.lower()] = enum_value
            mapping[enum_value.value.lower()] = enum_value
            mapping[enum_value.value] = enum_value
        return mapping

    def _build_category_mapping(self) -> Dict[str, EdgeCategory]:
        """Build mapping from string category names to EdgeCategory enum"""
        return self._build_enum_mapping(EdgeCategory)

    def _build_intensity_mapping(self) -> Dict[str, IntensityLevel]:
        """Build mapping from string intensity names to IntensityLevel enum"""
        return self._build_enum_mapping(IntensityLevel)

    def _normalize_category(self, category: Union[str, EdgeCategory]) -> EdgeCategory:
        """
        Normalize category input to EdgeCategory enum.

        Uses strict exact matching only - no fuzzy or substring matching.
        This ensures reliable dataset classification and fails fast for unknown categories.
        For example, "crisis" will fail rather than ambiguously matching multiple
        categories like "borderline_personality_crisis", "addiction_crisis", or "ptsd_crisis".
        """
        if isinstance(category, EdgeCategory):
            return category

        category_str = str(category).lower()
        if category_str in self._category_mapping:
            return self._category_mapping[category_str]

        raise ValueError(f"Unknown edge category: {category}")

    def _normalize_intensity(self, intensity: Union[str, IntensityLevel]) -> IntensityLevel:
        """Normalize intensity input to IntensityLevel enum"""
        if isinstance(intensity, IntensityLevel):
            return intensity

        intensity_str = str(intensity).lower()
        if intensity_str in self._intensity_mapping:
            return self._intensity_mapping[intensity_str]

        # Map common variations
        intensity_map = {
            "very high": IntensityLevel.VERY_HIGH,
            "very_high": IntensityLevel.VERY_HIGH,
            "veryhigh": IntensityLevel.VERY_HIGH,
            "extreme": IntensityLevel.EXTREME,
            "high": IntensityLevel.HIGH,
            "moderate": IntensityLevel.MODERATE,
            "medium": IntensityLevel.MODERATE,
            "low": IntensityLevel.LOW,
        }

        if intensity_str in intensity_map:
            return intensity_map[intensity_str]

        raise ValueError(f"Unknown intensity level: {intensity}")

    def _normalize_conversation(self, conversation: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """Normalize conversation input to standard message list format"""
        if isinstance(conversation, list):
            # Validate list format
            normalized = []
            for msg in conversation:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    normalized.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                else:
                    logger.warning(f"Skipping invalid message format: {msg}")
            # Ensure at least one message is returned (consistent with string parsing)
            if not normalized:
                raise ValueError("Conversation list is empty or contains no valid messages")
            return normalized

        # If string, try to parse or create simple format
        if isinstance(conversation, str):
            return self._parse_string_conversation(conversation)

        raise ValueError(f"Invalid conversation format: {type(conversation)}")

    def _parse_string_conversation(self, conversation: str) -> List[Dict[str, str]]:
        """Parse string conversation into message list format"""
        # Check for role markers (case-insensitive)
        has_role_markers = (
            "Therapist:" in conversation or "Client:" in conversation or
            "therapist:" in conversation.lower() or "client:" in conversation.lower()
        )
        
        if not has_role_markers:
            # Log info when treating multi-line string as single message
            if "\n" in conversation and len(conversation.split("\n")) > 3:
                logger.info(
                    "Multi-line conversation treated as single user message "
                    "(no role markers detected)"
                )
            return [{"role": "user", "content": conversation}]

        # Parse multi-turn conversation with role markers
        messages = []
        lines = conversation.split("\n")
        current_role = None
        current_content = []

        def save_message():
            """Helper to save current message"""
            if current_role and current_content:
                messages.append({
                    "role": current_role.lower(),
                    "content": "\n".join(current_content).strip()
                })

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Case-insensitive role detection
            line_lower = line.lower()
            if line_lower.startswith("therapist:") or line_lower.startswith("client:"):
                save_message()
                # Start new message - use startswith to match the conditional check
                current_role = "therapist" if line_lower.startswith("therapist:") else "client"
                parts = line.split(":", 1)
                current_content = [parts[1].strip()] if len(parts) > 1 else []
            elif current_role:
                # Continue current message
                current_content.append(line)

        # Save last message
        save_message()

        return messages if messages else [{"role": "user", "content": conversation}]

    def build_edge_example(
        self,
        raw_example: RawEdgeExample,
        profile_id: Optional[str] = None
    ) -> EdgeExample:
        """
        Build a single edge example with EdgeProfile metadata.

        Args:
            raw_example: Raw input data for the edge example
            profile_id: Optional profile ID (generated if not provided)

        Returns:
            EdgeExample with attached EdgeProfile
        """
        # Normalize inputs
        category = self._normalize_category(raw_example.category)
        intensity = self._normalize_intensity(raw_example.intensity)
        conversation = self._normalize_conversation(raw_example.conversation)

        # Generate profile ID if not provided
        if not profile_id:
            profile_id = f"edge-{category.value}-{intensity.value}-{uuid.uuid4().hex[:8]}"

        # Create EdgeProfile
        edge_profile = EdgeProfile(
            profile_id=profile_id,
            category=category,
            intensity=intensity,
            tone=raw_example.tone,
            stage=raw_example.stage,
            scenario_type=raw_example.scenario_type,
            challenge_type=raw_example.challenge_type,
            requires_supervision=True,  # Edge cases always require supervision
            crisis_language_preserved=True,  # Preserve crisis language for edge cases
            metadata={
                "source": raw_example.source or "edge_builder",
                **raw_example.metadata
            }
        )

        # Validate profile
        is_valid, error = validate_edge_profile(edge_profile)
        if not is_valid:
            raise ValueError(f"Invalid edge profile: {error}")

        # Create EdgeExample
        return EdgeExample(
            example_id=f"ex-{uuid.uuid4().hex[:12]}",
            conversation=conversation,
            edge_profile=edge_profile,
            raw_source=raw_example.source,
        )

    def build_edge_dataset(
        self,
        raw_examples: List[RawEdgeExample],
        output_path: Optional[Union[str, Path]] = None
    ) -> List[EdgeExample]:
        """
        Build a complete edge dataset from multiple raw examples.

        Args:
            raw_examples: List of raw input examples
            output_path: Optional path to save the dataset as JSONL

        Returns:
            List of EdgeExample objects with EdgeProfile metadata
        """
        logger.info(f"Building edge dataset from {len(raw_examples)} raw examples")

        edge_examples = []
        errors = []

        for idx, raw_ex in enumerate(raw_examples):
            try:
                example = self.build_edge_example(raw_ex)
                edge_examples.append(example)
            except Exception as e:
                error_msg = f"Error building example {idx}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue

        logger.info(f"Built {len(edge_examples)} edge examples ({len(errors)} errors)")

        if errors:
            logger.warning(f"Encountered {len(errors)} errors during building")

        # Save to file if path provided
        if output_path:
            self._save_dataset(edge_examples, output_path)

        return edge_examples

    def _save_dataset(self, examples: List[EdgeExample], output_path: Union[str, Path]):
        """Save edge dataset to JSONL file"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                for example in examples:
                    try:
                        f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                    except (TypeError, ValueError) as e:
                        logger.error(f"Failed to serialize example {example.example_id}: {e}")
                        continue
        except IOError as e:
            logger.error(f"Failed to write dataset to {path}: {e}")
            raise

        logger.info(f"Saved {len(examples)} edge examples to {path}")

    def tag_existing_example(
        self,
        conversation: Union[str, List[Dict[str, str]]],
        category: Union[str, EdgeCategory],
        intensity: Union[str, IntensityLevel],
        tone: str = "CRISIS_DIRECT",
        stage: int = 3,
        **kwargs
    ) -> EdgeExample:
        """
        Tag an existing conversation with edge profile metadata.
        Convenience method for quick tagging.
        """
        raw_example = RawEdgeExample(
            conversation=conversation,
            category=category,
            intensity=intensity,
            tone=tone,
            stage=stage,
            **kwargs
        )
        return self.build_edge_example(raw_example)

    def get_statistics(self, examples: List[EdgeExample]) -> Dict[str, Any]:
        """Get statistics about a built edge dataset"""
        if not examples:
            return {
                "total_examples": 0,
                "by_category": {},
                "by_intensity": {},
                "by_stage": {},
            }

        by_category = {}
        by_intensity = {}
        by_stage = {}

        for ex in examples:
            cat = ex.edge_profile.category.value
            intensity = ex.edge_profile.intensity.value
            stage = ex.edge_profile.stage

            by_category[cat] = by_category.get(cat, 0) + 1
            by_intensity[intensity] = by_intensity.get(intensity, 0) + 1
            by_stage[stage] = by_stage.get(stage, 0) + 1

        return {
            "total_examples": len(examples),
            "by_category": by_category,
            "by_intensity": by_intensity,
            "by_stage": by_stage,
        }

