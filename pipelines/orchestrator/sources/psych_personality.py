#!/usr/bin/env python3
"""
Psychology Books, Sarcasm Detection, and Personality Adaptation Integration
Unified interface for generating personality-aware, sarcasm-aware training examples.
Based on the expanded project brief's requirements for psychology books, sarcasm detection,
Big Five personality adaptation, and therapeutic approach customization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from pathlib import Path
import json
import re

from ..utils.logger import get_logger

logger = get_logger("dataset_pipeline.sources.psych_personality")


class TherapeuticApproach(Enum):
    """Therapeutic approaches for customization"""
    CBT = "cbt"  # Cognitive Behavioral Therapy
    DBT = "dbt"  # Dialectical Behavior Therapy
    HUMANISTIC = "humanistic"
    PSYCHODYNAMIC = "psychodynamic"
    ACT = "act"  # Acceptance and Commitment Therapy
    EMDR = "emdr"  # Eye Movement Desensitization and Reprocessing
    MINDFULNESS = "mindfulness"


class CommunicationStyle(Enum):
    """Communication styles for adaptation"""
    DIRECT = "direct"
    SUPPORTIVE = "supportive"
    ANALYTICAL = "analytical"
    EMPATHETIC = "empathetic"


class BigFiveTrait(Enum):
    """Big Five personality traits"""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


@dataclass
class PersonalityProfile:
    """Big Five personality profile"""
    openness: float = 0.5  # 0.0 to 1.0
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    def get_dominant_trait(self) -> BigFiveTrait:
        """Get the highest-scoring trait"""
        scores = {
            BigFiveTrait.OPENNESS: self.openness,
            BigFiveTrait.CONSCIENTIOUSNESS: self.conscientiousness,
            BigFiveTrait.EXTRAVERSION: self.extraversion,
            BigFiveTrait.AGREEABLENESS: self.agreeableness,
            BigFiveTrait.NEUROTICISM: self.neuroticism,
        }
        return max(scores.items(), key=lambda x: x[1])[0]

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary"""
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }


@dataclass
class SarcasmDetection:
    """Sarcasm detection result"""
    is_sarcastic: bool
    confidence: float  # 0.0 to 1.0
    indicators: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None


class PersonalityAdapter:
    """
    Adapts therapeutic approach and communication style based on personality traits.
    Based on Big Five personality model and therapeutic approach customization.
    """

    # Mapping from personality traits to preferred therapeutic approaches
    TRAIT_TO_APPROACH = {
        BigFiveTrait.OPENNESS: [TherapeuticApproach.ACT, TherapeuticApproach.HUMANISTIC],
        BigFiveTrait.CONSCIENTIOUSNESS: [TherapeuticApproach.CBT, TherapeuticApproach.DBT],
        BigFiveTrait.EXTRAVERSION: [TherapeuticApproach.HUMANISTIC, TherapeuticApproach.CBT],
        BigFiveTrait.AGREEABLENESS: [TherapeuticApproach.HUMANISTIC, TherapeuticApproach.EMDR],
        BigFiveTrait.NEUROTICISM: [TherapeuticApproach.DBT, TherapeuticApproach.MINDFULNESS],
    }

    # Mapping from personality traits to communication styles
    TRAIT_TO_STYLE = {
        BigFiveTrait.OPENNESS: CommunicationStyle.ANALYTICAL,
        BigFiveTrait.CONSCIENTIOUSNESS: CommunicationStyle.DIRECT,
        BigFiveTrait.EXTRAVERSION: CommunicationStyle.SUPPORTIVE,
        BigFiveTrait.AGREEABLENESS: CommunicationStyle.EMPATHETIC,
        BigFiveTrait.NEUROTICISM: CommunicationStyle.SUPPORTIVE,
    }

    def select_therapeutic_approach(self, profile: PersonalityProfile) -> TherapeuticApproach:
        """
        Select therapeutic approach based on personality profile.

        Args:
            profile: Big Five personality profile

        Returns:
            Recommended therapeutic approach
        """
        dominant_trait = profile.get_dominant_trait()
        approaches = self.TRAIT_TO_APPROACH.get(dominant_trait, [TherapeuticApproach.CBT])
        return approaches[0]  # Return first recommended approach

    def select_communication_style(self, profile: PersonalityProfile) -> CommunicationStyle:
        """
        Select communication style based on personality profile.

        Args:
            profile: Big Five personality profile

        Returns:
            Recommended communication style
        """
        dominant_trait = profile.get_dominant_trait()
        return self.TRAIT_TO_STYLE.get(dominant_trait, CommunicationStyle.EMPATHETIC)


class SarcasmDetector:
    """
    Detects sarcasm in mental health contexts.
    Advanced personality analysis with sarcasm recognition capabilities.
    """

    # Sarcasm indicators (patterns that suggest sarcasm)
    SARCASM_PATTERNS = [
        r"oh.*great",  # "oh great"
        r"that.*s.*wonderful",  # "that's wonderful" (sarcastic)
        r"yeah.*right",  # "yeah right"
        r"real.*helpful",  # "real helpful"
        r"thanks.*a.*lot",  # "thanks a lot" (sarcastic)
        r"just.*what.*i.*needed",  # "just what I needed"
        r"perfect.*timing",  # "perfect timing" (sarcastic)
        r"exactly.*what.*i.*wanted",  # "exactly what I wanted"
    ]

    # Context clues that increase sarcasm likelihood
    CONTEXT_CLUES = [
        r"obviously",
        r"clearly",
        r"of.*course",
        r"naturally",
    ]

    def __init__(self):
        """Initialize sarcasm detector"""
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self._sarcasm_re = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.SARCASM_PATTERNS
        ]
        self._context_re = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.CONTEXT_CLUES
        ]

    def detect_sarcasm(self, text: str, context: Optional[Dict[str, Any]] = None) -> SarcasmDetection:
        """
        Detect sarcasm in text.

        Args:
            text: Text to analyze
            context: Optional context (e.g., conversation history, emotional state)

        Returns:
            SarcasmDetection result
        """
        text_lower = text.lower()

        # Count sarcasm pattern matches
        sarcasm_matches = []
        for pattern in self._sarcasm_re:
            if pattern.search(text_lower):
                sarcasm_matches.append(pattern.pattern)

        # Count context clues
        context_matches = []
        for pattern in self._context_re:
            if pattern.search(text_lower):
                context_matches.append(pattern.pattern)

        # Calculate confidence
        base_score = len(sarcasm_matches) * 0.3
        context_score = len(context_matches) * 0.1

        # Check for explicit sarcasm markers
        explicit_markers = ["/s", "[sarcasm]", "sarcasm", "sarcastic"]
        has_explicit = any(marker in text_lower for marker in explicit_markers)

        if has_explicit:
            confidence = 0.95
        else:
            confidence = min(0.9, base_score + context_score)

        is_sarcastic = confidence > 0.4 or has_explicit

        reasoning = None
        if is_sarcastic:
            if has_explicit:
                reasoning = "Explicit sarcasm marker detected"
            elif sarcasm_matches:
                reasoning = f"Sarcasm patterns detected: {', '.join(sarcasm_matches[:3])}"
            else:
                reasoning = "Context clues suggest sarcasm"

        return SarcasmDetection(
            is_sarcastic=is_sarcastic,
            confidence=confidence,
            indicators=sarcasm_matches + context_matches,
            reasoning=reasoning,
        )


class PsychologyBookLoader:
    """
    Loader for psychology textbook corpus (xmu_psych_books).
    Integrates with existing dataset loading infrastructure.
    """

    def __init__(self, dataset_path: Optional[Union[str, Path]] = None):
        """
        Initialize psychology book loader.

        Args:
            dataset_path: Path to xmu_psych_books dataset (defaults to common locations)
        """
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            # Try common locations
            possible_paths = [
                Path("ai/datasets/external/xmu_psych_books"),
                Path("ai/datasets/xmu_psych_books"),
                Path("ai/data/xmu_psych_books"),
            ]
            self.dataset_path = None
            for path in possible_paths:
                if path.exists():
                    self.dataset_path = path
                    break

        if not self.dataset_path or not self.dataset_path.exists():
            logger.warning(f"Psychology books dataset not found at {self.dataset_path}")

    def load_psychology_books(self) -> List[Dict[str, Any]]:
        """
        Load psychology textbook corpus.

        Returns:
            List of psychology book entries
        """
        if not self.dataset_path or not self.dataset_path.exists():
            logger.warning("Psychology books dataset path not available")
            return []

        entries = []

        # Try to load from HuggingFace format or JSON
        try:
            # Check for common HuggingFace dataset formats
            if (self.dataset_path / "dataset_info.json").exists():
                # HuggingFace dataset format
                logger.info("Loading from HuggingFace dataset format")
                # Would need datasets library for full HF support
                # For now, try JSON/JSONL files
                pass

            # Try JSONL format
            jsonl_files = list(self.dataset_path.glob("*.jsonl"))
            if jsonl_files:
                for jsonl_file in jsonl_files:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                entries.append(entry)
                            except json.JSONDecodeError:
                                continue

            # Try JSON format
            json_files = list(self.dataset_path.glob("*.json"))
            if json_files and not entries:
                for json_file in json_files:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            entries.extend(data)
                        elif isinstance(data, dict):
                            entries.append(data)

            logger.info(f"Loaded {len(entries)} psychology book entries")
            return entries

        except Exception as e:
            logger.error(f"Error loading psychology books: {e}")
            return []


class PsychPersonalityIntegrator:
    """
    Unified integrator for psychology books, sarcasm detection, and personality adaptation.
    Generates personality-aware, sarcasm-aware training examples.
    """

    def __init__(
        self,
        psychology_books_path: Optional[Union[str, Path]] = None,
        enable_sarcasm_detection: bool = True,
        enable_personality_adaptation: bool = True,
    ):
        """
        Initialize the integrator.

        Args:
            psychology_books_path: Path to psychology books dataset
            enable_sarcasm_detection: Enable sarcasm detection
            enable_personality_adaptation: Enable personality-based adaptation
        """
        self.psych_loader = PsychologyBookLoader(psychology_books_path)
        self.sarcasm_detector = SarcasmDetector() if enable_sarcasm_detection else None
        self.personality_adapter = PersonalityAdapter() if enable_personality_adaptation else None

    def generate_personality_aware_example(
        self,
        conversation: Union[str, List[Dict[str, str]]],
        personality_profile: Optional[PersonalityProfile] = None,
        detect_sarcasm: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a personality-aware, sarcasm-aware training example.

        Args:
            conversation: Raw conversation text or message list
            personality_profile: Optional Big Five personality profile
            detect_sarcasm: Whether to detect sarcasm in the conversation

        Returns:
            Training example with personality and sarcasm metadata
        """
        # Normalize conversation
        if isinstance(conversation, str):
            messages = [{"role": "user", "content": conversation}]
        else:
            messages = conversation

        # Detect sarcasm if enabled
        sarcasm_result = None
        if self.sarcasm_detector and detect_sarcasm:
            # Check all user messages for sarcasm
            user_messages = [msg["content"] for msg in messages if msg.get("role") == "user"]
            if user_messages:
                combined_text = " ".join(user_messages)
                sarcasm_result = self.sarcasm_detector.detect_sarcasm(combined_text)

        # Determine therapeutic approach and communication style
        therapeutic_approach = TherapeuticApproach.CBT  # Default
        communication_style = CommunicationStyle.EMPATHETIC  # Default

        if self.personality_adapter and personality_profile:
            therapeutic_approach = self.personality_adapter.select_therapeutic_approach(
                personality_profile
            )
            communication_style = self.personality_adapter.select_communication_style(
                personality_profile
            )

        # Build metadata
        metadata = {
            "source": "psych_personality_integrator",
            "therapeutic_approach": therapeutic_approach.value,
            "communication_style": communication_style.value,
        }

        if personality_profile:
            metadata["personality_profile"] = personality_profile.to_dict()
            metadata["dominant_trait"] = personality_profile.get_dominant_trait().value

        if sarcasm_result:
            metadata["sarcasm_detected"] = sarcasm_result.is_sarcastic
            metadata["sarcasm_confidence"] = sarcasm_result.confidence
            if sarcasm_result.indicators:
                metadata["sarcasm_indicators"] = sarcasm_result.indicators

        return {
            "conversation": messages,
            "metadata": metadata,
        }

    def load_psychology_knowledge_examples(
        self,
        max_examples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load psychology book examples for training.

        Args:
            max_examples: Maximum number of examples to load

        Returns:
            List of psychology knowledge examples
        """
        entries = self.psych_loader.load_psychology_books()

        if max_examples:
            entries = entries[:max_examples]

        examples = []
        for entry in entries:
            # Convert entry to training format
            # Entry format may vary, so handle common structures
            if isinstance(entry, dict):
                text = entry.get("text", entry.get("content", str(entry)))
                title = entry.get("title", entry.get("concept", "Psychology Concept"))

                example = {
                    "conversation": [
                        {"role": "user", "content": f"Explain {title} in a therapeutic context."},
                        {"role": "assistant", "content": text},
                    ],
                    "metadata": {
                        "source": "psychology_books",
                        "title": title,
                        "therapeutic_approach": TherapeuticApproach.CBT.value,
                        "communication_style": CommunicationStyle.ANALYTICAL.value,
                    },
                }
                examples.append(example)

        logger.info(f"Generated {len(examples)} psychology knowledge examples")
        return examples


# Convenience functions
def detect_sarcasm(text: str, context: Optional[Dict[str, Any]] = None) -> SarcasmDetection:
    """Convenience function to detect sarcasm"""
    detector = SarcasmDetector()
    return detector.detect_sarcasm(text, context)


def select_therapeutic_approach(profile: PersonalityProfile) -> TherapeuticApproach:
    """Convenience function to select therapeutic approach"""
    adapter = PersonalityAdapter()
    return adapter.select_therapeutic_approach(profile)


def select_communication_style(profile: PersonalityProfile) -> CommunicationStyle:
    """Convenience function to select communication style"""
    adapter = PersonalityAdapter()
    return adapter.select_communication_style(profile)

