"""
PersonaManager - AI Patient Persona Scaffolding & Validation

Manages the injection of distinct clinical archetypes (e.g., Avoidant, Anxious, BPD)
and validates raw text generations to reject robotic phrasing and "LLM-isms", ensuring
high-fidelity human-like dialogue for the Empathy Gym simulator.

Archetypes are defined in `config/clinical_archetypes.json` alongside this module.
To add or modify a persona, edit that file — no Python changes are required.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Resolved relative to this file so it works regardless of CWD.
_ARCHETYPES_CONFIG = Path(__file__).parent / "config" / "clinical_archetypes.json"

# Heuristics used to prevent typical AI safety/assistant behaviors that ruin immersion
# These are hard blockers - text containing these phrases will be rejected
ROBOTIC_PHRASING_PENALTIES = [
    # AI self-identification
    "as an ai",
    "as a language model",
    "i'm an ai",
    "i am an ai",
    "as a therapeutic ai",
    # Generic helper phrases
    "i'm here to help",
    "i'm here to support you",
    "i understand that",  # Overused AI empathy marker
    "it sounds like you are saying",
    "let me know if you need anything else",
    "is there anything else i can help with",
    # Lecture/pedantic markers
    "it is important to remember",
    "it's important to note",
    "keep in mind that",
    # Formal transition words (common in AI, rare in casual speech)
    "firstly, ",
    "secondly, ",
    "in conclusion,",
    "to summarize,",
    "in summary,",
    # Over-apologizing patterns
    "i apologize if",
    "i'm sorry if this",
    "i apologize for any confusion",
    # Hedge phrases
    "it's worth noting that",
    "it may be helpful to",
    "please note that",
]

# Softer patterns that trigger a score penalty but not immediate rejection
SOFT_LLM_PATTERNS = [
    # Excessive politeness
    "i hope this helps",
    "please let me know",
    "feel free to",
    # Generic therapeutic clichés (can be legitimate but often AI-generated)
    "it takes courage",
    "seeking help is a sign of strength",
    "you are not alone",
    "there is hope",
]

# Maximum soft penalty score before rejection
MAX_SOFT_PENALTY_SCORE = 3


@dataclass
class Persona:
    archetype_id: str
    name: str
    description: str
    traits: Dict[str, float]
    default_defense: str

    def generate_system_prompt(self) -> str:
        """Generates the base persona section of the LLM prompt."""
        return (
            "You are roleplaying as a therapy patient. Do not break character. "
            f"Your clinical archetype is: {self.name}. "
            f"Profile: {self.description} "
            f"Default Defense Mechanism: {self.default_defense}. "
            "Do not act like an AI. Keep your responses grounded, human, and "
            "conversational."
        )


class PersonaManager:
    """
    Loads clinical archetypes from `config/clinical_archetypes.json` and exposes
    them as validated Persona objects. Validates LLM output against known
    robotic phrasing patterns to enforce high-fidelity immersion.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or _ARCHETYPES_CONFIG
        self._personas = self._load_personas()

    def _load_personas(self) -> Dict[str, Persona]:
        """Deserializes archetypes from JSON config into Persona objects."""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Clinical archetypes config not found: {self._config_path}"
            )

        with self._config_path.open("r", encoding="utf-8") as fh:
            raw: Dict = json.load(fh)

        # First, validate all archetypes
        for archetype_id, data in raw.items():
            self._validate_archetype_schema(archetype_id, data)

        # Then, create personas using a dictionary comprehension
        personas: Dict[str, Persona] = {
            archetype_id: Persona(
                archetype_id=archetype_id,
                name=data["name"],
                description=data["description"],
                traits=data["traits"],
                default_defense=data["default_defense"],
            )
            for archetype_id, data in raw.items()
        }

        if not personas:
            raise ValueError(
                f"Clinical archetypes config is empty: {self._config_path}"
            )

        logger.info(
            "Loaded %d clinical archetypes from %s",
            len(personas),
            self._config_path,
        )
        return personas

    @staticmethod
    def _validate_archetype_schema(archetype_id: str, data: Dict) -> None:
        """
        Ensures every archetype has the required fields and that all OCEAN
        trait scores are normalised within [0.0, 1.0].

        Raises ValueError on the first violation found so misconfigured
        archetypes are caught at startup, not at runtime.
        """
        required_keys = {"name", "description", "traits", "default_defense"}
        ocean_traits = {
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        }

        if missing := required_keys - data.keys():
            raise ValueError(
                f"Archetype '{archetype_id}' is missing required keys: {missing}"
            )

        traits: Dict = data["traits"]
        if missing_traits := ocean_traits - traits.keys():
            raise ValueError(
                f"Archetype '{archetype_id}' is missing OCEAN traits: {missing_traits}"
            )

        for trait, score in traits.items():
            if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Archetype '{archetype_id}': trait '{trait}' score "
                    f"{score!r} is out of range [0.0, 1.0]."
                )

    def get_persona(self, archetype_id: str) -> Optional[Persona]:
        """Fetch a specific persona by ID. Returns None if the ID is unknown."""
        return self._personas.get(archetype_id)

    def get_random_persona(self) -> Persona:
        """Fetch a randomly selected persona."""
        return random.choice(list(self._personas.values()))

    def get_available_archetypes(self) -> List[str]:
        """List all available archetype IDs."""
        return list(self._personas.keys())

    def validate_human_likeness(self, text: str) -> bool:
        """
        Validate generated text against known LLM-isms.

        Uses a two-tier system:
        1. Hard blockers: Immediate rejection for obvious AI phrases
        2. Soft penalties: Accumulate points for suspicious patterns

        Returns True if the text passes validation (is human-like),
        False if it is rejected.
        """
        text_lower = text.lower()
        soft_penalty_score = 0

        # Tier 1: Hard blockers - immediate rejection
        for phrase in ROBOTIC_PHRASING_PENALTIES:
            if phrase in text_lower:
                logger.debug("Validation failed due to robotic phrasing: '%s'", phrase)
                return False

        # Tier 2: Soft penalties - accumulate and check threshold
        for pattern in SOFT_LLM_PATTERNS:
            if pattern in text_lower:
                soft_penalty_score += 1
                logger.debug(
                    "Soft penalty triggered: '%s' (score: %d)",
                    pattern,
                    soft_penalty_score,
                )

        if soft_penalty_score >= MAX_SOFT_PENALTY_SCORE:
            logger.debug(
                "Validation failed due to accumulated soft penalties: %d",
                soft_penalty_score,
            )
            return False

        # Check for mechanical list-making (3+ numbered items)
        list_patterns = ["\n1.", "\n2.", "\n3."]
        if all(p in text for p in list_patterns):
            # Allow if it's a legitimate structured response (e.g., homework, exercises)
            # by checking for therapeutic context keywords
            therapeutic_context = any(
                kw in text_lower
                for kw in ["exercise", "practice", "step", "homework", "assignment"]
            )
            if not therapeutic_context:
                logger.debug(
                    "Validation failed due to mechanical list-making without therapeutic context."
                )
                return False

        # Check for excessive bullet points (more than 4)
        bullet_count = text.count("\n- ") + text.count("\n* ")
        if bullet_count > 4:
            logger.debug(
                "Validation failed due to excessive bullet points: %d", bullet_count
            )
            return False

        return True
