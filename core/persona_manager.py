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
ROBOTIC_PHRASING_PENALTIES = [
    "as an ai",
    "i'm here to help",
    "i understand that",
    "it sounds like you are saying",
    "let me know if you need anything else",
    "is there anything else i can help with",
    "it is important to remember",
    "firstly, ",
    "secondly, ",
    "in conclusion,",
]


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
        Allows for tuning: rejects robotic phrasing and list-making (e.g. 1. 2. 3.)
        to avoid rejecting genuinely structured human thought while blocking LLM-isms.

        Returns True if the text passes validation (is human-like),
        False if it is rejected.
        """
        text_lower = text.lower()

        # Immediate fail for typical AI disclaimers
        for phrase in ROBOTIC_PHRASING_PENALTIES:
            if phrase in text_lower:
                logger.debug("Validation failed due to robotic phrasing: '%s'", phrase)
                return False

        # Immediate fail for excessive bullet points or lists in a conversation
        if "\n1." in text and "\n2." in text and "\n3." in text:
            # Humans rarely speak in pristine 3-part bullet points off the cuff
            logger.debug("Validation failed due to mechanical list-making.")
            return False

        return True
