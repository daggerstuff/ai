"""PATIENT-Ψ CCD injection prompt template.

Structured prompt builder that interleaves CCD components into LLM
system/prompt pipeline in the format:

    [PATIENT_IDENTITY] → [CCD_PROFILE] → [SITUATION_CONTEXT]
    → [CONVERSATIONAL_STYLE] → [HISTORY_SUMMARY]

Supports dynamic difficulty scaling (low/medium/high) and outputs
prompts compatible with existing inference pipelines.
"""

from __future__ import annotations

from typing import Literal

from ai.tools.utilities.platform.patient_psi.profiles import ClinicalProfile, ProfileRegistry
from ai.tools.utilities.platform.patient_psi.styles import (
    ConversationalStyle,
    StyleRegistry,
)

# ---------------------------------------------------------------------------
# Difficulty-level constants
# ---------------------------------------------------------------------------

_DifficultyLevel = Literal["low", "medium", "high"]

# Max characters per history-snippet entry (avoids bloating the prompt)
_HISTORY_SNIPPET_CHARS = 120

# Thresholds for auto-classifying a raw float difficulty
_LOW_THRESHOLD = 0.35
_HIGH_THRESHOLD = 0.7




def _classify_difficulty(difficulty: float) -> _DifficultyLevel:
    """Map a 0-1 float difficulty to a categorical level."""
    if difficulty < _LOW_THRESHOLD:
        return "low"
    if difficulty < _HIGH_THRESHOLD:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_identity_section(
    profile: ClinicalProfile,
    patient_name: str,
) -> str:
    """[PATIENT_IDENTITY] section."""
    diag_str = "; ".join(profile.diagnoses)
    symptoms = profile.typical_symptoms[:5]
    symptom_str = ", ".join(symptoms)
    return (
        f"[PATIENT_IDENTITY]\n"
        f"You are simulating a patient named {patient_name}.\n"
        f"Clinical diagnosis: {profile.display_name} ({diag_str}).\n"
        f"Presenting symptoms: {symptom_str}.\n"
        f"Treatment history: {profile.treatment_history}\n"
    )


def _show(difficulty: _DifficultyLevel, section: str) -> bool:
    """True if *section* should be included at *difficulty*."""
    _visibility: dict[str, set[_DifficultyLevel]] = {
        "core_beliefs": {"low", "medium", "high"},
        "intermediate_beliefs": {"medium", "high"},
        "coping_strategies": {"medium", "high"},
        "emotional_responses": {"medium", "high"},
        "cognitive_triads": {"high"},
        "situation_interpretations": {"high"},
        "behavioral_responses": {"high"},
    }
    return difficulty in _visibility.get(section, set())


def _append_section(
    lines: list[str],
    items: list[dict],
    header: str,
    fmt: str,
    max_items: int = 8,
) -> None:
    """Append a formatted CCD sub-section to *lines* if *items* is non-empty."""
    if not items:
        return
    lines.append(header)
    for item in items[:max_items]:
        lines.append(f"  - {fmt.format(**item)}")
    lines.append("")


def _build_ccd_section(
    profile: ClinicalProfile,
    difficulty: _DifficultyLevel,
) -> str:
    """[CCD_PROFILE] section — scaled by difficulty.

    Uses a data-driven slot system to keep branch count low.
    """
    ccd = profile.ccd_config
    lines: list[str] = ["[CCD_PROFILE]"]

    sections = [
        ("core_beliefs", "Core beliefs:", "{content} (conviction: {conviction:.0%})"),
        (
            "intermediate_beliefs",
            "Intermediate beliefs (rules / attitudes / assumptions):",
            '[{rule_type}] "{content}"',
        ),
        (
            "coping_strategies",
            "Coping strategies:",
            "{content} ({strategy_type}, effectiveness: {effectiveness:.0%})",
        ),
        (
            "emotional_responses",
            "Emotional patterns:",
            "{emotion} (intensity: {intensity:.0%}, valence: {valence})",
        ),
        (
            "cognitive_triads",
            "Cognitive triads (negative views of self / world / future):",
            "self: {self_views:.0%}, world: {world_views:.0%}, future: {future_views:.0%}",
        ),
        (
            "situation_interpretations",
            "Typical interpretations of situations (with cognitive distortions):",
            'When "{situation}" → "{interpretation}"{dist}',
        ),
        (
            "behavioral_responses",
            "Behavioral patterns:",
            "{behavior} (triggered by: {triggered_by})",
        ),
    ]

    for key, header, fmt in sections:
        if not _show(difficulty, key):
            continue
        items = list(ccd.get(key, []))
        # Augment situation_interpretations with a "dist" key for the template
        if key == "situation_interpretations":
            new_items = []
            for i in items:
                new_i = dict(i)
                new_i["dist"] = f" [distortion: {new_i['distortion_type']}]" if new_i.get("distortion_type") else ""
                new_items.append(new_i)
            items = new_items
        max_items = 4 if key in {"situation_interpretations", "behavioral_responses", "cognitive_triads"} else 8
        _append_section(lines, items, header, fmt, max_items=max_items)

    return "\n".join(lines)


def _build_situation_section(situation_context: str | None) -> str:
    """[SITUATION_CONTEXT] section."""
    ctx = situation_context or "The patient is in a standard therapy session."
    return f"[SITUATION_CONTEXT]\n{ctx}\n"


def _build_style_section(
    style: ConversationalStyle,
    style_registry: StyleRegistry,
) -> str:
    """[CONVERSATIONAL_STYLE] section.

    Produces behavioural instructions for the LLM to match the style.
    """
    try:
        markers = style_registry.get_style_markers(style)
    except KeyError:
        markers = {"formality": 0.5, "emotional_valence": 0.5}

    style_desc = {
        ConversationalStyle.NEUTRAL: ("neutral, balanced therapeutic presence. Neither overly warm nor cold."),
        ConversationalStyle.FRIENDLY: ("warm and encouraging. Uses supportive language and expresses appreciation."),
        ConversationalStyle.HOSTILE: (
            "dismissive and confrontational. Shows resistance, challenges the therapist, minimal disclosure."
        ),
        ConversationalStyle.ANXIOUS: (
            "apprehensive and worried. Seeks reassurance, expresses doubt and fear about the process."
        ),
        ConversationalStyle.MELANCHOLIC: (
            "somber and low-energy. Expresses hopelessness, emptiness, and lack of motivation."
        ),
        ConversationalStyle.MANIC: (
            "energetic and tangential. Rapid speech, grand ideas, difficulty staying on one topic."
        ),
    }

    desc = style_desc.get(style, "neutral clinical style.")
    marker_str = "; ".join(f"{k}: {v:.0%}" for k, v in sorted(markers.items()))

    return f"[CONVERSATIONAL_STYLE]\nYou communicate in a {desc}\nLinguistic markers — {marker_str}.\n"


def _build_history_section(
    history: list[dict[str, str]] | None,
) -> str:
    """[HISTORY_SUMMARY] section.

    Args:
        history: A chronological list of {role, content} dicts.
    """
    if not history:
        return "[HISTORY_SUMMARY]\nThis is the beginning of the session.\n"

    # Keep last ~6 exchanges to stay within context budget
    recent = history[-6:]
    lines = ["[HISTORY_SUMMARY]"]
    for msg in recent:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        snippet = content[:_HISTORY_SNIPPET_CHARS] + ("…" if len(content) > _HISTORY_SNIPPET_CHARS else "")
        lines.append(f'{role}: "{snippet}"')
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


class CCDPromptBuilder:
    """Builds structured CCD-injected prompts for LLM patient simulation.

    Usage::

        builder = CCDPromptBuilder()
        prompt = builder.build_full_prompt(
            profile=registry.get_profile("generalized_anxiety"),
            style=ConversationalStyle.ANXIOUS,
            difficulty=0.7,
            situation_context="Client missed a work deadline.",
            history=[{"role": "therapist", "content": "How was your week?"}],
        )
    """

    def __init__(
        self,
        style_registry: StyleRegistry | None = None,
        profile_registry: ProfileRegistry | None = None,
    ) -> None:
        self._style_registry = style_registry or StyleRegistry()
        self._profile_registry = profile_registry or ProfileRegistry()

    # ------------------------------------------------------------------
    # Public helpers — build individual prompt sections
    # ------------------------------------------------------------------

    def build_identity_section(
        self,
        profile: ClinicalProfile,
        patient_name: str = "Client",
    ) -> str:
        """Return the [PATIENT_IDENTITY] section."""
        return _build_identity_section(profile, patient_name)

    def build_ccd_profile_section(
        self,
        profile: ClinicalProfile,
        difficulty: float | _DifficultyLevel = 0.5,
    ) -> str:
        """Return the [CCD_PROFILE] section at the given difficulty."""
        level: _DifficultyLevel = (
            _classify_difficulty(difficulty) if isinstance(difficulty, (int, float)) else difficulty
        )
        return _build_ccd_section(profile, level)

    def build_situation_section(self, situation_context: str | None = None) -> str:
        """Return the [SITUATION_CONTEXT] section."""
        return _build_situation_section(situation_context)

    def build_style_section(self, style: ConversationalStyle) -> str:
        """Return the [CONVERSATIONAL_STYLE] section for *style*."""
        return _build_style_section(style, self._style_registry)

    def build_history_section(self, history: list[dict[str, str]] | None = None) -> str:
        """Return the [HISTORY_SUMMARY] section from the conversation log."""
        return _build_history_section(history)

    # ------------------------------------------------------------------
    # Assembled prompt
    # ------------------------------------------------------------------

    def build_full_prompt(
        self,
        profile: ClinicalProfile,
        *,
        style: ConversationalStyle | None = None,
        difficulty: float = 0.5,
        situation_context: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Assemble the full multi-section CCD injection prompt.

        Args:
            profile: The ClinicalProfile to simulate.
            style: Conversational style override (defaults to profile.default_style).
            difficulty: 0.0-1.0 — controls detail depth of the CCD profile
                injected into the prompt.
            situation_context: Free-text description of the current scenario.
            history: Previous conversation turns as ``{"role": …, "content": …}``.

        Returns:
            A complete system-prompt string ready for LLM consumption.
        """
        style = style or profile.default_style
        patient_name = "Client"

        sections = [
            self.build_identity_section(profile, patient_name),
            self.build_ccd_profile_section(profile, difficulty),
            self.build_situation_section(situation_context),
            self.build_style_section(style),
            self.build_history_section(history),
        ]

        return "\n".join(sections)

    build_system_prompt = build_full_prompt
