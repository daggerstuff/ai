"""Tests for patient_psi CCD injection prompt template."""

from __future__ import annotations

import pytest

from ai.tools.utilities.platform.patient_psi.profiles import ClinicalProfile, ProfileRegistry
from ai.prompts.patient_psi.prompt_template import CCDPromptBuilder
from ai.tools.utilities.platform.patient_psi.styles import ConversationalStyle


@pytest.fixture
def builder() -> CCDPromptBuilder:
    return CCDPromptBuilder()


@pytest.fixture
def profile() -> ClinicalProfile:
    return ProfileRegistry().get_profile("generalized_anxiety")


# ---------------------------------------------------------------------------
# Identity section
# ---------------------------------------------------------------------------


class TestIdentitySection:
    def test_contains_patient_name(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        section = builder.build_identity_section(profile, patient_name="Alex")
        assert "[PATIENT_IDENTITY]" in section
        assert "Alex" in section

    def test_contains_diagnosis_info(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        section = builder.build_identity_section(profile)
        assert "F41.1" in section
        assert profile.display_name in section

    def test_default_patient_name(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        section = builder.build_identity_section(profile)
        assert "Client" in section


# ---------------------------------------------------------------------------
# CCD profile section — difficulty scaling
# ---------------------------------------------------------------------------


class TestCCDProfileSection:
    def test_core_beliefs_at_low_difficulty(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        section = builder.build_ccd_profile_section(profile, difficulty="low")
        assert "[CCD_PROFILE]" in section
        assert "Core beliefs:" in section
        # Medium+ sections should be absent at low
        assert "Intermediate beliefs" not in section
        assert "Coping strategies" not in section

    def test_intermediate_sections_at_medium(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        section = builder.build_ccd_profile_section(profile, difficulty="medium")
        assert "Intermediate beliefs" in section
        assert "Coping strategies" in section
        # High-only sections absent
        assert "Cognitive triads" not in section
        assert "Typical interpretations" not in section

    def test_all_sections_at_high(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        section = builder.build_ccd_profile_section(profile, difficulty="high")
        assert "Cognitive triads" in section
        assert "Typical interpretations" in section
        assert "Behavioral patterns" in section

    def test_float_difficulty_low(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        section = builder.build_ccd_profile_section(profile, difficulty=0.2)
        assert "[CCD_PROFILE]" in section
        assert "Core beliefs:" in section

    def test_float_difficulty_medium(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        section = builder.build_ccd_profile_section(profile, difficulty=0.5)
        assert "Intermediate beliefs" in section

    def test_float_difficulty_high(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        section = builder.build_ccd_profile_section(profile, difficulty=0.9)
        assert "Cognitive triads" in section

    def test_empty_profile_does_not_crash(self, builder: CCDPromptBuilder) -> None:
        """Edge case: very minimal profile with no ccd_config content."""
        profile = ClinicalProfile(
            name="empty",
            display_name="Empty",
            description="Minimal test profile.",
            diagnoses=["Z00.0"],
            typical_symptoms=["none"],
            default_style=ConversationalStyle.NEUTRAL,
            ccd_config={},
            linguistic_features={},
            severity_range=(0.0, 0.5),
            common_triggers=[],
            treatment_history="N/A",
        )
        section = builder.build_ccd_profile_section(profile, difficulty="high")
        assert "[CCD_PROFILE]" in section


# ---------------------------------------------------------------------------
# Situation section
# ---------------------------------------------------------------------------


class TestSituationSection:
    def test_custom_context(self, builder: CCDPromptBuilder) -> None:
        section = builder.build_situation_section("Client lost their job.")
        assert "[SITUATION_CONTEXT]" in section
        assert "lost their job" in section

    def test_default_context(self, builder: CCDPromptBuilder) -> None:
        section = builder.build_situation_section(None)
        assert "standard therapy session" in section


# ---------------------------------------------------------------------------
# Style section
# ---------------------------------------------------------------------------


class TestStyleSection:
    def test_each_style_has_unique_description(self, builder: CCDPromptBuilder) -> None:
        descriptions = set()
        for style in ConversationalStyle:
            section = builder.build_style_section(style)
            descriptions.add(section)
        # All 6 styles should produce unique outputs
        assert len(descriptions) == 6

    def test_style_markers_present(self, builder: CCDPromptBuilder) -> None:
        section = builder.build_style_section(ConversationalStyle.HOSTILE)
        assert "[CONVERSATIONAL_STYLE]" in section
        assert "confrontational" in section.lower() or "dismissive" in section.lower()


# ---------------------------------------------------------------------------
# History section
# ---------------------------------------------------------------------------


class TestHistorySection:
    def test_empty_history(self, builder: CCDPromptBuilder) -> None:
        section = builder.build_history_section(None)
        assert "beginning of the session" in section

    def test_recent_exchanges_included(self, builder: CCDPromptBuilder) -> None:
        history = [
            {"role": "therapist", "content": "How are you?"},
            {"role": "patient", "content": "I've been anxious all week."},
        ]
        section = builder.build_history_section(history)
        assert "Therapist" in section  # capitalized role
        assert "Patient" in section
        assert "How are you?" in section
        assert "anxious all week" in section

    def test_long_history_truncated(self, builder: CCDPromptBuilder) -> None:
        history = [{"role": "therapist", "content": f"Turn {i}"} for i in range(20)]
        section = builder.build_history_section(history)
        # Only last 6 should appear
        assert "Turn 0" not in section
        assert "Turn 14" in section

    def test_long_content_truncated(self, builder: CCDPromptBuilder) -> None:
        long = "word " * 200
        history = [{"role": "patient", "content": long}]
        section = builder.build_history_section(history)
        assert "…" in section  # ellipsis indicator


# ---------------------------------------------------------------------------
# Full prompt assembly
# ---------------------------------------------------------------------------


class TestFullPrompt:
    def test_all_sections_present(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        prompt = builder.build_full_prompt(profile)
        assert "[PATIENT_IDENTITY]" in prompt
        assert "[CCD_PROFILE]" in prompt
        assert "[SITUATION_CONTEXT]" in prompt
        assert "[CONVERSATIONAL_STYLE]" in prompt
        assert "[HISTORY_SUMMARY]" in prompt

    def test_happy_path(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        prompt = builder.build_full_prompt(
            profile,
            style=ConversationalStyle.ANXIOUS,
            difficulty=0.7,
            situation_context="Client is worried about a health scare.",
            history=[{"role": "therapist", "content": "What brings you here today?"}],
        )
        assert "apprehensive" in prompt or "worried" in prompt
        assert "health scare" in prompt
        assert profile.display_name in prompt

    def test_difficulty_controls_verbosity(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        low = builder.build_full_prompt(profile, difficulty=0.2)
        high = builder.build_full_prompt(profile, difficulty=0.9)
        assert len(high) > len(low)

    def test_build_system_prompt_alias(self, builder: CCDPromptBuilder, profile: ClinicalProfile) -> None:
        """build_system_prompt should work identically to build_full_prompt ."""
        sp = builder.build_system_prompt(profile, difficulty=0.5)
        fp = builder.build_full_prompt(profile, difficulty=0.5, history=None)
        assert sp == fp
