#!/usr/bin/env python3
"""Test suite for PATIENT-Ψ clinical profile definitions."""

import unittest

import pytest

from ai.pkg_mera.platform.patient_psi.profiles import ClinicalProfile, ProfileRegistry
from ai.pkg_mera.platform.patient_psi.styles import ConversationalStyle


class TestProfileRegistry(unittest.TestCase):
    """Test ProfileRegistry singleton and profile retrieval."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ProfileRegistry()

    def test_contains_exactly_twenty_profiles(self):
        profiles = self.registry.list_profiles()
        assert len(profiles) == 20, f"Expected 20 profiles, got {len(profiles)}"

    def test_all_profiles_have_non_empty_fields(self):
        for name in self.registry.list_profiles():
            profile = self.registry.get_profile(name)
            assert profile.name, f"Profile {name!r} has empty name"
            assert profile.display_name, f"Profile {name!r} has empty display_name"
            assert profile.description, f"Profile {name!r} has empty description"
            assert len(profile.diagnoses) > 0, f"Profile {name!r} has no diagnoses"
            assert len(profile.typical_symptoms) > 0, f"Profile {name!r} has no typical_symptoms"
            assert len(profile.ccd_config) > 0, f"Profile {name!r} has empty ccd_config"
            assert len(profile.linguistic_features) > 0, f"Profile {name!r} has no linguistic_features"
            assert len(profile.common_triggers) > 0, f"Profile {name!r} has no common_triggers"
            assert profile.treatment_history, f"Profile {name!r} has empty treatment_history"

    def test_get_profile_returns_valid_profile(self):
        profile = self.registry.get_profile("generalized_anxiety")
        assert isinstance(profile, ClinicalProfile)
        assert profile.display_name == "Generalized Anxiety Disorder"
        assert profile.default_style == ConversationalStyle.ANXIOUS

    def test_get_profile_raises_key_error_for_missing(self):
        with pytest.raises(KeyError):
            self.registry.get_profile("non_existent_disorder")

    def test_get_profiles_by_diagnosis(self):
        results = self.registry.get_profiles_by_diagnosis("F32.2")
        assert isinstance(results, list)
        assert len(results) > 0, "Expected at least one profile for F32.2"
        assert results[0].name == "major_depressive_disorder"

    def test_get_profiles_by_diagnosis_returns_empty_for_unknown(self):
        results = self.registry.get_profiles_by_diagnosis("X99.999")
        assert results == []

    def test_get_default_profile(self):
        default = self.registry.get_default_profile()
        assert isinstance(default, ClinicalProfile)
        assert default.name == "major_depressive_disorder"

    def test_no_duplicate_profile_names(self):
        names = self.registry.list_profiles()
        assert len(names) == len(set(names)), "Duplicate profile names found"

    def test_profile_list_is_sorted(self):
        names = self.registry.list_profiles()
        assert names == sorted(names), "Profile names should be sorted alphabetically"

    def test_each_profile_has_valid_severity_range(self):
        for name in self.registry.list_profiles():
            profile = self.registry.get_profile(name)
            min_sev, max_sev = profile.severity_range
            assert 0.0 <= min_sev <= 1.0, f"Profile {name!r} min severity {min_sev} out of range"
            assert 0.0 <= max_sev <= 1.0, f"Profile {name!r} max severity {max_sev} out of range"
            assert min_sev <= max_sev, f"Profile {name!r} min > max ({min_sev} > {max_sev})"

    def test_each_profile_has_all_linguistic_features(self):
        required_keys = {"hedging", "negation_density", "first_person_singular", "cause_words", "absolutist_words"}
        for name in self.registry.list_profiles():
            profile = self.registry.get_profile(name)
            actual_keys = set(profile.linguistic_features.keys())
            assert actual_keys == required_keys, (
                f"Profile {name!r} has unexpected linguistic features: "
                f"missing={required_keys - actual_keys}, extra={actual_keys - required_keys}"
            )

    def test_each_profile_linguistic_features_in_range(self):
        for name in self.registry.list_profiles():
            profile = self.registry.get_profile(name)
            for key, value in profile.linguistic_features.items():
                assert 0.0 <= value <= 1.0, f"Profile {name!r} linguistic feature {key!r} = {value} out of range"

    def test_each_profile_default_style_is_valid(self):
        valid_styles = set(ConversationalStyle)
        for name in self.registry.list_profiles():
            profile = self.registry.get_profile(name)
            assert profile.default_style in valid_styles, f"Profile {name!r} has invalid style: {profile.default_style}"

    def test_ccd_config_has_all_required_keys(self):
        required_keys = {
            "core_beliefs",
            "intermediate_beliefs",
            "coping_strategies",
            "compensatory_strategies",
            "situation_interpretations",
            "emotional_responses",
            "behavioral_responses",
            "cognitive_triads",
        }
        for name in self.registry.list_profiles():
            profile = self.registry.get_profile(name)
            actual_keys = set(profile.ccd_config.keys())
            assert actual_keys == required_keys, (
                f"Profile {name!r} CCD config missing: {required_keys - actual_keys}, "
                f"extra: {actual_keys - required_keys}"
            )

    def test_ccd_config_has_minimum_entries(self):
        for name in self.registry.list_profiles():
            profile = self.registry.get_profile(name)
            for key, entries in profile.ccd_config.items():
                min_required = 1 if key == "cognitive_triads" else 2
                assert len(entries) >= min_required, (
                    f"Profile {name!r} CCD key {key!r} has only {len(entries)} entries "
                    f"(minimum {min_required} required)"
                )

    def test_no_two_profiles_have_identical_ccd_config(self):
        names = self.registry.list_profiles()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p1 = self.registry.get_profile(names[i])
                p2 = self.registry.get_profile(names[j])
                first_belief_1 = p1.ccd_config["core_beliefs"][0]["content"]
                first_belief_2 = p2.ccd_config["core_beliefs"][0]["content"]
                assert first_belief_1 != first_belief_2, (
                    f"Profiles {names[i]!r} and {names[j]!r} share identical core belief: {first_belief_1!r}"
                )


class TestClinicalProfile(unittest.TestCase):
    """Test ClinicalProfile model integrity."""

    def test_profile_model_creation(self):
        profile = ClinicalProfile(
            name="test_profile",
            display_name="Test Profile",
            description="For testing purposes.",
            diagnoses=["F99"],
            typical_symptoms=["testing"],
            default_style=ConversationalStyle.NEUTRAL,
            ccd_config={
                "core_beliefs": [{"content": "Test belief", "domain": "self", "conviction": 0.5}],
                "intermediate_beliefs": [{"content": "Test belief", "rule_type": "rule", "conviction": 0.5}],
                "coping_strategies": [{"content": "Test strategy", "strategy_type": "avoidance", "effectiveness": 0.5}],
                "compensatory_strategies": [{"content": "Test", "behavior": "testing"}],
                "situation_interpretations": [{"situation": "Test", "interpretation": "testing"}],
                "emotional_responses": [{"emotion": "test", "intensity": 0.5, "valence": "negative"}],
                "behavioral_responses": [{"behavior": "Test", "triggered_by": "testing"}],
                "cognitive_triads": [{"self_views": 0.5, "world_views": 0.5, "future_views": 0.5}],
            },
            linguistic_features={
                "hedging": 0.5,
                "negation_density": 0.5,
                "first_person_singular": 0.5,
                "cause_words": 0.5,
                "absolutist_words": 0.5,
            },
            severity_range=(0.0, 1.0),
            common_triggers=["test trigger"],
            treatment_history="No treatment history.",
        )
        assert profile.name == "test_profile"
        assert profile.display_name == "Test Profile"
        assert profile.default_style == ConversationalStyle.NEUTRAL

    def test_to_dict_round_trip(self):
        registry = ProfileRegistry()
        original = registry.get_default_profile()
        d = original.model_dump()
        restored = ClinicalProfile.model_validate(d)
        assert restored.name == original.name
        assert restored.display_name == original.display_name
        assert restored.description == original.description
        assert restored.diagnoses == original.diagnoses
        assert restored.severity_range == original.severity_range
        assert restored.linguistic_features == original.linguistic_features
        assert restored.common_triggers == original.common_triggers
        assert restored.treatment_history == original.treatment_history


if __name__ == "__main__":
    unittest.main()
