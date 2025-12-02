#!/usr/bin/env python3
"""
Tests for edge category and profile types.
"""

import pytest
from ai.dataset_pipeline.types.edge_categories import (
    EdgeCategory,
    IntensityLevel,
    EdgeProfile,
    get_all_edge_categories,
    get_all_intensity_levels,
    get_categories_by_intensity,
    validate_edge_profile,
)
from ai.dataset_pipeline.style.less_chipper import Tone


def test_edge_category_enum():
    """Test that EdgeCategory enum has expected values."""
    assert EdgeCategory.SUICIDAL_IDEATION.value == "suicidal_ideation"
    assert EdgeCategory.DOMESTIC_VIOLENCE.value == "domestic_violence"
    assert EdgeCategory.BORDERLINE_PERSONALITY_CRISIS.value == "borderline_personality_crisis"

    # Verify we have at least 20 categories (brief mentions 25+)
    categories = get_all_edge_categories()
    assert len(categories) >= 20


def test_intensity_level_enum():
    """Test that IntensityLevel enum has expected values."""
    assert IntensityLevel.LOW.value == "low"
    assert IntensityLevel.MODERATE.value == "moderate"
    assert IntensityLevel.HIGH.value == "high"
    assert IntensityLevel.VERY_HIGH.value == "very_high"
    assert IntensityLevel.EXTREME.value == "extreme"


def test_edge_profile_creation():
    """Test creating and serializing an EdgeProfile."""
    profile = EdgeProfile(
        profile_id="test-profile-1",
        category=EdgeCategory.SUICIDAL_IDEATION,
        intensity=IntensityLevel.VERY_HIGH,
        tone=Tone.CRISIS_DIRECT,
        stage=3,
        scenario_type="crisis_intervention",
        requires_supervision=True,
    )

    assert profile.profile_id == "test-profile-1"
    assert profile.category == EdgeCategory.SUICIDAL_IDEATION
    assert profile.intensity == IntensityLevel.VERY_HIGH
    assert profile.tone == Tone.CRISIS_DIRECT
    assert profile.stage == 3
    assert profile.requires_supervision is True


def test_edge_profile_serialization():
    """Test serializing and deserializing EdgeProfile."""
    profile = EdgeProfile(
        profile_id="test-profile-2",
        category=EdgeCategory.DOMESTIC_VIOLENCE,
        intensity=IntensityLevel.HIGH,
        tone=Tone.CLINICAL,
        stage=3,
        metadata={"source": "synthetic", "validation": "passed"},
    )

    # Serialize
    data = profile.to_dict()
    assert data["profile_id"] == "test-profile-2"
    assert data["category"] == "domestic_violence"
    assert data["intensity"] == "high"
    assert data["tone"] == "clinical"

    # Deserialize
    restored = EdgeProfile.from_dict(data)
    assert restored.profile_id == profile.profile_id
    assert restored.category == profile.category
    assert restored.intensity == profile.intensity
    assert restored.tone == profile.tone
    assert restored.metadata == profile.metadata


def test_validate_edge_profile():
    """Test edge profile validation."""
    # Valid profile
    valid_profile = EdgeProfile(
        profile_id="valid-1",
        category=EdgeCategory.PSYCHOTIC_EPISODES,
        intensity=IntensityLevel.HIGH,
        tone=Tone.CRISIS_DIRECT,
        stage=3,
    )
    is_valid, error = validate_edge_profile(valid_profile)
    assert is_valid is True
    assert error is None

    # Invalid stage
    invalid_stage = EdgeProfile(
        profile_id="invalid-1",
        category=EdgeCategory.SUICIDAL_IDEATION,
        intensity=IntensityLevel.HIGH,
        tone=Tone.CRISIS_DIRECT,
        stage=5,  # Invalid
    )
    is_valid, error = validate_edge_profile(invalid_stage)
    assert is_valid is False
    assert error is not None
    assert "Stage must be 1-4" in error

    # Missing profile_id
    invalid_id = EdgeProfile(
        profile_id="",  # Empty
        category=EdgeCategory.SELF_HARM,
        intensity=IntensityLevel.MODERATE,
        tone=Tone.FOUNDATION,
        stage=2,
    )
    is_valid, error = validate_edge_profile(invalid_id)
    assert is_valid is False
    assert error is not None
    assert "profile_id is required" in error


def test_get_categories_by_intensity():
    """Test helper function for getting categories by intensity."""
    very_high_cats = get_categories_by_intensity(IntensityLevel.VERY_HIGH)
    assert EdgeCategory.SUICIDAL_IDEATION in very_high_cats
    assert EdgeCategory.HOMICIDAL_IDEATION in very_high_cats

    high_cats = get_categories_by_intensity(IntensityLevel.HIGH)
    assert EdgeCategory.DOMESTIC_VIOLENCE in high_cats
    assert EdgeCategory.SELF_HARM in high_cats

    moderate_cats = get_categories_by_intensity(IntensityLevel.MODERATE)
    assert EdgeCategory.EATING_DISORDERS in moderate_cats

    # Test that LOW intensity raises ValueError
    with pytest.raises(ValueError, match="LOW intensity is not associated"):
        get_categories_by_intensity(IntensityLevel.LOW)


def test_edge_profile_invalid_tone_deserialization():
    """Test that from_dict raises ValueError for invalid tone strings."""
    data = {
        "profile_id": "test-profile",
        "category": "self_harm",
        "intensity": "moderate",
        "tone": "invalid_tone",
        "stage": 2,
    }

    with pytest.raises(ValueError, match="Invalid tone"):
        EdgeProfile.from_dict(data)


def test_edge_profile_invalid_category_deserialization():
    """Test that from_dict raises ValueError for invalid category strings."""
    data = {
        "profile_id": "test-profile",
        "category": "invalid_category",
        "intensity": "moderate",
        "tone": "foundation",
        "stage": 2,
    }

    with pytest.raises(ValueError, match="Invalid category"):
        EdgeProfile.from_dict(data)

    # Verify error message includes allowed categories
    try:
        EdgeProfile.from_dict(data)
    except ValueError as e:
        assert "Allowed categories" in str(e)


def test_edge_profile_invalid_intensity_deserialization():
    """Test that from_dict raises ValueError for invalid intensity strings."""
    data = {
        "profile_id": "test-profile",
        "category": "self_harm",
        "intensity": "invalid_intensity",
        "tone": "foundation",
        "stage": 2,
    }

    with pytest.raises(ValueError, match="Invalid intensity"):
        EdgeProfile.from_dict(data)

    # Verify error message includes allowed intensities
    try:
        EdgeProfile.from_dict(data)
    except ValueError as e:
        assert "Allowed intensities" in str(e)


def test_edge_profile_missing_required_fields():
    """Test that from_dict raises ValueError for missing required fields."""
    base_data = {
        "profile_id": "test-profile",
        "category": "self_harm",
        "intensity": "moderate",
        "tone": "foundation",
        "stage": 2,
    }

    # Test missing category
    data = base_data.copy()
    del data["category"]
    with pytest.raises(ValueError, match="Missing required field 'category'"):
        EdgeProfile.from_dict(data)

    # Test missing intensity
    data = base_data.copy()
    del data["intensity"]
    with pytest.raises(ValueError, match="Missing required field 'intensity'"):
        EdgeProfile.from_dict(data)

    # Test missing tone
    data = base_data.copy()
    del data["tone"]
    with pytest.raises(ValueError, match="Missing required field 'tone'"):
        EdgeProfile.from_dict(data)

    # Test missing profile_id
    data = base_data.copy()
    del data["profile_id"]
    with pytest.raises(ValueError, match="Missing required field 'profile_id'"):
        EdgeProfile.from_dict(data)

    # Test missing stage
    data = base_data.copy()
    del data["stage"]
    with pytest.raises(ValueError, match="Missing required field 'stage'"):
        EdgeProfile.from_dict(data)


@pytest.mark.parametrize("tone_enum", [Tone.FOUNDATION, Tone.CLINICAL, Tone.CRISIS_DIRECT])
def test_edge_profile_tone_serialization_roundtrip(tone_enum):
    """Test that tone serialization/deserialization preserves all Tone enum values."""
    profile = EdgeProfile(
        profile_id="test-profile",
        category=EdgeCategory.SELF_HARM,
        intensity=IntensityLevel.MODERATE,
        tone=tone_enum,
        stage=2,
    )

    # Serialize and deserialize
    data = profile.to_dict()
    restored = EdgeProfile.from_dict(data)

    assert restored.tone == tone_enum
    assert restored.tone.value == tone_enum.value

