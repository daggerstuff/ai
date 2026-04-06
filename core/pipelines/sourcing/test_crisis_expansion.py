"""
Basic functional tests for the CrisisExpansion module.
"""

from ai.core.pipelines.sourcing.crisis_expansion import (
    CrisisCategory,
    CrisisExpansion,
    CrisisExpansionConfig,
    Language,
)


def test_basic_functionality():
    """Test basic functionality of CrisisExpansion."""
    print("Testing basic functionality...")

    # Create expansion instance
    expansion = CrisisExpansion()

    # Test expand_term
    print("\n1. Testing expand_term:")
    expansions = expansion.expand_term("kill myself", "suicidal_ideation")
    print(f"   'kill myself' expands to {len(expansions)} terms")

    # Check that key synonyms are present by looking at term info
    term_info = expansion.get_term_info("kill myself")
    assert term_info is not None
    assert "end it all" in term_info.synonyms
    assert "not worth living" in term_info.synonyms
    assert "better off dead" in term_info.synonyms

    # Test expansion with higher limit
    config_high_limit = CrisisExpansionConfig()
    config_high_limit.max_expansion_terms = 50
    expansion_high_limit = CrisisExpansion(config_high_limit)
    expansions = expansion_high_limit.expand_term("kill myself", "suicidal_ideation")
    assert len(expansions) > 20

    # Test categorize_term
    print("\n2. Testing categorize_term:")
    category = expansion.categorize_term("cut myself")
    print(f"   'cut myself' category: {category}")
    assert category == CrisisCategory.SELF_HARM.value

    category = expansion.categorize_term("suicidal")
    print(f"   'suicidal' category: {category}")
    assert category == CrisisCategory.SUICIDAL_IDEATION.value

    # Test get_intensity
    print("\n3. Testing get_intensity:")
    intensity = expansion.get_intensity("suicidal")
    print(f"   'suicidal' intensity: {intensity}")
    assert intensity == 0.95

    intensity = expansion.get_intensity("kill myself")
    print(f"   'kill myself' intensity: {intensity}")
    assert intensity == 1.0

    # Test is_crisis_term
    print("\n4. Testing is_crisis_term:")
    assert expansion.is_crisis_term("kill myself") == True
    assert expansion.is_crisis_term("hello world") == False
    assert expansion.is_crisis_term("cut myself") == True

    # Test get_all_crisis_terms
    print("\n5. Testing get_all_crisis_terms:")
    all_terms = expansion.get_all_crisis_terms()
    print(f"   Total crisis terms: {len(all_terms)}")
    assert len(all_terms) > 20  # Should have many terms

    # Test negation handling
    print("\n6. Testing negation handling:")
    neg_expansions = expansion.expand_term("kill myself", "suicidal_ideation")
    neg_terms = [
        t
        for t in neg_expansions
        if "not" in t
        or "don't" in t
        or "never" in t
        or "would never" in t
        or "could never" in t
        or "should never" in t
    ]
    print(f"   Found {len(neg_terms)} negation variants")
    print(f"   Sample negation terms: {neg_terms[:5]}")
    assert len(neg_terms) > 0
    # Check for any negation form rather than specific ones
    has_negation = any(
        "not" in t or "don't" in t or "never" in t for t in neg_expansions
    )
    assert has_negation

    # Test phrase variants
    print("\n7. Testing phrase variants:")
    variant_expansions = expansion.expand_term("kill myself", "suicidal_ideation")
    variant_terms = [
        t for t in variant_expansions if "I want to" in t or "I'm going to" in t
    ]
    print(f"   Found {len(variant_terms)} intention variants")
    assert len(variant_terms) > 0

    # Test Spanish terms (if enabled)
    print("\n8. Testing Spanish terms:")
    spanish_expansion = CrisisExpansion()
    spanish_term = spanish_expansion.expand_term(
        "quitarme la vida", "suicidal_ideation"
    )
    print(f"   Spanish term 'quitarme la vida' expands to {len(spanish_term)} terms")
    print(f"   Sample terms: {list(spanish_term)[:10]}")
    # Check that we get some expansions (the exact terms may vary based on implementation)
    assert len(spanish_term) > 0
    # The base term should be present
    assert "quitarme la vida" in spanish_term

    # Test term info
    print("\n9. Testing get_term_info:")
    term_info = expansion.get_term_info("cut myself")
    assert term_info is not None
    assert term_info.term == "cut myself"
    assert term_info.category == CrisisCategory.SELF_HARM
    print(f"   'cut myself' intensity: {term_info.intensity}")
    assert term_info.intensity == 0.9
    assert "cutting myself" in term_info.synonyms

    print("\n✅ All tests passed!")


def test_config_options():
    """Test different configuration options."""
    print("\n\nTesting configuration options...")

    # Test with synonyms disabled
    print("\n1. Testing with synonyms disabled:")
    config_no_synonyms = CrisisExpansionConfig()
    config_no_synonyms.include_synonyms = False
    expansion_no_synonyms = CrisisExpansion(config_no_synonyms)

    expansions = expansion_no_synonyms.expand_term("cut myself", "self_harm")
    print(f"   'cut myself' with synonyms disabled: {len(expansions)} terms")
    # Should still have the base term and variants, but fewer synonyms
    assert "cut myself" in expansions
    assert "cutting myself" not in expansions  # This is a synonym

    # Test with phrase variants disabled
    print("\n2. Testing with phrase variants disabled:")
    config_no_variants = CrisisExpansionConfig()
    config_no_variants.generate_phrase_variants = False
    expansion_no_variants = CrisisExpansion(config_no_variants)

    expansions = expansion_no_variants.expand_term("kill myself", "suicidal_ideation")
    print(f"   'kill myself' with variants disabled: {len(expansions)} terms")
    assert "kill myself" in expansions
    assert "I want to kill myself" not in expansions  # This is a variant

    # Test with negation handling disabled
    print("\n3. Testing with negation handling disabled:")
    config_no_negation = CrisisExpansionConfig()
    config_no_negation.handle_negation = False
    expansion_no_negation = CrisisExpansion(config_no_negation)

    expansions = expansion_no_negation.expand_term("kill myself", "suicidal_ideation")
    neg_terms = [t for t in expansions if "not" in t or "don't" in t]
    print(f"   'kill myself' with negation disabled: {len(neg_terms)} negation terms")
    assert len(neg_terms) == 0  # Should have no negation terms

    print("\n✅ Configuration tests passed!")


def test_custom_terms():
    """Test loading custom terms from file."""
    print("\n\nTesting custom term loading...")

    # Create a temporary YAML file with custom terms
    custom_terms_yaml = """
- term: "custom crisis term"
  category: "self_harm"
  intensity: 0.7
  synonyms:
    - "custom synonym 1"
    - "custom synonym 2"
  language: "en"
- term: "otro término personalizado"
  category: "substance_abuse"
  intensity: 0.8
  synonyms:
    - "sinónimo personalizado 1"
  language: "es"
"""

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(custom_terms_yaml)
        temp_filepath = f.name

    try:
        # Load expansion with custom terms
        expansion = CrisisExpansion()
        expansion.load_from_file(temp_filepath)

        # Test English custom term
        assert expansion.is_crisis_term("custom crisis term") == True
        category = expansion.categorize_term("custom crisis term")
        assert category == CrisisCategory.SELF_HARM.value
        intensity = expansion.get_intensity("custom crisis term")
        assert intensity == 0.7

        # Test Spanish custom term
        assert expansion.is_crisis_term("otro término personalizado") == True
        category = expansion.categorize_term("otro término personalizado")
        assert category == CrisisCategory.SUBSTANCE_ABUSE.value
        intensity = expansion.get_intensity("otro término personalizado")
        assert intensity == 0.8

        # Test synonyms
        assert expansion.is_crisis_term("custom synonym 1") == True
        assert expansion.is_crisis_term("sinónimo personalizado 1") == True

        print("✅ Custom term loading tests passed!")

    finally:
        # Clean up temp file
        os.unlink(temp_filepath)


if __name__ == "__main__":
    test_basic_functionality()
    test_config_options()
    test_custom_terms()
    print("\n🎉 All tests completed successfully!")
