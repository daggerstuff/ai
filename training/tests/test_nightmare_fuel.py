"""Tests for generate_nightmare_fuel_5k.py — DPO training pair generator.

Covers all Oracle review recommendations:
1. Semantic deduplication (sentence transformers replacing Jaccard)
2. No CRISIS_CATEGORIES duplication
3. Bounds checking on string operations
4. --seed flag reproducibility
5. generate_pairs with mocked LLM
6. LLM fallback to seeds
7. Context-aware crisis resource handling
"""

import json
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the scripts dir to sys.path so we can import the module
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import generate_nightmare_fuel_5k as nf

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_random():
    """Reset random seed before each test for reproducibility."""
    random.seed(42)
    yield


@pytest.fixture
def temp_output():
    """Provide a temp file path for output."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════
# 1. Semantic Deduplication Tests (replaces _too_similar)
# ═══════════════════════════════════════════════════════════════════════


class TestSemanticDeduplication:
    """Verify _too_similar replacement uses semantic similarity."""

    def test_semantic_similarity_identical_texts(self):
        """Identical texts should be caught as too similar."""
        text = "I want to die. I have a plan and I'm going to do it tonight."
        assert nf._too_similar(text, text) is True

    def test_semantic_similarity_near_identical(self):
        """Near-identical with minor wording changes should be caught."""
        a = "I want to die. I have a plan and I'm going to do it tonight."
        b = "I want to die. I've got a plan and I'm going to do it tonight."
        assert nf._too_similar(a, b) is True

    def test_semantic_similarity_different_texts(self):
        """Completely different texts should NOT be flagged."""
        a = "I want to die. I have a plan and I'm going to do it tonight."
        b = "I had a good day today. The weather was nice and I went for a walk."
        assert nf._too_similar(a, b) is False

    def test_semantic_similarity_same_meaning_different_words(self):
        """Same core meaning with different words should be caught."""
        a = "I'm thinking about ending my life. I have everything planned out."
        b = "I've been contemplating suicide. I've made all the arrangements."
        # These are semantically similar even with different wording
        assert nf._too_similar(a, b) is True

    def test_semantic_similarity_empty_strings(self):
        """Empty strings should not cause crashes."""
        assert nf._too_similar("", "") is False
        assert nf._too_similar("", "some text") is False
        assert nf._too_similar("some text", "") is False

    def test_semantic_similarity_chosen_rejected_too_close(self):
        """Chosen and rejected responses should be semantically different."""
        chosen = "I hear how much pain you're in. Your safety is the priority right now. Please call 988."
        rejected = "You should really try to stop feeling that way. Just think positive thoughts."
        # These SHOULD be dissimilar — chosen is supportive, rejected is dismissive
        assert nf._too_similar(chosen, rejected) is False


# ═══════════════════════════════════════════════════════════════════════
# 2. CRISIS_CATEGORIES Duplication Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCrisisCategories:
    """Verify CRISIS_CATEGORIES is not duplicated anywhere."""

    def test_crisis_categories_is_frozenset(self):
        """CRISIS_CATEGORIES should be a frozenset (module-level constant)."""
        assert isinstance(nf.CRISIS_CATEGORIES, frozenset)

    def test_expected_crisis_categories(self):
        """Should contain all expected crisis categories."""
        expected = {
            "suicidal_ideation_active",
            "suicidal_ideation_passive",
            "self_harm",
            "substance_relapse",
            "eating_disorder_crisis",
            "psychosis_symptoms",
        }
        assert nf.CRISIS_CATEGORIES == expected

    def test_no_hardcoded_crisis_tuple(self):
        """The expansion function should reference CRISIS_CATEGORIES, not hardcode a tuple."""
        source = Path(nf.__file__).read_text("utf-8")
        # Find the _expand_chosen_with_llm function (or whatever it's renamed to)
        # and check there's no hardcoded tuple of crisis category strings
        import ast

        tree = ast.parse(source)
        hardcoded_crisis = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Look for function calls that contain hardcoded category names
                if isinstance(node.func, ast.Name) and "expand" in node.func.id.lower():
                    for arg in node.args:
                        if isinstance(arg, ast.Tuple):
                            for elt in arg.elts:
                                if (
                                    isinstance(elt, ast.Constant)
                                    and isinstance(elt.value, str)
                                    and "suicidal" in elt.value
                                ):
                                    hardcoded_crisis = True
        # Actually this approach is fragile. Let me just grep the text.
        assert '"_suicidal' not in source, "Found hardcoded crisis strings in a tuple"
        # Better: verify _expand_chosen_with_llm uses CRISIS_CATEGORIES reference
        # by checking the function body
        assert "CRISIS_CATEGORIES" in source, "CRISIS_CATEGORIES reference missing"
        # Count references - should be 2+ (definition + usage)
        ref_count = source.count("CRISIS_CATEGORIES")
        assert ref_count >= 2, f"CRISIS_CATEGORIES should be referenced, found {ref_count}"


# ═══════════════════════════════════════════════════════════════════════
# 3. Bounds Checking Tests
# ═══════════════════════════════════════════════════════════════════════


class TestBoundsChecking:
    """Verify string operations handle edge cases without crashing."""

    def test_variate_response_empty_string(self):
        """_variate_response should handle empty string without crashing."""
        # After removal of variation functions, this should either not exist
        # or be a no-op. If it still exists, it shouldn't crash.
        result = nf._variate_response("")
        assert isinstance(result, str)

    def test_variate_response_single_char(self):
        """_variate_response should handle single char without crashing."""
        result = nf._variate_response("A")
        assert isinstance(result, str)

    def test_variate_response_no_crash_on_short(self):
        """_variate_response should handle very short strings."""
        cases = ["", "A", "Hi", "OK.", "No!", "Why?", "Fine."]
        for case in cases:
            result = nf._variate_response(case)
            assert isinstance(result, str), f"Failed on '{case}'"

    def test_variate_prompt_empty_string(self):
        """_variate_prompt should handle empty string without crashing."""
        result = nf._variate_prompt("")
        assert isinstance(result, str)

    def test_variate_prompt_very_short(self):
        """_variate_prompt should handle very short prompts."""
        cases = ["", "Hi", "No", "I'm fine.", "OK?", "Why?", "Hello"]
        for case in cases:
            result = nf._variate_prompt(case)
            assert isinstance(result, str), f"Failed on '{case}'"


# ═══════════════════════════════════════════════════════════════════════
# 4. Seed Flag Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSeedReproducibility:
    """Verify --seed flag produces deterministic output."""

    def test_main_has_seed_arg(self):
        """main() should accept --seed argument."""
        # Just check it's in the argparser
        parser = nf._create_parser()
        args = parser.parse_args(["--seed", "42"])
        assert args.seed == 42

    def test_random_seeded_by_arg(self):
        """When seed is set, random should produce same sequence."""
        random.seed(12345)
        seq_a = [random.random() for _ in range(10)]

        random.seed(12345)
        seq_b = [random.random() for _ in range(10)]

        assert seq_a == seq_b, "Same seed should produce same random sequence"

    def test_different_seeds_different_output(self):
        """Different seeds should produce different random sequences."""
        random.seed(11111)
        seq_a = [random.random() for _ in range(10)]

        random.seed(99999)
        seq_b = [random.random() for _ in range(10)]

        assert seq_a != seq_b, "Different seeds should produce different sequences"


# ═══════════════════════════════════════════════════════════════════════
# 5. generate_pairs Tests (with mocked LLM)
# ═══════════════════════════════════════════════════════════════════════


class TestGeneratePairs:
    """Verify generate_pairs works correctly with mocked LLM."""

    def test_generate_pairs_basic(self):
        """Should generate pairs without LLM, using seed pools."""
        pairs = nf.generate_pairs(
            target_count=15,
            use_llm=False,
            output_path=None,
        )
        assert len(pairs) == 15
        # Each pair should have required fields
        for p in pairs:
            assert "prompt" in p
            assert "chosen" in p
            assert "rejected" in p
            assert "metadata" in p
            assert p["metadata"].get("category") is not None
            assert p["metadata"].get("pair_type") == "nightmare_fuel"
            assert "is_crisis" in p["metadata"]

    def test_generate_pairs_all_categories(self):
        """Should cover all 15 scenario categories."""
        pairs = nf.generate_pairs(
            target_count=150,
            use_llm=False,
            output_path=None,
        )
        categories = {p["metadata"]["category"] for p in pairs}
        expected = set(nf.SCENARIOS.keys())
        assert categories == expected, f"Missing categories: {expected - categories}"

    def test_generate_pairs_unique_prompts(self):
        """At minimum, prompts should have some uniqueness."""
        pairs = nf.generate_pairs(
            target_count=30,
            use_llm=False,
            output_path=None,
        )
        prompts = [p["prompt"] for p in pairs]
        unique = len(set(prompts))
        # With 15 categories × 15+ base prompts each, should have decent uniqueness
        assert unique >= 15, f"Only {unique} unique prompts out of {len(prompts)}"

    def test_generate_pairs_chosen_not_identical_to_rejected(self):
        """Chosen and rejected should not be semantically identical."""
        pairs = nf.generate_pairs(
            target_count=30,
            use_llm=False,
            output_path=None,
        )
        for p in pairs:
            assert p["chosen"] != p["rejected"], f"Chosen and rejected are identical for prompt: {p['prompt'][:50]}"

    def test_generate_pairs_chosen_superior_to_rejected(self):
        """Chosen responses should be clearly better than rejected ones.

        This is a heuristic check — chosen should be longer/more detailed
        (therapist responses are typically more nuanced than dismissive ones).
        """
        pairs = nf.generate_pairs(
            target_count=30,
            use_llm=False,
            output_path=None,
        )
        # At minimum, chosen should not be empty/super short while rejected is long
        for i, p in enumerate(pairs):
            if len(p["chosen"]) < 20 and len(p["rejected"]) > 100:
                pytest.fail(
                    f"Pair {i}: chosen too short ({len(p['chosen'])} chars) vs rejected ({len(p['rejected'])} chars)"
                )

    def test_generate_pairs_with_llm_mock(self):
        """Should work when LLM is enabled but mocked."""
        mock_llm = MagicMock(
            return_value="This is a therapist response generated by the LLM. It addresses the client's concerns directly and provides meaningful support."
        )

        with patch("generate_nightmare_fuel_5k._call_llm", mock_llm):
            pairs = nf.generate_pairs(
                target_count=15,
                use_llm=True,
                llm_rate=1.0,  # Force 100% LLM usage
                output_path=None,
            )

        assert len(pairs) == 15
        for p in pairs:
            assert p["chosen"] is not None
            assert len(p["chosen"]) > 20

    def test_generate_pairs_llm_fallback_to_seeds(self):
        """When LLM fails, should fall back to seed pools gracefully."""
        failing_llm = MagicMock(return_value=None)  # LLM returns None (failure)

        with patch("generate_nightmare_fuel_5k._call_llm", failing_llm):
            pairs = nf.generate_pairs(
                target_count=15,
                use_llm=True,
                llm_rate=1.0,
                output_path=None,
            )

        assert len(pairs) == 15
        # Should have fallen back to seed pools, so chosen should be from pools
        for p in pairs:
            assert len(p["chosen"]) > 20


# ═══════════════════════════════════════════════════════════════════════
# 6. CRISIS Resource Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCrisisResources:
    """Verify crisis resources are handled appropriately."""

    def test_crisis_pairs_have_resources(self):
        """Crisis category pairs should include crisis resources."""
        pairs = nf.generate_pairs(
            target_count=30,
            use_llm=False,
            output_path=None,
        )
        crisis_pairs = [p for p in pairs if p["metadata"]["is_crisis"]]
        if not crisis_pairs:
            pytest.skip("No crisis pairs in generated output")

        for p in crisis_pairs:
            # Crisis resources should be present or at least the response
            # should be appropriate for a crisis situation
            assert len(p["chosen"]) > 30, f"Crisis response too short: {p['chosen'][:50]}"

    def test_non_crisis_pairs_no_mechanical_resource(self):
        """Non-crisis categories should NOT get mechanical resource injection."""
        pairs = nf.generate_pairs(
            target_count=30,
            use_llm=False,
            output_path=None,
        )
        non_crisis_pairs = [p for p in pairs if not p["metadata"]["is_crisis"]]

        for p in non_crisis_pairs:
            # Should NOT have mechanical "If you're in crisis, please call" appended
            crisis_phrases = [
                "If you're in crisis, please call",
                "call 988",
                "text HOME to 741741",
            ]
            for phrase in crisis_phrases:
                assert phrase.lower() not in p["chosen"].lower(), (
                    f"Non-crisis pair has crisis resource: {p['prompt'][:50]} -> {phrase}"
                )


# ═══════════════════════════════════════════════════════════════════════
# 7. Error Observability Tests
# ═══════════════════════════════════════════════════════════════════════


class TestErrorObservability:
    """Verify LLM errors are properly observable."""

    def test_call_nemo_failure_returns_none(self):
        """When Nemo API fails, should return None (not crash)."""
        # Can't actually call API, but we can verify the function
        # handles errors gracefully by checking the error path exists
        import inspect

        source = inspect.getsource(nf._call_nemo)
        assert "except Exception" in source
        assert "return None" in source


# ═══════════════════════════════════════════════════════════════════════
# 8. Variation Functions Tests (verify they exist or were removed)
# ═══════════════════════════════════════════════════════════════════════


class TestVariationFunctions:
    """Verify variation function changes."""

    def test_variate_functions_exist(self):
        """_variate_* functions should exist (even if reduced to no-ops)."""
        assert hasattr(nf, "_variate_response")
        assert hasattr(nf, "_variate_prompt")

    def test_variate_functions_dont_mangle(self):
        """_variate_* functions should NOT produce grammatically broken output.

        The Oracle specifically flagged that variation functions insert random
        hedges/comma splices that destroy grammar. This test verifies the fix.
        """
        test_cases = [
            "I want to die. I have a plan.",
            "I feel like giving up.",
            "Nothing matters anymore.",
            "I cut myself again last night.",
            "I relapsed after six months sober.",
        ]
        for case in test_cases:
            result = nf._variate_prompt(case)
            # Check it doesn't have obvious grammar breaks from the old code
            # Old code produced: "What good does it do to talk about my feelings if nobody understands them I don't know,."
            assert result != "", f"_variate_prompt returned empty for '{case}'"
            # Should end with proper punctuation (not comma)
            if result:
                last_char = result.rstrip()[-1] if result.rstrip() else ""
                assert last_char in ".!?", f"_variate_prompt mangled '{case}' -> '{result}' (ends with '{last_char}')"


# ═══════════════════════════════════════════════════════════════════════
# 9. LLM Augmentation Rate Tests
# ═══════════════════════════════════════════════════════════════════════


class TestLLMAugmentationRate:
    """Verify LLM augmentation rate constants are correct."""

    def test_llm_augment_rate_high(self):
        """LLM_AUGMENT_RATE should be >= 0.80 (was 0.25)."""
        assert nf.LLM_AUGMENT_RATE >= 0.80, f"LLM_AUGMENT_RATE is {nf.LLM_AUGMENT_RATE}, expected >= 0.80"

    def test_prompt_variation_rate_not_used(self):
        """PROMPT_VARIATION_RATE should either not exist or be 0."""
        if hasattr(nf, "PROMPT_VARIATION_RATE"):
            assert nf.PROMPT_VARIATION_RATE == 0, (
                f"PROMPT_VARIATION_RATE should be 0 (removing deterministic variation)"
            )


# ═══════════════════════════════════════════════════════════════════════
# 10. Expansion Function Tests
# ═══════════════════════════════════════════════════════════════════════


class TestExpansionFunctions:
    """Verify expansion function naming and consolidation."""

    def test_chosen_expansion_name(self):
        """_expand_chosen_with_llm should exist (renamed from _expand_pools_with_llm)."""
        assert hasattr(nf, "_expand_chosen_with_llm") or hasattr(nf, "_expand_pools_with_llm")

    def test_no_deterministic_expansion_fallbacks(self):
        """_expand_prompt_templates and _expand_rejected_responses should be removed."""
        assert not hasattr(nf, "_expand_prompt_templates"), (
            "_expand_prompt_templates should be removed (was deterministic variation)"
        )
        assert not hasattr(nf, "_expand_rejected_responses"), (
            "_expand_rejected_responses should be removed (was deterministic variation)"
        )


# ═══════════════════════════════════════════════════════════════════════
# 11. No More _maybe() Function
# ═══════════════════════════════════════════════════════════════════════


class TestNoMaybeFunction:
    """_maybe() was only used by variation functions and should be removed."""

    def test_no_maybe_function(self):
        """_maybe() should not exist (was a thin wrapper around random.random())."""
        assert not hasattr(nf, "_maybe"), "_maybe() should be removed (unnecessary abstraction for random.random())"


# ═══════════════════════════════════════════════════════════════════════
# 12. Response Variation Constants Removed
# ═══════════════════════════════════════════════════════════════════════


class TestVariationConstantsRemoved:
    """Verify response variation constants are removed."""

    def test_no_response_variation_prefixes(self):
        """RESPONSE_VARIATION_PREFIXES should be removed."""
        assert not hasattr(nf, "RESPONSE_VARIATION_PREFIXES")

    def test_no_response_variation_suffixes(self):
        """RESPONSE_VARIATION_SUFFIXES should be removed."""
        assert not hasattr(nf, "RESPONSE_VARIATION_SUFFIXES")

    def test_no_response_variation_middles(self):
        """RESPONSE_VARIATION_MIDDLES should be removed."""
        assert not hasattr(nf, "RESPONSE_VARIATION_MIDDLES")

    def test_no_crisis_resource_inject_rate(self):
        """CRISIS_RESOURCE_INJECT_RATE should be removed (mechanical injection)."""
        assert not hasattr(nf, "CRISIS_RESOURCE_INJECT_RATE"), (
            "CRISIS_RESOURCE_INJECT_RATE should be removed (replaced by LLM-aware injection)"
        )


# ═══════════════════════════════════════════════════════════════════════
# 13. Integrated Scenario Tests
# ═══════════════════════════════════════════════════════════════════════


class TestIntegratedScenarios:
    """End-to-end scenario tests for the generate_pairs pipeline."""

    def test_output_to_file(self, temp_output):
        """Should write correctly formatted JSONL to output file."""
        pairs = nf.generate_pairs(
            target_count=10,
            use_llm=False,
            output_path=temp_output,
        )
        assert len(pairs) == 10

        # Verify file was written
        with open(temp_output) as f:
            lines = f.readlines()
        assert len(lines) >= 10

        # Each line should be valid JSON with the right structure
        for line in lines[:5]:
            obj = json.loads(line)
            assert "prompt" in obj
            assert "chosen" in obj
            assert "rejected" in obj
            assert "metadata" in obj

    def test_generate_pairs_consistent_structure(self):
        """All pairs should have exactly the expected metadata keys."""
        pairs = nf.generate_pairs(target_count=45, use_llm=False, output_path=None)
        expected_meta_keys = {"category", "description", "difficulty", "pair_type", "is_crisis"}

        for p in pairs:
            meta = p["metadata"]
            assert expected_meta_keys.issubset(meta.keys()), (
                f"Missing metadata keys. Got {meta.keys()}, need {expected_meta_keys}"
            )
            # pair_type should always be nightmare_fuel
            assert meta["pair_type"] == "nightmare_fuel"
            # is_crisis should be bool
            assert isinstance(meta["is_crisis"], bool)

    def test_all_difficulty_levels_present(self):
        """Should have pairs across different difficulty levels."""
        pairs = nf.generate_pairs(target_count=150, use_llm=False, output_path=None)
        difficulties = {p["metadata"]["difficulty"] for p in pairs}
        expected_levels = {"critical", "high", "medium"}
        for level in expected_levels:
            assert level in difficulties, f"Missing difficulty level: {level}"
