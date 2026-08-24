"""P0-1: Edge-case safety filter bypass.

Training records flagged ``is_training_edge_case=True`` carry deliberate
clinical edge-case signal (crisis scenarios, high-risk content) that the
safety filter currently strips. This destroys training value per
``hidden1-gap-analysis.md`` (P0 critical training gap).

Test asserts: ``EnhancedSafetyFilter.check_output_safety`` passes
edge-case content through unchanged when ``request_metadata`` carries
``is_training_edge_case: True``.
"""

from __future__ import annotations

from ai.qa.validation.enhanced_safety_filter import (
    EnhancedSafetyFilter,
    SafetyCategory,
    SafetyLevel,
)


def _edge_case_content() -> str:
    """High-crisis content that safety filter would normally strip."""
    return "I want to kill myself right now. I have a gun and I am going to shoot myself in the head."


def test_edge_case_record_passes_through_unfiltered() -> None:
    """Edge-case training record must survive safety filtering unchanged."""
    filter_ = EnhancedSafetyFilter(safety_level=SafetyLevel.STRICT)
    content = _edge_case_content()
    metadata = {
        "is_training_edge_case": True,
        "crisis_intensity": "extreme",
        "edge_category": "suicidal_ideation",
    }

    result = filter_.check_output_safety(
        content=content,
        _user_context={"role": "training"},
        _request_metadata=metadata,
    )

    # The load-bearing assertion: content not modified and not flagged.
    assert result.is_safe is True
    assert result.filtered_content == content or result.filtered_content is None
    assert SafetyCategory.CRISIS not in result.flagged_categories
