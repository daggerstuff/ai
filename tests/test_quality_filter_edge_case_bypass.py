"""P0-1: Edge-case safety filter bypass for QualityFilter.

Training records flagged ``is_training_edge_case=True`` carry deliberate
clinical edge-case signal that the quality filter would otherwise strip
via deduplication. This destroys training value per the master plan.

Test asserts: ``QualityFilter.passes_filter`` passes edge-case content
through unchanged when record carries ``is_training_edge_case: True``.
"""

from __future__ import annotations

from ai.dataset_pipeline.processors.quality_filter import QualityFilter


def _edge_case_record() -> dict:
    """High-crisis content that quality filter would normally deduplicate."""
    return {
        "is_training_edge_case": True,
        "messages": [
            {"role": "system", "content": "You are a crisis therapist"},
            {"role": "user", "content": "I want to kill myself right now. I have a gun."},
            {"role": "assistant", "content": "I am deeply concerned about your safety. Please call 988."},
        ],
    }


def _normal_record() -> dict:
    """Normal therapeutic record."""
    return {
        "messages": [
            {"role": "system", "content": "You are a therapist"},
            {"role": "user", "content": "I am feeling anxious today"},
            {"role": "assistant", "content": "I hear you. Tell me more about that anxiety."},
        ],
    }


def test_quality_filter_edge_case_bypass() -> None:
    """Edge-case training record must survive quality filtering (no deduplication)."""
    filter_ = QualityFilter()
    record = _edge_case_record()

    # First pass
    assert filter_.passes_filter(record) is True

    # Second pass with SAME content - should NOT be deduplicated for edge cases
    assert filter_.passes_filter(record) is True


def test_quality_filter_normal_record_deduplicated() -> None:
    """Normal records should still be deduplicated."""
    filter_ = QualityFilter()
    record = _normal_record()

    # First pass
    assert filter_.passes_filter(record) is True

    # Second pass with SAME content - SHOULD be deduplicated
    assert filter_.passes_filter(record) is False


def test_quality_filter_edge_case_still_validates_format() -> None:
    """Edge cases still must pass format validation (length, roles, non-empty)."""
    filter_ = QualityFilter()

    # Too few messages
    bad_record = {
        "is_training_edge_case": True,
        "messages": [
            {"role": "user", "content": "hi"},
        ],
    }
    assert filter_.passes_filter(bad_record) is False

    # Empty content
    empty_record = {
        "is_training_edge_case": True,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "response"},
        ],
    }
    assert filter_.passes_filter(empty_record) is False

    # Same role consecutive
    role_record = {
        "is_training_edge_case": True,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "again"},
        ],
    }
    assert filter_.passes_filter(role_record) is False
