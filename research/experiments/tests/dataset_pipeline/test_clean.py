"""
Test suite for ai/dataset_pipeline/clean.py

Covers:
- PII detection/removal
- Normalization (text, whitespace, casing)
- Deduplication (exact, near-duplicate)
- Error handling (bad input, empty, malformed)
- Compliance edge cases (privacy, audit)
- No secrets or env vars hardcoded

TDD anchors: Red phase (tests must fail until implemented)
"""

from collections.abc import Callable
from unittest import mock

import pandas as pd
import pytest

from ai.pkg_mera.core.pipelines.processing import clean

# pytest ships no type stubs in this checking environment; bind the fixture
# decorator to a typed alias so decorated fixtures satisfy mypy --strict.
_FixtureFn = Callable[[], list[dict[str, str]]]
_typed_fixture: Callable[[_FixtureFn], _FixtureFn] = pytest.fixture


@_typed_fixture
def sample_data() -> list[dict[str, str]]:
    return [
        {"text": "Hello world", "user": "alice", "email": "alice@example.com"},
        {"text": "hello  world ", "user": "alice", "email": "alice@example.com"},
        {"text": "Goodbye", "user": "bob", "email": "bob@example.com"},
        {"text": "Hello world", "user": "alice", "email": "alice@example.com"},
        {
            "text": "Sensitive info: 555-12-3456",
            "user": "eve",
            "email": "eve@example.com",
        },
    ]


def test_remove_pii_fields(sample_data: list[dict[str, str]]) -> None:
    df = pd.DataFrame(sample_data)
    cleaned = clean.clean_and_deduplicate(df)
    # Should remove email and any PII fields
    assert "email" not in cleaned.columns
    # Should redact or remove PII in text
    assert not cleaned["text"].str.contains(r"\d{3}-\d{2}-\d{4}").any()


def test_normalization_whitespace_and_case(sample_data: list[dict[str, str]]) -> None:
    df = pd.DataFrame(sample_data)
    cleaned = clean.clean_and_deduplicate(df)
    # All text should be normalized (lowercase, single space)
    assert all("  " not in t for t in cleaned["text"])
    assert all(t == t.strip() for t in cleaned["text"])
    assert all(t == t.lower() for t in cleaned["text"])


def test_deduplication_exact_and_near(sample_data: list[dict[str, str]]) -> None:
    df = pd.DataFrame(sample_data)
    cleaned = clean.clean_and_deduplicate(df)
    # Should deduplicate exact and near-duplicate rows
    texts = cleaned["text"].tolist()
    assert texts.count("hello world") == 1
    assert len(cleaned) < len(df)


def test_empty_dataframe_returns_empty() -> None:
    df = pd.DataFrame(columns=["text", "user"])
    cleaned = clean.clean_and_deduplicate(df)
    assert cleaned.empty


def test_malformed_input_raises() -> None:
    # Not a DataFrame
    with pytest.raises(ValueError):
        clean.clean_and_deduplicate([{"text": "foo"}])
    # Missing required columns
    df = pd.DataFrame([{"foo": "bar"}])
    with pytest.raises(ValueError):
        clean.clean_and_deduplicate(df)


def test_privacy_audit_logging(sample_data: list[dict[str, str]]) -> None:
    df = pd.DataFrame(sample_data)
    with mock.patch.object(clean.logger, "info") as info_mock:
        clean.clean_and_deduplicate(df)
        assert any("privacy" in str(call) or "audit" in str(call) for call in info_mock.call_args_list)


def test_compliance_edge_cases() -> None:
    # PII in unexpected columns
    df = pd.DataFrame([{"text": "ok", "notes": "ssn: 123-45-6789"}])
    cleaned = clean.clean_and_deduplicate(df)
    assert not cleaned["notes"].str.contains(r"\d{3}-\d{2}-\d{4}").any()


# TDD anchor: Add more tests for unicode normalization, emoji, and rare PII patterns if needed.
