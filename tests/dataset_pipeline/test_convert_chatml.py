# test_convert_chatml.py
# TDD: ChatML conversion pipeline tests for Pixelated Empathy
from unittest.mock import patch

import pandas as pd
import pytest

# TDD Anchor: Import target function (to be implemented)
try:
    from ai.pipelines.orchestrator.processing.convert_chatml import convert_to_chatml
except ImportError:
    convert_to_chatml = None


@pytest.mark.skipif(
    convert_to_chatml is None, reason="convert_to_chatml not implemented"
)
class TestChatMLConversion:
    def test_basic_conversion(self):
        """
        Given a DataFrame with 'prompt' and 'response', when converting,
        then output is valid ChatML format.
        """
        df = pd.DataFrame([{"prompt": "Hello", "response": "Hi there!"}])
        chatml = convert_to_chatml(df)
        assert isinstance(chatml, list)
        assert chatml[0]["role"] == "user"
        assert chatml[0]["content"] == "Hello"
        assert chatml[1]["role"] == "assistant"
        assert chatml[1]["content"] == "Hi there!"

    def test_empty_dataframe(self):
        """Given an empty DataFrame, when converting, then output is an empty list."""
        df = pd.DataFrame(columns=["prompt", "response"])
        chatml = convert_to_chatml(df)
        assert chatml == []

    def test_missing_columns_raises(self):
        """
        Given a DataFrame missing required columns, when converting,
        then ValueError is raised.
        """
        df = pd.DataFrame([{"input": "foo"}])
        with pytest.raises(ValueError):
            convert_to_chatml(df)

    def test_multilingual_content(self):
        """
        Given multilingual prompts/responses, when converting,
        then output preserves content.
        """
        df = pd.DataFrame(
            [
                {"prompt": "你好", "response": "Hello!"},
                {"prompt": "¿Cómo estás?", "response": "¡Bien!"},
            ]
        )
        chatml = convert_to_chatml(df)
        assert chatml[0]["content"] == "你好"
        assert chatml[2]["content"] == "¿Cómo estás?"

    def test_privacy_redaction_integration(self):
        """
        Given PII in prompts/responses, when converting,
        then redacted tokens are present.
        """
        df = pd.DataFrame(
            [{"prompt": "My email is john@example.com", "response": "Thanks, John."}]
        )
        redacted_df = df.copy()
        redacted_df["prompt"] = "My email is [REDACTED]"
        with patch(
            "ai.pipelines.orchestrator.processing.clean.redact_pii_in_text_fields",
            return_value=redacted_df,
        ):
            chatml = convert_to_chatml(df)
            assert any("[REDACTED" in m["content"] for m in chatml)

    def test_malformed_rows_skipped(self):
        """
        Given rows with null/NaN or non-string, when converting,
        then only valid rows are included.
        """
        df = pd.DataFrame(
            [
                {"prompt": "Hi", "response": "Hello"},
                {"prompt": None, "response": "Missing prompt"},
                {"prompt": "Missing response", "response": None},
                {"prompt": 123, "response": "Numeric prompt"},
            ]
        )
        chatml = convert_to_chatml(df)
        assert all(isinstance(m["content"], str) for m in chatml)
        assert all(m["content"] not in [None, 123] for m in chatml)

    def test_compliance_structure(self):
        """
        Given valid input, when converting, then output alternates
        user/assistant and is ChatML-compliant.
        """
        df = pd.DataFrame(
            [{"prompt": "A", "response": "B"}, {"prompt": "C", "response": "D"}]
        )
        chatml = convert_to_chatml(df)
        roles = [m["role"] for m in chatml]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert all("content" in m for m in chatml)
