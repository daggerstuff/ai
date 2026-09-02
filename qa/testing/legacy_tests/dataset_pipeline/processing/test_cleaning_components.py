import logging
import unittest

import pandas as pd

from ai.tools.utilities.pipelines.processing.clean import (
    clean_and_deduplicate,
    find_pii_columns,
    normalize_text_columns,
    redact_pii_in_text_fields,
    remove_pii,
)


class TestCleaningComponents(unittest.TestCase):
    """
    Test suite for Task 5.7.1.1: Component Unit Tests (Cleaning focus)
    Validates PII detection, text normalization, and deduplication.
    """

    def setUp(self):
        """Set up test data."""
        self.sample_df = pd.DataFrame(
            {
                "text": ["  Hello World  ", "Contact me at 555-010-9999", "Duplicated"],
                "user_email": [
                    "test@example.com",
                    "other@example.com",
                    "test@example.com",
                ],
                "secret_ssn": ["123-45-6789", "None", "123-45-6789"],
            }
        )
        self.logger = logging.getLogger("test_cleaning")

    def test_find_pii_columns(self):
        """Test identification of PII columns based on names."""
        cols = ["id", "text", "user_email", "phone_number", "ssn_field"]
        pii_cols = find_pii_columns(cols)
        assert "user_email" in pii_cols
        assert "phone_number" in pii_cols
        assert "ssn_field" in pii_cols
        assert "text" not in pii_cols

    def test_normalize_text(self):
        """Test whitespace stripping and lowercasing."""
        df = pd.DataFrame({"msg": ["  Mixed CASE  ", "\tTabs\nNewline  "]})
        normalized = normalize_text_columns(df, ["msg"])
        assert normalized["msg"][0] == "mixed case"
        assert normalized["msg"][1] == "tabs newline"

    def test_pii_redaction(self):
        """Test regex-based redaction of SSNs in text content."""
        df = pd.DataFrame({"content": ["My SSN is 123-45-6789", "Just a number 12345"]})
        redacted = redact_pii_in_text_fields(df, ["content"])
        assert "[REDACTED-SSN]" in redacted["content"][0]
        assert "123-45-6789" not in redacted["content"][0]
        assert redacted["content"][1] == "Just a number 12345"

    def test_pii_removal(self):
        """Test dropping of identified PII columns."""
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "ssn": ["secret"]})
        pii_cols = {"name", "ssn"}
        cleaned = remove_pii(df, pii_cols, self.logger)
        assert "name" not in cleaned.columns
        assert "ssn" not in cleaned.columns
        assert "id" in cleaned.columns

    def test_full_clean_and_deduplicate(self):
        """Test end-to-end cleaning and deduplication logic."""
        df1 = pd.DataFrame({"text": ["hello", "world"], "email": ["a@b.com", "c@d.com"]})
        df2 = pd.DataFrame({"text": ["hello", "third"], "email": ["a@b.com", "e@f.com"]})

        config = {"dedup_columns": ["text"], "required_columns": ["text"]}

        result = clean_and_deduplicate([df1, df2], config=config)

        # 'hello' is duplicated across df1 and df2, should have 3 unique rows: hello, world, third
        assert len(result) == 3
        assert "email" not in result.columns  # email matches pii pattern 'email'
        assert all(col in result.columns for col in ["text"])


if __name__ == "__main__":
    unittest.main()
