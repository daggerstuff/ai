import logging
import unittest

import pandas as pd
from ai.dataset_pipeline.processing.clean import (
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
        self.assertIn("user_email", pii_cols)
        self.assertIn("phone_number", pii_cols)
        self.assertIn("ssn_field", pii_cols)
        self.assertNotIn("text", pii_cols)

    def test_normalize_text(self):
        """Test whitespace stripping and lowercasing."""
        df = pd.DataFrame({"msg": ["  Mixed CASE  ", "\tTabs\nNewline  "]})
        normalized = normalize_text_columns(df, ["msg"])
        self.assertEqual(normalized["msg"][0], "mixed case")
        self.assertEqual(normalized["msg"][1], "tabs newline")

    def test_pii_redaction(self):
        """Test regex-based redaction of SSNs in text content."""
        df = pd.DataFrame({"content": ["My SSN is 123-45-6789", "Just a number 12345"]})
        redacted = redact_pii_in_text_fields(df, ["content"])
        self.assertIn("[REDACTED-SSN]", redacted["content"][0])
        self.assertNotIn("123-45-6789", redacted["content"][0])
        self.assertEqual(redacted["content"][1], "Just a number 12345")

    def test_pii_removal(self):
        """Test dropping of identified PII columns."""
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "ssn": ["secret"]})
        pii_cols = {"name", "ssn"}
        cleaned = remove_pii(df, pii_cols, self.logger)
        self.assertNotIn("name", cleaned.columns)
        self.assertNotIn("ssn", cleaned.columns)
        self.assertIn("id", cleaned.columns)

    def test_full_clean_and_deduplicate(self):
        """Test end-to-end cleaning and deduplication logic."""
        df1 = pd.DataFrame(
            {"text": ["hello", "world"], "email": ["a@b.com", "c@d.com"]}
        )
        df2 = pd.DataFrame(
            {"text": ["hello", "third"], "email": ["a@b.com", "e@f.com"]}
        )

        config = {"dedup_columns": ["text"], "required_columns": ["text"]}

        result = clean_and_deduplicate([df1, df2], config=config)

        # 'hello' is duplicated across df1 and df2, should have 3 unique rows: hello, world, third
        self.assertEqual(len(result), 3)
        self.assertNotIn("email", result.columns)  # email matches pii pattern 'email'
        self.assertTrue(all(col in result.columns for col in ["text"]))


if __name__ == "__main__":
    unittest.main()
