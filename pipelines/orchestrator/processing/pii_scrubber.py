import logging
import re

logger = logging.getLogger(__name__)


class PIIScrubber:
    """
    Identifies and scrubs Personally Identifiable Information (PII) from text.
    Handles regex-based detection for names, emails, phone numbers, etc.
    """

    def __init__(self):
        # Compiled regex patterns for performance
        self.patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"\b(?:\+?1[-.]?)?(?:\(?\d{3}\)?[-.]?)?\d{3}[-.]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            # A very basic name pattern - heavily reliant on context or capitalization in real scenarios
            # This is a placeholder for more advanced NER-based name detection if needed later.
            # "name": re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b')
        }

    def scrub_text(self, text: str) -> str:
        """
        Redacts PII from the input text string.
        """
        if not text or not isinstance(text, str):
            return text

        scrubbed_text = text
        for pii_type, pattern in self.patterns.items():
            scrubbed_text = pattern.sub(f"[{pii_type.upper()}_REDACTED]", scrubbed_text)

        return scrubbed_text

    def scrub_dataset(self, dataset: list[dict], fields_to_scrub: list[str] | None = None) -> list[dict]:
        """
        Iterates through a dataset and scrubs PII from specified fields.
        If no fields specified, attempts to scrub all string fields.
        """
        logger.info(f"Starting PII scrub on {len(dataset)} records...")
        cleaned_data = []

        for entry in dataset:
            clean_entry = entry.copy()

            # If specific fields are provided, scrub only those
            if fields_to_scrub:
                for field in fields_to_scrub:
                    if field in clean_entry and isinstance(clean_entry[field], str):
                        clean_entry[field] = self.scrub_text(clean_entry[field])

            # Otherwise scrub all string values
            else:
                for key, value in clean_entry.items():
                    if isinstance(value, str):
                        clean_entry[key] = self.scrub_text(value)

            cleaned_data.append(clean_entry)

        logger.info("PII scrubbing completed.")
        return cleaned_data
