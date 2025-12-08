import logging
import unicodedata

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Standardizes text encoding, whitespace, and structural formatting.
    """

    def normalize_text(self, text: str) -> str:
        """
        Normalizes unicode characters and whitespace.
        """
        if not text or not isinstance(text, str):
            return text

        # Normalize unicode to NFKD form (compatibility decomposition) to separate accents
        # Then encode to ASCII (ignoring errors) and decode back to remove non-ascii chars if desired,
        # OR just keep it utf-8 but standardized. Let's stick to standardizing unicode.
        normalized = unicodedata.normalize("NFKC", text)

        # Replace non-breaking spaces and other whitespace variations with a single space
        normalized = " ".join(normalized.split())

        return normalized.strip()

    def standardize_keys(self, dataset: list[dict]) -> list[dict]:
        """
        Ensures consistent key naming (e.g., lower_snake_case).
        """
        logger.info("Normalizing data keys...")
        normalized_dataset = []

        for entry in dataset:
            new_entry = {}
            for key, value in entry.items():
                clean_key = key.strip().lower().replace(" ", "_").replace("-", "_")

                # Recursively normalize string values while we are at it
                if isinstance(value, str):
                    new_entry[clean_key] = self.normalize_text(value)
                else:
                    new_entry[clean_key] = value

            normalized_dataset.append(new_entry)

        return normalized_dataset
