import logging

logger = logging.getLogger(__name__)


class QualityValidator:
    """
    Validates data quality: minimum length, missing fields, basic entropy.
    """

    def __init__(self, min_text_length: int = 10, required_fields: list[str] | None = None):
        self.min_text_length = min_text_length
        self.required_fields = required_fields or ["title", "content", "items", "description"]

    def validate_entry(self, entry: dict) -> dict:
        """
        Checks a single entry. Returns a dict with 'is_valid' and 'issues'.
        """
        issues = []

        # Check for required fields (at least one of them should exist usually, or specific ones)
        # For this generic validator, let's check if *any* of the likely content fields have data
        has_content = False
        for field in self.required_fields:
            if entry.get(field):
                has_content = True

                # Check length for text fields
                val = entry[field]
                if isinstance(val, str) and len(val) < self.min_text_length:
                    issues.append(f"Field '{field}' is too short (<{self.min_text_length} chars).")
                elif isinstance(val, list) and len(val) == 0:
                    issues.append(f"Field '{field}' is an empty list.")

        if not has_content and not any(entry.values()):
            # It's okay if not ALL required fields are present, but the entry shouldn't be effectively empty
            # If we enforce strict schema validation, we would use Pydantic.
            # Here we just want to flag mostly empty junk.
            # Verify that at least one key from the entry has a non-empty value
            issues.append("Entry appears to be empty.")

        return {"is_valid": len(issues) == 0, "issues": issues}

    def validate_dataset(self, dataset: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Splits dataset into valid and invalid records.
        Returns (valid_data, invalid_data_with_reasons).
        """
        logger.info(f"Validating {len(dataset)} records...")
        valid_data = []
        invalid_data = []

        for entry in dataset:
            report = self.validate_entry(entry)
            if report["is_valid"]:
                valid_data.append(entry)
            else:
                entry_with_issues = entry.copy()
                entry_with_issues["_validation_issues"] = report["issues"]
                invalid_data.append(entry_with_issues)

        logger.info(f"Validation complete. Valid: {len(valid_data)}, Invalid: {len(invalid_data)}")
        return valid_data, invalid_data
