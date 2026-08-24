import os

import pytest
from fastapi import HTTPException

# Set the secret key before importing from the API
os.environ["AUTH_SECRET_KEY"] = "test-secret-key"

from inference.api.dataset_api import validate_identifier  # noqa: I001

def test_validate_identifier_valid():
    assert validate_identifier("valid_id_123") == "valid_id_123"
    assert validate_identifier("VALID_123") == "VALID_123"
    assert validate_identifier("valid") == "valid"
    assert validate_identifier("_valid_") == "_valid_"
    assert validate_identifier("12345") == "12345"

def test_validate_identifier_invalid():
    invalid_ids = [
        "",
        "invalid id",
        "invalid-id",
        "invalid@id",
        "id_with_'",
        "id;",
        "SELECT *",
        "DROP TABLE",
    ]

    for invalid_id in invalid_ids:
        with pytest.raises(HTTPException) as exc_info:
            validate_identifier(invalid_id)

        assert exc_info.value.status_code == 400  # noqa: PLR2004
        assert exc_info.value.detail == f"Invalid identifier format: {invalid_id}"
