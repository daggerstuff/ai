import importlib

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from configs.api_authentication import AuthenticationSystem, PermissionLevel

TEST_SECRET = "test-secret-key-for-authentication-0123456789"
DEFAULT_SECRET = "your-secret-key-here"
SHORT_SECRET = "too-short"


@pytest.fixture
def dataset_api(tmp_path, monkeypatch):
    """Import the dataset API after auth and database environment is configured."""
    monkeypatch.setenv("AUTH_SECRET_KEY", TEST_SECRET)
    monkeypatch.setenv("DATABASE_URL", ":memory:")
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth.sqlite3"))

    dataset_api_module = importlib.import_module("inference.api.dataset_api")

    return importlib.reload(dataset_api_module)


def test_api_key_auth_with_bearer_header(dataset_api):
    """An API key alone authenticates even when clients also send a bearer header."""
    api_key, _ = dataset_api.auth_system.create_api_key("test", [PermissionLevel.READ])

    with TestClient(dataset_api.app) as client:
        response = client.get(
            "/datasets",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-API-Key": api_key,
            },
        )

    assert response.status_code == status.HTTP_200_OK


def test_api_key_auth_without_authorization_header(dataset_api):
    """An API key alone authenticates when no bearer header is present."""
    api_key, _ = dataset_api.auth_system.create_api_key("test", [PermissionLevel.READ])

    with TestClient(dataset_api.app) as client:
        response = client.get("/datasets", headers={"X-API-Key": api_key})

    assert response.status_code == status.HTTP_200_OK


def test_api_key_persistence_across_restart(tmp_path):
    """API keys stored in SQLite remain valid after a new AuthenticationSystem instance."""
    auth_db = tmp_path / "auth.sqlite3"
    first = AuthenticationSystem(TEST_SECRET, auth_database=auth_db)
    api_key, _ = first.create_api_key("persisted", [PermissionLevel.READ])

    restarted = AuthenticationSystem(TEST_SECRET, auth_database=auth_db)
    loaded_key = restarted.authenticate_api_key(api_key)

    assert loaded_key is not None
    assert loaded_key.name == "persisted"
    assert PermissionLevel.READ in loaded_key.permissions


@pytest.mark.parametrize("secret", [DEFAULT_SECRET, SHORT_SECRET])
def test_authentication_system_rejects_default_or_short_secret(secret):
    """Default placeholders and short secrets are rejected before storage is created."""
    with pytest.raises(ValueError, match="AUTH_SECRET_KEY"):
        AuthenticationSystem(secret)
