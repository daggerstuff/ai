import importlib
import os

os.environ.setdefault("AUTH_SECRET_KEY", "test-secret-key-for-authentication-0123456789")
os.environ.setdefault("DATABASE_URL", ":memory:")
import os

from fastapi.testclient import TestClient

from configs.api_authentication import PermissionLevel
from inference.api.dataset_api import app as dataset_app

def dataset_api(tmp_path, monkeypatch):
    """Import the API with isolated auth and database state."""
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key-for-authentication-0123456789")
    monkeypatch.setenv("DATABASE_URL", ":memory:")
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth.sqlite3"))
    from inference.api import dataset_api

    return importlib.reload(dataset_api)


def test_api_key_auth_with_bearer_header(tmp_path, monkeypatch):
    """An API key alone authenticates even when clients also send a bearer header."""
    module = dataset_api(tmp_path, monkeypatch)
    api_key, _ = module.auth_system.create_api_key("test", [PermissionLevel.READ])

    with TestClient(module.app) as client:
        response = client.get(
            "/datasets",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-API-Key": api_key,
            },
        )

    assert response.status_code != 401


def test_api_key_auth_without_authorization_header(tmp_path, monkeypatch):
    """An API key alone authenticates when no bearer header is present."""
    module = dataset_api(tmp_path, monkeypatch)
    api_key, _ = module.auth_system.create_api_key("test", [PermissionLevel.READ])

    with TestClient(module.app) as client:
        response = client.get("/datasets", headers={"X-API-Key": api_key})

    assert response.status_code != 401
