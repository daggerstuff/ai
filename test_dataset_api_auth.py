import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.dataset_api import app, PermissionLevel, get_current_active_user_or_api_key

client = TestClient(app)


@pytest.fixture
def auth_override():
    """Fixture that ensures dependency override cleanup happens regardless of test outcome."""
    yield
    # Cleanup happens in finally to ensure it runs even if test fails
    finally:
        app.dependency_overrides.clear()


@patch("api.dataset_api.get_db_connection")
def test_unauthorized_access(mock_db):
    response = client.get("/datasets", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401


@patch("api.dataset_api.get_db_connection")
def test_authorized_but_insufficient_permissions(mock_db, auth_override):
    app.dependency_overrides[get_current_active_user_or_api_key] = lambda: {"username": "test", "scopes": [PermissionLevel.WRITE], "auth_type": "api_key"}

    response = client.get("/datasets")
    assert response.status_code == 403


@patch("api.dataset_api.get_db_connection")
def test_authorized_with_read_permission(mock_db, auth_override):
    app.dependency_overrides[get_current_active_user_or_api_key] = lambda: {"username": "test", "scopes": [PermissionLevel.READ], "auth_type": "api_key"}

    response = client.get("/datasets")
    assert response.status_code == 200


@patch("api.dataset_api.get_db_connection")
def test_unauthorized_access_metadata(mock_db):
    response = client.get("/datasets/test_dataset/metadata", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401


@patch("api.dataset_api.get_db_connection")
def test_authorized_but_insufficient_permissions_metadata(mock_db, auth_override):
    app.dependency_overrides[get_current_active_user_or_api_key] = lambda: {"username": "test", "scopes": [PermissionLevel.WRITE], "auth_type": "api_key"}

    response = client.get("/datasets/test_dataset/metadata")
    assert response.status_code == 403


@patch("api.dataset_api.get_db_connection")
def test_authorized_with_read_permission_metadata(mock_db, auth_override):
    app.dependency_overrides[get_current_active_user_or_api_key] = lambda: {"username": "test", "scopes": [PermissionLevel.READ], "auth_type": "api_key"}

    response = client.get("/datasets/test_dataset/metadata")
    assert response.status_code == 200


@patch("api.dataset_api.get_db_connection")
def test_unauthorized_access_query(mock_db):
    response = client.get("/datasets/query", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401


@patch("api.dataset_api.get_db_connection")
def test_authorized_but_insufficient_permissions_query(mock_db, auth_override):
    app.dependency_overrides[get_current_active_user_or_api_key] = lambda: {"username": "test", "scopes": [PermissionLevel.WRITE], "auth_type": "api_key"}

    response = client.get("/datasets/query")
    assert response.status_code == 403


@patch("api.dataset_api.get_db_connection")
def test_authorized_with_read_permission_query(mock_db, auth_override):
    app.dependency_overrides[get_current_active_user_or_api_key] = lambda: {"username": "test", "scopes": [PermissionLevel.READ], "auth_type": "api_key"}

    response = client.get("/datasets/query")
    assert response.status_code == 200