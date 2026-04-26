import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.dataset_api import app, PermissionLevel, get_current_active_user_or_api_key

client = TestClient(app)

@patch("api.dataset_api.get_db_connection")
def test_unauthorized_access(mock_db):
    response = client.get("/datasets", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401

@patch("api.dataset_api.get_db_connection")
def test_authorized_but_insufficient_permissions(mock_db):
    app.dependency_overrides[get_current_active_user_or_api_key] = lambda: {"username": "test", "scopes": [PermissionLevel.WRITE], "auth_type": "api_key"}

    response = client.get("/datasets")
    assert response.status_code == 403
    app.dependency_overrides.clear()
