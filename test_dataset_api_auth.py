import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.dataset_api import app, PermissionLevel, get_current_active_user_or_api_key

client = TestClient(app)

@patch("api.dataset_api.get_db_connection")
def test_unauthorized_access(mock_db):
    response = client.get("/datasets", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401

    response_metadata = client.get("/datasets/test_dataset/metadata", headers={"Authorization": "Bearer invalid_token"})
    assert response_metadata.status_code == 401

    response_query = client.post("/datasets/test_dataset/query", headers={"Authorization": "Bearer invalid_token"})
    assert response_query.status_code == 401

@patch("api.dataset_api.get_db_connection")
def test_authorized_but_insufficient_permissions(mock_db):
    app.dependency_overrides[get_current_active_user_or_api_key] = lambda: {"username": "test", "scopes": [PermissionLevel.WRITE], "auth_type": "api_key"}

    response = client.get("/datasets")
    assert response.status_code == 403

    response_metadata = client.get("/datasets/test_dataset/metadata")
    assert response_metadata.status_code == 403

    response_query = client.post("/datasets/test_dataset/query")
    assert response_query.status_code == 403

    app.dependency_overrides.clear()

@patch("api.dataset_api.get_db_connection")
def test_authorized_with_sufficient_permissions(mock_db):
    # Mock the database cursor properly to avoid returning None and causing TypeError when unpacking
    mock_cursor = mock_db.return_value.cursor.return_value
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = [0]

    app.dependency_overrides[get_current_active_user_or_api_key] = lambda: {"username": "test", "scopes": [PermissionLevel.READ], "auth_type": "api_key"}

    response = client.get("/datasets")
    assert response.status_code == 200

    # Needs to return a row-like object that can be indexed both by name and by integer
    class MockRow:
        def __init__(self, data, list_data=None):
            self.data = data
            self.list_data = list_data or [0]
        def __getitem__(self, key):
            if isinstance(key, int):
                return self.list_data[key]
            return self.data.get(key)

    mock_cursor.fetchone.side_effect = [MockRow({"name": "test_dataset"}), MockRow({}, [0])]
    response_metadata = client.get("/datasets/test_dataset/metadata")
    assert response_metadata.status_code == 200

    mock_cursor.fetchone.side_effect = [MockRow({"name": "test_dataset"}), MockRow({}, [0])]
    response_query = client.post("/datasets/test_dataset/query")
    assert response_query.status_code == 200

    app.dependency_overrides.clear()
