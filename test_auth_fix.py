import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from api.dataset_api import (
    get_current_active_user_or_api_key,
    list_datasets,
    get_dataset_metadata,
    query_dataset,
    app,
    PermissionLevel,
)

client = TestClient(app)


def _override_auth(permission_level: PermissionLevel):
    """Return a dependency override for get_current_active_user_or_api_key that yields a user/api-key object with the given permission level."""

    def _dependency_override():
        mock_user = MagicMock()
        mock_user.permission_level = permission_level
        return mock_user

    return _dependency_override


def test_unauthorized_access_to_datasets_returns_error():
    """ Basic authorization behavior: accessing datasets without auth should not succeed (expect 401 or 403). """
    response = client.get("/datasets")
    assert response.status_code in (401, 403)


def test_unauthorized_get_dataset_metadata_returns_error():
    """ Basic authorization behavior: accessing dataset metadata without auth should not succeed (expect 401 or 403). """
    response = client.get("/datasets/test-dataset/metadata")
    assert response.status_code in (401, 403)


def test_unauthorized_query_dataset_returns_error():
    """ Basic authorization behavior: querying a dataset without auth should not succeed (expect 401 or 403). """
    payload = {
        "dataset_id": "dummy-id",
        "query": "SELECT 1",
    }
    response = client.post("/datasets/test-dataset/query", json=payload)
    assert response.status_code in (401, 403)


class TestDatasetEndpointsWithoutReadPermission:
    """Tests for dataset endpoints when user lacks READ permission."""

    def test_list_datasets_returns_403_without_read_permission(self):
        """GET /datasets should return 403 when user lacks READ permission."""
        app.dependency_overrides[get_current_active_user_or_api_key] = _override_auth(
            permission_level=PermissionLevel.WRITE
        )
        try:
            response = client.get("/datasets")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 403

    def test_get_dataset_metadata_returns_403_without_read_permission(self):
        """GET /datasets/{id}/metadata should return 403 when user lacks READ permission."""
        app.dependency_overrides[get_current_active_user_or_api_key] = _override_auth(
            permission_level=PermissionLevel.WRITE
        )
        try:
            response = client.get("/datasets/test-dataset/metadata")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 403

    def test_query_dataset_returns_403_without_read_permission(self):
        """POST /datasets/{id}/query should return 403 when user lacks READ permission."""
        app.dependency_overrides[get_current_active_user_or_api_key] = _override_auth(
            permission_level=PermissionLevel.WRITE
        )
        try:
            response = client.post("/datasets/test-dataset/query", json={})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 403


class TestDatasetEndpointsWithReadPermission:
    """Tests for dataset endpoints when user has READ permission."""

    def test_list_datasets_returns_200_with_read_permission(self):
        """GET /datasets should return 200 when user has READ permission."""
        app.dependency_overrides[get_current_active_user_or_api_key] = _override_auth(
            permission_level=PermissionLevel.READ
        )
        try:
            response = client.get("/datasets")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200

    def test_get_dataset_metadata_returns_200_with_read_permission(self):
        """GET /datasets/{id}/metadata should return 200 when user has READ permission."""
        app.dependency_overrides[get_current_active_user_or_api_key] = _override_auth(
            permission_level=PermissionLevel.READ
        )
        try:
            response = client.get("/datasets/test-dataset/metadata")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200

    def test_query_dataset_returns_200_with_read_permission(self):
        """POST /datasets/{id}/query should return 200 when user has READ permission."""
        app.dependency_overrides[get_current_active_user_or_api_key] = _override_auth(
            permission_level=PermissionLevel.READ
        )
        try:
            response = client.post("/datasets/test-dataset/query", json={})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200


@pytest.mark.parametrize(
    "method,path,permission_level,expected_status",
    [
        # Permitted cases - should return 200
        ("GET", "/datasets", PermissionLevel.READ, 200),
        ("GET", "/datasets/test-dataset/metadata", PermissionLevel.READ, 200),
        ("POST", "/datasets/test-dataset/query", PermissionLevel.READ, 200),
        # Forbidden cases - should return 403
        ("GET", "/datasets", PermissionLevel.WRITE, 403),
        ("GET", "/datasets/test-dataset/metadata", PermissionLevel.WRITE, 403),
        ("POST", "/datasets/test-dataset/query", PermissionLevel.WRITE, 403),
    ],
)
def test_dataset_endpoints_parametrized(method, path, permission_level, expected_status):
    """ Parametrized test for all dataset endpoints with both permitted and forbidden permission levels. This verifies that the auth guard is correctly wired through the routing layer. """
    app.dependency_overrides[get_current_active_user_or_api_key] = _override_auth(
        permission_level=permission_level
    )
    try:
        response = client.request(
            method, path, json={"query": "SELECT 1"} if method == "POST" else None
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == expected_status