import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.dataset_api import app, PermissionLevel, get_current_active_user_or_api_key

client = TestClient(app)


@pytest.fixture
def override_active_user():
    """Fixture to override get_current_active_user_or_api_key with given scopes.

    Usage in tests:
        def test_something(override_active_user):
            override_active_user([PermissionLevel.WRITE])
            ...
    """
    def _override(scopes):
        app.dependency_overrides[get_current_active_user_or_api_key] = (
            lambda: {"username": "test", "scopes": scopes, "auth_type": "api_key"}
        )

    yield _override

    # Ensure overrides are always cleared after the test, even on failure
    app.dependency_overrides.clear()


@patch("api.dataset_api.get_db_connection")
def test_unauthorized_access(mock_db):
    response = client.get("/datasets", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401

    response_metadata = client.get(
        "/datasets/test_dataset/metadata",
        headers={"Authorization": "Bearer invalid_token"},
    )
    assert response_metadata.status_code == 401

    response_query = client.post(
        "/datasets/test_dataset/query",
        headers={"Authorization": "Bearer invalid_token"},
    )
    assert response_query.status_code == 401


@patch("api.dataset_api.get_db_connection")
def test_no_authorization_header(mock_db):
    """Test that requests without any Authorization header also return 401."""
    # Test /datasets endpoint without Authorization header
    response = client.get("/datasets")
    assert response.status_code == 401

    # Test /datasets/test_dataset/metadata endpoint without Authorization header
    response_metadata = client.get("/datasets/test_dataset/metadata")
    assert response_metadata.status_code == 401

    # Test /datasets/test_dataset/query endpoint without Authorization header
    response_query = client.post("/datasets/test_dataset/query")
    assert response_query.status_code == 401


@patch("api.dataset_api.get_db_connection")
def test_authorized_but_insufficient_permissions(mock_db, override_active_user):
    """Authenticated user with WRITE scope only; endpoints are expected to require higher/different permissions."""
    override_active_user([PermissionLevel.WRITE])

    response = client.get("/datasets")
    assert response.status_code == 403

    response_metadata = client.get("/datasets/test_dataset/metadata")
    assert response_metadata.status_code == 403

    response_query = client.post("/datasets/test_dataset/query")
    assert response_query.status_code == 403


@patch("api.dataset_api.get_db_connection")
def test_authorized_with_sufficient_permissions(mock_db, override_active_user):
    # Mock the database cursor properly to avoid returning None and causing TypeError when unpacking
    mock_cursor = mock_db.return_value.cursor.return_value
    mock_cursor.fetchall.return_value = []

    # Needs to return a row-like object that can be indexed both by name and by integer
    class MockRow:
        def __init__(self, data, list_data=None):
            self.data = data
            self.list_data = list_data or [0]

        def __getitem__(self, key):
            if isinstance(key, int):
                return self.list_data[key]
            return self.data.get(key)

    # Set up the side_effect to return appropriate mock rows
    override_active_user([PermissionLevel.READ])

    # Test /datasets endpoint
    mock_cursor.fetchone.side_effect = [MockRow({"name": "test_dataset"}), MockRow({}, [0])]
    response = client.get("/datasets")
    assert response.status_code == 200

    # Verify response body is a list containing the expected dataset
    response_json = response.json()
    assert isinstance(response_json, list)
    assert len(response_json) > 0
    assert response_json[0].get("name") == "test_dataset"

    # Test /datasets/test_dataset/metadata endpoint
    mock_cursor.fetchone.side_effect = [MockRow({"name": "test_dataset"}), MockRow({}, [0])]
    response_metadata = client.get("/datasets/test_dataset/metadata")
    assert response_metadata.status_code == 200

    # Verify metadata response has expected structure
    metadata_json = response_metadata.json()
    assert isinstance(metadata_json, dict)
    assert "name" in metadata_json
    assert metadata_json["name"] == "test_dataset"

    # Test /datasets/test_dataset/query endpoint
    mock_cursor.fetchone.side_effect = [MockRow({"name": "test_dataset"}), MockRow({}, [0])]
    response_query = client.post("/datasets/test_dataset/query")
    assert response_query.status_code == 200

    # Verify query response returns valid JSON structure
    query_json = response_query.json()
    assert isinstance(query_json, (list, dict))


@patch("api.dataset_api.get_db_connection")
@pytest.mark.parametrize(
    "scopes,expected_status",
    [
        # insufficient permissions: WRITE only
        ([PermissionLevel.WRITE], 403),
        # edge cases: empty / missing scopes
        ([], 403),
        (None, 403),
        # valid permissions: READ present
        ([PermissionLevel.READ], 200),
        ([PermissionLevel.READ, PermissionLevel.WRITE], 200),
    ],
)
def test_authorized_scopes_edge_cases(mock_db, override_active_user, scopes, expected_status):
    override_active_user(scopes)

    response = client.get("/datasets")
    assert response.status_code == expected_status

    response_metadata = client.get("/datasets/test_dataset/metadata")
    assert response_metadata.status_code == expected_status

    response_query = client.post("/datasets/test_dataset/query")
    assert response_query.status_code == expected_status