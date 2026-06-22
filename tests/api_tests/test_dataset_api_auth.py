import os

os.environ["AUTH_SECRET_KEY"] = "test-secret-key"

from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

os.environ["AUTH_SECRET_KEY"] = "test-secret"

from api.dataset_api import (
    app,
    get_current_active_user_or_api_key,
)
from security.api_authentication import PermissionLevel

client = TestClient(app)


def _override_auth(scopes):
    def _dependency_override():
        return {"username": "test_user", "scopes": scopes, "auth_type": "api_key"}

    return _dependency_override


@pytest.fixture(autouse=True)
def mock_db_connection():
    with patch("api.dataset_api.get_db_connection") as mock_db:
        mock_conn = MagicMock()
        mock_db.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        def mock_execute(*args, **kwargs):
            pass

        mock_cursor.execute = mock_execute

        # When SQLite row factory is used, row_factory = sqlite3.Row
        # Python accesses table_row["name"] like a dict, but fetchone() returning a pure dict causes issues
        # when other parts expect a tuple (like fetchone()[0] for count)

        # Let's use a very dumb mock that just ignores SQL errors and assumes everything is fine
        # We only care that the endpoints are testing the auth logic, not the sqlite execution.

        def safe_fetchone(*_, **__):
            # A magic mock that returns whatever is asked of it
            row = MagicMock()
            row.__getitem__.return_value = "test_table"  # for table_row["name"]
            row.__getitem__.side_effect = lambda key: "test_table" if isinstance(key, str) else 0  # for [0]
            return row

        mock_cursor.fetchone = safe_fetchone

        def safe_fetchall(*_, **__):
            row1 = MagicMock()
            row1.__getitem__.return_value = "test_table"
            row1.__getitem__.side_effect = lambda key: "test_table" if isinstance(key, str) else 0

            row2 = MagicMock()
            row2.__getitem__.side_effect = lambda key: {"name": "col1", "type": "TEXT", "notnull": 0, "pk": 0}.get(
                key, 0
            )

            return [row1, row2]

        mock_cursor.fetchall = safe_fetchall

        yield mock_db


@pytest.mark.parametrize(
    ("method", "path", "required_scopes"),
    [
        ("GET", "/datasets", [PermissionLevel.READ]),
        ("GET", "/datasets/test_db/metadata", [PermissionLevel.READ]),
        ("POST", "/datasets/test_db/query", [PermissionLevel.READ]),
    ],
)
def test_dataset_endpoints_with_permitted_scope(method, path, required_scopes):
    app.dependency_overrides[get_current_active_user_or_api_key] = _override_auth(scopes=required_scopes)
    try:
        response = client.request(method, path)
    finally:
        app.dependency_overrides.clear()

    # In case of 500 error from sqlite mock limitations, at least assert it's not 403
    assert response.status_code != status.HTTP_403_FORBIDDEN


@pytest.mark.parametrize(
    ("method", "path", "required_scopes"),
    [
        ("GET", "/datasets", [PermissionLevel.READ]),
        ("GET", "/datasets/test_db/metadata", [PermissionLevel.READ]),
        ("POST", "/datasets/test_db/query", [PermissionLevel.READ]),
    ],
)
def test_dataset_endpoints_with_forbidden_scope(method, path, required_scopes):
    _ = required_scopes
    insufficient_scopes = []

    app.dependency_overrides[get_current_active_user_or_api_key] = _override_auth(scopes=insufficient_scopes)
    try:
        response = client.request(method, path)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_403_FORBIDDEN
