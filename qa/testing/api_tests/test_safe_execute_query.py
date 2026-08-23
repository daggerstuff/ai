# Needs AUTH_SECRET_KEY for api.dataset_api import
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def safe_execute_query(monkeypatch):
    """Provide safe_execute_query with AUTH_SECRET_KEY set via monkeypatch.

    This avoids mutating os.environ at import time and scopes the env change
    to tests that use this fixture.
    """
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret")
    from inference.api.dataset_api import safe_execute_query as _safe_execute_query

    return _safe_execute_query


def test_safe_execute_query_valid_where(safe_execute_query):
    cursor = MagicMock()
    query_template = "SELECT * FROM {table} WHERE {where}"
    safe_execute_query(cursor, query_template, params=("test",), table="users", where="id = ?")
    cursor.execute.assert_called_once_with("SELECT * FROM users WHERE id = ?", ("test",))


def test_safe_execute_query_invalid_where_semicolon(safe_execute_query):
    cursor = MagicMock()
    query_template = "SELECT * FROM {table} WHERE {where}"
    with pytest.raises(ValueError, match="Invalid where clause"):
        safe_execute_query(cursor, query_template, table="users", where="id = 1; DROP TABLE users")

    cursor.execute.assert_not_called()


def test_safe_execute_query_valid_identifier(safe_execute_query):
    cursor = MagicMock()
    query_template = "SELECT * FROM {table}"
    safe_execute_query(cursor, query_template, table="users_123")
    cursor.execute.assert_called_once_with("SELECT * FROM users_123")


def test_safe_execute_query_invalid_identifier_space(safe_execute_query):
    cursor = MagicMock()
    query_template = "SELECT * FROM {table}"
    with pytest.raises(ValueError, match="Invalid identifier"):
        safe_execute_query(cursor, query_template, table="users space")

    cursor.execute.assert_not_called()


def test_safe_execute_query_invalid_identifier_dash(safe_execute_query):
    cursor = MagicMock()
    query_template = "SELECT * FROM {table}"
    with pytest.raises(ValueError, match="Invalid identifier"):
        safe_execute_query(cursor, query_template, table="users-name")

    cursor.execute.assert_not_called()


def test_safe_execute_query_no_params(safe_execute_query):
    cursor = MagicMock()
    query_template = "SELECT * FROM {table}"
    safe_execute_query(cursor, query_template, table="users")
    cursor.execute.assert_called_once_with("SELECT * FROM users")


def test_safe_execute_query_complex_valid_where(safe_execute_query):
    cursor = MagicMock()
    query_template = "SELECT * FROM {table} WHERE {where}"
    safe_execute_query(cursor, query_template, table="users", where="name = 'Alice' AND age = 30 OR role = \"admin\"")
    cursor.execute.assert_called_once_with("SELECT * FROM users WHERE name = 'Alice' AND age = 30 OR role = \"admin\"")

