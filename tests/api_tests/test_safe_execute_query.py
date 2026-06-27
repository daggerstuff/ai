# Needs AUTH_SECRET_KEY for api.dataset_api import
import os
from unittest.mock import MagicMock

import pytest

os.environ["AUTH_SECRET_KEY"] = "test-secret"

from api.dataset_api import safe_execute_query


def test_safe_execute_query_valid_where():
    cursor = MagicMock()
    query_template = "SELECT * FROM {table} WHERE {where}"
    safe_execute_query(cursor, query_template, params=("test",), table="users", where="id = ?")
    cursor.execute.assert_called_once_with("SELECT * FROM users WHERE id = ?", ("test",))


def test_safe_execute_query_invalid_where_semicolon():
    cursor = MagicMock()
    query_template = "SELECT * FROM {table} WHERE {where}"
    with pytest.raises(ValueError, match="Invalid where clause"):
        safe_execute_query(cursor, query_template, table="users", where="id = 1; DROP TABLE users")


def test_safe_execute_query_valid_identifier():
    cursor = MagicMock()
    query_template = "SELECT * FROM {table}"
    safe_execute_query(cursor, query_template, table="users_123")
    cursor.execute.assert_called_once_with("SELECT * FROM users_123")


def test_safe_execute_query_invalid_identifier_space():
    cursor = MagicMock()
    query_template = "SELECT * FROM {table}"
    with pytest.raises(ValueError, match="Invalid identifier"):
        safe_execute_query(cursor, query_template, table="users space")


def test_safe_execute_query_invalid_identifier_dash():
    cursor = MagicMock()
    query_template = "SELECT * FROM {table}"
    with pytest.raises(ValueError, match="Invalid identifier"):
        safe_execute_query(cursor, query_template, table="users-name")


def test_safe_execute_query_no_params():
    cursor = MagicMock()
    query_template = "SELECT * FROM {table}"
    safe_execute_query(cursor, query_template, table="users")
    cursor.execute.assert_called_once_with("SELECT * FROM users")


def test_safe_execute_query_complex_valid_where():
    cursor = MagicMock()
    query_template = "SELECT * FROM {table} WHERE {where}"
    safe_execute_query(cursor, query_template, table="users", where="name = 'Alice' AND age = 30 OR role = \"admin\"")
    cursor.execute.assert_called_once_with("SELECT * FROM users WHERE name = 'Alice' AND age = 30 OR role = \"admin\"")
