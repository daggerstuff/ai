"""Tests for the API authentication system."""

import pytest

from ai.tools.scripts.api_authentication import (
    AuthenticationSystem,
    PermissionLevel,
    UserRole,
)

SECRET = "test-secret-key-0123456789abcdef"


@pytest.fixture
def auth() -> AuthenticationSystem:
    return AuthenticationSystem(SECRET)


def test_password_hash_is_salted_and_verifiable(auth: AuthenticationSystem) -> None:
    """A stored hash verifies its own password and rejects any other."""
    digest = auth.hash_password("correct horse battery staple")

    assert digest != "correct horse battery staple"
    assert auth.verify_password("correct horse battery staple", digest) is True
    assert auth.verify_password("wrong password", digest) is False


def test_authentication_requires_the_exact_password(auth: AuthenticationSystem) -> None:
    """A registered user authenticates only with their own password."""
    auth.create_user("alice", "alice@example.com", "pw-Sup3r-secret")

    assert auth.authenticate_user("alice", "pw-Sup3r-secret") is not None
    assert auth.authenticate_user("alice", "pw-Sup3r-secreT") is None
    assert auth.authenticate_user("nobody", "pw-Sup3r-secret") is None


def test_jwt_round_trip_and_revocation(auth: AuthenticationSystem) -> None:
    """A token resolves to its user until it is revoked."""
    user = auth.create_user("bob", "bob@example.com", "pw-Sup3r-secret")
    token = auth.generate_jwt_token(user)

    claims = auth.verify_jwt_token(token)
    assert claims is not None
    assert claims["username"] == "bob"

    assert auth.revoke_token(token) is True
    assert auth.verify_jwt_token(token) is None


def test_malformed_tokens_are_rejected(auth: AuthenticationSystem) -> None:
    """Garbage is not a credential."""
    assert auth.verify_jwt_token("not.a.token") is None
    assert auth.verify_jwt_token("") is None


def test_permission_ladder_grants_downward_only(auth: AuthenticationSystem) -> None:
    """A role reaches its own permission level and everything below it."""
    assert auth.check_permission(UserRole.ADMIN, PermissionLevel.ADMIN) is True
    assert auth.check_permission(UserRole.ADMIN, PermissionLevel.WRITE) is True
    assert auth.check_permission(UserRole.USER, PermissionLevel.READ) is True
    assert auth.check_permission(UserRole.USER, PermissionLevel.ADMIN) is False


def test_api_key_authenticates_only_the_key_that_was_issued(auth: AuthenticationSystem) -> None:
    """The plaintext key resolves once; the stored record keeps only a hash."""
    raw_key, record = auth.create_api_key("ci", [PermissionLevel.READ])

    assert record.name == "ci"
    assert record.permissions == [PermissionLevel.READ]
    assert record.key_hash != raw_key
    assert auth.authenticate_api_key(raw_key) is not None
    assert auth.authenticate_api_key("bogus-key") is None
