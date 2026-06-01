"""Tests for the API authentication system."""

from security.api_authentication import (
    AuthenticationSystem,
    PermissionLevel,
    UserRole,
)


def make_auth() -> AuthenticationSystem:
    return AuthenticationSystem(secret_key="test-secret-key")


class TestPasswordHashing:
    def test_hash_and_verify(self):
        auth = make_auth()
        hash_val = auth.hash_password("my_password")
        assert auth.verify_password("my_password", hash_val) is True
        assert auth.verify_password("wrong", hash_val) is False

    def test_unique_salts(self):
        auth = make_auth()
        h1 = auth.hash_password("same")
        h2 = auth.hash_password("same")
        # bcrypt generates unique salts each time
        assert h1 != h2


class TestUserCreation:
    def test_create_user(self):
        auth = make_auth()
        user = auth.create_user("alice", "alice@test.com", "secret", UserRole.ADMIN)
        assert user.username == "alice"
        assert user.email == "alice@test.com"
        assert user.role == UserRole.ADMIN
        assert user.is_active is True
        assert user.created_at is not None

    def test_user_id_is_unique(self):
        auth = make_auth()
        u1 = auth.create_user("alice", "a@t.com", "p")
        u2 = auth.create_user("bob", "b@t.com", "p")
        assert u1.user_id != u2.user_id


class TestJWTToken:
    def test_generate_and_verify(self):
        auth = make_auth()
        user = auth.create_user("jwt_user", "j@t.com", "p")
        token = auth.generate_jwt_token(user)
        payload = auth.verify_jwt_token(token)
        assert payload is not None
        assert payload["user_id"] == user.user_id
        assert payload["username"] == "jwt_user"

    def test_invalid_token_returns_none(self):
        auth = make_auth()
        assert auth.verify_jwt_token("totally-fake-token") is None

    def test_revoked_token_rejected(self):
        auth = make_auth()
        user = auth.create_user("rev_user", "r@t.com", "p")
        token = auth.generate_jwt_token(user)
        assert auth.verify_jwt_token(token) is not None
        auth.revoke_token(token)
        assert auth.verify_jwt_token(token) is None

    def test_token_for_inactive_user_rejected(self):
        auth = make_auth()
        user = auth.create_user("ia_user", "i@t.com", "p")
        token = auth.generate_jwt_token(user)
        user.is_active = False
        assert auth.verify_jwt_token(token) is None


class TestAPIKey:
    def test_create_and_authenticate(self):
        auth = make_auth()
        raw_key, key_obj = auth.create_api_key("test", [PermissionLevel.READ])
        assert key_obj.name == "test"
        result = auth.authenticate_api_key(raw_key)
        assert result is not None
        assert result.key_id == key_obj.key_id

    def test_invalid_key_rejected(self):
        auth = make_auth()
        result = auth.authenticate_api_key("invalid_key")
        assert result is None

    def test_expired_key_rejected(self):
        auth = make_auth()
        # Create key with negative expiry — it expires immediately
        raw_key, _key_obj = auth.create_api_key("short", [PermissionLevel.READ], expires_in_days=-1)
        result = auth.authenticate_api_key(raw_key)
        assert result is None

    def test_inactive_key_rejected(self):
        auth = make_auth()
        raw_key, key_obj = auth.create_api_key("inactive", [PermissionLevel.READ])
        key_obj.is_active = False
        result = auth.authenticate_api_key(raw_key)
        assert result is None

    def test_multiple_keys(self):
        auth = make_auth()
        k1_raw, k1_obj = auth.create_api_key("k1", [PermissionLevel.READ])
        k2_raw, k2_obj = auth.create_api_key("k2", [PermissionLevel.WRITE])
        assert k1_obj.key_id != k2_obj.key_id
        assert auth.authenticate_api_key(k1_raw) is not None
        assert auth.authenticate_api_key(k2_raw) is not None


class TestRBAC:
    def test_admin_has_all_permissions(self):
        auth = make_auth()
        for perm in PermissionLevel:
            assert auth.check_permission(UserRole.ADMIN, perm) is True

    def test_readonly_cannot_write(self):
        auth = make_auth()
        assert auth.check_permission(UserRole.READONLY, PermissionLevel.READ) is True
        assert auth.check_permission(UserRole.READONLY, PermissionLevel.WRITE) is False
        assert auth.check_permission(UserRole.READONLY, PermissionLevel.DELETE) is False

    def test_api_key_permission_check(self):
        auth = make_auth()
        _raw, key_obj = auth.create_api_key("k", [PermissionLevel.READ])
        assert auth.check_api_key_permission(key_obj, PermissionLevel.READ) is True
        assert auth.check_api_key_permission(key_obj, PermissionLevel.WRITE) is False


class TestUserAuthentication:
    def test_valid_credentials(self):
        auth = make_auth()
        auth.create_user("login_user", "l@t.com", "correct_pw")
        user = auth.authenticate_user("login_user", "correct_pw")
        assert user is not None
        assert user.username == "login_user"

    def test_wrong_password(self):
        auth = make_auth()
        auth.create_user("fail_user", "f@t.com", "correct_pw")
        user = auth.authenticate_user("fail_user", "wrong_pw")
        assert user is None

    def test_nonexistent_user(self):
        auth = make_auth()
        assert auth.authenticate_user("ghost", "p") is None
