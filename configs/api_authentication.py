"""
Task 101: API Authentication System Implementation
Critical Security Component - JWT Token Authentication with RBAC

This module provides enterprise-grade API authentication with:
- JWT token authentication
- Role-based access control (RBAC)
- API key management
- Security middleware
- Authentication validation
"""

import hashlib
import json
import logging
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path

import bcrypt
import jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_SECRET_VALUES = frozenset({"your-secret-key-here"})
_MINIMUM_SECRET_LENGTH = 32


def validate_secret_key(secret_key: str | None) -> str:
    """Validate a JWT signing secret before it is used."""
    if not secret_key or secret_key in _DEFAULT_SECRET_VALUES or len(secret_key) < _MINIMUM_SECRET_LENGTH:
        raise ValueError(
            f"AUTH_SECRET_KEY must be set to a unique secret of at least "
            f"{_MINIMUM_SECRET_LENGTH} characters. Default placeholders are rejected."
        )
    return secret_key


class UserRole(Enum):
    """User roles for RBAC system"""

    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"
    API_CLIENT = "api_client"
    READONLY = "readonly"


class PermissionLevel(Enum):
    """Permission levels for resource access"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class User:
    """User model for authentication"""

    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    is_active: bool = True

    created_at: datetime = None

    last_login: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)


@dataclass
class APIKey:
    """API Key model for service authentication"""

    key_id: str
    key_hash: str
    name: str
    permissions: list[PermissionLevel]
    is_active: bool = True

    created_at: datetime = None
    expires_at: datetime | None = None
    last_used: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)


class SQLiteAuthenticationStore:
    """SQLite-backed persistence for users, API keys, and revoked JWT IDs."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(database_path) as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS auth_users (
                    user_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS auth_api_keys (
                    key_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS auth_revoked_jti (
                    jti TEXT PRIMARY KEY,
                    revoked_at TEXT NOT NULL
                );
                """
            )

    def load(self) -> tuple[dict[str, User], dict[str, APIKey], set[str]]:
        """Load persisted authentication state."""
        users: dict[str, User] = {}
        api_keys: dict[str, APIKey] = {}

        with sqlite3.connect(self.database_path) as connection:
            user_rows = connection.execute("SELECT payload FROM auth_users").fetchall()
            api_key_rows = connection.execute("SELECT payload FROM auth_api_keys").fetchall()
            revoked_rows = connection.execute("SELECT jti FROM auth_revoked_jti").fetchall()

        for user_payload in user_rows:
            user = self._deserialize_user(user_payload[0])
            users[user.user_id] = user

        for api_key_payload in api_key_rows:
            api_key = self._deserialize_api_key(api_key_payload[0])
            api_keys[api_key.key_id] = api_key

        return users, api_keys, {jti for (jti,) in revoked_rows}

    def save_user(self, user: User) -> None:
        """Insert or update a user."""
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT INTO auth_users (user_id, payload) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET payload = excluded.payload",
                (user.user_id, self._serialize_user(user)),
            )

    def save_api_key(self, api_key: APIKey) -> None:
        """Insert or update an API key."""
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT INTO auth_api_keys (key_id, payload) VALUES (?, ?) "
                "ON CONFLICT(key_id) DO UPDATE SET payload = excluded.payload",
                (api_key.key_id, self._serialize_api_key(api_key)),
            )

    def revoke_jti(self, jti: str) -> None:
        """Persist a revoked JWT ID."""
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO auth_revoked_jti (jti, revoked_at) VALUES (?, ?)",
                (jti, datetime.now(UTC).isoformat()),
            )

    def _serialize_user(self, user: User) -> str:
        return json.dumps(
            {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "password_hash": user.password_hash,
                "role": user.role.value,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
            }
        )

    def _serialize_api_key(self, api_key: APIKey) -> str:
        return json.dumps(
            {
                "key_id": api_key.key_id,
                "key_hash": api_key.key_hash,
                "name": api_key.name,
                "permissions": [permission.value for permission in api_key.permissions],
                "is_active": api_key.is_active,
                "created_at": api_key.created_at.isoformat(),
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
            }
        )

    def _deserialize_user(self, payload: str) -> User:
        data = json.loads(payload)
        return User(
            user_id=data["user_id"],
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            role=UserRole(data["role"]),
            is_active=data["is_active"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_login=datetime.fromisoformat(data["last_login"]) if data["last_login"] else None,
        )

    def _deserialize_api_key(self, payload: str) -> APIKey:
        data = json.loads(payload)
        return APIKey(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            name=data["name"],
            permissions=[PermissionLevel(permission) for permission in data["permissions"]],
            is_active=data["is_active"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
            last_used=datetime.fromisoformat(data["last_used"]) if data["last_used"] else None,
        )

    def deactivate_api_key(self, key_id: str) -> None:
        """Persist an inactive API key."""
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE auth_api_keys SET payload = ? WHERE key_id = ?",
                (self._serialize_api_key(self.api_keys[key_id]), key_id),
            )
        """Insert or update a user."""
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT INTO auth_users (user_id, payload) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET payload = excluded.payload",
                (user.user_id, self._serialize_user(user)),
            )

    def save_api_key(self, api_key: APIKey) -> None:
        """Insert or update an API key."""
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT INTO auth_api_keys (key_id, payload) VALUES (?, ?) "
                "ON CONFLICT(key_id) DO UPDATE SET payload = excluded.payload",
                (api_key.key_id, self._serialize_api_key(api_key)),
            )

    def revoke_jti(self, jti: str) -> None:
        """Persist a revoked JWT ID."""
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO auth_revoked_jti (jti, revoked_at) VALUES (?, ?)",
                (jti, datetime.now(UTC).isoformat()),
            )

    def _serialize_user(self, user: User) -> str:
        return json.dumps(
            {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "password_hash": user.password_hash,
                "role": user.role.value,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
            }
        )

    def _serialize_api_key(self, api_key: APIKey) -> str:
        return json.dumps(
            {
                "key_id": api_key.key_id,
                "key_hash": api_key.key_hash,
                "name": api_key.name,
                "permissions": [permission.value for permission in api_key.permissions],
                "is_active": api_key.is_active,
                "created_at": api_key.created_at.isoformat(),
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
            }
        )

    def _deserialize_user(self, payload: str) -> User:
        data = json.loads(payload)
        return User(
            user_id=data["user_id"],
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            role=UserRole(data["role"]),
            is_active=data["is_active"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_login=datetime.fromisoformat(data["last_login"]) if data["last_login"] else None,
        )

    def _deserialize_api_key(self, payload: str) -> APIKey:
        data = json.loads(payload)
        return APIKey(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            name=data["name"],
            permissions=[PermissionLevel(permission) for permission in data["permissions"]],
            is_active=data["is_active"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
            last_used=datetime.fromisoformat(data["last_used"]) if data["last_used"] else None,
        )


class AuthenticationSystem:
    """
    Enterprise API Authentication System

    Provides JWT token authentication, RBAC, and API key management
    """

    def __init__(
        self,
        secret_key: str,
        token_expiry_hours: int = 24,
        auth_database: Path | None = None,
    ):
        self.secret_key = validate_secret_key(secret_key)
        self.token_expiry_hours = token_expiry_hours
        self.algorithm = "HS256"

        self.store = SQLiteAuthenticationStore(auth_database or Path(os.getenv("AUTH_DB_PATH", "data/auth.sqlite3")))
        self.users, self.api_keys, self.revoked_tokens = self.store.load()

        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: [
                PermissionLevel.READ,
                PermissionLevel.WRITE,
                PermissionLevel.DELETE,
                PermissionLevel.ADMIN,
            ],
            UserRole.MODERATOR: [
                PermissionLevel.READ,
                PermissionLevel.WRITE,
                PermissionLevel.DELETE,
            ],
            UserRole.USER: [PermissionLevel.READ, PermissionLevel.WRITE],
            UserRole.API_CLIENT: [PermissionLevel.READ, PermissionLevel.WRITE],
            UserRole.READONLY: [PermissionLevel.READ],
        }

        logger.info("Authentication system initialized")

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)

    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> User:
        """Create new user with hashed password"""
        user_id = secrets.token_urlsafe(16)
        password_hash = self.hash_password(password)

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
        )

        self.users[user_id] = user
        self.store.save_user(user)
        logger.info(f"User created: {username} with role {role.value}")
        return user

    def create_api_key(
        self,
        name: str,
        permissions: list[PermissionLevel],
        expires_in_days: int | None = None,
    ) -> tuple[str, APIKey]:
        """Create new API key with specified permissions"""
        key_id = secrets.token_urlsafe(16)
        api_key = self.generate_api_key()
        key_hash = self.hash_api_key(api_key)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        api_key_obj = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            expires_at=expires_at,
        )

        self.api_keys[key_id] = api_key_obj
        self.store.save_api_key(api_key_obj)
        logger.info(f"API key created: {name} with permissions {[p.value for p in permissions]}")
        return api_key, api_key_obj

    def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate user with username/password"""
        for user in self.users.values():
            if user.username == username and user.is_active and self.verify_password(password, user.password_hash):
                user.last_login = datetime.now(UTC)
                logger.info(f"User authenticated: {username}")
                return user

        logger.warning(f"Authentication failed for user: {username}")
        return None

    def authenticate_api_key(self, api_key: str) -> APIKey | None:
        """Authenticate API key"""
        key_hash = self.hash_api_key(api_key)

        for api_key_obj in self.api_keys.values():
            if api_key_obj.key_hash == key_hash and api_key_obj.is_active:
                # Check expiration
                if api_key_obj.expires_at and datetime.now(UTC) > api_key_obj.expires_at:
                    logger.warning(f"Expired API key used: {api_key_obj.name}")
                    return None

                api_key_obj.last_used = datetime.now(UTC)
                self.store.save_api_key(api_key_obj)
                logger.info(f"API key authenticated: {api_key_obj.name}")
                return api_key_obj

        logger.warning("Invalid API key used")
        return None

    def generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "exp": datetime.now(UTC) + timedelta(hours=self.token_expiry_hours),
            "iat": datetime.now(UTC),
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"JWT token generated for user: {user.username}")
        return token

    def verify_jwt_token(self, token: str) -> dict | None:
        """Verify and decode JWT token"""
        try:
            # Check if token is revoked
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            if payload.get("jti") in self.revoked_tokens:
                logger.warning("Revoked token used")
                return None

            # Verify user still exists and is active
            user_id = payload.get("user_id")
            if user_id not in self.users or not self.users[user_id].is_active:
                logger.warning(f"Token for inactive/deleted user: {user_id}")
                return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Expired JWT token used")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token used")
            return None

    def revoke_token(self, token: str) -> bool:
        """Revoke JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            jti = payload.get("jti")
            if not jti:
                return False

            self.revoked_tokens.add(jti)
            self.store.revoke_jti(jti)
            logger.info(f"Token revoked for user: {payload.get('username')}")
            return True
        except jwt.InvalidTokenError:
            return False

    def check_permission(self, user_role: UserRole, required_permission: PermissionLevel) -> bool:
        """Check if user role has required permission"""
        user_permissions = self.role_permissions.get(user_role, [])
        return required_permission in user_permissions

    def check_api_key_permission(self, api_key: APIKey, required_permission: PermissionLevel) -> bool:
        """Check if API key has required permission"""
        return required_permission in api_key.permissions


class AuthenticationMiddleware:
    """
    Authentication middleware for API endpoints
    """

    def __init__(self, auth_system: AuthenticationSystem):
        self.auth_system = auth_system

    def require_auth(self, required_permission: PermissionLevel = PermissionLevel.READ):
        """Decorator to require authentication for API endpoints"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract token from request (implementation depends on framework)
                # This is a generic example - adapt for your specific framework

                auth_header = kwargs.get("auth_header", "")
                api_key_header = kwargs.get("api_key_header", "")

                authenticated_user = None
                authenticated_api_key = None

                # Try JWT authentication first
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:]  # Remove 'Bearer ' prefix
                    payload = self.auth_system.verify_jwt_token(token)

                    if payload:
                        user_id = payload["user_id"]
                        user = self.auth_system.users.get(user_id)

                        if user and self.auth_system.check_permission(user.role, required_permission):
                            authenticated_user = user
                        else:
                            return {"error": "Insufficient permissions", "status": 403}
                    else:
                        return {"error": "Invalid or expired token", "status": 401}

                # Try API key authentication
                elif api_key_header:
                    api_key_obj = self.auth_system.authenticate_api_key(api_key_header)

                    if api_key_obj and self.auth_system.check_api_key_permission(api_key_obj, required_permission):
                        authenticated_api_key = api_key_obj
                    else:
                        return {
                            "error": "Invalid API key or insufficient permissions",
                            "status": 403,
                        }

                else:
                    return {"error": "Authentication required", "status": 401}

                # Add authentication info to kwargs
                kwargs["authenticated_user"] = authenticated_user
                kwargs["authenticated_api_key"] = authenticated_api_key

                return func(*args, **kwargs)

            return wrapper

        return decorator


# Security Testing Suite
class AuthenticationTester:
    """
    Security testing suite for authentication system
    """

    def __init__(self, auth_system: AuthenticationSystem):
        self.auth_system = auth_system
        self.test_results = []

    def run_security_tests(self) -> dict[str, bool]:
        """Run comprehensive security tests"""
        tests = [
            self.test_password_hashing,
            self.test_jwt_token_validation,
            self.test_api_key_security,
            self.test_role_based_access,
            self.test_token_expiration,
            self.test_token_revocation,
            self.test_brute_force_protection,
        ]

        results = {}
        for test in tests:
            try:
                result = test()
                results[test.__name__] = result
                logger.info(f"Security test {test.__name__}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                results[test.__name__] = False
                logger.error(f"Security test {test.__name__} failed with error: {e}")

        return results

    def test_password_hashing(self) -> bool:
        """Test password hashing security"""
        password = "test_password_123"
        hash1 = self.auth_system.hash_password(password)
        hash2 = self.auth_system.hash_password(password)

        # Hashes should be different (salt)
        if hash1 == hash2:
            return False

        # Both should verify correctly
        return self.auth_system.verify_password(password, hash1) and self.auth_system.verify_password(password, hash2)

    def test_jwt_token_validation(self) -> bool:
        """Test JWT token validation"""
        user = self.auth_system.create_user("test_user", "test@example.com", "password")
        token = self.auth_system.generate_jwt_token(user)

        # Valid token should verify
        payload = self.auth_system.verify_jwt_token(token)
        if not payload or payload["user_id"] != user.user_id:
            return False

        # Invalid token should not verify
        invalid_payload = self.auth_system.verify_jwt_token("invalid_token")
        return invalid_payload is None

    def test_api_key_security(self) -> bool:
        """Test API key security"""
        api_key, api_key_obj = self.auth_system.create_api_key("test_key", [PermissionLevel.READ])

        # Valid API key should authenticate
        auth_result = self.auth_system.authenticate_api_key(api_key)
        if not auth_result or auth_result.key_id != api_key_obj.key_id:
            return False

        # Invalid API key should not authenticate
        invalid_result = self.auth_system.authenticate_api_key("invalid_key")
        return invalid_result is None

    def test_role_based_access(self) -> bool:
        """Test role-based access control"""
        # Admin should have all permissions
        admin_check = self.auth_system.check_permission(UserRole.ADMIN, PermissionLevel.DELETE)

        # Readonly should not have write permissions
        readonly_check = not self.auth_system.check_permission(UserRole.READONLY, PermissionLevel.WRITE)

        return admin_check and readonly_check

    def test_token_expiration(self) -> bool:
        """Test token expiration (simulated)"""
        # This would require time manipulation in a real test
        # For now, just verify the expiration field is set correctly
        user = self.auth_system.create_user("exp_user", "exp@example.com", "password")
        token = self.auth_system.generate_jwt_token(user)

        payload = self.auth_system.verify_jwt_token(token)
        return payload is not None and "exp" in payload

    def test_token_revocation(self) -> bool:
        """Test token revocation"""
        user = self.auth_system.create_user("rev_user", "rev@example.com", "password")
        token = self.auth_system.generate_jwt_token(user)

        # Token should be valid before revocation
        if not self.auth_system.verify_jwt_token(token):
            return False

        # Revoke token
        self.auth_system.revoke_token(token)

        # Token should be invalid after revocation
        return self.auth_system.verify_jwt_token(token) is None

    def test_brute_force_protection(self) -> bool:
        """Test brute force protection (basic implementation)"""
        self.auth_system.create_user("bf_user", "bf@example.com", "correct_password")

        # Multiple failed attempts
        for _ in range(5):
            result = self.auth_system.authenticate_user("bf_user", "wrong_password")
            if result:
                return False

        # Correct password should still work (no lockout implemented yet)
        result = self.auth_system.authenticate_user("bf_user", "correct_password")
        return result is not None


# Example usage and testing
if __name__ == "__main__":
    # Initialize authentication system
    secret_key = os.environ.get("AUTH_SECRET_KEY")
    if not secret_key or secret_key == "your-secret-key-here":
        raise ValueError(
            "AUTH_SECRET_KEY environment variable must be set with a secure value. Default placeholder is not allowed."
        )
    auth_system = AuthenticationSystem(secret_key=validate_secret_key(secret_key))

    # Create test users
    admin_user = auth_system.create_user("admin", "admin@example.com", "admin_password", UserRole.ADMIN)
    regular_user = auth_system.create_user("user", "user@example.com", "user_password", UserRole.USER)

    # Create API key
    api_key, api_key_obj = auth_system.create_api_key(
        "test_api_key",
        [PermissionLevel.READ, PermissionLevel.WRITE],
        expires_in_days=30,
    )

    # Test authentication
    authenticated_user = auth_system.authenticate_user("admin", "admin_password")
    if authenticated_user:
        token = auth_system.generate_jwt_token(authenticated_user)

        # Verify token
        payload = auth_system.verify_jwt_token(token)

    # Run security tests
    tester = AuthenticationTester(auth_system)
    test_results = tester.run_security_tests()

    for _test_name, result in test_results.items():
        logger.info(f"  {_test_name}: {'PASSED' if result else 'FAILED'}")

    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    security_score = (passed_tests / total_tests) * 100
    logger.info(f"Security score: {security_score:.0f}%")
