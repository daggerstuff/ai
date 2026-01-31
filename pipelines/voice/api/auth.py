"""
Authentication and authorization for Pixel Voice API.
"""

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum

import jwt
import structlog
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

logger = structlog.get_logger(__name__)

# Security configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class UserRole(str, Enum):
    """User roles."""

    ADMIN = "admin"
    PREMIUM = "premium"
    STANDARD = "standard"
    READONLY = "readonly"


class Permission(str, Enum):
    """System permissions."""

    READ_JOBS = "read:jobs"
    CREATE_JOBS = "create:jobs"
    DELETE_JOBS = "delete:jobs"
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    UNLIMITED_QUOTA = "unlimited:quota"


# Role permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.READ_JOBS,
        Permission.CREATE_JOBS,
        Permission.DELETE_JOBS,
        Permission.ADMIN_USERS,
        Permission.ADMIN_SYSTEM,
        Permission.UNLIMITED_QUOTA,
    ],
    UserRole.PREMIUM: [
        Permission.READ_JOBS,
        Permission.CREATE_JOBS,
        Permission.DELETE_JOBS,
    ],
    UserRole.STANDARD: [
        Permission.READ_JOBS,
        Permission.CREATE_JOBS,
    ],
    UserRole.READONLY: [
        Permission.READ_JOBS,
    ],
}


class User(BaseModel):
    """User model."""

    id: str
    email: EmailStr
    username: str
    role: UserRole
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime
    last_login: datetime | None = None
    api_key: str | None = None
    quota_limit: int | None = None


class UserCreate(BaseModel):
    """User creation model."""

    email: EmailStr
    username: str
    password: str
    role: UserRole = UserRole.STANDARD


class UserUpdate(BaseModel):
    """User update model."""

    email: EmailStr | None = None
    username: str | None = None
    role: UserRole | None = None
    is_active: bool | None = None
    is_verified: bool | None = None
    quota_limit: int | None = None


class Token(BaseModel):
    """Token model."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data model."""

    user_id: str | None = None
    permissions: list[str] = []


class AuthManager:
    """Authentication and authorization manager."""

    def __init__(self):
        self.users: dict[str, User] = {}
        self.api_keys: dict[str, str] = {}  # api_key -> user_id
        self.revoked_tokens: set = set()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash password."""
        return pwd_context.hash(password)

    def generate_api_key(self) -> str:
        """Generate API key."""
        return f"pv_{secrets.token_urlsafe(32)}"

    def create_access_token(self, data: dict, expires_delta: timedelta | None = None):
        """Create access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def create_refresh_token(self, data: dict):
        """Create refresh token."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def verify_token(self, token: str) -> TokenData | None:
        """Verify and decode token."""
        try:
            if token in self.revoked_tokens:
                return None

            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")

            if user_id is None:
                return None

            user = self.get_user(user_id)
            if not user or not user.is_active:
                return None

            permissions = ROLE_PERMISSIONS.get(user.role, [])
            return TokenData(user_id=user_id, permissions=[p.value for p in permissions])

        except jwt.PyJWTError:
            return None

    def create_user(self, user_data: UserCreate) -> User:
        """Create new user."""
        user_id = str(uuid.uuid4())
        api_key = self.generate_api_key()

        user = User(
            id=user_id,
            email=user_data.email,
            username=user_data.username,
            role=user_data.role,
            created_at=datetime.now(timezone.utc),
            api_key=api_key,
        )

        self.users[user_id] = user
        self.api_keys[api_key] = user_id

        logger.info("User created", user_id=user_id, username=user_data.username)
        return user

    def get_user(self, user_id: str) -> User | None:
        """Get user by ID."""
        return self.users.get(user_id)

    def get_user_by_email(self, email: str) -> User | None:
        """Get user by email."""
        for user in self.users.values():
            if user.email == email:
                return user
        return None

    def get_user_by_api_key(self, api_key: str) -> User | None:
        """Get user by API key."""
        user_id = self.api_keys.get(api_key)
        if user_id:
            return self.get_user(user_id)
        return None

    def update_user(self, user_id: str, user_data: UserUpdate) -> User | None:
        """Update user."""
        user = self.get_user(user_id)
        if not user:
            return None

        update_data = user_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        logger.info("User updated", user_id=user_id, fields=list(update_data.keys()))
        return user

    def revoke_token(self, token: str):
        """Revoke token."""
        self.revoked_tokens.add(token)

    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has permission."""
        user_permissions = ROLE_PERMISSIONS.get(user.role, [])
        return permission in user_permissions


# Global auth manager
auth_manager = AuthManager()


async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> User:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = auth_manager.verify_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception

    user = auth_manager.get_user(token_data.user_id) if token_data.user_id else None
    if user is None:
        raise credentials_exception

    return user


async def get_current_user_from_api_key(
    api_key: str | None = Security(api_key_header),
) -> User | None:
    """Get current user from API key."""
    if not api_key:
        return None

    user = auth_manager.get_user_by_api_key(api_key)
    if not user or not user.is_active:
        return None

    return user


async def get_current_user(
    token_user: User | None = Depends(get_current_user_from_token),
    api_key_user: User | None = Depends(get_current_user_from_api_key),
) -> User:
    """Get current user from either token or API key."""
    user = token_user or api_key_user

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    return user


def require_permission(permission: Permission):
    """Decorator to require specific permission."""

    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not auth_manager.has_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}",
            )
        return current_user

    return permission_checker


def require_role(role: UserRole):
    """Decorator to require specific role."""

    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in (role, UserRole.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f"Role required: {role.value}"
            )
        return current_user

    return role_checker


# Convenience dependencies
RequireAdmin = Depends(require_role(UserRole.ADMIN))
RequirePremium = Depends(require_role(UserRole.PREMIUM))
RequireCreateJobs = Depends(require_permission(Permission.CREATE_JOBS))
RequireDeleteJobs = Depends(require_permission(Permission.DELETE_JOBS))
RequireAdminUsers = Depends(require_permission(Permission.ADMIN_USERS))
