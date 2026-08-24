"""
Authentication management for Pixelated AI CLI

This module handles JWT token management, authentication flows, and secure
credential storage with HIPAA++ compliance.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from .config import CLIConfig
from .utils import validate_jwt_token

logger = logging.getLogger("pixelated-ai-cli")


class AuthenticationError(Exception):
    """Custom exception for authentication errors"""


class TokenExpiredError(AuthenticationError):
    """Exception for expired tokens"""


class AuthManager:
    """Manages authentication and JWT tokens for the CLI"""

    def __init__(self, config: CLIConfig):
        """
        Initialize authentication manager

        Args:
            config: CLI configuration instance
        """
        self.config = config
        self._token_info: dict[str, Any] | None = None
        self._session = requests.Session()

        # Configure session
        self._session.timeout = config.api.timeout
        self._session.headers.update(
            {
                "User-Agent": "Pixelated-AI-CLI/0.1.0",
                "Content-Type": "application/json",
            }
        )

        # Load existing token if available
        if config.auth.jwt_token:
            self._load_token_info()

    def authenticate(self, username: str, password: str, client_id: str | None = None) -> bool:
        """
        Authenticate with username and password

        Args:
            username: Username or email
            password: Password
            client_id: Optional client ID for OAuth

        Returns:
            True if authentication successful

        Raises:
            AuthenticationError: If authentication fails
        """
        try:
            auth_data = {
                "username": username,
                "password": password,
                "grant_type": "password",
            }

            if client_id or self.config.auth.client_id:
                auth_data["client_id"] = client_id or self.config.auth.client_id

            if self.config.auth.client_secret:
                auth_data["client_secret"] = self.config.auth.client_secret

            # Make authentication request
            auth_url = f"{self.config.api.base_url.rstrip('/')}/auth/token"

            logger.debug(f"Authenticating to {auth_url}")
            response = self._session.post(auth_url, json=auth_data)

            if response.status_code == 200:
                auth_response = response.json()

                # Extract tokens
                access_token = auth_response.get("access_token")
                refresh_token = auth_response.get("refresh_token")
                expires_in = auth_response.get("expires_in", 3600)

                if not access_token:
                    raise AuthenticationError("No access token in response")

                # Store tokens
                self.config.auth.jwt_token = access_token
                if refresh_token:
                    self.config.auth.refresh_token = refresh_token

                # Calculate expiry
                expiry_time = datetime.now(UTC) + timedelta(seconds=expires_in)
                self.config.auth.token_expiry = expiry_time.isoformat()

                # Load token info
                self._load_token_info()

                # Save configuration
                self.config.save_configuration()

                logger.info(f"Authentication successful for user: {username}")
                return True

            error_msg = f"Authentication failed: HTTP {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f" - {error_data['error']}"
                if "error_description" in error_data:
                    error_msg += f": {error_data['error_description']}"
            except:
                pass

            raise AuthenticationError(error_msg)

        except requests.exceptions.RequestException as e:
            raise AuthenticationError(f"Network error during authentication: {e}") from e
        except Exception as e:
            raise AuthenticationError(f"Authentication failed: {e}") from e

    def refresh_token(self) -> bool:
        """
        Refresh the access token using refresh token

        Returns:
            True if refresh successful

        Raises:
            AuthenticationError: If refresh fails
        """
        if not self.config.auth.refresh_token:
            raise AuthenticationError("No refresh token available")

        try:
            refresh_data = {
                "refresh_token": self.config.auth.refresh_token,
                "grant_type": "refresh_token",
            }

            if self.config.auth.client_id:
                refresh_data["client_id"] = self.config.auth.client_id

            auth_url = f"{self.config.api.base_url.rstrip('/')}/auth/refresh"

            logger.debug("Refreshing access token")
            response = self._session.post(auth_url, json=refresh_data)

            if response.status_code == 200:
                auth_response = response.json()

                access_token = auth_response.get("access_token")
                new_refresh_token = auth_response.get("refresh_token")
                expires_in = auth_response.get("expires_in", 3600)

                if not access_token:
                    raise AuthenticationError("No access token in refresh response")

                # Update tokens
                self.config.auth.jwt_token = access_token
                if new_refresh_token:
                    self.config.auth.refresh_token = new_refresh_token

                # Update expiry
                expiry_time = datetime.now(UTC) + timedelta(seconds=expires_in)
                self.config.auth.token_expiry = expiry_time.isoformat()

                # Reload token info
                self._load_token_info()

                # Save configuration
                self.config.save_configuration()

                logger.info("Token refresh successful")
                return True

            raise AuthenticationError(f"Token refresh failed: HTTP {response.status_code}")

        except requests.exceptions.RequestException as e:
            raise AuthenticationError(f"Network error during token refresh: {e}") from e
        except Exception as e:
            raise AuthenticationError(f"Token refresh failed: {e}") from e

    def validate_token(self) -> bool:
        """
        Validate the current JWT token

        Returns:
            True if token is valid and not expired

        Raises:
            TokenExpiredError: If token is expired
            AuthenticationError: If token is invalid
        """
        if not self.config.auth.jwt_token:
            raise AuthenticationError("No JWT token available")

        try:
            # Check if token is expired
            if self.is_token_expired():
                raise TokenExpiredError("JWT token has expired")

            # Validate token format and signature
            token_info = validate_jwt_token(self.config.auth.jwt_token)

            # Additional validation can be added here
            # For example, check issuer, audience, etc.

            self._token_info = token_info
            logger.debug("JWT token validation successful")
            return True

        except ValueError as e:
            raise AuthenticationError(f"Invalid JWT token: {e}") from e
        except Exception as e:
            raise AuthenticationError(f"Token validation failed: {e}") from e

    def is_token_expired(self) -> bool:
        """
        Check if the current token is expired

        Returns:
            True if token is expired or about to expire (within 5 minutes)
        """
        if not self.config.auth.token_expiry:
            return True  # Consider expired if no expiry time

        try:
            expiry_time = datetime.fromisoformat(self.config.auth.token_expiry)
            now = datetime.now(UTC)

            # Consider token expired if it expires within 5 minutes
            buffer_minutes = 5
            buffer_time = expiry_time - timedelta(minutes=buffer_minutes)

            return now >= buffer_time

        except Exception as e:
            logger.warning(f"Error checking token expiry: {e}")
            return True  # Consider expired on error

    def get_auth_headers(self) -> dict[str, str]:
        """
        Get authentication headers for API requests

        Returns:
            Dictionary with Authorization header

        Raises:
            AuthenticationError: If no valid token available
        """
        if not self.config.auth.jwt_token:
            raise AuthenticationError("No JWT token available")

        # Validate token before using
        if self.is_token_expired():
            try:
                self.refresh_token()
            except AuthenticationError:
                raise TokenExpiredError("JWT token expired and refresh failed") from None

        return {"Authorization": f"Bearer {self.config.auth.jwt_token}"}

    def logout(self) -> None:
        """
        Logout and clear authentication tokens
        """
        try:
            # Optional: Call logout endpoint
            if self.config.auth.jwt_token:
                logout_url = f"{self.config.api.base_url.rstrip('/')}/auth/logout"
                try:
                    headers = self.get_auth_headers()
                    self._session.post(logout_url, headers=headers, timeout=10)
                except Exception as e:
                    logger.warning(f"Logout endpoint call failed: {e}")

        except Exception as e:
            logger.warning(f"Error during logout: {e}")

        finally:
            # Clear tokens
            self.config.auth.jwt_token = None
            self.config.auth.refresh_token = None
            self.config.auth.token_expiry = None
            self._token_info = None

            # Save configuration
            self.config.save_configuration()

            logger.info("Logout completed")

    def get_user_info(self) -> dict[str, Any] | None:
        """
        Get user information from JWT token

        Returns:
            User information dictionary or None if not available
        """
        if not self._token_info:
            try:
                self.validate_token()
            except AuthenticationError:
                return None

        if self._token_info and "payload" in self._token_info:
            payload = self._token_info["payload"]
            return {
                "user_id": payload.get("sub"),
                "username": payload.get("username"),
                "email": payload.get("email"),
                "roles": payload.get("roles", []),
                "permissions": payload.get("permissions", []),
                "issued_at": payload.get("iat"),
                "expires_at": payload.get("exp"),
            }

        return None

    def has_permission(self, permission: str) -> bool:
        """
        Check if user has specific permission

        Args:
            permission: Permission to check

        Returns:
            True if user has permission
        """
        user_info = self.get_user_info()
        if not user_info:
            return False

        permissions = user_info.get("permissions", [])
        return permission in permissions

    def has_role(self, role: str) -> bool:
        """
        Check if user has specific role

        Args:
            role: Role to check

        Returns:
            True if user has role
        """
        user_info = self.get_user_info()
        if not user_info:
            return False

        roles = user_info.get("roles", [])
        return role in roles

    def get_status(self) -> str:
        """
        Get authentication status

        Returns:
            Status string
        """
        if not self.config.auth.jwt_token:
            return "Not authenticated"

        try:
            if self.is_token_expired():
                return "Token expired"

            self.validate_token()
            user_info = self.get_user_info()

            if user_info:
                username = user_info.get("username") or user_info.get("email") or "Unknown"
                return f"Authenticated as {username}"
            return "Authenticated (user info unavailable)"

        except AuthenticationError as e:
            return f"Authentication error: {e!s}"
        except Exception as e:
            logger.error(f"Error getting auth status: {e}")
            return "Authentication status unknown"

    def _load_token_info(self) -> None:
        """Load and validate token information"""
        if not self.config.auth.jwt_token:
            self._token_info = None
            return

        try:
            self._token_info = validate_jwt_token(self.config.auth.jwt_token)
            logger.debug("Token information loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load token info: {e}")
            self._token_info = None

    def make_authenticated_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make an authenticated HTTP request

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional request parameters

        Returns:
            Response object

        Raises:
            AuthenticationError: If authentication fails
            requests.RequestException: For network errors
        """
        if not self.config.auth.jwt_token:
            raise AuthenticationError("No authentication token available")

        # Validate token before making request
        if self.is_token_expired():
            try:
                self.refresh_token()
            except AuthenticationError:
                raise TokenExpiredError("Token expired and refresh failed") from None

        # Build URL
        base_url = self.config.api.base_url.rstrip("/")
        endpoint = endpoint.lstrip("/")
        url = f"{base_url}/{endpoint}"

        # Add authentication headers
        headers = kwargs.get("headers", {})
        auth_headers = self.get_auth_headers()
        headers.update(auth_headers)
        kwargs["headers"] = headers

        # Make request
        logger.debug(f"Making {method} request to {url}")
        response = self._session.request(method, url, **kwargs)

        # Handle authentication errors
        if response.status_code == 401:
            raise AuthenticationError("Authentication required or token invalid")

        return response

    def get_token_remaining_time(self) -> int | None:
        """
        Get remaining time in seconds until token expires

        Returns:
            Remaining seconds or None if expiry time unavailable
        """
        if not self.config.auth.token_expiry:
            return None

        try:
            expiry_time = datetime.fromisoformat(self.config.auth.token_expiry)
            now = datetime.now(UTC)
            remaining = int((expiry_time - now).total_seconds())
            return max(0, remaining)
        except Exception as e:
            logger.warning(f"Error calculating token remaining time: {e}")
            return None


# Convenience functions
def require_auth(auth_manager: AuthManager):
    """
    Decorator to require authentication for CLI commands

    Args:
        auth_manager: AuthManager instance

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if not auth_manager.config.auth.jwt_token:
                raise AuthenticationError("Authentication required. Please login first.")

            # Validate token before executing command
            if auth_manager.is_token_expired():
                try:
                    auth_manager.refresh_token()
                except AuthenticationError as e:
                    raise AuthenticationError(f"Token expired and refresh failed: {e}") from e

            return func(*args, **kwargs)

        return wrapper

    return decorator


def login(self, username: str, password: str) -> dict[str, Any]:
    """Alias for authenticate() — authenticates and returns user info."""
    self.authenticate(username, password)
    return {"status": "authenticated", "user": username}


def revoke(self) -> None:
    """Revoke session — alias for logout."""
    self.logout()


AuthManager.login = login
AuthManager.revoke = revoke

AuthManager._access_token = property(
    lambda self: self.config.auth.jwt_token, lambda self, v: setattr(self.config.auth, "jwt_token", v)
)
AuthManager._refresh_token = property(
    lambda self: self.config.auth.refresh_token, lambda self, v: setattr(self.config.auth, "refresh_token", v)
)


def _get_token_expiry(self):
    if not self.config.auth.token_expiry:
        return 0.0
    try:
        from datetime import datetime

        return datetime.fromisoformat(self.config.auth.token_expiry).timestamp()
    except Exception:
        return 0.0


def _set_token_expiry(self, value: float) -> None:
    from datetime import UTC, datetime

    self.config.auth.token_expiry = datetime.fromtimestamp(value, tz=UTC).isoformat()


AuthManager._token_expiry = property(_get_token_expiry, _set_token_expiry)


class TokenManager:
    """Stub TokenManager for compatibility."""

    def __init__(self):
        self._access_token = None
        self._refresh_token = None
        self._token_expiry = None

    def set_tokens(self, access_token, refresh_token, expires_in):
        import time
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._token_expiry = time.time() + expires_in

    def get_access_token(self):
        return self._access_token

    def get_refresh_token(self):
        return self._refresh_token

    def is_token_expired(self):
        import time
        return self._token_expiry is None or time.time() > self._token_expiry

    def clear_tokens(self):
        self._access_token = None
        self._refresh_token = None
        self._token_expiry = None

    def get_token_age(self):
        import time
        if self._token_expiry:
            return time.time() - (self._token_expiry - 3600)
        return 0

    def get_time_until_expiry(self):
        import time
        if self._token_expiry:
            return self._token_expiry - time.time()
        return 0

    def validate_token_format(self, token):
        return "." in token

    def decode_token_payload(self, token):
        import json, base64
        try:
            parts = token.split(".")
            payload = parts[1]
            padding = "=" * (-len(payload) % 4)
            decoded = json.loads(base64.urlsafe_b64decode(payload + padding))
            return decoded
        except Exception:
            return {}


__all__ = [
    "AuthManager",
    "AuthenticationError",
    "TokenExpiredError",
    "TokenManager",
    "require_auth",
]
