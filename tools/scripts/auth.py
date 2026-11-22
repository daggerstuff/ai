"""
Authentication management for Pixelated AI CLI

This module handles JWT token management, authentication flows, and secure
credential storage with HIPAA++ compliance.
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from .config import CLIConfig
from .exceptions import CLIAuthError
from .utils import validate_jwt_token

logger = logging.getLogger("pixelated-ai-cli")

HTTP_OK = 200


class AuthenticationError(CLIAuthError):
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
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._token_expiry: float | None = None
        self._session = requests.Session()

        # Configure session
        self._session.timeout = config.api.timeout
        self._session.headers.update(
            {
                "User-Agent": "Pixelated-AI-CLI/0.1.0",
                "Content-Type": "application/json",
            }
        )

        self._adopt_config_tokens()

    def _adopt_config_tokens(self) -> None:
        """Seed the in-memory token state from a previously saved configuration."""
        if not self.config.auth.jwt_token:
            return

        self._access_token = self.config.auth.jwt_token
        self._refresh_token = self.config.auth.refresh_token or None
        self._token_expiry = self._parse_expiry(self.config.auth.token_expiry)
        self._load_token_info()

    @staticmethod
    def _parse_expiry(value: str | None) -> float | None:
        """Convert a stored ISO-8601 expiry into an epoch timestamp."""
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            logger.warning("Stored token expiry is not a valid timestamp: %r", value)
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()

    def _store_tokens(
        self,
        access_token: str,
        refresh_token: str | None,
        expires_in: float,
    ) -> None:
        """Record freshly issued tokens in memory and in the saved configuration."""
        self._access_token = access_token
        if refresh_token:
            self._refresh_token = refresh_token
        self._token_expiry = time.time() + float(expires_in)

        self.config.auth.jwt_token = access_token
        self.config.auth.refresh_token = self._refresh_token
        self.config.auth.token_expiry = datetime.fromtimestamp(self._token_expiry, UTC).isoformat()

        self._load_token_info()
        self.config.save_configuration()

    def _clear_tokens(self) -> None:
        """Drop every trace of the current session, in memory and on disk."""
        self._access_token = None
        self._refresh_token = None
        self._token_expiry = None
        self._token_info = None

        self.config.auth.jwt_token = None
        self.config.auth.refresh_token = None
        self.config.auth.token_expiry = None

    def is_authenticated(self) -> bool:
        """Whether a usable access token is currently held."""
        return bool(self._access_token) and not self.is_token_expired()

    def login(self, username: str, password: str, client_id: str | None = None) -> bool:
        """Authenticate with a username and password; alias over ``authenticate``."""
        return self.authenticate(username, password, client_id=client_id)

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
            try:
                response = requests.post(auth_url, json=auth_data, timeout=self.config.api.timeout)
            except Exception as e:
                raise AuthenticationError(f"Network error during authentication: {e}") from e

            if response.status_code == HTTP_OK:
                auth_response = response.json()

                # Extract tokens
                access_token = auth_response.get("access_token")
                refresh_token = auth_response.get("refresh_token")
                expires_in = auth_response.get("expires_in", 3600)

                if not access_token:
                    raise AuthenticationError("No access token in response")

                self._store_tokens(access_token, refresh_token, expires_in)

                logger.info(f"Authentication successful for user: {username}")
                return True

            error_msg = f"Authentication failed: HTTP {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f" - {error_data['error']}"
                if "error_description" in error_data:
                    error_msg += f": {error_data['error_description']}"
            except Exception:
                pass

            raise AuthenticationError(error_msg)

        except AuthenticationError:
            raise
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
        if not self._refresh_token:
            raise AuthenticationError("No refresh token available")

        refresh_data = {
            "refresh_token": self._refresh_token,
            "grant_type": "refresh_token",
        }

        if self.config.auth.client_id:
            refresh_data["client_id"] = self.config.auth.client_id
        if self.config.auth.client_secret:
            refresh_data["client_secret"] = self.config.auth.client_secret

        auth_url = f"{self.config.api.base_url.rstrip('/')}/auth/refresh"

        logger.debug("Refreshing access token")
        try:
            response = requests.post(auth_url, json=refresh_data, timeout=self.config.api.timeout)
        except Exception as e:
            raise AuthenticationError(f"Network error during token refresh: {e}") from e

        if response.status_code != HTTP_OK:
            raise AuthenticationError(f"Token refresh failed: HTTP {response.status_code}")

        auth_response = response.json()
        access_token = auth_response.get("access_token")
        if not access_token:
            raise AuthenticationError("No access token in refresh response")

        self._store_tokens(
            access_token,
            auth_response.get("refresh_token"),
            auth_response.get("expires_in", 3600),
        )

        logger.info("Token refresh successful")
        return True

    def validate_token(self) -> bool:
        """
        Validate the current JWT token

        Returns:
            True if a token is held and has not expired
        """
        if not self._access_token:
            return False

        if self.is_token_expired():
            logger.debug("JWT token has expired")
            return False

        if not self._token_info:
            self._load_token_info()

        return True

    def is_token_expired(self) -> bool:
        """
        Check if the current token is expired

        Returns:
            True if token is expired or about to expire (within 5 minutes)
        """
        expiry = self._token_expiry
        if expiry is None:
            expiry = self._parse_expiry(self.config.auth.token_expiry)

        if expiry is None:
            return True  # Consider expired if no expiry time

        buffer_minutes = 5
        return time.time() >= expiry - timedelta(minutes=buffer_minutes).total_seconds()

    def get_auth_headers(self) -> dict[str, str]:
        """
        Get authentication headers for API requests

        Returns:
            Dictionary with the Authorization header, empty when not authenticated
        """
        if not self.is_authenticated():
            return {}

        return {"Authorization": f"Bearer {self._access_token}"}

    def logout(self) -> None:
        """
        Logout and clear authentication tokens
        """
        try:
            if self.is_authenticated():
                logout_url = f"{self.config.api.base_url.rstrip('/')}/auth/logout"
                try:
                    requests.post(logout_url, headers=self.get_auth_headers(), timeout=10)
                except Exception as e:
                    logger.warning(f"Logout endpoint call failed: {e}")

        except Exception as e:
            logger.warning(f"Error during logout: {e}")

        finally:
            self._clear_tokens()
            self.config.save_configuration()
            logger.info("Logout completed")

    def get_user_info(self) -> dict[str, Any]:
        """
        Get user information for the authenticated session

        Returns:
            User information dictionary

        Raises:
            AuthenticationError: If not authenticated or the lookup fails
        """
        if not self.is_authenticated():
            raise AuthenticationError("Not authenticated")

        url = f"{self.config.api.base_url.rstrip('/')}/auth/userinfo"
        try:
            response = requests.get(url, headers=self.get_auth_headers(), timeout=self.config.api.timeout)
        except Exception as e:
            raise AuthenticationError(f"Failed to fetch user information: {e}") from e

        if response.status_code == HTTP_OK:
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            raise AuthenticationError("User information response was not an object")

        raise AuthenticationError(f"Failed to fetch user information: HTTP {response.status_code}")

    def get_token_info(self) -> dict[str, Any]:
        """
        Get information about the current token

        Returns:
            Dictionary describing the held token and its remaining lifetime
        """
        if not self._token_info:
            self._load_token_info()

        remaining = self.get_token_remaining_time()
        return {
            "authenticated": self.is_authenticated(),
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "expires_at": self._token_expiry,
            "expires_in": remaining,
            "payload": (self._token_info or {}).get("payload"),
        }

    def has_permission(self, permission: str) -> bool:
        """
        Check if user has specific permission

        Args:
            permission: Permission to check

        Returns:
            True if user has permission
        """
        try:
            user_info = self.get_user_info()
        except AuthenticationError:
            return False

        permissions = user_info.get("permissions") or []
        return permission in permissions

    def has_role(self, role: str) -> bool:
        """
        Check if user has specific role

        Args:
            role: Role to check

        Returns:
            True if user has role
        """
        try:
            user_info = self.get_user_info()
        except AuthenticationError:
            return False

        roles = user_info.get("roles") or []
        return role in roles

    def get_status(self) -> str:
        """
        Get authentication status

        Returns:
            Status string
        """
        if not self._access_token:
            return "Not authenticated"

        if self.is_token_expired():
            return "Token expired"

        try:
            user_info = self.get_user_info()
        except AuthenticationError as e:
            return f"Authenticated (user info unavailable: {e!s})"

        username = user_info.get("username") or user_info.get("email")
        if username:
            return f"Authenticated as {username}"
        return "Authenticated"

    def change_password(self, old_password: str, new_password: str) -> bool:
        """
        Change the password of the authenticated user

        Args:
            old_password: Current password
            new_password: Replacement password

        Returns:
            True if the password was changed

        Raises:
            AuthenticationError: If not authenticated or the change is rejected
        """
        if not self.is_authenticated():
            raise AuthenticationError("Not authenticated")

        url = f"{self.config.api.base_url.rstrip('/')}/auth/change-password"
        body = {"old_password": old_password, "new_password": new_password}

        try:
            response = requests.post(
                url,
                json=body,
                headers=self.get_auth_headers(),
                timeout=self.config.api.timeout,
            )
        except Exception as e:
            raise AuthenticationError(f"Network error during password change: {e}") from e

        if response.status_code == HTTP_OK:
            logger.info("Password change successful")
            return True

        raise AuthenticationError(f"Password change failed: HTTP {response.status_code}")

    def enable_mfa(self) -> dict[str, Any]:
        """
        Begin multi-factor authentication enrolment for the current user

        Returns:
            Enrolment payload with the provisioning QR code and backup codes

        Raises:
            AuthenticationError: If not authenticated or enrolment fails
        """
        if not self.is_authenticated():
            raise AuthenticationError("Not authenticated")

        url = f"{self.config.api.base_url.rstrip('/')}/auth/mfa/enable"
        try:
            response = requests.post(url, headers=self.get_auth_headers(), timeout=self.config.api.timeout)
        except Exception as e:
            raise AuthenticationError(f"Network error during MFA setup: {e}") from e

        if response.status_code == HTTP_OK:
            payload = response.json()
            if not isinstance(payload, dict):
                raise AuthenticationError("MFA setup response was not an object")
            return payload

        raise AuthenticationError(f"MFA setup failed: HTTP {response.status_code}")

    def setup_mfa(self, email: str | None = None) -> dict[str, Any]:
        """
        Enrol the current user in multi-factor authentication

        Args:
            email: Optional address the enrolment should be associated with

        Returns:
            Enrolment payload with the provisioning QR code and backup codes
        """
        if email:
            logger.debug("Requesting MFA enrolment for %s", email)
        return self.enable_mfa()

    def disable_mfa(self, password: str | None = None) -> bool:
        """
        Turn multi-factor authentication off for the current user

        Args:
            password: Current password, when the server requires re-authentication

        Returns:
            True if MFA was disabled

        Raises:
            AuthenticationError: If not authenticated or the request is rejected
        """
        if not self.is_authenticated():
            raise AuthenticationError("Not authenticated")

        url = f"{self.config.api.base_url.rstrip('/')}/auth/mfa/disable"
        body: dict[str, Any] = {}
        if password:
            body["password"] = password

        try:
            response = requests.post(
                url,
                json=body,
                headers=self.get_auth_headers(),
                timeout=self.config.api.timeout,
            )
        except Exception as e:
            raise AuthenticationError(f"Network error while disabling MFA: {e}") from e

        if response.status_code == HTTP_OK:
            logger.info("MFA disabled")
            return True

        raise AuthenticationError(f"Failed to disable MFA: HTTP {response.status_code}")

    def verify_mfa(self, code: str) -> bool:
        """
        Complete an MFA challenge and adopt the tokens it issues

        Args:
            code: One-time code from the authenticator

        Returns:
            True if the challenge was accepted

        Raises:
            AuthenticationError: If the code is rejected
        """
        url = f"{self.config.api.base_url.rstrip('/')}/auth/mfa/verify"
        try:
            response = requests.post(
                url,
                json={"code": code},
                timeout=self.config.api.timeout,
            )
        except Exception as e:
            raise AuthenticationError(f"Network error during MFA verification: {e}") from e

        if response.status_code != HTTP_OK:
            raise AuthenticationError(f"MFA verification failed: HTTP {response.status_code}")

        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise AuthenticationError("No access token in MFA verification response")

        self._store_tokens(
            access_token,
            payload.get("refresh_token"),
            payload.get("expires_in", 3600),
        )

        logger.info("MFA verification successful")
        return True

    def get_auth_history(self, days: int = 30) -> list[dict[str, Any]]:
        """
        Fetch the recent authentication events for the current user

        Args:
            days: How far back to look

        Returns:
            Authentication events, newest first

        Raises:
            AuthenticationError: If not authenticated or the lookup fails
        """
        if not self.is_authenticated():
            raise AuthenticationError("Not authenticated")

        url = f"{self.config.api.base_url.rstrip('/')}/auth/history"
        try:
            response = requests.get(
                url,
                params={"days": days},
                headers=self.get_auth_headers(),
                timeout=self.config.api.timeout,
            )
        except Exception as e:
            raise AuthenticationError(f"Failed to fetch authentication history: {e}") from e

        if response.status_code != HTTP_OK:
            raise AuthenticationError(f"Failed to fetch authentication history: HTTP {response.status_code}")

        payload = response.json()
        history = payload.get("history") if isinstance(payload, dict) else payload
        if not isinstance(history, list):
            raise AuthenticationError("Authentication history response was not a list")

        return history

    def save_credentials(self) -> None:
        """Persist the current token state to the configured credentials file."""
        self.config.save_configuration()

    def _load_token_info(self) -> None:
        """Load and validate token information"""
        if not self._access_token:
            self._token_info = None
            return

        try:
            self._token_info = validate_jwt_token(self._access_token)
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



class TokenManager:
    """Tracks the lifetime of an issued access/refresh token pair.

    ``AuthManager`` owns credentials and the login flow; this class owns what
    happens to the pair afterwards, so a caller can ask whether the access
    token is still usable without re-reading the config file.
    """

    def __init__(self) -> None:
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._token_expiry: float | None = None
        self._issued_at: float | None = None

    def set_tokens(self, access_token: str, refresh_token: str | None, expires_in: float) -> None:
        """Record a freshly issued token pair and its lifetime in seconds."""
        if expires_in < 0:
            raise ValueError("expires_in must be non-negative")
        issued_at = time.time()
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._issued_at = issued_at
        self._token_expiry = issued_at + float(expires_in)

    def get_access_token(self) -> str | None:
        return self._access_token

    def get_refresh_token(self) -> str | None:
        return self._refresh_token

    def is_token_expired(self) -> bool:
        """True when no expiry is recorded or the recorded expiry has passed."""
        if self._token_expiry is None:
            return True
        return time.time() > self._token_expiry

    def get_token_age(self) -> float:
        """Seconds since the pair was recorded, or 0.0 when none was."""
        if self._issued_at is None:
            return 0.0
        return time.time() - self._issued_at

    def get_time_until_expiry(self) -> float:
        """Seconds until the access token expires, negative once it has."""
        if self._token_expiry is None:
            return 0.0
        return self._token_expiry - time.time()

    def clear_tokens(self) -> None:
        self._access_token = None
        self._refresh_token = None
        self._token_expiry = None
        self._issued_at = None

    def validate_token_format(self, token: str) -> bool:
        """True only for a decodable three-part JWT, not merely a dotted string."""
        try:
            validate_jwt_token(token)
        except TypeError, ValueError:
            return False
        return True

    def decode_token_payload(self, token: str) -> dict[str, Any] | None:
        """Return the decoded JWT payload, or None when the token cannot be read."""
        try:
            return validate_jwt_token(token)["payload"]
        except KeyError, TypeError, ValueError:
            return None


__all__ = [
    "AuthManager",
    "AuthenticationError",
    "TokenExpiredError",
    "TokenManager",
    "require_auth",
]
