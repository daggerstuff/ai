import unittest
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import jwt
from flask import Flask, g

# Ensure we can import the module without unsafe PATH mods
# We assume this script is run as a module:
# uv run python -m ai.tests.manual_test_middleware
# But if run directly, we might need to add project root to sys.path safely.
# However, user explicitly warned against permissions. We will stick to mocking.

# Mocking the dependencies if they aren't available in the test environment immediately
with suppress(ImportError):
    from ai.pkg_mera.core.api.techdeck_integration.auth.middleware import JWTAuthMiddleware


class TestJWTAuthMiddleware(unittest.TestCase):
    def setUp(self):
        # Create minimal Flask app and push context
        self.flask_app = Flask(__name__)
        self.app_context = self.flask_app.app_context()
        self.app_context.push()

        self.app = MagicMock()
        self.config = MagicMock()
        self.config.JWT_SECRET_KEY = "test_secret"
        self.config.JWT_ALGORITHM = "HS256"
        self.config.RATE_LIMIT_PER_MINUTE = 100

        # We need to patch get_logger since we might not have the logging setup
        self.logger_patcher = patch("ai.api.techdeck_integration.auth.middleware.get_logger")
        self.mock_get_logger = self.logger_patcher.start()
        self.mock_logger = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger

    def tearDown(self):
        self.logger_patcher.stop()
        self.app_context.pop()

    def test_public_endpoint_skips_auth(self):
        # Arrange
        middleware = JWTAuthMiddleware(self.app, self.config)
        environ = {"PATH_INFO": "/api/v1/system/health", "REQUEST_METHOD": "GET"}
        start_response = MagicMock()

        # Act
        middleware(environ, start_response)

        # Assert
        # Should call the wrapped app directly without auth checks
        self.app.assert_called_with(environ, start_response)

    def test_valid_token_allows_access(self):
        # Arrange
        middleware = JWTAuthMiddleware(self.app, self.config)

        # Create a valid token
        payload = {
            "sub": "user123",
            "email": "test@example.com",
            "role": "admin",
            "iat": datetime.now(UTC).timestamp(),
            "exp": (datetime.now(UTC) + timedelta(hours=1)).timestamp(),
        }
        token = jwt.encode(payload, self.config.JWT_SECRET_KEY, algorithm=self.config.JWT_ALGORITHM)

        environ = {
            "PATH_INFO": "/api/v1/protected/resource",
            "REQUEST_METHOD": "GET",
            "HTTP_AUTHORIZATION": f"Bearer {token}",
        }
        start_response = MagicMock()

        # Act
        # We need to patch the internal validation methods or just let them run
        # if pure logic. But Request object needs to be mocked or we rely on werkzeug
        with patch("ai.api.techdeck_integration.auth.middleware.Request") as MockRequest:
            mock_request = MockRequest.return_value
            mock_request.path = "/api/v1/protected/resource"
            mock_request.headers = {"Authorization": f"Bearer {token}"}

            middleware(environ, start_response)

        # Assert
        self.app.assert_called()
        # Verify logger was called for success
        self.mock_logger.info.assert_called_with("Authentication successful", extra=unittest.mock.ANY)
        # Verify g was populated
        assert g.user["role"] == "admin"
        assert g.user_id == "user123"

    def test_missing_token_returns_401(self):
        # Arrange
        middleware = JWTAuthMiddleware(self.app, self.config)
        environ = {
            "PATH_INFO": "/api/v1/protected/resource",
            "REQUEST_METHOD": "GET",
            # No Authorization header
        }
        start_response = MagicMock()

        # Act
        with patch("ai.api.techdeck_integration.auth.middleware.Request") as MockRequest:
            mock_request = MockRequest.return_value
            mock_request.path = "/api/v1/protected/resource"
            mock_request.headers = {}

            middleware(environ, start_response)

        # Assert
        self.app.assert_not_called()
        start_response.assert_called_with("401 Unauthorized", unittest.mock.ANY)


if __name__ == "__main__":
    unittest.main()
