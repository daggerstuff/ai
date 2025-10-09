"""
JWT Authentication middleware for TechDeck Flask service.

This module implements JWT-based authentication with role-based access control,
rate limiting, and comprehensive security measures for HIPAA++ compliance.
"""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List
from functools import wraps

from flask import request, g, current_app
from werkzeug.wrappers import Request, Response
from werkzeug.exceptions import Unauthorized, TooManyRequests, Forbidden

from ..utils.logger import get_logger
from ..error_handling.custom_errors import (
    AuthenticationError, AuthorizationError, RateLimitError
)


class JWTAuthMiddleware:
    """
    JWT Authentication middleware with rate limiting and role-based access control.

    This middleware provides comprehensive authentication and authorization
    with HIPAA++ compliant audit logging and security measures.
    """

    def __init__(self, app: Optional[Callable] = None, config: Optional[Any] = None):
        """
        Initialize JWT authentication middleware.

        Args:
            app: WSGI application to wrap
            config: Configuration object with auth settings
        """
        self.app = app
        self.config = config or current_app.config.get('TECHDECK_CONFIG')
        self.logger = get_logger(__name__)

        if app is not None:
            self.app = app

    def __call__(self, environ: Dict[str, Any], start_response: Callable) -> Any:
        """
        WSGI application interface.

        Args:
            environ: WSGI environment dictionary
            start_response: WSGI start response callable

        Returns:
            WSGI response iterator
        """
        request = Request(environ)

        try:
            # Skip authentication for health check and public endpoints
            if self._is_public_endpoint(request.path):
                return self.app(environ, start_response)

            # Validate JWT token
            auth_result = self._validate_jwt_token(request)

            if not auth_result['valid']:
                return self._handle_auth_error(auth_result['error'], start_response)

            # Set user context
            g.user = auth_result['user']
            g.user_id = auth_result['user']['id']
            g.user_role = auth_result['user']['role']
            g.request_id = self._generate_request_id()

            # Check rate limiting
            rate_limit_result = self._check_rate_limit(g.user_id, request.path)

            if not rate_limit_result['allowed']:
                return self._handle_rate_limit_error(rate_limit_result, start_response)

            # Log authentication success for audit
            self._log_auth_success(g.user_id, request.path, g.request_id)

            return self.app(environ, start_response)

        except Exception as e:
            self.logger.error(f"Authentication middleware error: {e}")
            return self._handle_internal_error(str(e), start_response)

    def _is_public_endpoint(self, path: str) -> bool:
        """
        Check if endpoint is public and doesn't require authentication.

        Args:
            path: Request path

        Returns:
            True if endpoint is public, False otherwise
        """
        public_endpoints = [
            '/api/v1/system/health',
            '/api/v1/system/ready',
            '/api/v1/system/status',
            '/docs',
            '/swagger',
            '/openapi.json'
        ]

        return any(path.startswith(endpoint) for endpoint in public_endpoints)

    def _validate_jwt_token(self, request: Request) -> Dict[str, Any]:
        """
        Validate JWT token from request headers.

        Args:
            request: Flask request object

        Returns:
            Dictionary with validation result and user data
        """
        try:
            # Extract token from Authorization header
            auth_header = request.headers.get('Authorization', '')

            if not auth_header.startswith('Bearer '):
                return {
                    'valid': False,
                    'error': 'Missing or invalid Authorization header'
                }

            token = auth_header.split(' ')[1]

            # Decode JWT token
            payload = jwt.decode(
                token,
                self.config.JWT_SECRET_KEY,
                algorithms=[self.config.JWT_ALGORITHM]
            )

            # Validate token claims
            validation_result = self._validate_token_claims(payload)

            if not validation_result['valid']:
                return validation_result

            # Extract user information
            user_data = {
                'id': payload.get('sub'),
                'email': payload.get('email'),
                'role': payload.get('role', 'user'),
                'permissions': payload.get('permissions', []),
                'session_id': payload.get('session_id'),
                'issued_at': datetime.fromtimestamp(payload.get('iat', 0)),
                'expires_at': datetime.fromtimestamp(payload.get('exp', 0))
            }

            # Validate user exists and is active
            user_validation = self._validate_user_active(user_data['id'])

            if not user_validation['valid']:
                return user_validation

            return {
                'valid': True,
                'user': user_data
            }

        except jwt.ExpiredSignatureError:
            return {
                'valid': False,
                'error': 'JWT token has expired'
            }
        except jwt.InvalidTokenError as e:
            return {
                'valid': False,
                'error': f'Invalid JWT token: {str(e)}'
            }
        except Exception as e:
            self.logger.error(f"JWT validation error: {e}")
            return {
                'valid': False,
                'error': 'Token validation failed'
            }

    def _validate_token_claims(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate JWT token claims.

        Args:
            payload: Decoded JWT payload

        Returns:
            Dictionary with validation result
        """
        required_claims = ['sub', 'email', 'exp', 'iat']

        for claim in required_claims:
            if claim not in payload:
                return {
                    'valid': False,
                    'error': f'Missing required claim: {claim}'
                }

        # Check token expiration
        exp_timestamp = payload.get('exp', 0)
        if datetime.utcnow().timestamp() > exp_timestamp:
            return {
                'valid': False,
                'error': 'Token has expired'
            }

        # Check token issued time (prevent future tokens)
        iat_timestamp = payload.get('iat', 0)
        if datetime.utcnow().timestamp() < iat_timestamp - 60:  # 1 minute grace period
            return {
                'valid': False,
                'error': 'Token issued in the future'
            }

        return {'valid': True}

    def _validate_user_active(self, user_id: str) -> Dict[str, Any]:
        """
        Validate that user exists and is active.

        Args:
            user_id: User ID from JWT token

        Returns:
            Dictionary with validation result
        """
        try:
            # This would typically query the database
            # For now, we'll simulate validation
            if not user_id or len(user_id) < 1:
                return {
                    'valid': False,
                    'error': 'Invalid user ID'
                }

            # Simulate user lookup (replace with actual database query)
            # user = User.query.get(user_id)
            # if not user or not user.is_active:
            #     return {
            #         'valid': False,
            #         'error': 'User not found or inactive'
            #     }

            return {'valid': True}

        except Exception as e:
            self.logger.error(f"User validation error: {e}")
            return {
                'valid': False,
                'error': 'User validation failed'
            }

    def _check_rate_limit(self, user_id: str, path: str) -> Dict[str, Any]:
        """
        Check rate limiting for user.

        Args:
            user_id: User ID
            path: Request path

        Returns:
            Dictionary with rate limit result
        """
        try:
            # This would typically use Redis or similar for rate limiting
            # For now, we'll simulate rate limiting logic

            rate_limit_key = f"rate_limit:{user_id}:{path}"

            # Simulate rate limit check (replace with actual Redis implementation)
            # current_count = redis_client.incr(rate_limit_key)
            # if current_count == 1:
            #     redis_client.expire(rate_limit_key, 60)  # 1 minute window

            # if current_count > self.config.RATE_LIMIT_PER_MINUTE:
            #     return {
            #         'allowed': False,
            #         'retry_after': 60,
            #         'limit': self.config.RATE_LIMIT_PER_MINUTE
            #     }

            return {
                'allowed': True,
                'current_count': 1,  # Simulated
                'limit': self.config.RATE_LIMIT_PER_MINUTE
            }

        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request if rate limiting fails
            return {'allowed': True}

    def _generate_request_id(self) -> str:
        """Generate unique request ID for tracking."""
        import uuid
        return str(uuid.uuid4())

    def _log_auth_success(self, user_id: str, path: str, request_id: str) -> None:
        """
        Log successful authentication for audit purposes.

        Args:
            user_id: User ID
            path: Request path
            request_id: Request ID
        """
        self.logger.info(
            f"Authentication successful",
            extra={
                'user_id': user_id,
                'path': path,
                'request_id': request_id,
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': 'auth_success'
            }
        )

    def _handle_auth_error(self, error_message: str, start_response: Callable) -> Any:
        """
        Handle authentication errors.

        Args:
            error_message: Error message
            start_response: WSGI start response callable

        Returns:
            WSGI response
        """
        self.logger.warning(f"Authentication failed: {error_message}")

        response_data = {
            'success': False,
            'error': {
                'code': 'AUTHENTICATION_FAILED',
                'message': error_message,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        response_body = str(response_data).encode('utf-8')

        start_response('401 Unauthorized', [
            ('Content-Type', 'application/json'),
            ('Content-Length', str(len(response_body))),
            ('WWW-Authenticate', 'Bearer')
        ])

        return [response_body]

    def _handle_rate_limit_error(self, rate_limit_result: Dict[str, Any],
                                start_response: Callable) -> Any:
        """
        Handle rate limit errors.

        Args:
            rate_limit_result: Rate limit result
            start_response: WSGI start response callable

        Returns:
            WSGI response
        """
        self.logger.warning(f"Rate limit exceeded: {rate_limit_result}")

        response_data = {
            'success': False,
            'error': {
                'code': 'RATE_LIMIT_EXCEEDED',
                'message': 'Rate limit exceeded',
                'retry_after': rate_limit_result.get('retry_after', 60),
                'limit': rate_limit_result.get('limit', 0),
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        response_body = str(response_data).encode('utf-8')

        start_response('429 Too Many Requests', [
            ('Content-Type', 'application/json'),
            ('Content-Length', str(len(response_body))),
            ('Retry-After', str(rate_limit_result.get('retry_after', 60)))
        ])

        return [response_body]

    def _handle_internal_error(self, error_message: str, start_response: Callable) -> Any:
        """
        Handle internal server errors.

        Args:
            error_message: Error message
            start_response: WSGI start response callable

        Returns:
            WSGI response
        """
        self.logger.error(f"Internal server error in auth middleware: {error_message}")

        response_data = {
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': 'Internal server error',
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        response_body = str(response_data).encode('utf-8')

        start_response('500 Internal Server Error', [
            ('Content-Type', 'application/json'),
            ('Content-Length', str(len(response_body)))
        ])

        return [response_body]


def require_auth(roles: Optional[List[str]] = None):
    """
    Decorator to require authentication for Flask routes.

    Args:
        roles: List of required roles (optional)

    Returns:
        Decorator function
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if user is authenticated
            if not hasattr(g, 'user') or not g.user:
                raise Unauthorized('Authentication required')

            # Check role requirements
            if roles and g.user_role not in roles:
                raise Forbidden(f'Insufficient permissions. Required roles: {roles}')

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def require_admin(f):
    """Decorator to require admin role."""
    return require_auth(['admin'])(f)


def require_moderator(f):
    """Decorator to require moderator or admin role."""
    return require_auth(['moderator', 'admin'])(f)
