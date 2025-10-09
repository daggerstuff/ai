"""
Main Flask application factory for TechDeck-Python Pipeline Integration Service.

This module implements the Flask application factory pattern with comprehensive
configuration management, middleware registration, and blueprint initialization.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from flask import Flask, request, g
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import TechDeckServiceConfig
from .auth.middleware import JWTAuthMiddleware
from .utils.logger import setup_logging
from .utils.error_handler import TechDeckErrorHandler
from .websocket.manager import WebSocketManager
from .integration.redis_client import RedisClient
from .error_handling.error_handler import ErrorHandler


def create_app(config: Optional[TechDeckServiceConfig] = None) -> Flask:
    """
    Create and configure Flask application for TechDeck integration service.
    
    Args:
        config: Optional configuration object. If None, uses default config.
        
    Returns:
        Configured Flask application instance
        
    Raises:
        ValueError: If required configuration is missing
        RuntimeError: If critical services cannot be initialized
    """
    app = Flask(__name__)
    
    # Load configuration
    if config is None:
        config = TechDeckServiceConfig()
    
    # Validate required configuration
    _validate_configuration(config)
    
    # Configure Flask app
    app.config.from_object(config)
    app.config['TECHDECK_CONFIG'] = config
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Initializing TechDeck Flask application")
    
    try:
        # Initialize extensions and services
        _init_extensions(app, config)
        
        # Register blueprints
        _register_blueprints(app)
        
        # Register error handlers
        _register_error_handlers(app, config)
        
        # Register middleware
        _register_middleware(app, config)
        
        # Register application hooks
        _register_hooks(app)
        
        logger.info("TechDeck Flask application initialized successfully")
        return app
        
    except Exception as e:
        logger.critical(f"Failed to initialize Flask application: {e}")
        raise RuntimeError(f"Application initialization failed: {e}")


def _validate_configuration(config: TechDeckServiceConfig) -> None:
    """Validate required configuration parameters."""
    required_vars = [
        'SECRET_KEY',
        'REDIS_URL',
        'MONGODB_URI'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not getattr(config, var, None):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required configuration variables: {missing_vars}")


def _init_extensions(app: Flask, config: TechDeckServiceConfig) -> None:
    """Initialize Flask extensions and external services."""
    logger = logging.getLogger(__name__)
    
    # Initialize CORS
    CORS(app, origins=getattr(config, 'ALLOWED_ORIGINS', ['*']))
    logger.debug("CORS initialized")
    
    # Initialize Redis client
    try:
        app.redis_client = RedisClient(config.REDIS_URL, config.REDIS_DB)
        logger.debug("Redis client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis client: {e}")
        raise
    
    # Initialize WebSocket manager
    try:
        app.websocket_manager = WebSocketManager(config)
        logger.debug("WebSocket manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize WebSocket manager: {e}")
        raise
    
    # Initialize error handler
    try:
        app.error_handler = ErrorHandler(config)
        logger.debug("Error handler initialized")
    except Exception as e:
        logger.error(f"Failed to initialize error handler: {e}")
        raise


def _register_blueprints(app: Flask) -> None:
    """Register API route blueprints."""
    logger = logging.getLogger(__name__)
    
    # Import and register blueprints
    from .routes.datasets import datasets_bp
    from .routes.pipeline import pipeline_bp
    
    blueprints = [
        (datasets_bp, '/api/v1/datasets'),
        (pipeline_bp, '/api/v1/pipeline')
    ]
    
    for blueprint, url_prefix in blueprints:
        app.register_blueprint(blueprint, url_prefix=url_prefix)
        logger.debug(f"Registered blueprint {blueprint.name} at {url_prefix}")


def _register_error_handlers(app: Flask, config: TechDeckServiceConfig) -> None:
    """Register comprehensive error handlers."""
    logger = logging.getLogger(__name__)
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors."""
        return {
            'success': False,
            'error': {
                'code': 'NOT_FOUND',
                'message': 'The requested resource was not found',
                'timestamp': datetime.utcnow().isoformat(),
                'path': request.path
            }
        }, 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 Method Not Allowed errors."""
        return {
            'success': False,
            'error': {
                'code': 'METHOD_NOT_ALLOWED',
                'message': f'Method {request.method} is not allowed for this endpoint',
                'timestamp': datetime.utcnow().isoformat(),
                'allowed_methods': getattr(error, 'valid_methods', [])
            }
        }, 405
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle 413 Request Entity Too Large errors."""
        return {
            'success': False,
            'error': {
                'code': 'REQUEST_ENTITY_TOO_LARGE',
                'message': 'The request payload is too large',
                'timestamp': datetime.utcnow().isoformat(),
                'max_size_mb': config.MAX_FILE_SIZE_MB
            }
        }, 413
    
    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 Internal Server Error."""
        error_id = str(datetime.utcnow().timestamp())
        logger.error(f"Internal server error {error_id}: {error}")
        
        return {
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': 'An internal server error occurred',
                'timestamp': datetime.utcnow().isoformat(),
                'error_id': error_id,
                'support_reference': error_id[:8]
            }
        }, 500
    
    logger.debug("Error handlers registered")


def _register_middleware(app: Flask, config: TechDeckServiceConfig) -> None:
    """Register middleware components."""
    logger = logging.getLogger(__name__)
    
    # Add ProxyFix for reverse proxy support
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Initialize JWT authentication middleware
    try:
        jwt_middleware = JWTAuthMiddleware(app.wsgi_app, config)
        app.wsgi_app = jwt_middleware
        logger.debug("JWT authentication middleware registered")
    except Exception as e:
        logger.error(f"Failed to register JWT middleware: {e}")
        raise


def _register_hooks(app: Flask) -> None:
    """Register application lifecycle hooks."""
    logger = logging.getLogger(__name__)
    
    @app.before_request
    def before_request():
        """Execute before each request."""
        g.start_time = datetime.utcnow()
        g.request_id = request.headers.get('X-Request-ID', str(datetime.utcnow().timestamp()))
        
        # Log request details
        logger.info(f"Request {g.request_id}: {request.method} {request.path}")
    
    @app.after_request
    def after_request(response):
        """Execute after each request."""
        if hasattr(g, 'start_time'):
            duration = (datetime.utcnow() - g.start_time).total_seconds()
            response.headers['X-Response-Time'] = f"{duration:.3f}s"
            response.headers['X-Request-ID'] = g.request_id
            
            # Log response details
            logger.info(f"Response {g.request_id}: {response.status_code} in {duration:.3f}s")
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response
    
    @app.teardown_request
    def teardown_request(exception=None):
        """Execute after request processing (even if exception occurred)."""
        if exception:
            logger.error(f"Exception in request {getattr(g, 'request_id', 'unknown')}: {exception}")


if __name__ == '__main__':
    # Create and run the application
    app = create_app()
    
    # Get configuration
    config = app.config['TECHDECK_CONFIG']
    
    # Run the application
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True,
        use_reloader=False  # Disable reloader for production stability
    )