"""
Main Flask application factory for MCP (Management Control Panel) Server.

This module implements the Flask application factory pattern with comprehensive
configuration management, middleware registration, and blueprint initialization
specifically for agent interaction management.
"""

import logging
from datetime import UTC, datetime
from typing import Any, cast

from flask import Flask, g, request
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.middleware.proxy_fix import ProxyFix

from .auth.middleware import MCPAuthMiddleware
from .config import MCPConfig
from .core.agent_manager import AgentManager
from .core.pipeline_integration import PipelineIntegrationManager
from .core.task_orchestrator import TaskOrchestrator
from .error_handling.error_handler import MCPErrorRecoveryManager
from .integration.mongodb_client import MCPMongoDBClient
from .integration.redis_client import MCPRedisClient
from .routes.agents import agents_bp
from .routes.pipeline import pipeline_bp
from .routes.system import system_bp
from .routes.tasks import tasks_bp
from .utils.logger import setup_logging
from .websocket.manager import WebSocketManager


class MCPFlask(Flask):
    """Custom Flask subclass with type hinting for MCP extensions."""

    @property
    def redis_client(self) -> "MCPRedisClient":
        return cast("MCPRedisClient", self.extensions.get("redis_client"))

    @property
    def mongodb_client(self) -> "MCPMongoDBClient":
        return cast("MCPMongoDBClient", self.extensions.get("mongodb_client"))

    @property
    def error_recovery_manager(self) -> "MCPErrorRecoveryManager":
        return cast("MCPErrorRecoveryManager", self.extensions.get("error_recovery_manager"))

    @property
    def agent_manager(self) -> Any:
        return self.extensions.get("agent_manager")

    @property
    def task_orchestrator(self) -> Any:
        return self.extensions.get("task_orchestrator")

    @property
    def pipeline_manager(self) -> Any:
        return self.extensions.get("pipeline_manager")

    @property
    def auth_middleware(self) -> Any:
        return self.extensions.get("auth_middleware")

    @property
    def websocket_manager(self) -> WebSocketManager:
        return cast(WebSocketManager, self.extensions.get("websocket_manager"))

    @property
    def socketio(self) -> SocketIO:
        return cast(SocketIO, self.extensions.get("socketio"))


def create_mcp_app(config: MCPConfig | None = None) -> MCPFlask:
    """
    Create and configure Flask application for MCP server.

    Args:
        config: Optional configuration object. If None, uses default config.

    Returns:
        Configured Flask application instance

    Raises:
        ValueError: If required configuration is missing
        RuntimeError: If critical services cannot be initialized
    """
    app = MCPFlask(__name__)

    # Load configuration
    if config is None:
        config = MCPConfig()

    # Validate required configuration
    _validate_configuration(config)

    # Configure Flask app
    app.config.from_object(config)
    app.config["MCP_CONFIG"] = config

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Initializing MCP Flask application")

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

        # Initialize WebSocket manager
        _init_websocket(app, config)

        logger.info("MCP Flask application initialized successfully")
        return app

    except Exception as e:
        logger.critical(f"Failed to initialize MCP Flask application: {e}")
        raise RuntimeError(f"Application initialization failed: {e}") from e


def _validate_configuration(config: MCPConfig) -> None:
    """Validate required configuration parameters."""
    required_vars = ["SECRET_KEY", "REDIS_URL", "MONGODB_URI"]

    missing_vars = []
    for var in required_vars:
        if not getattr(config, var, None):
            missing_vars.append(var)

    if missing_vars:
        raise ValueError(f"Missing required configuration variables: {missing_vars}")


def _init_extensions(app: MCPFlask, config: MCPConfig) -> None:
    """Initialize Flask extensions and external services."""
    logger = logging.getLogger(__name__)

    # Initialize CORS
    CORS(app, origins=getattr(config, "ALLOWED_ORIGINS", ["*"]))
    logger.debug("CORS initialized")

    # Initialize Redis client (separate from main TechDeck Redis)
    try:
        app.extensions["redis_client"] = MCPRedisClient(config.REDIS_URL, config.REDIS_DB)
        logger.debug("MCP Redis client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MCP Redis client: {e}")
        raise

    # Initialize MongoDB client (separate from main TechDeck MongoDB)
    try:
        app.extensions["mongodb_client"] = MCPMongoDBClient(config.MONGODB_URI, config.MONGODB_DATABASE)
        logger.debug("MCP MongoDB client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MCP MongoDB client: {e}")
        raise

    # Initialize error recovery manager
    try:
        app.extensions["error_recovery_manager"] = MCPErrorRecoveryManager(config)
        logger.debug("MCP error recovery manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MCP error recovery manager: {e}")
        raise

    # Initialize Core Managers

    try:
        agent_manager = AgentManager(app.redis_client, app.mongodb_client)
        task_orchestrator = TaskOrchestrator(agent_manager, app.redis_client, app.mongodb_client)
        pipeline_manager = PipelineIntegrationManager(task_orchestrator, agent_manager)

        # Inject into extensions for MCPFlask property access
        app.extensions["agent_manager"] = agent_manager
        app.extensions["task_orchestrator"] = task_orchestrator
        app.extensions["pipeline_manager"] = pipeline_manager

        # Inject into app.config for serialization/discovery as requested by plan
        app.config["AGENT_MANAGER"] = agent_manager
        app.config["TASK_ORCHESTRATOR"] = task_orchestrator
        app.config["PIPELINE_MANAGER"] = pipeline_manager

        logger.debug("MCP core managers initialized and injected into extensions/config")
    except Exception as e:
        logger.error(f"Failed to initialize MCP core managers: {e}")
        raise


def _register_blueprints(app: MCPFlask) -> None:
    """Register API route blueprints."""
    logger = logging.getLogger(__name__)

    # Import and register blueprints

    blueprints = [
        (agents_bp, "/api/v1/agents"),
        (tasks_bp, "/api/v1/tasks"),
        (pipeline_bp, "/api/v1/pipeline"),
        (system_bp, "/api/v1/system"),
    ]

    for blueprint, url_prefix in blueprints:
        app.register_blueprint(blueprint, url_prefix=url_prefix)
        logger.debug(f"Registered blueprint {blueprint.name} at {url_prefix}")


def _register_error_handlers(app: MCPFlask, config: MCPConfig) -> None:
    """Register comprehensive error handlers."""
    logger = logging.getLogger(__name__)

    @app.errorhandler(404)
    def not_found(_error):
        """Handle 404 Not Found errors."""
        return {
            "success": False,
            "error": {
                "code": "NOT_FOUND",
                "message": "The requested resource was not found",
                "timestamp": datetime.now(UTC).isoformat(),
                "path": request.path,
            },
        }, 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 Method Not Allowed errors."""
        method_message = f"Method {request.method} is not allowed for this endpoint"
        return {
            "success": False,
            "error": {
                "code": "METHOD_NOT_ALLOWED",
                "message": method_message,
                "timestamp": datetime.now(UTC).isoformat(),
                "allowed_methods": getattr(error, "valid_methods", []),
            },
        }, 405

    @app.errorhandler(413)
    def request_entity_too_large(_error):
        """Handle 413 Request Entity Too Large errors."""
        return {
            "success": False,
            "error": {
                "code": "REQUEST_ENTITY_TOO_LARGE",
                "message": "The request payload is too large",
                "timestamp": datetime.now(UTC).isoformat(),
                # Using agent limit as size reference
                "max_size_mb": config.MAX_AGENTS_PER_USER,
            },
        }, 413

    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 Internal Server Error."""
        error_id = str(datetime.now(UTC).timestamp())
        logger.error(f"Internal server error {error_id}: {error}")

        return {
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred",
                "timestamp": datetime.now(UTC).isoformat(),
                "error_id": error_id,
                "support_reference": error_id[:8],
            },
        }, 500

    logger.debug("MCP error handlers registered")


def _register_middleware(app: MCPFlask, config: MCPConfig) -> None:
    """Register middleware components."""
    logger = logging.getLogger(__name__)

    # Add ProxyFix for reverse proxy support
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # Initialize MCP authentication middleware
    try:
        mcp_auth_middleware = MCPAuthMiddleware(app.wsgi_app, config)
        app.extensions["auth_middleware"] = mcp_auth_middleware
        app.wsgi_app = mcp_auth_middleware
        logger.debug("MCP authentication middleware registered")
    except Exception as e:
        logger.error(f"Failed to register MCP authentication middleware: {e}")
        raise


def _register_hooks(app: MCPFlask) -> None:
    """Register application lifecycle hooks."""
    logger = logging.getLogger(__name__)

    @app.before_request
    def before_request():
        """Execute before each request."""
        g.start_time = datetime.now(UTC)
        request_id_header = request.headers.get("X-Request-ID")
        g.request_id = request_id_header or str(datetime.now(UTC).timestamp())
        g.agent_id = request.headers.get("X-Agent-ID")  # For agent-specific requests

        # Inject managers into g context for easier access in routes
        g.agent_manager = app.agent_manager
        g.task_orchestrator = app.task_orchestrator
        g.pipeline_manager = app.pipeline_manager

        # Log request details
        logger.info(f"MCP Request {g.request_id}: {request.method} {request.path}")

    @app.after_request
    def after_request(response):
        """Execute after each request."""
        if hasattr(g, "start_time"):
            duration = (datetime.now(UTC) - g.start_time).total_seconds()
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Request-ID"] = g.request_id

            # Log response details
            logger.info(f"MCP Response {g.request_id}: {response.status_code} in {duration:.3f}s")

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Add MCP-specific headers
        response.headers["X-MCP-Version"] = "1.0.0"
        response.headers["X-MCP-Timestamp"] = datetime.now(UTC).isoformat()

        return response

    @app.teardown_request
    def teardown_request(exception=None):
        """Execute after request processing (even if exception occurred)."""
        if exception:
            request_id = getattr(g, "request_id", "unknown")
            logger.error(f"Exception in MCP request {request_id}: {exception}")


def _init_websocket(app: MCPFlask, config: MCPConfig) -> None:
    """Initialize WebSocket manager for real-time communication."""
    logger = logging.getLogger(__name__)

    if not config.WEBSOCKET_ENABLED:
        logger.info("WebSocket support disabled in configuration")
        return

    try:
        # Initialize SocketIO
        app.extensions["socketio"] = SocketIO(
            app,
            cors_allowed_origins=config.ALLOWED_ORIGINS,
            ping_interval=config.WEBSOCKET_PING_INTERVAL,
            ping_timeout=config.WEBSOCKET_PING_TIMEOUT,
            max_http_buffer_size=1024 * 1024,  # 1MB buffer
        )

        # Initialize WebSocket manager
        app.extensions["websocket_manager"] = WebSocketManager(app.socketio, config)

        # Register WebSocket event handlers
        _register_websocket_events(app)

        logger.info("WebSocket manager initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize WebSocket manager: {e}")
        raise


def _register_websocket_events(app: MCPFlask) -> None:
    """Register WebSocket event handlers."""
    logger = logging.getLogger(__name__)

    @app.socketio.on("connect")
    def handle_connect():
        """Handle client connection."""
        sid = getattr(request, "sid", "unknown")
        logger.info(f"WebSocket client connected: {sid}")

    @app.socketio.on("disconnect")
    def handle_disconnect():
        """Handle client disconnection."""
        sid = getattr(request, "sid", "unknown")
        logger.info(f"WebSocket client disconnected: {sid}")

    @app.socketio.on("agent_status_subscribe")
    def handle_agent_status_subscribe(data):
        """Handle agent status subscription."""
        app.websocket_manager.handle_agent_status_subscribe(getattr(request, "sid", "unknown"), data)

    @app.socketio.on("task_progress_subscribe")
    def handle_task_progress_subscribe(data):
        """Handle task progress subscription."""
        app.websocket_manager.handle_task_progress_subscribe(getattr(request, "sid", "unknown"), data)

    @app.socketio.on("pipeline_updates_subscribe")
    def handle_pipeline_updates_subscribe(data):
        """Handle pipeline updates subscription."""
        app.websocket_manager.handle_pipeline_updates_subscribe(getattr(request, "sid", "unknown"), data)

    logger.debug("WebSocket event handlers registered")


if __name__ == "__main__":
    # Create and run the application
    app = create_mcp_app()

    # Get configuration
    config = app.config["MCP_CONFIG"]

    # Run the application
    if config.WEBSOCKET_ENABLED:
        # Run with SocketIO support
        app.socketio.run(
            app,
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG,
            use_reloader=False,  # Disable reloader for production stability
        )
    else:
        # Run standard Flask
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG,
            threaded=True,
            use_reloader=False,  # Disable reloader for production stability
        )
