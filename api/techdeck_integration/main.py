"""
Main entry point for TechDeck-Python Pipeline Integration Flask Service.

This module provides the main application runner with proper initialization,
error handling, and graceful shutdown capabilities.
"""

import os
import sys
import signal
import logging
from typing import Optional

from .app import create_app
from .config import get_config
from .utils.logger import setup_logging, get_logger
from .integration.redis_client import RedisClient


def setup_signal_handlers(app) -> None:
    """
    Setup signal handlers for graceful shutdown.
    
    Args:
        app: Flask application instance
    """
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logger = get_logger(__name__)
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        
        try:
            # Close Redis connections
            if hasattr(app, 'redis_client'):
                app.redis_client.close()
                logger.info("Redis connections closed")
            
            # Perform any other cleanup
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def validate_environment() -> None:
    """
    Validate required environment variables and configuration.
    
    Raises:
        ValueError: If required configuration is missing
    """
    required_vars = [
        'SECRET_KEY',
        'JWT_SECRET_KEY',
        'MONGODB_URI',
        'REDIS_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")


def main(host: Optional[str] = None, port: Optional[int] = None, debug: Optional[bool] = None) -> None:
    """
    Main application entry point.
    
    Args:
        host: Host to bind to (overrides config)
        port: Port to bind to (overrides config)
        debug: Debug mode (overrides config)
    """
    # Setup logging first
    logger = get_logger(__name__)
    
    try:
        logger.info("Starting TechDeck Flask service")
        
        # Validate environment
        validate_environment()
        
        # Get configuration
        config = get_config()
        
        # Override config with command line arguments if provided
        if host:
            config.HOST = host
        if port:
            config.PORT = port
        if debug is not None:
            config.DEBUG = debug
        
        # Setup logging with config
        setup_logging(config)
        
        # Create Flask application
        app = create_app(config)
        
        # Setup signal handlers
        setup_signal_handlers(app)
        
        logger.info(f"TechDeck Flask service starting on {config.HOST}:{config.PORT}")
        logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")
        logger.info(f"Debug mode: {config.DEBUG}")
        
        # Run the application
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG,
            threaded=True,
            use_reloader=False  # Disable reloader for production stability
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Application failed to start: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # Allow command line overrides
    import argparse
    
    parser = argparse.ArgumentParser(description='TechDeck Flask Service')
    parser.add_argument('--host', type=str, help='Host to bind to')
    parser.add_argument('--port', type=int, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    main(
        host=args.host,
        port=args.port,
        debug=args.debug
    )