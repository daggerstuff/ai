"""
Utility functions for Pixelated AI CLI

This module provides utility functions for logging setup, environment validation,
banner printing, and other common operations.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import click
import requests
import json
from datetime import datetime

from cli.config import CLIConfig


def setup_logging(level: str = "INFO", config: Optional[CLIConfig] = None) -> logging.Logger:
    """
    Setup logging configuration for the CLI
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        config: Optional CLIConfig for advanced logging settings
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("pixelated-ai-cli")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if config and config.logging.format:
        formatter = logging.Formatter(config.logging.format)
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if configured
    if config and config.logging.file_path:
        try:
            log_file = Path(config.logging.file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=config.logging.max_file_size,
                backupCount=config.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")
    
    return logger


def validate_environment() -> bool:
    """
    Validate the environment for CLI operation
    
    Returns:
        True if environment is valid, False otherwise
        
    Raises:
        ValueError: If critical environment issues are found
    """
    logger = logging.getLogger("pixelated-ai-cli")
    
    try:
        # Check Python version
        if sys.version_info < (3, 11):
            raise ValueError("Python 3.11 or higher is required")
        
        # Check required environment variables
        required_vars = []
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
        
        # Check for required directories
        required_dirs = [
            Path.home() / ".pixelated-ai",
        ]
        
        for dir_path in required_dirs:
            try:
                dir_path.mkdir(exist_ok=True)
            except Exception as e:
                logger.warning(f"Cannot create directory {dir_path}: {e}")
        
        # Check network connectivity (optional)
        try:
            response = requests.get("https://httpbin.org/status/200", timeout=5)
            if response.status_code == 200:
                logger.debug("Network connectivity verified")
        except Exception as e:
            logger.warning(f"Network connectivity issue: {e}")
        
        logger.info("Environment validation completed successfully")
        return True
        
    except ValueError as e:
        logger.error(f"Environment validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during environment validation: {e}")
        raise ValueError(f"Environment validation failed: {e}")


def print_banner():
    """Print the CLI banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██████╗ ██╗███╗   ██╗ ██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗    ║
║    ██╔══██╗██║████╗  ██║██╔════╝ ██╔════╝██║     ██║████╗  ██║██╔════╝    ║
║    ██████╔╝██║██╔██╗ ██║██║  ███╗█████╗  ██║     ██║██╔██╗ ██║█████╗      ║
║    ██╔═══╝ ██║██║╚██╗██║██║   ██║██╔══╝  ██║     ██║██║╚██╗██║██╔══╝      ║
║    ██║     ██║██║ ╚████║╚██████╔╝██║     ███████╗██║██║ ╚████║███████╗    ║
║    ╚═╝     ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝    ║
║                                                                              ║
║    AI-Powered Mental Health & Empathy Platform                               ║
║    TechDeck-Python Pipeline Integration CLI                                  ║
║                                                                              ║
║    HIPAA++ Compliant • Bias Detection • FHE Encryption                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    click.echo(click.style(banner, fg="cyan", bold=True))


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe filesystem usage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:\"/\\|?*]', '_', filename)
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)  # Remove control characters
    filename = filename.strip('. ')  # Remove leading/trailing dots and spaces
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    return filename or "unnamed"


def validate_jwt_token(token: str) -> Dict[str, Any]:
    """
    Validate JWT token format and extract basic information
    
    Args:
        token: JWT token string
        
    Returns:
        Dictionary with token information
        
    Raises:
        ValueError: If token is invalid
    """
    try:
        # Basic JWT format validation
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")
        
        # Decode header and payload
        import base64
        
        # Add padding if needed
        def add_padding(s):
            return s + '=' * (4 - len(s) % 4)
        
        header = json.loads(base64.urlsafe_b64decode(add_padding(parts[0])))
        payload = json.loads(base64.urlsafe_b64decode(add_padding(parts[1])))
        
        # Basic validation
        if 'exp' not in payload:
            logger = logging.getLogger("pixelated-ai-cli")
            logger.warning("JWT token missing expiration")
        
        return {
            'header': header,
            'payload': payload,
            'valid': True,
            'expires_at': payload.get('exp'),
            'issued_at': payload.get('iat'),
            'subject': payload.get('sub'),
            'issuer': payload.get('iss'),
        }
        
    except Exception as e:
        raise ValueError(f"Invalid JWT token: {e}")


def check_api_health(base_url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Check API health status
    
    Args:
        base_url: API base URL
        timeout: Request timeout
        
    Returns:
        Dictionary with health status information
    """
    logger = logging.getLogger("pixelated-ai-cli")
    
    try:
        health_url = f"{base_url.rstrip('/')}/health"
        response = requests.get(health_url, timeout=timeout)
        
        if response.status_code == 200:
            try:
                health_data = response.json()
                return {
                    'status': 'healthy',
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'details': health_data
                }
            except json.JSONDecodeError:
                return {
                    'status': 'healthy',
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'details': {'message': 'API is responding'}
                }
        else:
            return {
                'status': 'unhealthy',
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'error': f"HTTP {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        return {
            'status': 'timeout',
            'error': f"Request timed out after {timeout} seconds"
        }
    except requests.exceptions.ConnectionError:
        return {
            'status': 'connection_error',
            'error': f"Cannot connect to {base_url}"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def format_datetime(dt: datetime) -> str:
    """
    Format datetime in a consistent, human-readable format
    
    Args:
        dt: Datetime object
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_datetime(dt_str: str) -> datetime:
    """
    Parse datetime string in various formats
    
    Args:
        dt_str: Datetime string
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If string cannot be parsed
    """
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Cannot parse datetime: {dt_str}")


def truncate_string(s: str, max_length: int = 100) -> str:
    """
    Truncate string to maximum length with ellipsis
    
    Args:
        s: Input string
        max_length: Maximum length
        
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string with fallback
    
    Args:
        json_str: JSON string
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent,
        'timestamp': datetime.utcnow().isoformat()
    }


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'


def colorize(text: str, color: str) -> str:
    """
    Colorize text for terminal output
    
    Args:
        text: Text to colorize
        color: Color name
        
    Returns:
        Colorized text
    """
    color_code = getattr(Colors, color.upper(), '')
    return f"{color_code}{text}{Colors.RESET}"


# Export commonly used functions
__all__ = [
    'setup_logging',
    'validate_environment',
    'print_banner',
    'format_file_size',
    'sanitize_filename',
    'validate_jwt_token',
    'check_api_health',
    'format_datetime',
    'parse_datetime',
    'truncate_string',
    'safe_json_loads',
    'get_system_info',
    'Colors',
    'colorize',
]