"""
Main CLI entry point for Pixelated AI CLI

This module provides the main command-line interface using Click framework
with three primary entry points: web-frontend, cli-interface, and mcp-connect.
"""

import click
import sys
import os
from pathlib import Path
from typing import Optional

# Add the ai directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.config import CLIConfig
from cli.auth import AuthManager
from cli.pipeline import PipelineManager
from cli.progress import ProgressTracker
from cli.utils import setup_logging, validate_environment, print_banner
from cli.commands import (
    web_frontend_group,
    cli_interface_group,
    mcp_connect_group,
    pipeline_group,
    config_group,
    auth_group,
)


@click.group()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Enable debug mode",
)
@click.option(
    "--profile",
    "-p",
    type=str,
    default="default",
    help="Configuration profile to use",
)
@click.pass_context
def cli(ctx: click.Context, config_file: Optional[Path], verbose: bool, debug: bool, profile: str):
    """
    Pixelated AI CLI - Command Line Interface for TechDeck-Python Pipeline Integration
    
    A comprehensive CLI tool for managing AI pipelines, configuration, and interactions
    with the Pixelated Empathy platform. Supports HIPAA++ compliance and secure
    communication with Flask backend services.
    
    Examples:
        pixelated-ai web-frontend start
        pixelated-ai cli-interface pipeline run --config my-config.yaml
        pixelated-ai mcp-connect agent --name my-agent
        pixelated-ai pipeline status --id pipeline-123
    """
    # Initialize context
    ctx.ensure_object(dict)
    
    # Setup logging first
    log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
    logger = setup_logging(log_level)
    
    # Print banner
    print_banner()
    
    # Validate environment
    try:
        validate_environment()
    except Exception as e:
        click.echo(f"Environment validation failed: {e}", err=True)
        sys.exit(1)
    
    # Initialize configuration
    try:
        config = CLIConfig(config_file=config_file, profile=profile)
        ctx.obj["config"] = config
        ctx.obj["logger"] = logger
    except Exception as e:
        click.echo(f"Failed to initialize configuration: {e}", err=True)
        sys.exit(1)
    
    # Initialize managers
    try:
        ctx.obj["auth_manager"] = AuthManager(config)
        ctx.obj["pipeline_manager"] = PipelineManager(config)
        ctx.obj["progress_tracker"] = ProgressTracker(config)
    except Exception as e:
        click.echo(f"Failed to initialize managers: {e}", err=True)
        sys.exit(1)
    
    logger.info("Pixelated AI CLI initialized successfully")


# Add command groups
cli.add_command(web_frontend_group)
cli.add_command(cli_interface_group)
cli.add_command(mcp_connect_group)
cli.add_command(pipeline_group)
cli.add_command(config_group)
cli.add_command(auth_group)


@cli.command()
@click.pass_context
def version(ctx: click.Context):
    """Display version information"""
    from cli import __version__, __description__, __author__
    
    click.echo(f"Pixelated AI CLI v{__version__}")
    click.echo(f"{__description__}")
    click.echo(f"Author: {__author__}")
    
    # Display additional version info
    config = ctx.obj["config"]
    click.echo(f"Config Profile: {config.profile}")
    click.echo(f"API Endpoint: {config.api_base_url}")


@cli.command()
@click.pass_context
def status(ctx: click.Context):
    """Display system status and health information"""
    logger = ctx.obj["logger"]
    config = ctx.obj["config"]
    auth_manager = ctx.obj["auth_manager"]
    pipeline_manager = ctx.obj["pipeline_manager"]
    
    try:
        click.echo("System Status:")
        click.echo("=" * 50)
        
        # Configuration status
        click.echo(f"✓ Configuration: {config.profile} profile active")
        
        # Authentication status
        auth_status = auth_manager.get_status()
        click.echo(f"✓ Authentication: {auth_status}")
        
        # Pipeline manager status
        pipeline_status = pipeline_manager.get_status()
        click.echo(f"✓ Pipeline Manager: {pipeline_status}")
        
        # API connectivity
        api_status = pipeline_manager.check_api_health()
        click.echo(f"✓ API Health: {api_status}")
        
        click.echo("=" * 50)
        click.echo("All systems operational ✓")
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        click.echo(f"❌ Status check failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()