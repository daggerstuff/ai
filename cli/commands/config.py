"""
Configuration Management Commands for Pixelated AI CLI

This module provides commands for managing CLI configuration, including profiles,
environment variables, and settings.
"""

import click
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from ..utils import setup_logging, get_logger, validate_environment
from ..config import get_config, CLIConfig, save_config


logger = get_logger(__name__)


@click.group(name='config')
@click.pass_context
def config_group(ctx):
    """Configuration management commands."""
    setup_logging(ctx.obj.get('verbose', False))
    logger.info("Config command group initialized")


@config_group.command()
@click.option('--profile', '-p', help='Configuration profile to show')
@click.option('--all', 'show_all', is_flag=True, help='Show all profiles')
@click.option('--decrypt', is_flag=True, help='Show decrypted sensitive values')
@click.pass_context
def show(ctx, profile: Optional[str], show_all: bool, decrypt: bool):
    """Display current configuration."""
    try:
        if show_all:
            # Show all profiles
            config_dir = Path.home() / '.pixelated' / 'config'
            if config_dir.exists():
                click.echo("📋 Available Configuration Profiles:")
                for config_file in config_dir.glob('*.yaml'):
                    profile_name = config_file.stem
                    if profile_name != 'default':
                        click.echo(f"  • {profile_name}")
            else:
                click.echo("❌ No configuration profiles found")
                
        else:
            # Show specific profile or current
            config = get_config(profile)
            
            click.echo(f"⚙️  Configuration Profile: {config.profile_name}")
            click.echo("-" * 50)
            
            # Display API configuration
            click.echo("🔌 API Configuration:")
            click.echo(f"  Base URL: {config.api.base_url}")
            click.echo(f"  Timeout: {config.api.timeout}s")
            click.echo(f"  Max retries: {config.api.max_retries}")
            
            # Display logging configuration
            click.echo("\n📝 Logging Configuration:")
            click.echo(f"  Level: {config.logging.level}")
            click.echo(f"  File: {config.logging.file}")
            click.echo(f"  Max size: {config.logging.max_size_mb}MB")
            click.echo(f"  Backup count: {config.logging.backup_count}")
            
            # Display upload configuration
            click.echo("\n📤 Upload Configuration:")
            click.echo(f"  Max file size: {config.upload.max_file_size_mb}MB")
            click.echo(f"  Allowed types: {', '.join(config.upload.allowed_types)}")
            
            # Display security settings (without sensitive data)
            click.echo("\n🔒 Security Settings:")
            click.echo(f"  Encrypt credentials: {config.security.encrypt_credentials}")
            click.echo(f"  Validate certificates: {config.security.validate_certificates}")
            
            if decrypt and config.security.encrypt_credentials:
                click.echo("\n🔓 Decrypted Sensitive Values:")
                # This would decrypt and show sensitive values
                click.echo("  (Decryption not implemented in this example)")
                
    except Exception as e:
        logger.error(f"Configuration display failed: {e}")
        click.echo(f"❌ Configuration display failed: {e}", err=True)
        raise click.Abort()


@config_group.command()
@click.option('--profile', '-p', help='Profile to edit (creates new if not exists)')
@click.option('--set', 'set_values', multiple=True, help='Set configuration value (key=value)')
@click.option('--interactive', '-i', is_flag=True, help='Interactive configuration mode')
@click.pass_context
def set(ctx, profile: Optional[str], set_values: tuple, interactive: bool):
    """Set configuration values."""
    try:
        config = get_config(profile)
        
        if interactive:
            _interactive_config(config)
        else:
            # Parse and apply set values
            updates = {}
            for value_str in set_values:
                if '=' not in value_str:
                    click.echo(f"❌ Invalid format: {value_str}. Use key=value", err=True)
                    continue
                
                key, value = value_str.split('=', 1)
                
                # Try to parse as JSON for complex values
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                
                updates[key] = parsed_value
            
            if updates:
                _apply_config_updates(config, updates)
                save_config(config)
                click.echo("✅ Configuration updated successfully!")
            else:
                click.echo("❌ No valid configuration updates provided")
                
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        click.echo(f"❌ Configuration update failed: {e}", err=True)
        raise click.Abort()


@config_group.command()
@click.option('--profile', '-p', help='Profile to reset')
@click.option('--force', is_flag=True, help='Force reset without confirmation')
@click.pass_context
def reset(ctx, profile: Optional[str], force: bool):
    """Reset configuration to defaults."""
    try:
        if not force:
            profile_msg = f" profile '{profile}'" if profile else " configuration"
            if not click.confirm(f"⚠️  Are you sure you want to reset{profile_msg} to defaults?"):
                click.echo("❌ Reset cancelled")
                return
        
        config = get_config(profile)
        config.reset_to_defaults()
        save_config(config)
        
        profile_name = profile or "default"
        click.echo(f"✅ Configuration profile '{profile_name}' reset to defaults")
        
    except Exception as e:
        logger.error(f"Configuration reset failed: {e}")
        click.echo(f"❌ Configuration reset failed: {e}", err=True)
        raise click.Abort()


@config_group.command()
@click.option('--from-profile', required=True, help='Source profile to copy from')
@click.option('--to-profile', required=True, help='Target profile to create/copy to')
@click.option('--overwrite', is_flag=True, help='Overwrite existing target profile')
@click.pass_context
def copy(ctx, from_profile: str, to_profile: str, overwrite: bool):
    """Copy configuration from one profile to another."""
    try:
        if from_profile == to_profile:
            click.echo("❌ Source and target profiles cannot be the same", err=True)
            return
        
        source_config = get_config(from_profile)
        
        # Check if target exists
        try:
            target_config = get_config(to_profile)
            if not overwrite:
                click.echo(f"❌ Target profile '{to_profile}' already exists. Use --overwrite to replace.", err=True)
                return
        except FileNotFoundError:
            # Target doesn't exist, which is fine
            pass
        
        # Create copy
        source_config.profile_name = to_profile
        save_config(source_config)
        
        click.echo(f"✅ Configuration copied from '{from_profile}' to '{to_profile}'")
        
    except Exception as e:
        logger.error(f"Configuration copy failed: {e}")
        click.echo(f"❌ Configuration copy failed: {e}", err=True)
        raise click.Abort()


@config_group.command()
@click.option('--profile', '-p', help='Profile to validate')
@click.option('--strict', is_flag=True, help='Enable strict validation')
@click.pass_context
def validate(ctx, profile: Optional[str], strict: bool):
    """Validate configuration settings."""
    try:
        config = get_config(profile)
        
        click.echo(f"🔍 Validating configuration profile: {config.profile_name}")
        
        validation_result = _validate_config(config, strict)
        
        if validation_result['valid']:
            click.echo("✅ Configuration is valid!")
            
            if validation_result.get('warnings'):
                click.echo("⚠️  Warnings:")
                for warning in validation_result['warnings']:
                    click.echo(f"  • {warning}")
                    
        else:
            click.echo("❌ Configuration validation failed!")
            
            if validation_result.get('errors'):
                click.echo("❌ Errors:")
                for error in validation_result['errors']:
                    click.echo(f"  • {error}")
        
        # Display validation summary
        click.echo(f"\n📊 Validation Summary:")
        click.echo(f"  Errors: {len(validation_result.get('errors', []))}")
        click.echo(f"  Warnings: {len(validation_result.get('warnings', []))}")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        click.echo(f"❌ Configuration validation failed: {e}", err=True)
        raise click.Abort()


@config_group.command()
@click.option('--profile', '-p', help='Profile to export')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output file path')
@click.option('--format', type=click.Choice(['json', 'yaml']), default='yaml',
              help='Export format')
@click.option('--include-secrets', is_flag=True, help='Include sensitive configuration values')
@click.pass_context
def export(ctx, profile: Optional[str], output: str, format: str, include_secrets: bool):
    """Export configuration to file."""
    try:
        config = get_config(profile)
        
        click.echo(f"📤 Exporting configuration profile: {config.profile_name}")
        click.echo(f"📁 Output: {output}")
        click.echo(f"📊 Format: {format}")
        
        # Prepare export data
        export_data = _prepare_export_data(config, include_secrets)
        
        # Write to file
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:  # yaml
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False)
        
        click.echo(f"✅ Configuration exported successfully!")
        
    except Exception as e:
        logger.error(f"Configuration export failed: {e}")
        click.echo(f"❌ Configuration export failed: {e}", err=True)
        raise click.Abort()


@config_group.command()
@click.option('--file', '-f', type=click.Path(exists=True), required=True,
              help='Configuration file to import')
@click.option('--profile', '-p', help='Target profile name')
@click.option('--overwrite', is_flag=True, help='Overwrite existing profile')
@click.pass_context
def import_config(ctx, file: str, profile: Optional[str], overwrite: bool):
    """Import configuration from file."""
    try:
        file_path = Path(file)
        
        if not file_path.exists():
            click.echo(f"❌ File not found: {file}", err=True)
            return
        
        click.echo(f"📥 Importing configuration from: {file_path}")
        
        # Load configuration data
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                config_data = json.load(f)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            click.echo("❌ Unsupported file format. Use JSON or YAML.", err=True)
            return
        
        # Determine target profile
        target_profile = profile or config_data.get('profile_name', 'imported')
        
        # Check if target exists
        try:
            existing_config = get_config(target_profile)
            if not overwrite:
                click.echo(f"❌ Profile '{target_profile}' already exists. Use --overwrite to replace.", err=True)
                return
        except FileNotFoundError:
            # Target doesn't exist, which is fine
            pass
        
        # Create new configuration
        new_config = CLIConfig(profile_name=target_profile)
        _apply_import_data(new_config, config_data)
        
        # Save configuration
        save_config(new_config)
        
        click.echo(f"✅ Configuration imported successfully!")
        click.echo(f"🆔 Profile: {target_profile}")
        
    except Exception as e:
        logger.error(f"Configuration import failed: {e}")
        click.echo(f"❌ Configuration import failed: {e}", err=True)
        raise click.Abort()


@config_group.command()
@click.pass_context
def env_info(ctx):
    """Display environment information and requirements."""
    try:
        click.echo("🌍 Environment Information")
        click.echo("-" * 50)
        
        # Python version
        import sys
        click.echo(f"🐍 Python Version: {sys.version}")
        
        # Check required environment variables
        click.echo("\n🔐 Required Environment Variables:")
        required_vars = [
            'PIXELATED_API_URL',
            'PIXELATED_API_KEY',
            'PIXELATED_REDIS_URL'
        ]
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                if 'KEY' in var or 'SECRET' in var:
                    masked_value = value[:8] + '...' + value[-4:] if len(value) > 12 else '***'
                    click.echo(f"  ✅ {var}: {masked_value}")
                else:
                    click.echo(f"  ✅ {var}: {value}")
            else:
                click.echo(f"  ❌ {var}: Not set")
        
        # Check optional variables
        click.echo("\n🔧 Optional Environment Variables:")
        optional_vars = [
            'PIXELATED_LOG_LEVEL',
            'PIXELATED_CONFIG_DIR',
            'PIXELATED_CACHE_DIR'
        ]
        
        for var in optional_vars:
            value = os.getenv(var)
            if value:
                click.echo(f"  ✅ {var}: {value}")
            else:
                click.echo(f"  ⚠️  {var}: Not set (using default)")
        
        # Configuration directory
        config_dir = Path.home() / '.pixelated'
        click.echo(f"\n📁 Configuration Directory: {config_dir}")
        click.echo(f"  Exists: {'✅' if config_dir.exists() else '❌'}")
        
        if config_dir.exists():
            config_files = list(config_dir.glob('*.yaml'))
            click.echo(f"  Config files: {len(config_files)}")
            for config_file in config_files:
                click.echo(f"    • {config_file.name}")
        
        # System information
        click.echo(f"\n💻 System Information:")
        click.echo(f"  Platform: {sys.platform}")
        click.echo(f"  Architecture: {sys.machine}")
        
        # Memory information (if available)
        try:
            import psutil
            memory = psutil.virtual_memory()
            click.echo(f"  Memory: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available")
        except ImportError:
            click.echo("  Memory: psutil not available")
        
    except Exception as e:
        logger.error(f"Environment info failed: {e}")
        click.echo(f"❌ Environment info failed: {e}", err=True)
        raise click.Abort()


# Helper functions

def _interactive_config(config: CLIConfig) -> None:
    """Interactive configuration mode."""
    click.echo("🔄 Interactive Configuration Mode")
    click.echo("Enter new values or press Enter to keep current values")
    click.echo("-" * 50)
    
    # API Configuration
    click.echo("\n🔌 API Configuration:")
    api_url = click.prompt("API Base URL", default=config.api.base_url)
    api_timeout = click.prompt("API Timeout (seconds)", default=config.api.timeout, type=int)
    api_retries = click.prompt("Max Retries", default=config.api.max_retries, type=int)
    
    # Logging Configuration
    click.echo("\n📝 Logging Configuration:")
    log_level = click.prompt("Log Level", default=config.logging.level)
    log_file = click.prompt("Log File", default=config.logging.file)
    log_size = click.prompt("Max Log Size (MB)", default=config.logging.max_size_mb, type=int)
    log_backups = click.prompt("Backup Count", default=config.logging.backup_count, type=int)
    
    # Apply updates
    updates = {
        'api': {
            'base_url': api_url,
            'timeout': api_timeout,
            'max_retries': api_retries
        },
        'logging': {
            'level': log_level,
            'file': log_file,
            'max_size_mb': log_size,
            'backup_count': log_backups
        }
    }
    
    _apply_config_updates(config, updates)
    save_config(config)
    
    click.echo("✅ Interactive configuration completed!")


def _apply_config_updates(config: CLIConfig, updates: Dict[str, Any]) -> None:
    """Apply configuration updates."""
    for key, value in updates.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif '.' in key:
            # Handle nested updates like 'api.base_url'
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                setattr(obj, parts[-1], value)


def _validate_config(config: CLIConfig, strict: bool) -> Dict[str, Any]:
    """Validate configuration settings."""
    errors = []
    warnings = []
    
    # Validate API configuration
    if not config.api.base_url:
        errors.append("API base URL is required")
    elif not config.api.base_url.startswith(('http://', 'https://')):
        errors.append("API base URL must start with http:// or https://")
    
    if config.api.timeout <= 0:
        errors.append("API timeout must be positive")
    
    if config.api.max_retries < 0:
        errors.append("API max retries cannot be negative")
    
    # Validate logging configuration
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if config.logging.level not in valid_log_levels:
        errors.append(f"Invalid log level: {config.logging.level}")
    
    if config.logging.max_size_mb <= 0:
        errors.append("Log max size must be positive")
    
    if config.logging.backup_count < 0:
        errors.append("Log backup count cannot be negative")
    
    # Validate upload configuration
    if config.upload.max_file_size_mb <= 0:
        errors.append("Max file size must be positive")
    
    if not config.upload.allowed_types:
        warnings.append("No allowed file types specified")
    
    # Validate security settings
    if strict and config.security.encrypt_credentials:
        # Additional validation for encrypted credentials
        pass
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def _prepare_export_data(config: CLIConfig, include_secrets: bool) -> Dict[str, Any]:
    """Prepare configuration data for export."""
    data = {
        'profile_name': config.profile_name,
        'api': {
            'base_url': config.api.base_url,
            'timeout': config.api.timeout,
            'max_retries': config.api.max_retries
        },
        'logging': {
            'level': config.logging.level,
            'file': config.logging.file,
            'max_size_mb': config.logging.max_size_mb,
            'backup_count': config.logging.backup_count
        },
        'upload': {
            'max_file_size_mb': config.upload.max_file_size_mb,
            'allowed_types': config.upload.allowed_types
        },
        'security': {
            'encrypt_credentials': config.security.encrypt_credentials,
            'validate_certificates': config.security.validate_certificates
        }
    }
    
    if include_secrets:
        # Add sensitive data (in real implementation, this would be properly handled)
        data['api']['api_key'] = config.api.api_key
        data['security']['encryption_key'] = config.security.encryption_key
    
    return data


def _apply_import_data(config: CLIConfig, data: Dict[str, Any]) -> None:
    """Apply imported configuration data."""
    # Update API configuration
    if 'api' in data:
        api_data = data['api']
        if 'base_url' in api_data:
            config.api.base_url = api_data['base_url']
        if 'timeout' in api_data:
            config.api.timeout = api_data['timeout']
        if 'max_retries' in api_data:
            config.api.max_retries = api_data['max_retries']
    
    # Update logging configuration
    if 'logging' in data:
        log_data = data['logging']
        if 'level' in log_data:
            config.logging.level = log_data['level']
        if 'file' in log_data:
            config.logging.file = log_data['file']
        if 'max_size_mb' in log_data:
            config.logging.max_size_mb = log_data['max_size_mb']
        if 'backup_count' in log_data:
            config.logging.backup_count = log_data['backup_count']
    
    # Update other sections as needed
    if 'upload' in data:
        upload_data = data['upload']
        if 'max_file_size_mb' in upload_data:
            config.upload.max_file_size_mb = upload_data['max_file_size_mb']
        if 'allowed_types' in upload_data:
            config.upload.allowed_types = upload_data['allowed_types']
    
    if 'security' in data:
        security_data = data['security']
        if 'encrypt_credentials' in security_data:
            config.security.encrypt_credentials = security_data['encrypt_credentials']
        if 'validate_certificates' in security_data:
            config.security.validate_certificates = security_data['validate_certificates']