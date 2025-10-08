"""
Authentication Commands for Pixelated AI CLI

This module provides commands for user authentication, including login, logout,
token management, and user profile operations.
"""

import click
import json
import time
from typing import Optional, Dict, Any

from ..utils import setup_logging, get_logger, validate_environment
from ..config import get_config
from ..auth import AuthManager


logger = get_logger(__name__)


@click.group(name='auth')
@click.pass_context
def auth_group(ctx):
    """Authentication and user management commands."""
    setup_logging(ctx.obj.get('verbose', False))
    logger.info("Auth command group initialized")


@auth_group.command()
@click.option('--username', '-u', prompt=True, help='Username for authentication')
@click.option('--password', '-p', prompt=True, hide_input=True, help='Password for authentication')
@click.option('--profile', help='Configuration profile to use')
@click.pass_context
def login(ctx, username: str, password: str, profile: Optional[str]):
    """Login to the Pixelated platform."""
    try:
        config = get_config(profile)
        auth_manager = AuthManager(config)
        
        click.echo(f"🔐 Logging in as {username}...")
        
        # Attempt login
        login_result = auth_manager.login(username, password)
        
        if login_result['success']:
            click.echo("✅ Login successful!")
            
            # Display user information
            user_info = login_result.get('user', {})
            click.echo(f"👤 Welcome, {user_info.get('name', username)}!")
            
            if user_info.get('role'):
                click.echo(f"🎭 Role: {user_info['role']}")
            
            # Display token information
            token_info = login_result.get('token_info', {})
            if token_info.get('expires_in'):
                expires_hours = token_info['expires_in'] / 3600
                click.echo(f"⏰ Token expires in: {expires_hours:.1f} hours")
            
            # Save login information
            auth_manager.save_credentials()
            
            click.echo("💾 Credentials saved securely")
            
        else:
            error_msg = login_result.get('error', 'Login failed')
            click.echo(f"❌ Login failed: {error_msg}", err=True)
            raise click.Abort()
            
    except Exception as e:
        logger.error(f"Login failed: {e}")
        click.echo(f"❌ Login failed: {e}", err=True)
        raise click.Abort()


@auth_group.command()
@click.option('--profile', help='Configuration profile to logout from')
@click.option('--all', 'logout_all', is_flag=True, help='Logout from all profiles')
@click.pass_context
def logout(ctx, profile: Optional[str], logout_all: bool):
    """Logout from the Pixelated platform."""
    try:
        if logout_all:
            # Logout from all profiles
            config_dir = Path.home() / '.pixelated' / 'config'
            if config_dir.exists():
                profiles_logged_out = 0
                for config_file in config_dir.glob('*.yaml'):
                    profile_name = config_file.stem
                    try:
                        config = get_config(profile_name)
                        auth_manager = AuthManager(config)
                        if auth_manager.is_authenticated():
                            auth_manager.logout()
                            profiles_logged_out += 1
                    except Exception as e:
                        logger.warning(f"Failed to logout from profile {profile_name}: {e}")
                
                click.echo(f"✅ Logged out from {profiles_logged_out} profiles")
            else:
                click.echo("❌ No configuration profiles found")
                
        else:
            # Logout from specific profile
            config = get_config(profile)
            auth_manager = AuthManager(config)
            
            if not auth_manager.is_authenticated():
                click.echo("❌ Not currently logged in")
                return
            
            click.echo("🔓 Logging out...")
            auth_manager.logout()
            click.echo("✅ Logout successful!")
            
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        click.echo(f"❌ Logout failed: {e}", err=True)
        raise click.Abort()


@auth_group.command()
@click.option('--profile', help='Configuration profile to check')
@click.pass_context
def status(ctx, profile: Optional[str]):
    """Check authentication status."""
    try:
        config = get_config(profile)
        auth_manager = AuthManager(config)
        
        click.echo(f"🔍 Authentication Status (Profile: {config.profile_name})")
        click.echo("-" * 50)
        
        if auth_manager.is_authenticated():
            user_info = auth_manager.get_user_info()
            token_info = auth_manager.get_token_info()
            
            click.echo("✅ Authenticated")
            click.echo(f"👤 User: {user_info.get('name', 'Unknown')}")
            click.echo(f"📧 Email: {user_info.get('email', 'Unknown')}")
            
            if user_info.get('role'):
                click.echo(f"🎭 Role: {user_info['role']}")
            
            # Token information
            if token_info:
                expires_at = token_info.get('expires_at')
                if expires_at:
                    import datetime
                    expires_datetime = datetime.datetime.fromtimestamp(expires_at)
                    time_remaining = expires_datetime - datetime.datetime.now()
                    
                    if time_remaining.total_seconds() > 0:
                        hours = time_remaining.total_seconds() / 3600
                        click.echo(f"⏰ Token expires in: {hours:.1f} hours")
                    else:
                        click.echo("⚠️  Token has expired")
                
                if token_info.get('scopes'):
                    click.echo(f"🔑 Scopes: {', '.join(token_info['scopes'])}")
            
            # Additional user info
            if user_info.get('organization'):
                click.echo(f"🏢 Organization: {user_info['organization']}")
            
            if user_info.get('permissions'):
                click.echo(f"🔐 Permissions: {', '.join(user_info['permissions'])}")
                
        else:
            click.echo("❌ Not authenticated")
            click.echo("💡 Use 'pixelated auth login' to authenticate")
            
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        click.echo(f"❌ Status check failed: {e}", err=True)
        raise click.Abort()


@auth_group.command()
@click.option('--profile', help='Configuration profile to use')
@click.pass_context
def refresh(ctx, profile: Optional[str]):
    """Refresh authentication token."""
    try:
        config = get_config(profile)
        auth_manager = AuthManager(config)
        
        if not auth_manager.is_authenticated():
            click.echo("❌ Not currently authenticated. Please login first.", err=True)
            return
        
        click.echo("🔄 Refreshing authentication token...")
        
        refresh_result = auth_manager.refresh_token()
        
        if refresh_result['success']:
            click.echo("✅ Token refreshed successfully!")
            
            token_info = refresh_result.get('token_info', {})
            if token_info.get('expires_in'):
                expires_hours = token_info['expires_in'] / 3600
                click.echo(f"⏰ New token expires in: {expires_hours:.1f} hours")
                
        else:
            error_msg = refresh_result.get('error', 'Token refresh failed')
            click.echo(f"❌ Token refresh failed: {error_msg}", err=True)
            
            # If refresh fails, user might need to login again
            if 'invalid' in error_msg.lower() or 'expired' in error_msg.lower():
                click.echo("💡 Please try logging in again with: pixelated auth login")
                
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        click.echo(f"❌ Token refresh failed: {e}", err=True)
        raise click.Abort()


@auth_group.command()
@click.option('--profile', help='Configuration profile to use')
@click.pass_context
def profile(ctx, profile: Optional[str]):
    """Display user profile information."""
    try:
        config = get_config(profile)
        auth_manager = AuthManager(config)
        
        if not auth_manager.is_authenticated():
            click.echo("❌ Not authenticated. Please login first.", err=True)
            return
        
        click.echo("👤 User Profile")
        click.echo("-" * 50)
        
        user_info = auth_manager.get_user_info()
        
        # Basic information
        click.echo(f"Name: {user_info.get('name', 'Not provided')}")
        click.echo(f"Email: {user_info.get('email', 'Not provided')}")
        click.echo(f"Username: {user_info.get('username', 'Not provided')}")
        
        if user_info.get('role'):
            click.echo(f"Role: {user_info['role']}")
        
        # Organization information
        if user_info.get('organization'):
            click.echo(f"Organization: {user_info['organization']}")
        
        if user_info.get('department'):
            click.echo(f"Department: {user_info['department']}")
        
        # Account information
        if user_info.get('created_at'):
            click.echo(f"Account created: {user_info['created_at']}")
        
        if user_info.get('last_login'):
            click.echo(f"Last login: {user_info['last_login']}")
        
        # Permissions and capabilities
        if user_info.get('permissions'):
            click.echo(f"\n🔐 Permissions:")
            for permission in user_info['permissions']:
                click.echo(f"  • {permission}")
        
        if user_info.get('capabilities'):
            click.echo(f"\n🎯 Capabilities:")
            for capability in user_info['capabilities']:
                click.echo(f"  • {capability}")
        
        # API usage information
        if user_info.get('api_usage'):
            usage = user_info['api_usage']
            click.echo(f"\n📊 API Usage:")
            click.echo(f"  Requests today: {usage.get('requests_today', 0)}")
            click.echo(f"  Requests this month: {usage.get('requests_this_month', 0)}")
            click.echo(f"  Rate limit: {usage.get('rate_limit', 'Unknown')}")
        
        # Subscription information
        if user_info.get('subscription'):
            sub = user_info['subscription']
            click.echo(f"\n💳 Subscription:")
            click.echo(f"  Plan: {sub.get('plan', 'Unknown')}")
            click.echo(f"  Status: {sub.get('status', 'Unknown')}")
            
            if sub.get('expires_at'):
                click.echo(f"  Expires: {sub['expires_at']}")
        
    except Exception as e:
        logger.error(f"Profile display failed: {e}")
        click.echo(f"❌ Profile display failed: {e}", err=True)
        raise click.Abort()


@auth_group.command()
@click.option('--profile', help='Configuration profile to use')
@click.option('--old-password', prompt=True, hide_input=True, help='Current password')
@click.option('--new-password', prompt=True, hide_input=True, help='New password')
@click.option('--confirm-password', prompt=True, hide_input=True, help='Confirm new password')
@click.pass_context
def change_password(ctx, profile: Optional[str], old_password: str, new_password: str, confirm_password: str):
    """Change user password."""
    try:
        config = get_config(profile)
        auth_manager = AuthManager(config)
        
        if not auth_manager.is_authenticated():
            click.echo("❌ Not authenticated. Please login first.", err=True)
            return
        
        # Validate new password
        if new_password != confirm_password:
            click.echo("❌ New passwords do not match", err=True)
            return
        
        if len(new_password) < 8:
            click.echo("❌ New password must be at least 8 characters long", err=True)
            return
        
        click.echo("🔑 Changing password...")
        
        result = auth_manager.change_password(old_password, new_password)
        
        if result['success']:
            click.echo("✅ Password changed successfully!")
            click.echo("💡 You may need to login again with your new password")
            
        else:
            error_msg = result.get('error', 'Password change failed')
            click.echo(f"❌ Password change failed: {error_msg}", err=True)
            
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        click.echo(f"❌ Password change failed: {e}", err=True)
        raise click.Abort()


@auth_group.command()
@click.option('--profile', help='Configuration profile to use')
@click.option('--days', default=30, help='Number of days to show')
@click.pass_context
def history(ctx, profile: Optional[str], days: int):
    """Display authentication history."""
    try:
        config = get_config(profile)
        auth_manager = AuthManager(config)
        
        if not auth_manager.is_authenticated():
            click.echo("❌ Not authenticated. Please login first.", err=True)
            return
        
        click.echo(f"📜 Authentication History (last {days} days)")
        click.echo("-" * 50)
        
        history_data = auth_manager.get_auth_history(days)
        
        if not history_data or not history_data.get('events'):
            click.echo("No authentication events found")
            return
        
        events = history_data['events']
        
        for event in events:
            timestamp = event.get('timestamp', 'Unknown')
            action = event.get('action', 'Unknown')
            ip_address = event.get('ip_address', 'Unknown')
            user_agent = event.get('user_agent', 'Unknown')
            success = event.get('success', False)
            
            status_icon = "✅" if success else "❌"
            click.echo(f"{status_icon} {timestamp}")
            click.echo(f"  Action: {action}")
            click.echo(f"  IP: {ip_address}")
            click.echo(f"  User Agent: {user_agent[:50]}{'...' if len(user_agent) > 50 else ''}")
            click.echo()
        
        # Summary
        total_events = len(events)
        successful_events = sum(1 for event in events if event.get('success', False))
        failed_events = total_events - successful_events
        
        click.echo(f"📊 Summary:")
        click.echo(f"  Total events: {total_events}")
        click.echo(f"  Successful: {successful_events}")
        click.echo(f"  Failed: {failed_events}")
        
    except Exception as e:
        logger.error(f"History display failed: {e}")
        click.echo(f"❌ History display failed: {e}", err=True)
        raise click.Abort()


@auth_group.command()
@click.option('--profile', help='Configuration profile to use')
@click.option('--email', prompt=True, help='Email address for MFA')
@click.pass_context
def setup_mfa(ctx, profile: Optional[str], email: str):
    """Setup multi-factor authentication."""
    try:
        config = get_config(profile)
        auth_manager = AuthManager(config)
        
        if not auth_manager.is_authenticated():
            click.echo("❌ Not authenticated. Please login first.", err=True)
            return
        
        click.echo(f"🔐 Setting up MFA for {email}...")
        
        setup_result = auth_manager.setup_mfa(email)
        
        if setup_result['success']:
            click.echo("✅ MFA setup successful!")
            
            if setup_result.get('backup_codes'):
                click.echo("\n🔑 Backup Codes (save these securely):")
                for i, code in enumerate(setup_result['backup_codes'], 1):
                    click.echo(f"  {i}. {code}")
                
                click.echo("\n⚠️  Important: Save these backup codes in a secure location!")
                click.echo("   You can use them if you lose access to your MFA device.")
            
            if setup_result.get('qr_code'):
                click.echo("\n📱 QR Code:")
                click.echo("  Scan this QR code with your authenticator app:")
                click.echo(f"  {setup_result['qr_code']}")
                
        else:
            error_msg = setup_result.get('error', 'MFA setup failed')
            click.echo(f"❌ MFA setup failed: {error_msg}", err=True)
            
    except Exception as e:
        logger.error(f"MFA setup failed: {e}")
        click.echo(f"❌ MFA setup failed: {e}", err=True)
        raise click.Abort()


@auth_group.command()
@click.option('--profile', help='Configuration profile to use')
@click.pass_context
def verify_mfa(ctx, profile: Optional[str]):
    """Verify multi-factor authentication setup."""
    try:
        config = get_config(profile)
        auth_manager = AuthManager(config)
        
        if not auth_manager.is_authenticated():
            click.echo("❌ Not authenticated. Please login first.", err=True)
            return
        
        # Prompt for MFA code
        mfa_code = click.prompt("Enter MFA code from your authenticator app", type=str)
        
        click.echo("🔍 Verifying MFA code...")
        
        verify_result = auth_manager.verify_mfa(mfa_code)
        
        if verify_result['success']:
            click.echo("✅ MFA verification successful!")
            click.echo("🔐 Your MFA is properly configured and working")
            
        else:
            error_msg = verify_result.get('error', 'MFA verification failed')
            click.echo(f"❌ MFA verification failed: {error_msg}", err=True)
            
    except Exception as e:
        logger.error(f"MFA verification failed: {e}")
        click.echo(f"❌ MFA verification failed: {e}", err=True)
        raise click.Abort()


@auth_group.command()
@click.option('--profile', help='Configuration profile to use')
@click.pass_context
def disable_mfa(ctx, profile: Optional[str]):
    """Disable multi-factor authentication."""
    try:
        config = get_config(profile)
        auth_manager = AuthManager(config)
        
        if not auth_manager.is_authenticated():
            click.echo("❌ Not authenticated. Please login first.", err=True)
            return
        
        if not click.confirm("⚠️  Are you sure you want to disable MFA? This will reduce your account security."):
            click.echo("❌ MFA disable cancelled")
            return
        
        click.echo("🔓 Disabling MFA...")
        
        disable_result = auth_manager.disable_mfa()
        
        if disable_result['success']:
            click.echo("✅ MFA disabled successfully!")
            click.echo("⚠️  Your account is now less secure. Consider re-enabling MFA.")
            
        else:
            error_msg = disable_result.get('error', 'MFA disable failed')
            click.echo(f"❌ MFA disable failed: {error_msg}", err=True)
            
    except Exception as e:
        logger.error(f"MFA disable failed: {e}")
        click.echo(f"❌ MFA disable failed: {e}", err=True)
        raise click.Abort()