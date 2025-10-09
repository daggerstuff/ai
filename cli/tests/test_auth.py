"""
Tests for CLI authentication management.
"""

import pytest
import time
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime, timedelta

from cli.auth import AuthManager, TokenManager
from cli.config import CLIConfig
from cli.exceptions import CLIAuthError


class TestAuthManager:
    """Test cases for authentication manager"""
    
    def test_auth_manager_initialization(self, mock_config):
        """Test authentication manager initialization"""
        auth_manager = AuthManager(mock_config)
        
        assert auth_manager.config == mock_config
        assert auth_manager._access_token is None
        assert auth_manager._refresh_token is None
        assert auth_manager._token_expiry is None
    
    def test_login_success(self, mock_config):
        """Test successful login"""
        auth_manager = AuthManager(mock_config)
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer"
        }
        
        with patch('cli.auth.requests.post', return_value=mock_response):
            result = auth_manager.login("testuser", "testpass")
            
            assert result is True
            assert auth_manager._access_token == "test_access_token"
            assert auth_manager._refresh_token == "test_refresh_token"
            assert auth_manager._token_expiry > time.time()
    
    def test_login_failure_invalid_credentials(self, mock_config):
        """Test login failure with invalid credentials"""
        auth_manager = AuthManager(mock_config)
        
        # Mock failed API response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid credentials"
        
        with patch('cli.auth.requests.post', return_value=mock_response):
            with pytest.raises(CLIAuthError) as exc_info:
                auth_manager.login("invaliduser", "invalidpass")
            
            assert "Authentication failed" in str(exc_info.value)
    
    def test_login_failure_network_error(self, mock_config):
        """Test login failure due to network error"""
        auth_manager = AuthManager(mock_config)
        
        with patch('cli.auth.requests.post', side_effect=Exception("Network error")):
            with pytest.raises(CLIAuthError) as exc_info:
                auth_manager.login("testuser", "testpass")
            
            assert "Network error during authentication" in str(exc_info.value)
    
    def test_logout(self, mock_auth_manager):
        """Test logout functionality"""
        # Ensure we have tokens to logout
        assert mock_auth_manager._access_token is not None
        
        mock_auth_manager.logout()
        
        assert mock_auth_manager._access_token is None
        assert mock_auth_manager._refresh_token is None
        assert mock_auth_manager._token_expiry is None
    
    def test_is_authenticated_true(self, mock_auth_manager):
        """Test authentication check when authenticated"""
        assert mock_auth_manager.is_authenticated() is True
    
    def test_is_authenticated_false_no_token(self, mock_config):
        """Test authentication check when not authenticated"""
        auth_manager = AuthManager(mock_config)
        assert auth_manager.is_authenticated() is False
    
    def test_is_authenticated_false_expired_token(self, mock_config):
        """Test authentication check with expired token"""
        auth_manager = AuthManager(mock_config)
        auth_manager._access_token = "expired_token"
        auth_manager._token_expiry = time.time() - 3600  # 1 hour ago
        
        assert auth_manager.is_authenticated() is False
    
    def test_refresh_token_success(self, mock_config):
        """Test successful token refresh"""
        auth_manager = AuthManager(mock_config)
        auth_manager._refresh_token = "valid_refresh_token"
        auth_manager._token_expiry = time.time() - 3600  # Expired
        
        # Mock successful refresh response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600
        }
        
        with patch('cli.auth.requests.post', return_value=mock_response):
            result = auth_manager.refresh_token()
            
            assert result is True
            assert auth_manager._access_token == "new_access_token"
            assert auth_manager._refresh_token == "new_refresh_token"
    
    def test_refresh_token_no_refresh_token(self, mock_config):
        """Test token refresh when no refresh token available"""
        auth_manager = AuthManager(mock_config)
        
        with pytest.raises(CLIAuthError) as exc_info:
            auth_manager.refresh_token()
        
        assert "No refresh token available" in str(exc_info.value)
    
    def test_refresh_token_failure(self, mock_config):
        """Test token refresh failure"""
        auth_manager = AuthManager(mock_config)
        auth_manager._refresh_token = "invalid_refresh_token"
        
        # Mock failed refresh response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid refresh token"
        
        with patch('cli.auth.requests.post', return_value=mock_response):
            with pytest.raises(CLIAuthError) as exc_info:
                auth_manager.refresh_token()
            
            assert "Token refresh failed" in str(exc_info.value)
    
    def test_get_auth_headers_authenticated(self, mock_auth_manager):
        """Test getting authentication headers when authenticated"""
        headers = mock_auth_manager.get_auth_headers()
        
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_access_token"
    
    def test_get_auth_headers_not_authenticated(self, mock_config):
        """Test getting authentication headers when not authenticated"""
        auth_manager = AuthManager(mock_config)
        
        headers = auth_manager.get_auth_headers()
        
        assert headers == {}
    
    def test_validate_token_valid(self, mock_auth_manager):
        """Test token validation with valid token"""
        is_valid = mock_auth_manager.validate_token()
        
        assert is_valid is True
    
    def test_validate_token_invalid(self, mock_config):
        """Test token validation with invalid token"""
        auth_manager = AuthManager(mock_config)
        auth_manager._access_token = "invalid_token"
        
        # Mock validation response
        mock_response = Mock()
        mock_response.status_code = 401
        
        with patch('cli.auth.requests.get', return_value=mock_response):
            is_valid = auth_manager.validate_token()
            
            assert is_valid is False
    
    def test_get_user_info_success(self, mock_auth_manager):
        """Test getting user information"""
        # Mock user info response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user", "admin"]
        }
        
        with patch('cli.auth.requests.get', return_value=mock_response):
            user_info = mock_auth_manager.get_user_info()
            
            assert user_info["username"] == "testuser"
            assert user_info["email"] == "test@example.com"
            assert "admin" in user_info["roles"]
    
    def test_get_user_info_not_authenticated(self, mock_config):
        """Test getting user information when not authenticated"""
        auth_manager = AuthManager(mock_config)
        
        with pytest.raises(CLIAuthError) as exc_info:
            auth_manager.get_user_info()
        
        assert "Not authenticated" in str(exc_info.value)
    
    def test_has_role_true(self, mock_auth_manager):
        """Test role checking when user has role"""
        # Mock user info
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"roles": ["user", "admin"]}
        
        with patch('cli.auth.requests.get', return_value=mock_response):
            has_admin = mock_auth_manager.has_role("admin")
            has_user = mock_auth_manager.has_role("user")
            
            assert has_admin is True
            assert has_user is True
    
    def test_has_role_false(self, mock_auth_manager):
        """Test role checking when user doesn't have role"""
        # Mock user info
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"roles": ["user"]}
        
        with patch('cli.auth.requests.get', return_value=mock_response):
            has_admin = mock_auth_manager.has_role("admin")
            
            assert has_admin is False
    
    def test_get_status_authenticated(self, mock_auth_manager):
        """Test getting authentication status when authenticated"""
        status = mock_auth_manager.get_status()
        
        assert "Authenticated" in status
        assert "testuser" in status
    
    def test_get_status_not_authenticated(self, mock_config):
        """Test getting authentication status when not authenticated"""
        auth_manager = AuthManager(mock_config)
        
        status = auth_manager.get_status()
        
        assert status == "Not authenticated"
    
    def test_change_password_success(self, mock_auth_manager):
        """Test successful password change"""
        # Mock successful password change response
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('cli.auth.requests.post', return_value=mock_response):
            result = mock_auth_manager.change_password("oldpass", "newpass")
            
            assert result is True
    
    def test_change_password_failure(self, mock_auth_manager):
        """Test failed password change"""
        # Mock failed password change response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid old password"
        
        with patch('cli.auth.requests.post', return_value=mock_response):
            with pytest.raises(CLIAuthError) as exc_info:
                mock_auth_manager.change_password("wrongold", "newpass")
            
            assert "Password change failed" in str(exc_info.value)
    
    def test_enable_mfa(self, mock_auth_manager):
        """Test enabling multi-factor authentication"""
        # Mock MFA setup response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "qr_code": "data:image/png;base64,testqr",
            "backup_codes": ["code1", "code2", "code3"]
        }
        
        with patch('cli.auth.requests.post', return_value=mock_response):
            mfa_data = mock_auth_manager.enable_mfa()
            
            assert "qr_code" in mfa_data
            assert "backup_codes" in mfa_data
            assert len(mfa_data["backup_codes"]) == 3
    
    def test_disable_mfa(self, mock_auth_manager):
        """Test disabling multi-factor authentication"""
        # Mock MFA disable response
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('cli.auth.requests.post', return_value=mock_response):
            result = mock_auth_manager.disable_mfa("current_password")
            
            assert result is True
    
    def test_verify_mfa(self, mock_config):
        """Test MFA verification"""
        auth_manager = AuthManager(mock_config)
        
        # Mock MFA verification response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "mfa_access_token",
            "refresh_token": "mfa_refresh_token",
            "expires_in": 3600
        }
        
        with patch('cli.auth.requests.post', return_value=mock_response):
            result = auth_manager.verify_mfa("123456")
            
            assert result is True
            assert auth_manager._access_token == "mfa_access_token"
    
    def test_get_auth_history(self, mock_auth_manager):
        """Test getting authentication history"""
        # Mock auth history response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "history": [
                {
                    "timestamp": "2023-01-01T12:00:00Z",
                    "ip_address": "192.168.1.1",
                    "user_agent": "Mozilla/5.0",
                    "success": True
                }
            ]
        }
        
        with patch('cli.auth.requests.get', return_value=mock_response):
            history = mock_auth_manager.get_auth_history()
            
            assert len(history) == 1
            assert history[0]["ip_address"] == "192.168.1.1"
            assert history[0]["success"] is True


class TestTokenManager:
    """Test cases for token management"""
    
    def test_token_manager_initialization(self):
        """Test token manager initialization"""
        token_manager = TokenManager()
        
        assert token_manager._access_token is None
        assert token_manager._refresh_token is None
        assert token_manager._token_expiry is None
    
    def test_set_tokens(self):
        """Test setting tokens"""
        token_manager = TokenManager()
        
        token_manager.set_tokens(
            access_token="test_access",
            refresh_token="test_refresh",
            expires_in=3600
        )
        
        assert token_manager._access_token == "test_access"
        assert token_manager._refresh_token == "test_refresh"
        assert token_manager._token_expiry > time.time()
    
    def test_get_access_token(self):
        """Test getting access token"""
        token_manager = TokenManager()
        token_manager.set_tokens("test_access", "test_refresh", 3600)
        
        assert token_manager.get_access_token() == "test_access"
    
    def test_get_refresh_token(self):
        """Test getting refresh token"""
        token_manager = TokenManager()
        token_manager.set_tokens("test_access", "test_refresh", 3600)
        
        assert token_manager.get_refresh_token() == "test_refresh"
    
    def test_is_token_expired_false(self):
        """Test token expiry check for valid token"""
        token_manager = TokenManager()
        token_manager.set_tokens("test_access", "test_refresh", 3600)
        
        assert token_manager.is_token_expired() is False
    
    def test_is_token_expired_true(self):
        """Test token expiry check for expired token"""
        token_manager = TokenManager()
        token_manager._token_expiry = time.time() - 3600  # 1 hour ago
        
        assert token_manager.is_token_expired() is True
    
    def test_clear_tokens(self):
        """Test clearing tokens"""
        token_manager = TokenManager()
        token_manager.set_tokens("test_access", "test_refresh", 3600)
        
        token_manager.clear_tokens()
        
        assert token_manager._access_token is None
        assert token_manager._refresh_token is None
        assert token_manager._token_expiry is None
    
    def test_get_token_age(self):
        """Test getting token age"""
        token_manager = TokenManager()
        token_manager.set_tokens("test_access", "test_refresh", 3600)
        
        # Token should be very young (just created)
        age = token_manager.get_token_age()
        assert age < 1  # Less than 1 second old
    
    def test_get_time_until_expiry(self):
        """Test getting time until token expiry"""
        token_manager = TokenManager()
        token_manager.set_tokens("test_access", "test_refresh", 3600)
        
        time_until_expiry = token_manager.get_time_until_expiry()
        assert time_until_expiry > 3500  # Should be close to 3600 seconds
    
    def test_get_time_until_expiry_expired(self):
        """Test getting time until expiry for expired token"""
        token_manager = TokenManager()
        token_manager._token_expiry = time.time() - 3600  # 1 hour ago
        
        time_until_expiry = token_manager.get_time_until_expiry()
        assert time_until_expiry < 0  # Should be negative (expired)
    
    def test_validate_token_format_valid(self):
        """Test token format validation with valid token"""
        token_manager = TokenManager()
        
        # Valid JWT format (header.payload.signature)
        valid_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        
        is_valid = token_manager.validate_token_format(valid_token)
        assert is_valid is True
    
    def test_validate_token_format_invalid(self):
        """Test token format validation with invalid token"""
        token_manager = TokenManager()
        
        # Invalid format
        invalid_token = "invalid_token_format"
        
        is_valid = token_manager.validate_token_format(invalid_token)
        assert is_valid is False
    
    def test_decode_token_payload(self):
        """Test decoding token payload"""
        token_manager = TokenManager()
        
        # Valid JWT with payload
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        
        payload = token_manager.decode_token_payload(token)
        
        assert payload is not None
        assert "sub" in payload
        assert "name" in payload
        assert payload["name"] == "John Doe"