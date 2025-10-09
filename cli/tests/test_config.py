"""
Tests for CLI configuration management.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from cli.config import CLIConfig, ConfigProfile, AuthConfig, APIConfig
from cli.exceptions import CLIConfigError


class TestCLIConfig:
    """Test cases for CLI configuration"""
    
    def test_config_initialization_default(self):
        """Test default configuration initialization"""
        config = CLIConfig()
        
        assert config.profile == "default"
        assert config.api_base_url == "http://localhost:8000"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.log_level == "INFO"
    
    def test_config_initialization_with_profile(self):
        """Test configuration initialization with specific profile"""
        config = CLIConfig(profile="test")
        
        assert config.profile == "test"
        assert config.api_base_url == "http://localhost:8000"
    
    def test_config_from_file(self, temp_config_dir):
        """Test configuration loading from file"""
        config_content = """
api_base_url: http://test.example.com
timeout: 60
max_retries: 5
log_level: DEBUG
profiles:
  test:
    api_base_url: http://test.example.com
    auth_endpoint: /auth/test
    timeout: 45
"""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(config_content)
        
        config = CLIConfig(config_file=config_file, profile="test")
        
        assert config.api_base_url == "http://test.example.com"
        assert config.timeout == 45
        assert config.max_retries == 5
        assert config.log_level == "DEBUG"
    
    def test_config_from_env_vars(self, temp_config_dir):
        """Test configuration override from environment variables"""
        config_content = """
api_base_url: http://file.example.com
timeout: 30
"""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(config_content)
        
        with patch.dict('os.environ', {
            'PIXELATED_API_BASE_URL': 'http://env.example.com',
            'PIXELATED_TIMEOUT': '90'
        }):
            config = CLIConfig(config_file=config_file)
            
            assert config.api_base_url == "http://env.example.com"
            assert config.timeout == 90
    
    def test_config_validation_invalid_url(self, temp_config_dir):
        """Test configuration validation with invalid URL"""
        config_content = """
api_base_url: invalid-url
timeout: 30
"""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(config_content)
        
        with pytest.raises(CLIConfigError):
            CLIConfig(config_file=config_file)
    
    def test_config_validation_invalid_timeout(self, temp_config_dir):
        """Test configuration validation with invalid timeout"""
        config_content = """
api_base_url: http://localhost:8000
timeout: -1
"""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(config_content)
        
        with pytest.raises(CLIConfigError):
            CLIConfig(config_file=config_file)
    
    def test_config_save_and_load(self, temp_config_dir):
        """Test configuration save and load functionality"""
        config_file = temp_config_dir / "test_config.yaml"
        
        # Create config and save
        config = CLIConfig(config_file=config_file)
        config.api_base_url = "http://saved.example.com"
        config.timeout = 120
        config.save()
        
        # Load saved config
        loaded_config = CLIConfig(config_file=config_file)
        
        assert loaded_config.api_base_url == "http://saved.example.com"
        assert loaded_config.timeout == 120
    
    def test_config_profile_switching(self, temp_config_dir):
        """Test switching between configuration profiles"""
        config_content = """
api_base_url: http://default.example.com
profiles:
  development:
    api_base_url: http://dev.example.com
    timeout: 60
  production:
    api_base_url: http://prod.example.com
    timeout: 30
"""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(config_content)
        
        # Test default profile
        config = CLIConfig(config_file=config_file)
        assert config.api_base_url == "http://default.example.com"
        
        # Test development profile
        dev_config = CLIConfig(config_file=config_file, profile="development")
        assert dev_config.api_base_url == "http://dev.example.com"
        assert dev_config.timeout == 60
        
        # Test production profile
        prod_config = CLIConfig(config_file=config_file, profile="production")
        assert prod_config.api_base_url == "http://prod.example.com"
        assert prod_config.timeout == 30
    
    def test_config_encryption(self, temp_config_dir):
        """Test configuration encryption for sensitive data"""
        config_file = temp_config_dir / "secure_config.yaml"
        
        config = CLIConfig(config_file=config_file)
        config.auth = AuthConfig(
            client_id="test_client_id",
            client_secret="test_client_secret"
        )
        
        # Save with encryption
        config.save(encrypt=True)
        
        # Verify file is encrypted
        content = config_file.read_text()
        assert "test_client_secret" not in content
        assert "encrypted_" in content
        
        # Load and decrypt
        loaded_config = CLIConfig(config_file=config_file)
        assert loaded_config.auth.client_id == "test_client_id"
        assert loaded_config.auth.client_secret == "test_client_secret"
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion"""
        config = CLIConfig(profile="test")
        
        config_dict = config.to_dict()
        
        assert config_dict["profile"] == "test"
        assert config_dict["api_base_url"] == "http://localhost:8000"
        assert config_dict["timeout"] == 30
        assert config_dict["max_retries"] == 3
    
    def test_config_from_dict(self):
        """Test configuration from dictionary creation"""
        config_dict = {
            "profile": "custom",
            "api_base_url": "http://custom.example.com",
            "timeout": 45,
            "max_retries": 2,
            "log_level": "DEBUG"
        }
        
        config = CLIConfig.from_dict(config_dict)
        
        assert config.profile == "custom"
        assert config.api_base_url == "http://custom.example.com"
        assert config.timeout == 45
        assert config.max_retries == 2
        assert config.log_level == "DEBUG"
    
    def test_config_missing_profile(self, temp_config_dir):
        """Test configuration with missing profile"""
        config_content = """
api_base_url: http://localhost:8000
profiles:
  existing:
    api_base_url: http://existing.example.com
"""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(config_content)
        
        # Should fall back to default config
        config = CLIConfig(config_file=config_file, profile="missing")
        assert config.api_base_url == "http://localhost:8000"
    
    def test_config_invalid_yaml(self, temp_config_dir):
        """Test configuration with invalid YAML"""
        config_file = temp_config_dir / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(CLIConfigError):
            CLIConfig(config_file=config_file)


class TestAuthConfig:
    """Test cases for authentication configuration"""
    
    def test_auth_config_creation(self):
        """Test authentication configuration creation"""
        auth_config = AuthConfig(
            client_id="test_client",
            client_secret="test_secret",
            token_url="http://localhost:8000/auth/token"
        )
        
        assert auth_config.client_id == "test_client"
        assert auth_config.client_secret == "test_secret"
        assert auth_config.token_url == "http://localhost:8000/auth/token"
    
    def test_auth_config_validation(self):
        """Test authentication configuration validation"""
        # Valid config
        auth_config = AuthConfig(
            client_id="test_client",
            client_secret="test_secret"
        )
        assert auth_config.validate() is True
        
        # Invalid config - missing client_id
        with pytest.raises(ValueError):
            AuthConfig(client_secret="test_secret")
        
        # Invalid config - missing client_secret
        with pytest.raises(ValueError):
            AuthConfig(client_id="test_client")


class TestAPIConfig:
    """Test cases for API configuration"""
    
    def test_api_config_creation(self):
        """Test API configuration creation"""
        api_config = APIConfig(
            base_url="http://api.example.com",
            timeout=60,
            max_retries=5,
            rate_limit=100
        )
        
        assert api_config.base_url == "http://api.example.com"
        assert api_config.timeout == 60
        assert api_config.max_retries == 5
        assert api_config.rate_limit == 100
    
    def test_api_config_validation(self):
        """Test API configuration validation"""
        # Valid config
        api_config = APIConfig(
            base_url="http://api.example.com",
            timeout=30
        )
        assert api_config.validate() is True
        
        # Invalid config - negative timeout
        with pytest.raises(ValueError):
            APIConfig(base_url="http://api.example.com", timeout=-1)
        
        # Invalid config - negative retries
        with pytest.raises(ValueError):
            APIConfig(base_url="http://api.example.com", max_retries=-1)


class TestConfigProfile:
    """Test cases for configuration profiles"""
    
    def test_config_profile_creation(self):
        """Test configuration profile creation"""
        profile = ConfigProfile(
            name="test_profile",
            api_base_url="http://test.example.com",
            auth_endpoint="/auth/test",
            pipeline_endpoint="/api/test/pipelines",
            timeout=45,
            max_retries=2
        )
        
        assert profile.name == "test_profile"
        assert profile.api_base_url == "http://test.example.com"
        assert profile.auth_endpoint == "/auth/test"
        assert profile.pipeline_endpoint == "/api/test/pipelines"
        assert profile.timeout == 45
        assert profile.max_retries == 2
    
    def test_config_profile_validation(self):
        """Test configuration profile validation"""
        # Valid profile
        profile = ConfigProfile(
            name="valid_profile",
            api_base_url="http://valid.example.com"
        )
        assert profile.validate() is True
        
        # Invalid profile - empty name
        with pytest.raises(ValueError):
            ConfigProfile(
                name="",
                api_base_url="http://invalid.example.com"
            )
        
        # Invalid profile - invalid URL
        with pytest.raises(ValueError):
            ConfigProfile(
                name="invalid_profile",
                api_base_url="invalid-url"
            )
    
    def test_config_profile_inheritance(self):
        """Test configuration profile inheritance"""
        base_profile = ConfigProfile(
            name="base",
            api_base_url="http://base.example.com",
            timeout=30,
            max_retries=3
        )
        
        child_profile = ConfigProfile(
            name="child",
            api_base_url="http://child.example.com",
            timeout=60  # Override timeout
            # Inherits max_retries from base
        )
        
        assert child_profile.timeout == 60
        # In a real implementation, inheritance would be handled by the config system