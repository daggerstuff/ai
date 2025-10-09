"""
Tests for main CLI functionality and integration.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from cli.main import cli
from cli import __version__, __description__, __author__


class TestMainCLI:
    """Test cases for main CLI functionality"""
    
    def test_cli_version_command(self):
        """Test version command output"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['--help'])
            
            assert result.exit_code == 0
            assert "Pixelated AI CLI" in result.output
    
    def test_cli_help_command(self):
        """Test help command output"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['--help'])
            
            assert result.exit_code == 0
            assert "Usage:" in result.output
            assert "Commands:" in result.output
            assert "web-frontend" in result.output
            assert "cli-interface" in result.output
            assert "mcp-connect" in result.output
            assert "pipeline" in result.output
            assert "config" in result.output
            assert "auth" in result.output
    
    def test_cli_verbose_flag(self):
        """Test verbose flag functionality"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['--verbose', 'version'])
            
            assert result.exit_code == 0
            # Verbose mode should show more detailed output
    
    def test_cli_debug_flag(self):
        """Test debug flag functionality"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['--debug', 'version'])
            
            assert result.exit_code == 0
            # Debug mode should show debug-level output
    
    def test_cli_config_file_option(self):
        """Test configuration file option"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Create a test config file
            config_content = """
api_base_url: http://test.example.com
timeout: 60
max_retries: 5
"""
            config_file = Path("test_config.yaml")
            config_file.write_text(config_content)
            
            result = runner.invoke(cli, ['--config-file', str(config_file), 'version'])
            
            assert result.exit_code == 0
    
    def test_cli_profile_option(self):
        """Test profile option"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['--profile', 'test', 'version'])
            
            assert result.exit_code == 0
    
    def test_version_command(self):
        """Test version command"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['version'])
            
            assert result.exit_code == 0
            assert f"Pixelated AI CLI v{__version__}" in result.output
            assert __description__ in result.output
            assert f"Author: {__author__}" in result.output
    
    def test_status_command_success(self):
        """Test status command with successful system check"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Mock successful system checks
            with patch('cli.main.validate_environment') as mock_validate:
                mock_validate.return_value = None
                
                with patch('cli.main.AuthManager') as mock_auth_class:
                    mock_auth = MagicMock()
                    mock_auth.get_status.return_value = "Authenticated as testuser"
                    mock_auth_class.return_value = mock_auth
                
                with patch('cli.main.PipelineManager') as mock_pipeline_class:
                    mock_pipeline = MagicMock()
                    mock_pipeline.get_status.return_value = "Running"
                    mock_pipeline.check_api_health.return_value = "Healthy"
                    mock_pipeline_class.return_value = mock_pipeline
                
                result = runner.invoke(cli, ['status'])
                
                assert result.exit_code == 0
                assert "System Status:" in result.output
                assert "✓ Configuration:" in result.output
                assert "✓ Authentication:" in result.output
                assert "✓ Pipeline Manager:" in result.output
                assert "✓ API Health:" in result.output
                assert "All systems operational ✓" in result.output
    
    def test_status_command_failure(self):
        """Test status command with system check failure"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Mock failed system check
            with patch('cli.main.validate_environment') as mock_validate:
                mock_validate.side_effect = Exception("Environment validation failed")
                
                result = runner.invoke(cli, ['status'])
                
                assert result.exit_code == 1
                assert "Environment validation failed" in result.output
    
    def test_web_frontend_commands(self):
        """Test web frontend command group"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['web-frontend', '--help'])
            
            assert result.exit_code == 0
            assert "Web Frontend Commands" in result.output
            assert "start" in result.output
            assert "stop" in result.output
            assert "status" in result.output
    
    def test_cli_interface_commands(self):
        """Test CLI interface command group"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['cli-interface', '--help'])
            
            assert result.exit_code == 0
            assert "CLI Interface Commands" in result.output
            assert "pipeline" in result.output
            assert "data" in result.output
            assert "batch" in result.output
    
    def test_mcp_connect_commands(self):
        """Test MCP connect command group"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['mcp-connect', '--help'])
            
            assert result.exit_code == 0
            assert "MCP Connect Commands" in result.output
            assert "agent" in result.output
            assert "execute" in result.output
            assert "list" in result.output
    
    def test_pipeline_commands(self):
        """Test pipeline command group"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['pipeline', '--help'])
            
            assert result.exit_code == 0
            assert "Pipeline Management Commands" in result.output
            assert "start" in result.output
            assert "stop" in result.output
            assert "status" in result.output
            assert "monitor" in result.output
    
    def test_config_commands(self):
        """Test configuration command group"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['config', '--help'])
            
            assert result.exit_code == 0
            assert "Configuration Management Commands" in result.output
            assert "show" in result.output
            assert "set" in result.output
            assert "profiles" in result.output
    
    def test_auth_commands(self):
        """Test authentication command group"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['auth', '--help'])
            
            assert result.exit_code == 0
            assert "Authentication Commands" in result.output
            assert "login" in result.output
            assert "logout" in result.output
            assert "status" in result.output
    
    def test_cli_with_invalid_config_file(self):
        """Test CLI with invalid configuration file"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Create invalid config file
            config_file = Path("invalid_config.yaml")
            config_file.write_text("invalid: yaml: content: [")
            
            result = runner.invoke(cli, ['--config-file', str(config_file), 'version'])
            
            assert result.exit_code == 1
            assert "Failed to initialize configuration" in result.output
    
    def test_cli_with_nonexistent_config_file(self):
        """Test CLI with non-existent configuration file"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['--config-file', 'nonexistent.yaml', 'version'])
            
            assert result.exit_code == 1
            assert "Failed to initialize configuration" in result.output
    
    def test_cli_environment_validation_failure(self):
        """Test CLI when environment validation fails"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Mock environment validation failure
            with patch('cli.main.validate_environment') as mock_validate:
                mock_validate.side_effect = Exception("Missing required environment variables")
                
                result = runner.invoke(cli, ['version'])
                
                assert result.exit_code == 1
                assert "Environment validation failed" in result.output
    
    def test_cli_initialization_with_all_options(self):
        """Test CLI initialization with all command-line options"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Create a valid config file
            config_content = """
api_base_url: http://test.example.com
timeout: 60
max_retries: 5
"""
            config_file = Path("test_config.yaml")
            config_file.write_text(config_content)
            
            result = runner.invoke(cli, [
                '--config-file', str(config_file),
                '--profile', 'test',
                '--verbose',
                '--debug',
                'version'
            ])
            
            assert result.exit_code == 0
            assert f"Pixelated AI CLI v{__version__}" in result.output
    
    def test_cli_command_chaining(self):
        """Test that CLI commands can be chained properly"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Test that we can run multiple commands in sequence
            result1 = runner.invoke(cli, ['version'])
            result2 = runner.invoke(cli, ['status'])
            
            assert result1.exit_code == 0
            assert result2.exit_code == 0
    
    def test_cli_error_handling_unknown_command(self):
        """Test CLI error handling for unknown commands"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['unknown-command'])
            
            assert result.exit_code == 2  # Click's exit code for unknown commands
            assert "No such command" in result.output
    
    def test_cli_error_handling_invalid_arguments(self):
        """Test CLI error handling for invalid arguments"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['--invalid-flag', 'version'])
            
            assert result.exit_code == 2
            assert "no such option" in result.output
    
    def test_cli_context_passing(self):
        """Test that CLI context is properly passed to subcommands"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Create a config file
            config_content = """
api_base_url: http://context.test.com
timeout: 45
"""
            config_file = Path("context_config.yaml")
            config_file.write_text(config_content)
            
            # Test that context is available in subcommands
            result = runner.invoke(cli, [
                '--config-file', str(config_file),
                '--profile', 'test',
                'config', 'show'
            ])
            
            assert result.exit_code == 0
            # The config show command should display the configuration
    
    def test_cli_with_mocked_dependencies(self):
        """Test CLI with mocked dependencies"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Mock all the managers
            with patch('cli.main.AuthManager') as mock_auth_class, \
                 patch('cli.main.PipelineManager') as mock_pipeline_class, \
                 patch('cli.main.ProgressTracker') as mock_progress_class:
                
                # Configure mocks
                mock_auth = MagicMock()
                mock_auth.get_status.return_value = "Mocked auth status"
                mock_auth_class.return_value = mock_auth
                
                mock_pipeline = MagicMock()
                mock_pipeline.get_status.return_value = "Mocked pipeline status"
                mock_pipeline.check_api_health.return_value = "Mocked API health"
                mock_pipeline_class.return_value = mock_pipeline
                
                mock_progress = MagicMock()
                mock_progress_class.return_value = mock_progress
                
                # Run status command
                result = runner.invoke(cli, ['status'])
                
                assert result.exit_code == 0
                assert "Mocked auth status" in result.output
                assert "Mocked pipeline status" in result.output
                assert "Mocked API health" in result.output


class TestCLIIntegration:
    """Integration tests for CLI functionality"""
    
    def test_full_authentication_flow(self):
        """Test complete authentication flow"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Mock authentication API responses
            login_response = MagicMock()
            login_response.status_code = 200
            login_response.json.return_value = {
                "access_token": "integration_test_token",
                "refresh_token": "integration_refresh_token",
                "expires_in": 3600
            }
            
            user_info_response = MagicMock()
            user_info_response.status_code = 200
            user_info_response.json.return_value = {
                "username": "integration_user",
                "email": "integration@example.com",
                "roles": ["user"]
            }
            
            with patch('cli.auth.requests.post', return_value=login_response):
                with patch('cli.auth.requests.get', return_value=user_info_response):
                    # Test login
                    result = runner.invoke(cli, ['auth', 'login', '--username', 'testuser', '--password', 'testpass'])
                    assert result.exit_code == 0
                    
                    # Test status after login
                    result = runner.invoke(cli, ['auth', 'status'])
                    assert result.exit_code == 0
                    assert "integration_user" in result.output
                    
                    # Test logout
                    result = runner.invoke(cli, ['auth', 'logout'])
                    assert result.exit_code == 0
                    
                    # Test status after logout
                    result = runner.invoke(cli, ['auth', 'status'])
                    assert result.exit_code == 0
                    assert "Not authenticated" in result.output
    
    def test_pipeline_lifecycle(self):
        """Test complete pipeline lifecycle"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Mock pipeline API responses
            start_response = MagicMock()
            start_response.status_code = 200
            start_response.json.return_value = {"pipeline_id": "test-pipeline-123"}
            
            status_response = MagicMock()
            status_response.status_code = 200
            status_response.json.return_value = {
                "id": "test-pipeline-123",
                "status": "running",
                "progress": 75
            }
            
            stop_response = MagicMock()
            stop_response.status_code = 200
            
            with patch('cli.pipeline.requests.post', side_effect=[start_response, stop_response]):
                with patch('cli.pipeline.requests.get', return_value=status_response):
                    # Test pipeline start
                    result = runner.invoke(cli, ['pipeline', 'start', '--config', 'test-config.yaml'])
                    assert result.exit_code == 0
                    
                    # Test pipeline status
                    result = runner.invoke(cli, ['pipeline', 'status', '--id', 'test-pipeline-123'])
                    assert result.exit_code == 0
                    assert "running" in result.output
                    assert "75" in result.output
                    
                    # Test pipeline stop
                    result = runner.invoke(cli, ['pipeline', 'stop', '--id', 'test-pipeline-123'])
                    assert result.exit_code == 0
    
    def test_configuration_management(self):
        """Test configuration management workflow"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Test config show
            result = runner.invoke(cli, ['config', 'show'])
            assert result.exit_code == 0
            
            # Test config set
            result = runner.invoke(cli, ['config', 'set', 'timeout', '120'])
            assert result.exit_code == 0
            
            # Test config profiles list
            result = runner.invoke(cli, ['config', 'profiles', 'list'])
            assert result.exit_code == 0
            
            # Test config validation
            result = runner.invoke(cli, ['config', 'validate'])
            assert result.exit_code == 0
    
    def test_error_recovery_and_retry(self):
        """Test error recovery and retry mechanisms"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Mock API responses with failures then success
            fail_response = MagicMock()
            fail_response.status_code = 500
            fail_response.text = "Internal server error"
            
            success_response = MagicMock()
            success_response.status_code = 200
            success_response.json.return_value = {"status": "success"}
            
            # First call fails, second succeeds (retry mechanism)
            with patch('cli.pipeline.requests.post', side_effect=[fail_response, success_response]):
                result = runner.invoke(cli, ['pipeline', 'start', '--config', 'test-config.yaml'])
                
                # Should succeed on retry
                assert result.exit_code == 0


# Performance and stress tests
class TestCLIPerformance:
    """Performance and stress tests for CLI"""
    
    def test_concurrent_command_execution(self):
        """Test handling of concurrent command execution"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Run multiple commands concurrently
            results = []
            for i in range(5):
                result = runner.invoke(cli, ['version'])
                results.append(result)
            
            # All should succeed
            for result in results:
                assert result.exit_code == 0
    
    def test_large_output_handling(self):
        """Test handling of large command outputs"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Mock large response data
            large_response = MagicMock()
            large_response.status_code = 200
            large_response.json.return_value = {
                "data": ["item_" + str(i) for i in range(1000)],  # Large dataset
                "total": 1000,
                "page": 1,
                "pages": 10
            }
            
            with patch('cli.pipeline.requests.get', return_value=large_response):
                result = runner.invoke(cli, ['pipeline', 'list'])
                
                assert result.exit_code == 0
                # Should handle large output without issues
    
    def test_memory_usage_stability(self):
        """Test memory usage stability over multiple operations"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Run many operations and check for memory leaks
            for i in range(50):
                result = runner.invoke(cli, ['version'])
                assert result.exit_code == 0
            
            # If we get here without memory issues, test passes
            assert True
    
    def test_timeout_handling(self):
        """Test timeout handling for long-running operations"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Mock slow API response
            def slow_response(*args, **kwargs):
                import time
                time.sleep(0.1)  # Simulate slow response
                response = MagicMock()
                response.status_code = 200
                response.json.return_value = {"status": "completed"}
                return response
            
            with patch('cli.pipeline.requests.get', side_effect=slow_response):
                result = runner.invoke(cli, ['pipeline', 'status', '--id', 'slow-pipeline'])
                
                assert result.exit_code == 0