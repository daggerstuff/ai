#!/usr/bin/env python3
"""
MCP Configuration Validator for Amazon Q Developer

Validates the MCP server configuration for compatibility with Amazon Q Developer
and tests connectivity to configured servers.
"""

import json
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MCPValidator:
    """Validates MCP configuration for Amazon Q Developer compatibility."""
    
    def __init__(self, config_path: str = None):
        """Initialize the MCP validator."""
        if config_path is None:
            config_path = Path(__file__).parent / "mcp.json"
        
        self.config_path = Path(config_path)
        self.config = None
        self.validation_results = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "server_status": {},
            "recommendations": []
        }
    
    def load_config(self) -> bool:
        """Load and parse the MCP configuration file."""
        try:
            if not self.config_path.exists():
                self.validation_results["errors"].append(f"MCP config file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            logger.info(f"✅ Loaded MCP configuration from {self.config_path}")
            return True
            
        except json.JSONDecodeError as e:
            self.validation_results["errors"].append(f"Invalid JSON in MCP config: {e}")
            return False
        except Exception as e:
            self.validation_results["errors"].append(f"Error loading MCP config: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """Validate the basic structure of the MCP configuration."""
        if not self.config:
            return False
        
        # Check required top-level keys
        if "mcpServers" not in self.config:
            self.validation_results["errors"].append("Missing required 'mcpServers' key")
            return False
        
        if not isinstance(self.config["mcpServers"], dict):
            self.validation_results["errors"].append("'mcpServers' must be an object")
            return False
        
        if len(self.config["mcpServers"]) == 0:
            self.validation_results["warnings"].append("No MCP servers configured")
        
        logger.info(f"✅ Configuration structure valid with {len(self.config['mcpServers'])} servers")
        return True
    
    def validate_servers(self) -> bool:
        """Validate individual server configurations."""
        if not self.config or "mcpServers" not in self.config:
            return False
        
        valid_servers = 0
        
        for server_name, server_config in self.config["mcpServers"].items():
            logger.info(f"🔍 Validating server: {server_name}")
            
            # Check server configuration structure
            if not isinstance(server_config, dict):
                self.validation_results["errors"].append(f"Server '{server_name}' config must be an object")
                continue
            
            # Validate server type
            server_type = server_config.get("type", "stdio")  # Default to stdio
            if server_type not in ["stdio", "sse", "streamable-http"]:
                self.validation_results["warnings"].append(
                    f"Server '{server_name}' has unknown type '{server_type}'"
                )
            
            # Validate stdio servers
            if server_type == "stdio" or "command" in server_config:
                if "command" not in server_config:
                    self.validation_results["errors"].append(
                        f"Server '{server_name}' missing required 'command' field"
                    )
                    continue
                
                # Check if command exists
                command = server_config["command"]
                if not self._check_command_available(command):
                    self.validation_results["warnings"].append(
                        f"Command '{command}' for server '{server_name}' may not be available"
                    )
            
            # Validate HTTP/SSE servers
            elif server_type in ["sse", "streamable-http"] or "url" in server_config:
                if "url" not in server_config:
                    self.validation_results["errors"].append(
                        f"Server '{server_name}' missing required 'url' field"
                    )
                    continue
                
                # Validate URL format
                url = server_config["url"]
                if not url.startswith(("http://", "https://")):
                    self.validation_results["errors"].append(
                        f"Server '{server_name}' has invalid URL format: {url}"
                    )
                    continue
            
            # Check for required environment variables
            if "env" in server_config:
                for env_var, env_value in server_config["env"].items():
                    if env_value.startswith("${") and env_value.endswith("}"):
                        # Environment variable reference
                        actual_env_var = env_value[2:-1]
                        if actual_env_var not in os.environ:
                            self.validation_results["warnings"].append(
                                f"Environment variable '{actual_env_var}' for server '{server_name}' not set"
                            )
            
            valid_servers += 1
            logger.info(f"  ✅ Server '{server_name}' configuration valid")
        
        logger.info(f"✅ Validated {valid_servers}/{len(self.config['mcpServers'])} servers")
        return valid_servers > 0
    
    def _check_command_available(self, command: str) -> bool:
        """Check if a command is available in the system PATH."""
        try:
            result = subprocess.run(
                ["which", command] if os.name != 'nt' else ["where", command],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def test_server_connectivity(self, server_name: str, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test connectivity to a specific MCP server."""
        result = {
            "name": server_name,
            "status": "unknown",
            "response_time": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            # Test stdio servers
            if server_config.get("type") == "stdio" or "command" in server_config:
                command = server_config["command"]
                args = server_config.get("args", [])
                env = dict(os.environ)
                env.update(server_config.get("env", {}))
                
                # Quick test - just check if command starts
                process = subprocess.Popen(
                    [command] + args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True
                )
                
                # Give it a moment to start
                time.sleep(1)
                
                if process.poll() is None:
                    # Process is still running, likely good
                    process.terminate()
                    result["status"] = "available"
                else:
                    # Process exited quickly, check return code
                    stdout, stderr = process.communicate()
                    if process.returncode == 0:
                        result["status"] = "available"
                    else:
                        result["status"] = "error"
                        result["error"] = f"Command exited with code {process.returncode}: {stderr}"
            
            # Test HTTP/SSE servers
            elif "url" in server_config:
                import urllib.request
                import urllib.error
                
                try:
                    with urllib.request.urlopen(server_config["url"], timeout=10) as response:
                        if response.status == 200:
                            result["status"] = "available"
                        else:
                            result["status"] = "error"
                            result["error"] = f"HTTP {response.status}"
                except urllib.error.URLError as e:
                    result["status"] = "error"
                    result["error"] = str(e)
            
            result["response_time"] = time.time() - start_time
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving the MCP configuration."""
        recommendations = []
        
        if not self.config:
            return ["Fix configuration loading errors first"]
        
        servers = self.config.get("mcpServers", {})
        
        # Check for essential servers
        essential_servers = ["filesystem", "git"]
        missing_essential = [s for s in essential_servers if s not in servers]
        if missing_essential:
            recommendations.append(
                f"Consider adding essential servers: {', '.join(missing_essential)}"
            )
        
        # Check for development vs production setup
        if "defaults" not in self.config:
            recommendations.append("Add 'defaults' section with timeout and retry settings")
        
        if "metadata" not in self.config:
            recommendations.append("Add 'metadata' section for better configuration management")
        
        # Check server priorities
        servers_with_priority = [s for s in servers.values() if "priority" in s]
        if len(servers_with_priority) != len(servers):
            recommendations.append("Add priority levels to all servers for better ordering")
        
        # Check for descriptions
        servers_without_desc = [name for name, config in servers.items() if "description" not in config]
        if servers_without_desc:
            recommendations.append("Add descriptions to servers for better documentation")
        
        return recommendations
    
    def validate(self) -> Dict[str, Any]:
        """Run complete validation of the MCP configuration."""
        logger.info("🔍 Starting MCP configuration validation...")
        
        # Load configuration
        if not self.load_config():
            self.validation_results["valid"] = False
            return self.validation_results
        
        # Validate structure
        if not self.validate_structure():
            self.validation_results["valid"] = False
            return self.validation_results
        
        # Validate servers
        if not self.validate_servers():
            self.validation_results["valid"] = False
            return self.validation_results
        
        # Test server connectivity (optional, can be slow)
        logger.info("🔗 Testing server connectivity...")
        for server_name, server_config in self.config["mcpServers"].items():
            if server_name in ["filesystem", "git", "time"]:  # Test key servers
                result = self.test_server_connectivity(server_name, server_config)
                self.validation_results["server_status"][server_name] = result
                
                if result["status"] == "available":
                    logger.info(f"  ✅ {server_name}: Available")
                elif result["status"] == "error":
                    logger.warning(f"  ⚠️ {server_name}: {result['error']}")
        
        # Generate recommendations
        self.validation_results["recommendations"] = self.generate_recommendations()
        
        # Determine overall validity
        has_critical_errors = any("missing required" in error.lower() for error in self.validation_results["errors"])
        self.validation_results["valid"] = not has_critical_errors
        
        return self.validation_results
    
    def print_report(self):
        """Print a formatted validation report."""
        results = self.validation_results
        
        print("\n" + "="*60)
        print("🔍 MCP CONFIGURATION VALIDATION REPORT")
        print("="*60)
        
        # Overall status
        status_icon = "✅" if results["valid"] else "❌"
        print(f"\n{status_icon} Overall Status: {'VALID' if results['valid'] else 'INVALID'}")
        
        # Errors
        if results["errors"]:
            print(f"\n❌ Errors ({len(results['errors'])}):")
            for error in results["errors"]:
                print(f"  • {error}")
        
        # Warnings
        if results["warnings"]:
            print(f"\n⚠️ Warnings ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"  • {warning}")
        
        # Server status
        if results["server_status"]:
            print(f"\n🔗 Server Connectivity:")
            for server_name, status in results["server_status"].items():
                status_icon = "✅" if status["status"] == "available" else "❌" if status["status"] == "error" else "❓"
                print(f"  {status_icon} {server_name}: {status['status'].upper()}")
                if status["error"]:
                    print(f"    Error: {status['error']}")
        
        # Recommendations
        if results["recommendations"]:
            print(f"\n💡 Recommendations ({len(results['recommendations'])}):")
            for rec in results["recommendations"]:
                print(f"  • {rec}")
        
        print("\n" + "="*60)

def main():
    """Main function to run MCP validation."""
    validator = MCPValidator()
    results = validator.validate()
    validator.print_report()
    
    # Exit with appropriate code
    sys.exit(0 if results["valid"] else 1)

if __name__ == "__main__":
    main()
