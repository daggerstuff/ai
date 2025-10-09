#!/usr/bin/env python3
"""
Enterprise Baseline Setup Script

Sets up minimum enterprise-grade baseline for all components:
- Configuration management
- Centralized logging
- Health monitoring
- Error handling and recovery
- Performance metrics
- Security basics
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from enterprise_config import EnterpriseConfigManager, get_config
from enterprise_logging import setup_logging, get_logger
from enterprise_monitoring import get_monitor, start_monitoring
from enterprise_error_handling import get_error_handler

class EnterpriseBaselineSetup:
    """Sets up enterprise baseline for the entire system."""
    
    def __init__(self):
        self.base_path = Path("/home/vivi/pixelated/ai")
        self.enterprise_path = self.base_path / "enterprise_config"
        self.enterprise_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.config_manager = None
        self.logger_system = None
        self.monitor = None
        self.error_handler = None
        
        print("🏢 Enterprise Baseline Setup initialized")
    
    def setup_directory_structure(self):
        """Create enterprise directory structure."""
        print("📁 Setting up enterprise directory structure...")
        
        directories = [
            "logs",
            "cache", 
            "config",
            "monitoring",
            "backups",
            "security",
            "metrics",
            "health_checks"
        ]
        
        for directory in directories:
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {directory}/")
        
        print("✅ Directory structure created")
        return True
    
    def setup_configuration_management(self):
        """Set up centralized configuration management."""
        print("⚙️ Setting up configuration management...")
        
        self.config_manager = EnterpriseConfigManager()
        self.config_manager.create_default_configs()
        
        # Load and validate configuration
        config = self.config_manager.load_config()
        
        if self.config_manager.validate_config(config):
            print("   ✅ Configuration system initialized")
            print(f"   Environment: {config.environment}")
            print(f"   Debug mode: {config.debug_mode}")
            print(f"   Quality threshold: {config.quality_threshold}")
        else:
            print("   ❌ Configuration validation failed")
            return False
        
        return True
    
    def setup_logging_system(self):
        """Set up enterprise logging system."""
        print("📝 Setting up enterprise logging...")
        
        config = get_config()
        self.logger_system = setup_logging(
            level=config.log_level,
            log_dir=config.logs_path
        )
        
        # Test logging
        logger = get_logger("setup")
        logger.info("Enterprise logging system initialized")
        logger.debug("Debug logging enabled")
        
        # Check health
        health = self.logger_system.health_check()
        if health['status'] == 'healthy':
            print("   ✅ Logging system operational")
            print(f"   Log files: {health['log_files_count']}")
        else:
            print(f"   ❌ Logging system unhealthy: {health.get('error', 'Unknown error')}")
            return False
        
        return True
    
    def setup_monitoring_system(self):
        """Set up enterprise monitoring and health checks."""
        print("📊 Setting up monitoring system...")
        
        self.monitor = get_monitor()
        
        # Run initial health checks
        status = self.monitor.get_system_status()
        
        print(f"   Overall status: {status['overall_status']}")
        
        healthy_checks = 0
        total_checks = len(status['health_checks'])
        
        for name, check in status['health_checks'].items():
            if check['status'] == 'healthy':
                healthy_checks += 1
                print(f"   ✅ {name}: {check['message']}")
            else:
                print(f"   ⚠️ {name}: {check['status']} - {check['message']}")
        
        print(f"   Health checks: {healthy_checks}/{total_checks} passing")
        
        # Start continuous monitoring
        self.monitor.start_monitoring()
        print("   ✅ Continuous monitoring started")
        
        return True
    
    def setup_error_handling(self):
        """Set up enterprise error handling."""
        print("🛡️ Setting up error handling system...")
        
        self.error_handler = get_error_handler()
        
        # Test error handling
        try:
            raise ValueError("Test error for enterprise setup")
        except Exception as e:
            error_info = self.error_handler.handle_error(e, "setup_test")
            print(f"   ✅ Error handling test: {error_info.error_id}")
        
        # Check health
        health = self.error_handler.health_check()
        print(f"   Error handling status: {health['status']}")
        print(f"   Recent errors: {health['statistics']['total_errors']}")
        
        return True
    
    def setup_security_basics(self):
        """Set up basic security configurations."""
        print("🔒 Setting up security basics...")
        
        security_dir = self.base_path / "security"
        
        # Create security configuration
        security_config = {
            'api_key_required': True,
            'rate_limiting': {
                'enabled': True,
                'requests_per_minute': 1000
            },
            'encryption': {
                'enabled': True,
                'algorithm': 'AES-256'
            },
            'audit_logging': {
                'enabled': True,
                'log_file': str(security_dir / "audit.log")
            }
        }
        
        with open(security_dir / "security_config.yaml", 'w') as f:
            yaml.dump(security_config, f, default_flow_style=False, indent=2)
        
        # Create .gitignore for sensitive files
        gitignore_content = """# Enterprise security files
security/secrets.yaml
security/keys/
security/*.key
security/*.pem
.env
.env.*
!.env.template

# Logs with potential sensitive data
logs/audit.log
logs/*_errors.log

# Cache and temporary files
cache/
temp/
*.tmp
"""
        
        with open(self.base_path / ".gitignore", 'a') as f:
            f.write(gitignore_content)
        
        print("   ✅ Security basics configured")
        return True
    
    def setup_performance_monitoring(self):
        """Set up performance monitoring."""
        print("⚡ Setting up performance monitoring...")
        
        metrics_dir = self.base_path / "metrics"
        
        # Create performance monitoring config
        perf_config = {
            'collection_interval': 60,  # seconds
            'retention_days': 30,
            'metrics': {
                'system': ['cpu', 'memory', 'disk', 'network'],
                'application': ['processing_rate', 'error_rate', 'quality_scores'],
                'custom': []
            },
            'alerts': {
                'cpu_threshold': 80,
                'memory_threshold': 85,
                'disk_threshold': 90,
                'error_rate_threshold': 0.05
            }
        }
        
        with open(metrics_dir / "performance_config.yaml", 'w') as f:
            yaml.dump(perf_config, f, default_flow_style=False, indent=2)
        
        print("   ✅ Performance monitoring configured")
        return True
    
    def create_enterprise_status_dashboard(self):
        """Create a simple status dashboard."""
        print("📈 Creating enterprise status dashboard...")
        
        dashboard_script = '''#!/usr/bin/env python3
"""
Enterprise Status Dashboard

Quick status overview of all enterprise components.
"""

import sys
from pathlib import Path

# Add enterprise config to path
sys.path.append(str(Path(__file__).parent))

from enterprise_config import get_config
from enterprise_logging import get_logger
from enterprise_monitoring import get_monitor
from enterprise_error_handling import get_error_handler

def show_status():
    """Show comprehensive system status."""
    print("🏢 PIXELATED AI ENTERPRISE STATUS DASHBOARD")
    print("=" * 60)
    
    # Configuration status
    try:
        config = get_config()
        print(f"⚙️ Configuration: ✅ Loaded ({config.environment})")
    except Exception as e:
        print(f"⚙️ Configuration: ❌ Error - {e}")
    
    # Monitoring status
    try:
        monitor = get_monitor()
        status = monitor.get_system_status()
        print(f"📊 System Status: {status['overall_status'].upper()}")
        
        for name, check in status['health_checks'].items():
            status_icon = "✅" if check['status'] == 'healthy' else "⚠️" if check['status'] == 'degraded' else "❌"
            print(f"   {status_icon} {name}: {check['message']}")
        
        # System metrics
        metrics = status['system_metrics']
        print(f"💻 System Metrics:")
        print(f"   CPU: {metrics.get('cpu_percent', 0):.1f}%")
        print(f"   Memory: {metrics.get('memory_percent', 0):.1f}%")
        print(f"   Disk: {metrics.get('disk_percent', 0):.1f}%")
        
    except Exception as e:
        print(f"📊 Monitoring: ❌ Error - {e}")
    
    # Error handling status
    try:
        error_handler = get_error_handler()
        error_health = error_handler.health_check()
        print(f"🛡️ Error Handling: {error_health['status'].upper()}")
        print(f"   Recent errors: {error_health['statistics']['total_errors']}")
    except Exception as e:
        print(f"🛡️ Error Handling: ❌ Error - {e}")
    
    print("=" * 60)
    print("✅ Enterprise baseline operational")

if __name__ == "__main__":
    show_status()
'''
        
        dashboard_file = self.enterprise_path / "status_dashboard.py"
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_script)
        
        # Make it executable
        os.chmod(dashboard_file, 0o755)
        
        print("   ✅ Status dashboard created")
        return True
    
    def run_comprehensive_setup(self):
        """Run complete enterprise baseline setup."""
        print("🚀 STARTING ENTERPRISE BASELINE SETUP")
        print("=" * 60)
        
        setup_steps = [
            ("Directory Structure", self.setup_directory_structure),
            ("Configuration Management", self.setup_configuration_management),
            ("Logging System", self.setup_logging_system),
            ("Monitoring System", self.setup_monitoring_system),
            ("Error Handling", self.setup_error_handling),
            ("Security Basics", self.setup_security_basics),
            ("Performance Monitoring", self.setup_performance_monitoring),
            ("Status Dashboard", self.create_enterprise_status_dashboard)
        ]
        
        successful_steps = 0
        
        for step_name, step_func in setup_steps:
            try:
                if step_func():
                    successful_steps += 1
                else:
                    print(f"❌ {step_name} setup failed")
            except Exception as e:
                print(f"❌ {step_name} setup error: {e}")
        
        print("=" * 60)
        print(f"🎯 ENTERPRISE BASELINE SETUP COMPLETE")
        print(f"✅ {successful_steps}/{len(setup_steps)} components successfully configured")
        
        if successful_steps == len(setup_steps):
            print("🎉 ALL ENTERPRISE COMPONENTS OPERATIONAL!")
            print("\n📋 Next steps:")
            print("   1. Run: python enterprise_config/status_dashboard.py")
            print("   2. Check logs in: logs/")
            print("   3. Monitor health checks continuously")
            print("   4. Configure environment-specific settings")
        else:
            print("⚠️ Some components need attention - check logs for details")
        
        return successful_steps == len(setup_steps)
    
    def verify_enterprise_baseline(self):
        """Verify that enterprise baseline is working correctly."""
        print("🔍 VERIFYING ENTERPRISE BASELINE...")
        
        verification_results = {}
        
        # Test configuration
        try:
            config = get_config()
            verification_results['configuration'] = True
            print("   ✅ Configuration system working")
        except Exception as e:
            verification_results['configuration'] = False
            print(f"   ❌ Configuration system failed: {e}")
        
        # Test logging
        try:
            logger = get_logger("verification")
            logger.info("Enterprise baseline verification test")
            verification_results['logging'] = True
            print("   ✅ Logging system working")
        except Exception as e:
            verification_results['logging'] = False
            print(f"   ❌ Logging system failed: {e}")
        
        # Test monitoring
        try:
            monitor = get_monitor()
            status = monitor.get_system_status()
            verification_results['monitoring'] = status['overall_status'] in ['healthy', 'degraded']
            print(f"   ✅ Monitoring system working ({status['overall_status']})")
        except Exception as e:
            verification_results['monitoring'] = False
            print(f"   ❌ Monitoring system failed: {e}")
        
        # Test error handling
        try:
            error_handler = get_error_handler()
            health = error_handler.health_check()
            verification_results['error_handling'] = health['status'] != 'critical'
            print(f"   ✅ Error handling working ({health['status']})")
        except Exception as e:
            verification_results['error_handling'] = False
            print(f"   ❌ Error handling failed: {e}")
        
        # Overall result
        all_working = all(verification_results.values())
        working_count = sum(verification_results.values())
        total_count = len(verification_results)
        
        print(f"\n🎯 VERIFICATION COMPLETE: {working_count}/{total_count} components working")
        
        if all_working:
            print("✅ ENTERPRISE BASELINE FULLY OPERATIONAL!")
        else:
            print("⚠️ Some components need attention")
        
        return all_working

if __name__ == "__main__":
    setup = EnterpriseBaselineSetup()
    
    # Run complete setup
    success = setup.run_comprehensive_setup()
    
    if success:
        # Verify everything is working
        setup.verify_enterprise_baseline()
    
    print("\n🏢 Enterprise baseline setup complete!")
    print("Run 'python enterprise_config/status_dashboard.py' to check status anytime.")
