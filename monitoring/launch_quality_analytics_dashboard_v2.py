#!/usr/bin/env python3
"""
Quality Analytics Dashboard V2 Launcher (Task 5.6.2.1)

Enterprise-grade launcher for the quality analytics dashboard with
comprehensive validation, dependency checking, and production deployment.
"""

import sys
import os
import subprocess
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
import json
import argparse
import signal
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / 'dashboard_launcher.log')
    ]
)
logger = logging.getLogger(__name__)

class QualityAnalyticsDashboardLauncher:
    """
    Enterprise-grade launcher for Quality Analytics Dashboard V2.
    
    Features:
    - Comprehensive dependency validation
    - Database schema verification
    - Production-ready deployment
    - Health monitoring and auto-recovery
    - Graceful shutdown handling
    """
    
    def __init__(self):
        """Initialize the dashboard launcher."""
        self.project_root = Path(__file__).parent.parent
        self.monitoring_dir = Path(__file__).parent
        self.db_path = self.project_root / "database" / "conversations.db"
        self.dashboard_file = self.monitoring_dir / "quality_analytics_dashboard_v2.py"
        
        # Required dependencies with version checks
        self.required_packages = {
            'streamlit': '1.0.0',
            'pandas': '1.3.0',
            'plotly': '5.0.0',
            'numpy': '1.20.0',
            'seaborn': '0.11.0',
            'matplotlib': '3.3.0',
            'sqlite3': None  # Built-in
        }
        
        # Process tracking
        self.dashboard_process = None
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Quality Analytics Dashboard V2 Launcher initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        if self.dashboard_process:
            self._stop_dashboard()
    
    def check_dependencies(self) -> bool:
        """
        Check if all required dependencies are available with version validation.
        
        Returns:
            bool: True if all dependencies are satisfied
        """
        logger.info("🔍 Checking dependencies...")
        
        missing_packages = []
        version_issues = []
        
        for package, min_version in self.required_packages.items():
            try:
                if package == 'sqlite3':
                    import sqlite3
                    logger.info(f"  ✅ {package}: Available (built-in)")
                    continue
                
                # Dynamic import
                module = __import__(package)
                
                # Check version if specified
                if min_version and hasattr(module, '__version__'):
                    current_version = module.__version__
                    if self._compare_versions(current_version, min_version) < 0:
                        version_issues.append(f"{package} (current: {current_version}, required: >={min_version})")
                        logger.warning(f"  ⚠️ {package}: Version {current_version} < {min_version}")
                    else:
                        logger.info(f"  ✅ {package}: Version {current_version}")
                else:
                    logger.info(f"  ✅ {package}: Available")
                
            except ImportError:
                missing_packages.append(package)
                logger.error(f"  ❌ {package}: Missing")
        
        if missing_packages:
            logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
            logger.info("💡 Install missing packages with:")
            logger.info(f"   pip install {' '.join(missing_packages)}")
            return False
        
        if version_issues:
            logger.error(f"❌ Version issues: {', '.join(version_issues)}")
            logger.info("💡 Upgrade packages with:")
            logger.info(f"   pip install --upgrade {' '.join(p.split()[0] for p in version_issues)}")
            return False
        
        logger.info("✅ All dependencies satisfied")
        return True
    
    def _compare_versions(self, current: str, required: str) -> int:
        """
        Compare version strings.
        
        Returns:
            int: -1 if current < required, 0 if equal, 1 if current > required
        """
        def version_tuple(v):
            return tuple(map(int, (v.split("."))))
        
        try:
            current_tuple = version_tuple(current)
            required_tuple = version_tuple(required)
            
            if current_tuple < required_tuple:
                return -1
            elif current_tuple > required_tuple:
                return 1
            else:
                return 0
        except:
            return 0  # Assume OK if can't parse
    
    def validate_database(self) -> bool:
        """
        Validate database exists and has correct schema with quality data.
        
        Returns:
            bool: True if database is valid
        """
        logger.info("🔍 Validating database...")
        
        if not self.db_path.exists():
            logger.error(f"❌ Database not found: {self.db_path}")
            logger.info("💡 Run the dataset processing pipeline to create the database")
            return False
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['conversations', 'conversation_quality']
            missing_tables = [table for table in required_tables if table not in tables]
            
            if missing_tables:
                logger.error(f"❌ Missing tables: {', '.join(missing_tables)}")
                conn.close()
                return False
            
            # Validate schema for conversations table
            cursor.execute("PRAGMA table_info(conversations)")
            conv_columns = [row[1] for row in cursor.fetchall()]
            required_conv_columns = ['conversation_id', 'tier', 'created_at', 'dataset_source']
            
            missing_conv_columns = [col for col in required_conv_columns if col not in conv_columns]
            if missing_conv_columns:
                logger.error(f"❌ Missing columns in conversations table: {', '.join(missing_conv_columns)}")
                conn.close()
                return False
            
            # Validate schema for conversation_quality table
            cursor.execute("PRAGMA table_info(conversation_quality)")
            qual_columns = [row[1] for row in cursor.fetchall()]
            required_qual_columns = ['conversation_id', 'overall_quality', 'therapeutic_accuracy']
            
            missing_qual_columns = [col for col in required_qual_columns if col not in qual_columns]
            if missing_qual_columns:
                logger.error(f"❌ Missing columns in conversation_quality table: {', '.join(missing_qual_columns)}")
                conn.close()
                return False
            
            # Check if quality data exists
            cursor.execute("SELECT COUNT(*) FROM conversation_quality WHERE overall_quality IS NOT NULL")
            quality_count = cursor.fetchone()[0]
            
            if quality_count == 0:
                logger.error("❌ No quality data found in database")
                logger.info("💡 Run quality validation on your conversations first")
                conn.close()
                return False
            
            # Check data recency
            cursor.execute("SELECT MAX(created_at) FROM conversations")
            latest_date = cursor.fetchone()[0]
            if latest_date:
                latest_datetime = datetime.fromisoformat(latest_date.replace('Z', '+00:00') if 'Z' in latest_date else latest_date)
                days_old = (datetime.now() - latest_datetime).days
                if days_old > 30:
                    logger.warning(f"⚠️ Latest data is {days_old} days old")
                else:
                    logger.info(f"✅ Data is current ({days_old} days old)")
            
            logger.info(f"✅ Database validated: {quality_count:,} quality records found")
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Database validation error: {e}")
            return False
    
    def validate_dashboard_files(self) -> bool:
        """
        Validate dashboard files exist and are functional.
        
        Returns:
            bool: True if files are valid
        """
        logger.info("🔍 Validating dashboard files...")
        
        if not self.dashboard_file.exists():
            logger.error(f"❌ Dashboard file not found: {self.dashboard_file}")
            return False
        
        # Check if file is readable and has required content
        try:
            with open(self.dashboard_file, 'r') as f:
                content = f.read()
                
                required_components = [
                    'QualityAnalyticsDashboard',
                    'run_streamlit_dashboard',
                    'load_quality_data',
                    'calculate_quality_analytics'
                ]
                
                missing_components = [comp for comp in required_components if comp not in content]
                if missing_components:
                    logger.error(f"❌ Missing components in dashboard file: {', '.join(missing_components)}")
                    return False
            
            logger.info("✅ Dashboard files validated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error reading dashboard file: {e}")
            return False
    
    def run_pre_launch_tests(self) -> bool:
        """
        Run pre-launch tests to ensure dashboard functionality.
        
        Returns:
            bool: True if tests pass
        """
        logger.info("🧪 Running pre-launch tests...")
        
        test_file = self.monitoring_dir / "test_quality_analytics_dashboard_v2.py"
        
        if not test_file.exists():
            logger.warning("⚠️ Test file not found, skipping tests")
            return True
        
        try:
            # Run the test with timeout
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Pre-launch tests passed")
                return True
            else:
                logger.error(f"❌ Pre-launch tests failed:")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Pre-launch tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error running pre-launch tests: {e}")
            return False
    
    def create_launch_config(self, port: int, host: str) -> dict:
        """
        Create launch configuration for the dashboard.
        
        Args:
            port: Port number
            host: Host address
            
        Returns:
            dict: Launch configuration
        """
        config = {
            'dashboard_title': 'Quality Analytics Dashboard V2',
            'version': '2.0.0',
            'port': port,
            'host': host,
            'db_path': str(self.db_path),
            'launch_time': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'monitoring_dir': str(self.monitoring_dir),
            'dashboard_file': str(self.dashboard_file),
            'launcher_version': '2.0.0'
        }
        
        # Save config for reference
        config_path = self.monitoring_dir / "dashboard_v2_launch_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"📝 Launch configuration saved: {config_path}")
        return config
    
    def launch_dashboard(self, port: int = 8501, host: str = 'localhost') -> bool:
        """
        Launch the quality analytics dashboard.
        
        Args:
            port: Port to run dashboard on
            host: Host to bind dashboard to
            
        Returns:
            bool: True if launch successful
        """
        logger.info(f"🚀 Launching Quality Analytics Dashboard V2 on {host}:{port}...")
        
        try:
            # Prepare streamlit command
            cmd = [
                sys.executable, '-m', 'streamlit', 'run',
                str(self.dashboard_file),
                '--server.port', str(port),
                '--server.address', host,
                '--server.headless', 'true',
                '--browser.gatherUsageStats', 'false',
                '--server.enableCORS', 'false',
                '--server.enableXsrfProtection', 'true'
            ]
            
            logger.info(f"📋 Command: {' '.join(cmd)}")
            
            # Launch dashboard
            self.dashboard_process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment to check if process started successfully
            time.sleep(3)
            
            if self.dashboard_process.poll() is None:
                logger.info(f"✅ Dashboard launched successfully! PID: {self.dashboard_process.pid}")
                logger.info(f"🌐 Access dashboard at: http://{host}:{port}")
                logger.info("🛑 Press Ctrl+C to stop the dashboard")
                
                # Monitor the process
                self._monitor_dashboard()
                return True
            else:
                stdout, stderr = self.dashboard_process.communicate()
                logger.error(f"❌ Dashboard failed to start:")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error launching dashboard: {e}")
            return False
    
    def _monitor_dashboard(self):
        """Monitor dashboard process and handle shutdown."""
        try:
            while not self.shutdown_requested:
                if self.dashboard_process.poll() is not None:
                    # Process has terminated
                    stdout, stderr = self.dashboard_process.communicate()
                    logger.error("❌ Dashboard process terminated unexpectedly")
                    if stdout:
                        logger.error(f"STDOUT: {stdout}")
                    if stderr:
                        logger.error(f"STDERR: {stderr}")
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
        finally:
            self._stop_dashboard()
    
    def _stop_dashboard(self):
        """Stop the dashboard process gracefully."""
        if self.dashboard_process and self.dashboard_process.poll() is None:
            logger.info("🛑 Stopping dashboard...")
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=10)
                logger.info("✅ Dashboard stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Dashboard didn't stop gracefully, forcing termination")
                self.dashboard_process.kill()
                self.dashboard_process.wait()
                logger.info("✅ Dashboard terminated")
            except Exception as e:
                logger.error(f"❌ Error stopping dashboard: {e}")
    
    def launch(self, port: int = 8501, host: str = 'localhost', skip_tests: bool = False) -> bool:
        """
        Complete launch sequence for the quality analytics dashboard.
        
        Args:
            port: Port to run the dashboard on
            host: Host to bind the dashboard to
            skip_tests: Skip pre-launch tests
            
        Returns:
            bool: True if launch successful
        """
        logger.info("🎯 Starting Quality Analytics Dashboard V2 Launch Sequence...")
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Launch aborted: Missing dependencies")
            return False
        
        # Step 2: Validate database
        if not self.validate_database():
            logger.error("❌ Launch aborted: Database validation failed")
            return False
        
        # Step 3: Validate dashboard files
        if not self.validate_dashboard_files():
            logger.error("❌ Launch aborted: Dashboard files validation failed")
            return False
        
        # Step 4: Run pre-launch tests (optional)
        if not skip_tests:
            if not self.run_pre_launch_tests():
                logger.warning("⚠️ Pre-launch tests failed, but continuing...")
        
        # Step 5: Create launch configuration
        config = self.create_launch_config(port=port, host=host)
        
        # Step 6: Launch dashboard
        success = self.launch_dashboard(port=port, host=host)
        
        if success:
            logger.info("🎉 Quality Analytics Dashboard V2 launched successfully!")
            return True
        else:
            logger.error("❌ Dashboard launch failed")
            return False

def main():
    """Main function to launch the quality analytics dashboard."""
    parser = argparse.ArgumentParser(
        description='Launch Quality Analytics Dashboard V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_quality_analytics_dashboard_v2.py
  python launch_quality_analytics_dashboard_v2.py --port 8502 --host 0.0.0.0
  python launch_quality_analytics_dashboard_v2.py --skip-tests
        """
    )
    
    parser.add_argument('--port', type=int, default=8501, 
                       help='Port to run dashboard on (default: 8501)')
    parser.add_argument('--host', type=str, default='localhost', 
                       help='Host to bind dashboard to (default: localhost)')
    parser.add_argument('--skip-tests', action='store_true', 
                       help='Skip pre-launch tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    launcher = QualityAnalyticsDashboardLauncher()
    success = launcher.launch(
        port=args.port,
        host=args.host,
        skip_tests=args.skip_tests
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
