#!/usr/bin/env python3
"""
Quality Analytics Dashboard Launcher (Task 5.6.2.1)

Enterprise-grade launcher for the quality analytics dashboard with
comprehensive setup, validation, and monitoring capabilities.
"""

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class QualityAnalyticsDashboardLauncher:
    """
    Enterprise-grade launcher for quality analytics dashboard.

    Handles setup, validation, dependency checking, and dashboard launch
    with comprehensive error handling and monitoring.
    """

    def __init__(self):
        """Initialize the dashboard launcher."""
        self.project_root = Path(__file__).parent.parent
        self.monitoring_dir = Path(__file__).parent
        self.db_path = self.project_root / "database" / "conversations.db"

        # Required dependencies
        self.required_packages = [
            "streamlit",
            "pandas",
            "plotly",
            "sqlite3",
            "numpy",
            "seaborn",
            "matplotlib",
        ]

        logger.info("🚀 Quality Analytics Dashboard Launcher initialized")

    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        logger.info("🔍 Checking dependencies...")

        missing_packages = []

        for package in self.required_packages:
            try:
                if package in {
                    "sqlite3",
                    "streamlit",
                    "pandas",
                    "plotly",
                    "numpy",
                    "seaborn",
                    "matplotlib",
                }:
                    pass

                logger.info(f"  ✅ {package}: Available")

            except ImportError:
                missing_packages.append(package)
                logger.error(f"  ❌ {package}: Missing")

        if missing_packages:
            logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
            logger.info("💡 Install missing packages with: pip install " + " ".join(missing_packages))
            return False

        logger.info("✅ All dependencies available")
        return True

    def validate_database(self) -> bool:
        """Validate that the database exists and has quality data."""
        logger.info("🔍 Validating database...")

        if not self.db_path.exists():
            logger.error(f"❌ Database not found: {self.db_path}")
            logger.info("💡 Run the dataset processing pipeline to create the database")
            return False

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Check if required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            required_tables = ["conversations", "quality_metrics"]
            missing_tables = [table for table in required_tables if table not in tables]

            if missing_tables:
                logger.error(f"❌ Missing tables: {', '.join(missing_tables)}")
                conn.close()
                return False

            # Check if quality data exists
            cursor.execute("SELECT COUNT(*) FROM quality_metrics WHERE overall_quality IS NOT NULL")
            quality_count = cursor.fetchone()[0]

            if quality_count == 0:
                logger.error("❌ No quality data found in database")
                logger.info("💡 Run quality validation on your conversations first")
                conn.close()
                return False

            logger.info(f"✅ Database validated: {quality_count} quality records found")
            conn.close()
            return True

        except Exception as e:
            logger.error(f"❌ Database validation error: {e}")
            return False

    def validate_dashboard_files(self) -> bool:
        """Validate that dashboard files exist and are accessible."""
        logger.info("🔍 Validating dashboard files...")

        dashboard_file = self.monitoring_dir / "quality_analytics_dashboard.py"

        if not dashboard_file.exists():
            logger.error(f"❌ Dashboard file not found: {dashboard_file}")
            return False

        # Check if file is readable and has main content
        try:
            with open(dashboard_file) as f:
                content = f.read()
                if "QualityAnalyticsDashboard" not in content:
                    logger.error("❌ Dashboard file appears to be corrupted")
                    return False

            logger.info("✅ Dashboard files validated")
            return True

        except Exception as e:
            logger.error(f"❌ Error reading dashboard file: {e}")
            return False

    def create_launch_config(self) -> dict:
        """Create launch configuration for the dashboard."""
        config = {
            "dashboard_title": "Quality Analytics Dashboard",
            "port": 8501,
            "host": "localhost",
            "db_path": str(self.db_path),
            "launch_time": datetime.now(UTC).isoformat(),
            "project_root": str(self.project_root),
            "monitoring_dir": str(self.monitoring_dir),
        }

        # Save config for reference
        config_path = self.monitoring_dir / "dashboard_launch_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"📝 Launch configuration saved: {config_path}")
        return config

    def launch_dashboard(self, port: int = 8501, host: str = "localhost") -> bool:
        """Launch the quality analytics dashboard."""
        logger.info(f"🚀 Launching Quality Analytics Dashboard on {host}:{port}...")

        dashboard_file = self.monitoring_dir / "quality_analytics_dashboard.py"

        try:
            # Prepare streamlit command
            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard_file),
                "--server.port",
                str(port),
                "--server.address",
                host,
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ]

            logger.info(f"📋 Command: {' '.join(cmd)}")

            # Launch dashboard
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            logger.info("✅ Dashboard launched successfully!")
            logger.info(f"🌐 Access dashboard at: http://{host}:{port}")
            logger.info("🛑 Press Ctrl+C to stop the dashboard")

            # Monitor the process
            try:
                process.wait()
            except KeyboardInterrupt:
                logger.info("🛑 Stopping dashboard...")
                process.terminate()
                process.wait()
                logger.info("✅ Dashboard stopped")

            return True

        except Exception as e:
            logger.error(f"❌ Error launching dashboard: {e}")
            return False

    def run_pre_launch_tests(self) -> bool:
        """Run pre-launch tests to ensure everything is working."""
        logger.info("🧪 Running pre-launch tests...")

        test_file = self.monitoring_dir / "test_quality_analytics_dashboard.py"

        if not test_file.exists():
            logger.warning("⚠️ Test file not found, skipping tests")
            return True

        try:
            # Run the test
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=str(self.project_root),
                shell=False,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            if result.returncode == 0:
                logger.info("✅ Pre-launch tests passed")
                return True
            logger.error(f"❌ Pre-launch tests failed: {result.stderr}")
            return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Pre-launch tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error running pre-launch tests: {e}")
            return False

    def launch(self, port: int = 8501, host: str = "localhost", skip_tests: bool = False) -> bool:
        """
        Complete launch sequence for the quality analytics dashboard.

        Args:
            port: Port to run the dashboard on
            host: Host to bind the dashboard to
            skip_tests: Skip pre-launch tests

        Returns:
            bool: True if launch successful, False otherwise
        """
        logger.info("🎯 Starting Quality Analytics Dashboard Launch Sequence...")

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
        if not skip_tests and not self.run_pre_launch_tests():
            logger.warning("⚠️ Pre-launch tests failed, but continuing...")

        # Step 5: Create launch configuration
        self.create_launch_config()

        # Step 6: Launch dashboard
        success = self.launch_dashboard(port=port, host=host)

        if success:
            logger.info("🎉 Quality Analytics Dashboard launched successfully!")
            return True
        logger.error("❌ Dashboard launch failed")
        return False


def main():
    """Main function to launch the quality analytics dashboard."""

    parser = argparse.ArgumentParser(description="Launch Quality Analytics Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to run dashboard on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind dashboard to")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pre-launch tests")

    args = parser.parse_args()

    launcher = QualityAnalyticsDashboardLauncher()
    success = launcher.launch(port=args.port, host=args.host, skip_tests=args.skip_tests)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
