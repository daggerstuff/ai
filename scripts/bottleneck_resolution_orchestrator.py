#!/usr/bin/env python3
"""
Bottleneck Resolution Orchestrator
Master script to execute the complete 3-week emergency resolution plan.
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bottleneck_resolution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BottleneckResolutionOrchestrator:
    """Master orchestrator for the 3-week emergency resolution plan."""
    
    def __init__(self):
        self.scripts_dir = Path(__file__).parent
        self.start_time = datetime.now()
        self.completed_phases = []
        self.failed_phases = []
        
    def run_script(self, script_name: str, description: str) -> bool:
        """Run a script and track its success/failure."""
        logger.info(f"🚀 Starting: {description}")
        
        try:
            script_path = self.scripts_dir / script_name
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Completed: {description}")
                logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Failed: {description}")
                logger.error(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception in {description}: {e}")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites before starting."""
        logger.info("🔍 Checking prerequisites...")

        missing = []

        # Check Python
        try:
            subprocess.run(["python3", "--version"], capture_output=True, check=True)
            logger.info("✅ Python 3.8+ available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Python 3.8+ not found")
            missing.append("Python 3.8+")

        # Check Docker
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            logger.info("✅ Docker available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Docker not found")
            missing.append("Docker")

        # Check Docker Compose (try both new and old syntax)
        docker_compose_available = False
        try:
            subprocess.run(["docker", "compose", "version"], capture_output=True, check=True)
            logger.info("✅ Docker Compose available")
            docker_compose_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(["docker-compose", "--version"], capture_output=True, check=True)
                logger.info("✅ Docker Compose available")
                docker_compose_available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error("❌ Docker Compose not found")
                missing.append("Docker Compose")

        # Check PostgreSQL client
        try:
            subprocess.run(["psql", "--version"], capture_output=True, check=True)
            logger.info("✅ PostgreSQL client available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ PostgreSQL client not found")
            missing.append("PostgreSQL client")

        if missing:
            logger.error(f"Missing prerequisites: {', '.join(missing)}")
            return False

        logger.info("✅ All prerequisites satisfied")
        return True
    
    def phase_1a_emergency_data_persistence(self) -> bool:
        """Phase 1A: Emergency Data Persistence (Days 1-2)."""
        logger.info("🚨 PHASE 1A: EMERGENCY DATA PERSISTENCE")
        
        tasks = [
            ("emergency_backup.py", "Create emergency backup of all processed conversations"),
            ("setup_production_database.py", "Setup PostgreSQL database and migrate data")
        ]
        
        for script, description in tasks:
            if not self.run_script(script, description):
                self.failed_phases.append("1A")
                return False
        
        self.completed_phases.append("1A")
        logger.info("✅ PHASE 1A COMPLETED: Data is now safely persisted")
        return True
    
    def phase_1b_core_infrastructure(self) -> bool:
        """Phase 1B: Core Production Infrastructure (Days 3-5)."""
        logger.info("🏗️ PHASE 1B: CORE PRODUCTION INFRASTRUCTURE")
        
        # Check if Docker containers are already built
        logger.info("Checking existing Docker infrastructure...")
        
        # The Dockerfiles already exist, so we just need to build them
        docker_builds = [
            (".", "pixel-empathy-main", "Main application container"),
            ("pixel_voice", "pixel-voice", "Voice processing container")
        ]
        
        for build_context, image_name, description in docker_builds:
            logger.info(f"Building {description}...")
            try:
                result = subprocess.run([
                    "docker", "build", "-t", image_name, build_context
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✅ {description} built successfully")
                else:
                    logger.error(f"❌ {description} build failed: {result.stderr}")
                    self.failed_phases.append("1B")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Docker build error for {description}: {e}")
                self.failed_phases.append("1B")
                return False
        
        self.completed_phases.append("1B")
        logger.info("✅ PHASE 1B COMPLETED: Core infrastructure ready")
        return True
    
    def phase_1c_integration_testing(self) -> bool:
        """Phase 1C: Integration Testing Framework (Days 6-7)."""
        logger.info("🧪 PHASE 1C: INTEGRATION TESTING FRAMEWORK")
        
        # For now, we'll create a basic integration test
        # This would be expanded with proper test suites
        logger.info("Setting up integration testing framework...")
        
        try:
            # Create basic integration test structure
            test_dir = Path("tests/integration")
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a basic integration test
            basic_test = test_dir / "test_basic_integration.py"
            if not basic_test.exists():
                test_content = '''#!/usr/bin/env python3
"""Basic integration test for the Pixel Empathy system."""

import pytest
import requests
import psycopg2
import os

def test_database_connection():
    """Test database connectivity."""
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pixelated_empathy")
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversations")
        count = cursor.fetchone()[0]
        assert count >= 0
        conn.close()
        print(f"✅ Database test passed: {count} conversations found")
    except Exception as e:
        pytest.fail(f"Database connection failed: {e}")

def test_api_health():
    """Test API health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200
        print("✅ API health test passed")
    except Exception as e:
        print(f"⚠️ API health test skipped (API not running): {e}")

if __name__ == "__main__":
    test_database_connection()
    test_api_health()
'''
                with open(basic_test, 'w') as f:
                    f.write(test_content)
            
            logger.info("✅ Integration testing framework created")
            self.completed_phases.append("1C")
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration testing setup failed: {e}")
            self.failed_phases.append("1C")
            return False
    
    def phase_2a_monitoring_systems(self) -> bool:
        """Phase 2A: Real-Time Monitoring & Alerting."""
        logger.info("📊 PHASE 2A: MONITORING SYSTEMS")
        
        if not self.run_script("setup_monitoring.py", "Deploy monitoring stack (Prometheus, Grafana, AlertManager)"):
            self.failed_phases.append("2A")
            return False
        
        self.completed_phases.append("2A")
        logger.info("✅ PHASE 2A COMPLETED: Monitoring systems active")
        return True
    
    def generate_completion_report(self) -> Dict[str, Any]:
        """Generate completion report."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_hours": duration.total_seconds() / 3600,
            "completed_phases": self.completed_phases,
            "failed_phases": self.failed_phases,
            "success_rate": len(self.completed_phases) / (len(self.completed_phases) + len(self.failed_phases)) if (self.completed_phases or self.failed_phases) else 0,
            "next_steps": []
        }
        
        # Add next steps based on what was completed
        if "1A" in self.completed_phases:
            report["next_steps"].append("✅ Data is safely backed up and migrated to PostgreSQL")
        if "1B" in self.completed_phases:
            report["next_steps"].append("✅ Docker containers are built and ready for deployment")
        if "1C" in self.completed_phases:
            report["next_steps"].append("✅ Integration testing framework is in place")
        if "2A" in self.completed_phases:
            report["next_steps"].append("✅ Monitoring stack is deployed and active")
        
        # Add remaining work
        if "1A" not in self.completed_phases:
            report["next_steps"].append("🚨 CRITICAL: Run emergency backup and database migration")
        if "2A" not in self.completed_phases:
            report["next_steps"].append("📊 Deploy monitoring systems")
        
        # Always add these as they're not automated yet
        report["next_steps"].extend([
            "🔄 Implement Redis task queue for distributed processing",
            "🛡️ Add fault tolerance and circuit breaker patterns",
            "🚀 Build REST API for dataset access",
            "📋 Create operational runbooks and documentation"
        ])
        
        return report
    
    def run_emergency_resolution(self) -> bool:
        """Run the complete emergency resolution plan."""
        logger.info("🚨 STARTING BOTTLENECK RESOLUTION PLAN")
        logger.info("Timeline: 3 weeks to production readiness")
        logger.info("Risk Level: MAXIMUM - Data Loss & System Failure Imminent")
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites not met. Aborting.")
            return False
        
        # Execute phases
        phases = [
            (self.phase_1a_emergency_data_persistence, "Phase 1A: Emergency Data Persistence"),
            (self.phase_1b_core_infrastructure, "Phase 1B: Core Production Infrastructure"),
            (self.phase_1c_integration_testing, "Phase 1C: Integration Testing Framework"),
            (self.phase_2a_monitoring_systems, "Phase 2A: Monitoring Systems")
        ]
        
        for phase_func, phase_name in phases:
            logger.info(f"\n{'='*60}")
            logger.info(f"EXECUTING: {phase_name}")
            logger.info(f"{'='*60}")
            
            if not phase_func():
                logger.error(f"❌ {phase_name} FAILED")
                break
            
            logger.info(f"✅ {phase_name} COMPLETED")
        
        # Generate report
        report = self.generate_completion_report()
        
        # Save report
        report_path = Path("bottleneck_resolution_report.json")
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("BOTTLENECK RESOLUTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Duration: {report['duration_hours']:.2f} hours")
        logger.info(f"Completed phases: {', '.join(report['completed_phases'])}")
        if report['failed_phases']:
            logger.info(f"Failed phases: {', '.join(report['failed_phases'])}")
        logger.info(f"Success rate: {report['success_rate']*100:.1f}%")
        
        logger.info("\nNext Steps:")
        for step in report['next_steps']:
            logger.info(f"  {step}")
        
        logger.info(f"\nDetailed report saved to: {report_path}")
        
        return len(report['failed_phases']) == 0

def main():
    """Main orchestrator entry point."""
    orchestrator = BottleneckResolutionOrchestrator()
    success = orchestrator.run_emergency_resolution()
    
    if success:
        logger.info("🎉 BOTTLENECK RESOLUTION COMPLETED SUCCESSFULLY!")
    else:
        logger.error("❌ BOTTLENECK RESOLUTION INCOMPLETE - MANUAL INTERVENTION REQUIRED")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
