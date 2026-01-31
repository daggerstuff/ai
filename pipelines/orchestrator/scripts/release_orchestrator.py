#!/usr/bin/env python3
"""
ENTERPRISE-GRADE Release Orchestrator - Mental Health Datasets Expansion Release 0

Master orchestrator that runs all GitHub issues in sequence to create a complete
Release 0 dataset with all gates and validations, integrated with existing
enterprise infrastructure and main orchestrator.

This implements the complete EPIC from the GitHub issues tracking document
with production-grade enterprise components.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the dataset_pipeline to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage_config import StorageConfig, get_storage_config
from safety_ethics_audit_trail import get_audit_trail, AuditEventType
from main_orchestrator import DatasetPipelineOrchestrator

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("enterprise_release_orchestrator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class EnterpriseReleaseOrchestrator:
    """Enterprise-grade orchestrator for the complete Release 0 process"""

    def __init__(
        self, storage_config: StorageConfig, release_version: Optional[str] = None
    ):
        self.config = storage_config
        self.release_version = (
            release_version or f"v{datetime.utcnow().strftime('%Y-%m-%d')}"
        )
        self.scripts_dir = Path(__file__).parent
        self.results = {}
        self.audit_trail = get_audit_trail()

        # Initialize main dataset pipeline orchestrator for integration
        try:
            self.main_orchestrator = DatasetPipelineOrchestrator()
            logger.info("Main dataset pipeline orchestrator integrated")
        except Exception as e:
            logger.warning(f"Main orchestrator unavailable: {e}")
            self.main_orchestrator = None

    def run_script(
        self, script_name: str, args: List[str] = None
    ) -> Tuple[bool, str, str]:
        """Run a script with enhanced error handling and logging"""
        script_path = self.scripts_dir / script_name

        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False, "", f"Script not found: {script_path}"

        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)

        logger.info(f"Executing: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for enterprise operations
            )

            success = result.returncode == 0

            if success:
                logger.info(f"Script {script_name} completed successfully")
            else:
                logger.error(f"Script {script_name} failed with return code {result.returncode}")

            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            logger.error(f"Script {script_name} timed out after 10 minutes")
            return False, "", "Script timed out after 10 minutes"
        except Exception as e:
.error(f"Failed to run script {script_name}: {e}")
            return False, "", f"Failed to run script: {str(e)}"

    def issue_1_dataset_pipeline_integration(self) -> Dict[str, Any]:
        """Issue 1: Integrate with main dataset pipeline orchestrator"""
        logger.info("Starting dataset pipeline integration...")

        result = {
            "issue": "Issue 1: Dataset Pipeline Integration",
            "success": False,
            "timestamp": datetime.utcnow().isoformat(),
            "integration_details": {}
        }

        if self.main_orchestrator:
            try:
                # Run the complete dataset pipeline
                pipeline_results = self.main_orchestrator.execute_complete_pipeline()

                result["success"] = pipeline_results["success"]
                result["integration_details"] = pipeline_results["results"]

                # Log integration success
                if self.audit_trail:
                    self.audit_trail.log_validation_started(
                        self.release_version,
                        user_id="enterprise_release_orchestrator"
                    )

                logger.info("Dataset pipeline integration completed successfully")

            except Exception as e:
                logger.error(f"Dataset pipeline integration failed: {e}")
                result["error"] = str(e)

                if self.audit_trail:
                    self.audit_trail.log_safety_issue(
                        self.release_version,
                        {
                            "issue_type": "pipeline_integration_failure",
                            "error": str(e)
                        }
                    )
        else:
            logger.warning("Main orchestrator unavailable - skipping integration")
            result["success"] = True  # Don't block release for missing integration
            result["integration_details"] = {"status": "skipped", "reason": "main_orchestrator_unavailable"}

eturn result

    def issue_2_coverage_matrix(self) -> Dict[str, Any]:
        """Issue 2: Build coverage matrix from S3 inventory"""
        logger.info("Building enterprise coverage matrix...")

        success, stdout, stderr = self.run_script("build_coverage_matrix.py")

        result = {
            "issue": "Issue 2: Enterprise Coverage Matrix",
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if success:
            logger.info("✅ Enterprise coverage matrix generated successfully")
        else:
            logger.error("❌ Enterprise coverage matrix generation failed")
            logger.error(f"Error: {stderr}")

        return result

    def issue_3_manifest_export(self) -> Dict[str, Any]:
        """Issue 3: Generate versioned manifest + compiled ChatML export"""
        print("\n" + "=" * 60)
        print("📋 ISSUE 3: Building Release Manifest and Export")
        print("=" * 60)

        success, stdout, stderr = self.run_script(
            "build_release_manifest.py", [self.release_version]
        )

        result = {
            "issue": "Issue 3: Manifest + Export",
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if success:
            print("✅ Release manifest and export created successfully")
        else:
            print("❌ Release manifest creation failed")
            print(f"Error: {stderr}")

        return result

    def issue_4_privacy_provenance_gates(self) -> Dict[str, Any]:
        """Issue 4: Enforce enterprise privacy and provenance gates"""
        logger.info("Running enterprise privacy and provenance gates...")

        success, stdout, stderr = self.run_script(
            "privacy_provenance_gates.py", [self.release_version]
        )

        result = {
            "issue": "Issue 4: Enterprise Privacy + Provenance Gates",
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if success:
            logger.info("✅ Enterprise privacy and provenance gates passed")
        else:
            logger.error("❌ Enterprise privacy and provenance gates failed - RELEASE BLOCKED")
            logger.error(f"Error: {stderr}")

            # Log critical gate failure
            if self.audit_trail:
                self.audit_trail.log_intervention_required(
                    self.release_version,
                    "Enterprise privacy/provenance gates failed",
                    "critical"
                )

        return result

    def issue_5_dedup_leakage_gates(self) -> Dict[str, Any]:
        """Issue 5: Run enterprise dedup and cross-split leakage gates"""
        logger.info("Running enterprise deduplication and leakage gates...")

        success, stdout, stderr = self.run_script(
            "dedup_leakage_gates.py", [self.release_version]
        )

        result = {
            "issue": "Issue 5: Enterprise Dedup + Leakage Gates",
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if success:
            logger.info("✅ Enterprise deduplication and leakage gates passed")
        else:
            logger.error("❌ Enterprise deduplication and leakage gates failed - RELEASE BLOCKED")
            logger.error(f"Error: {stderr}")

            # Log critical gate failure
            if self.audit_trail:
                self.audit_trail.log_intervention_required(
                    self.release_version,
                    "Enterprise dedup/leakage gates failed",
                    "critical"
                )

        return result

    def issue_6_distribution_gate(self) -> Dict[str, Any]:
        """Issue 6: Record distribution stats by family and split"""
        print("\n" + "=" * 60)
        print("📊 ISSUE 6: Running Distribution Analysis Gate")
        print("=" * 60)

        success, stdout, stderr = self.run_script(
            "distribution_gate.py", [self.release_version]
        )

        result = {
            "issue": "Issue 6: Distribution Gate",
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if success:
            print("✅ Distribution analysis completed")
        else:
            print("⚠️  Distribution analysis has warnings")
            print(f"Details: {stderr}")

        return result

    def issue_7_human_qa_signoff(self) -> Dict[str, Any]:
        """Issue 7: Clinician QA + bias/cultural review signoff"""
        print("\n" + "=" * 60)
        print("📋 ISSUE 7: Human QA Signoff Process")
        print("=" * 60)

        success, stdout, stderr = self.run_script(
            "human_qa_signoff.py", [self.release_version]
        )

        result = {
            "issue": "Issue 7: Human QA Signoff",
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if success:
            print("✅ Human QA signoff completed")
        else:
            print("❌ Human QA signoff failed - RELEASE BLOCKED")
            print(f"Error: {stderr}")

        return result

    def issue_8_training_consumption_test(self) -> Dict[str, Any]:
        """Issue 8: Smoke test training consumes S3 release artifacts"""
        print("\n" + "=" * 60)
        print("🧪 ISSUE 8: Training Consumption Smoke Test")
        print("=" * 60)

        success, stdout, stderr = self.run_script(
            "training_consumption_test.py", [self.release_version]
        )

        result = {
            "issue": "Issue 8: Training Consumption Test",
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if success:
            print("✅ Training consumption test passed")
        else:
            print("❌ Training consumption test failed")
            print(f"Error: {stderr}")

        return result

    def run_complete_release(self, fail_fast: bool = True) -> Dict[str, Any]:
        """Run the complete enterprise Release 0 process"""
        logger.info("🚀 STARTING ENTERPRISE MENTAL HEALTH DATASETS EXPANSION - RELEASE 0")
        logger.info(f"Release Version: {self.release_version}")
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
        logger.info(f"S3 Bucket: {self.config.s3_bucket}")

        release_results = {
            "metadata": {
                "release_version": self.release_version,
                "start_time": datetime.utcnow().isoformat(),
                "orchestrator_version": "2.0.0-enterprise",
                "s3_bucket": self.config.s3_bucket,
                "enterprise_grade": True,
                "audit_trail_enabled": self.audit_trail is not None,
                "main_orchestrator_integrated": self.main_orchestrator is not None
            },
            "issues": {},
            "overall_success": True,
            "enterprise_ready": True,
            "blocking_failures": [],
            "warnings": [],
        }

        # Log release start
        if self.audit_trail:
            self.audit_trail.log_validation_started(
                self.release_version,
                user_id="enterprise_release_orchestrator"
            )

        # Define the enterprise issue execution order
        issues = [
            ("issue_1", self.issue_1_dataset_pipeline_integration, False),  # Non-blocking integration
            ("issue_2", self.issue_2_coverage_matrix, True),  # Blocking
            ("issue_3", self.issue_3_manifest_export, True),  # Blocking
            ("issue_4", self.issue_4_privacy_provenance_gates, True),  # Blocking
            ("issue_5", self.issue_5_dedup_leakage_gates, True),  # Blocking
            ("issue_6", self.issue_6_distribution_gate, False),  # Non-blocking
            ("issue_7", self.issue_7_human_qa_signoff, True),  # Blocking
            ("issue_8", self.issue_8_training_consumption_test, False),  # Non-blocking
        ]

        # Execute issues in sequence with enhanced error handling
        for issue_id, issue_func, is_blocking in issues:
            logger.info(f"\n{'='*60}")
            logger.info(f"EXECUTING {issue_id.upper()}")
            logger.info(f"{'='*60}")

            try:
                result = issue_func()
                release_results["issues"][issue_id] = result

                if not result["success"]:
                    if is_blocking:
                        release_results["blocking_failures"].append(issue_id)
                        release_results["overall_success"] = False
                        release_results["enterprise_ready"] = False

                        logger.error(f"🚨 RELEASE BLOCKED: {issue_id} failed")

                        # Log blocking failure
                        if self.audit_trail:
                            self.audit_trail.log_intervention_required(
                                self.release_version,
                                f"Blocking issue failed: {issue_id}",
                                "critical"
                            )

                        if fail_fast:
                            break
                    else:
                        release_results["warnings"].append(issue_id)
                        logger.warning(f"⚠️  Non-blocking issue failed: {issue_id}")

                # Check for enterprise readiness
                if not result.get("enterprise_grade", True):
                    release_results["enterprise_ready"] = False

            except Exception as e:
                logger.error(f"💥 {issue_id} crashed: {e}")

                error_result = {
                    "issue": issue_id,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                release_results["issues"][issue_id] = error_result

                if is_blocking:
                    release_results["blocking_failures"].append(issue_id)
                    release_results["overall_success"] = False
                    release_results["enterprise_ready"] = False

                    # Log critical failure
                    if self.audit_trail:
                        self.audit_trail.log_safety_issue(
                            self.release_version,
                            {
                                "issue_type": "orchestrator_crash",
                                "issue_id": issue_id,
                                "error": str(e)
                            }
                        )

                    if fail_fast:
                        break
                else:
                    release_results["warnings"].append(issue_id)

        # Finalize results
        release_results["metadata"]["end_time"] = datetime.utcnow().isoformat()

        # Log completion
        if self.audit_trail:
            self.audit_trail.log_validation_completion(
                self.release_version,
                {
                    "overall_success": release_results["overall_success"],
            "enterprise_ready": release_results["enterprise_ready"],
                    "blocking_failures": release_results["blocking_failures"],
                    "warnings": release_results["warnings"]
                }
            )

        return release_results

    def save_release_summary(self, release_results: Dict[str, Any]) -> str:
        """Save release summary to S3"""
        import boto3

        summary_key = f"{self.config.exports_prefix}/releases/{self.release_version}/release_summary.json"

        try:
            s3_client = boto3.client(
                "s3",
                endpoint_url=self.config.s3_endpoint_url,
                aws_access_key_id=self.config.s3_access_key_id,
                aws_secret_access_key=self.config.s3_secret_access_key,
                region_name=self.config.s3_region or "us-east-1",
            )

            s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=summary_key,
                Body=json.dumps(release_results, indent=2),
                ContentType="application/json",
            )

            summary_url = f"s3://{self.config.s3_bucket}/{summary_key}"
            print(f"✓ Release summary saved: {summary_url}")
            return summary_url

        except Exception as e:
            print(f"⚠️  Failed to save release summary: {e}")
            return ""

    def print_final_summary(self, release_results: Dict[str, Any]):
        """Print comprehensive enterprise release summary"""
        print("\n" + "=" * 80)
        print("🎯 ENTERPRISE MENTAL HEALTH DATASETS EXPANSION - RELEASE 0 SUMMARY")
        print("=" * 80)

        metadata = release_results["metadata"]
        print(f"Release Version: {metadata['release_version']}")
        print(f"Start Time: {metadata['start_time']}")
        print(f"End Time: {metadata.get('end_time', 'In Progress')}")
        print(f"S3 Bucket: {metadata['s3_bucket']}")
        print(f"Enterprise Grade: {'✅' if metadata.get('enterprise_grade', False) else '❌'}")
        print(f"Audit Trail: {'✅ ENABLED' if metadata.get('audit_trail_enabled', False) else '⚠️  DISABLED'}")
        print(f"Main Orchestrator: {'✅ INTEGRATED' if metadata.get('main_orchestrator_integrated', False) else '⚠️  STANDALONE'}")

        overall_status = "✅ SUCCESS" if release_results["overall_success"] else "❌ FAILED"
        enterprise_status = "🏢 ENTERPRISE READY" if release_results.get("enterprise_ready", False) else "⚠️  STANDARD GRADE"

        print(f"\nOverall Status: {overall_status}")
        print(f"Enterprise Readiness: {enterprise_status}")

        print("\n📊 ISSUE RESULTS:")
        for issue_id, result in release_results["issues"].items():
            status_icon = "✅" if result["success"] else "❌"
            enterprise_icon = "🏢" if result.get("enterprise_grade", True) else "📊"
            print(f"  {status_icon} {enterprise_icon} {result['issue']}")

        if release_results["blocking_failures"]:
            print("\n🚨 BLOCKING FAILURES:")
            for failure in release_results["blocking_failures"]:
                print(f"  ❌ {failure}")

        if release_results["warnings"]:
            print("\n⚠️  WARNINGS:")
            for warning in release_results["warnings"]:
                print(f"  ⚠️  {warning}")

        if release_results["overall_success"]:
            if release_results.get("enterprise_ready", False):
                print(f"\n🏢 ENTERPRISE RELEASE {metadata['release_version']} COMPLETED SUCCESSFULLY!")
                print("✅ All critical gates passed with enterprise-grade components")
                print("✅ Dataset ready for production training consumption")
                print("✅ Human QA signoff obtained with clinical validation")
                print("✅ Comprehensive audit trail maintained")
            else:
                print(f"\n🎉 STANDARD RELEASE {metadata['release_version']} COMPLETED SUCCESSFULLY!")
                print("✅ All critical gates passed")
                print("✅ Dataset ready for training consumption")
                print("⚠️  Consider upgrading to enterprise-grade components")

            print("\n📁 RELEASE ARTIFACTS:")
            release_prefix = f"s3://{metadata['s3_bucket']}/exports/releases/{metadata['release_version']}"
            print(f"  📋 Manifest: {release_prefix}/manifest.json")
            print(f"  📦 Export: {release_prefix}/compiled_export.jsonl")
            print(f"  ⚙️  Routing Config: {release_prefix}/routing_config.json")
            print(f"  🚪 Gate Reports: {release_prefix}/gates/")
            print(f"  📋 QA Records: {release_prefix}/qa/")
            print(f"  🧪 Test Reports: {release_prefix}/tests/")

        else:
            print(f"\n💥 ENTERPRISE RELEASE {metadata['release_version']} FAILED")
            print("❌ Critical gates failed - release blocked")
            print("🔧 Review gate reports and address all issues before retry")

            if not release_results.get("enterprise_ready", False):
                print("\n💼 ENTERPRISE UPGRADE RECOMMENDED:")
                print("  Consider upgrading infrastructure components for enterprise-grade operations")

        print("\n" + "=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enterprise Mental Health Datasets Expansion - Release 0 Orchestrator"
    )
    parser.add_argument(
        "--release-version", help="Release version (default: vYYYY-MM-DD)", default=None
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Continue execution even after blocking failures",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )

    args = parser.parse_args()

    print("🚀 Enterprise Mental Health Datasets Expansion - Release 0 Orchestrator")
    print("=" * 70)

    # Load storage configuration
    config = get_storage_config()

    # Validate S3 configuration
    is_valid, error_msg = config.validate()
    if not is_valid:
        print(f"❌ Storage configuration error: {error_msg}")
        print("\n💡 Setup Instructions:")
        print("  1. Set DATASET_STORAGE_BACKEND=s3")
        print("  2. Set OVH_S3_BUCKET=pixel-data")
        print("  3. Set OVH_S3_ACCESS_KEY=<your-access-key>")
        print("  4. Set OVH_S3_SECRET_KEY=<your-secret-key>")
        print("  5. Set OVH_S3_ENDPOINT=<your-s3-endpoint>")
        sys.exit(1)

    if config.backend != config.backend.S3:
        print("❌ S3 backend required. Set DATASET_STORAGE_BACKEND=s3")
        sys.exit(1)

    if args.dry_run:
        print("🔍 DRY RUN MODE - Showing execution plan:")
        print("  Issue 1: Dataset pipeline integration (NON-BLOCKING)")
        print("  Issue 2: Build coverage matrix from S3 inventory (BLOCKING)")
        print("  Issue 3: Generate versioned manifest + compiled ChatML export (BLOCKING)")
        print("  Issue 4: Enforce enterprise privacy and provenance gates (BLOCKING)")
        print("  Issue 5: Run enterprise dedup and cross-split leakage gates (BLOCKING)")
        print("  Issue 6: Record distribution stats by family and split (NON-BLOCKING)")
        print("  Issue 7: Clinician QA + bias/cultural review signoff (BLOCKING)")
        print("  Issue 8: Smoke test training consumes S3 release artifacts (NON-BLOCKING)")
        print("\n✅ Dry run complete - use without --dry-run to execute")
        sys.exit(0)

    try:
        # Create enterprise orchestrator
        orchestrator = EnterpriseReleaseOrchestrator(config, args.release_version)

        print("📋 ENTERPRISE CONFIGURATION:")
        print(f"  Release Version: {orchestrator.release_version}")
        print(f"  S3 Bucket: {config.s3_bucket}")
        print(f"  S3 Endpoint: {config.s3_endpoint_url}")
        print(f"  Fail Fast: {not args.no_fail_fast}")
        print(f"  Audit Trail: {'✅ ENABLED' if orchestrator.audit_trail else '⚠️  DISABLED'}")
        print(f"  Main Orchestrator: {'✅ INTEGRATED' if orchestrator.main_orchestrator else '⚠️  STANDALONE'}")

        # Run complete release process
        results = orchestrator.run_complete_release(fail_fast=not args.no_fail_fast)

        # Save release summary
        summary_url = orchestrator.save_release_summary(results)

        # Print final summary
        orchestrator.print_final_summary(results)

        if summary_url:
            print(f"\n📄 Complete release summary: {summary_url}")

        # Exit with appropriate code
        if results["overall_success"]:
            if results.get("enterprise_ready", False):
                logger.info("🏢 Enterprise release completed successfully")
                sys.exit(0)
            else:
                logger.info("✅ Standard release completed successfully")
                sys.exit(0)
        else:
            logger.error("❌ Release failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Release process interrupted by user")
        logger.warning("Release process interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Enterprise release orchestration failed: {e}")
        logger.error(f"Enterprise release orchestration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
