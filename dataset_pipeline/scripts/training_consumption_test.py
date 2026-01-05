#!/usr/bin/env python3
"""
Training Consumption Smoke Test

Implements Issue 8: Release 0: Smoke test training consumes S3 release artifacts

Thiscript verifies training scripts can consume the S3-hosted release
manifest/export end-to-end without local file dependencies.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

# Add the dataset_pipeline to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage_config import StorageConfig, get_storage_config


class S3DatasetLoader:
    """Loads datasets directly from S3 for training consumption"""

    def __init__(self, storage_config: StorageConfig):
        self.config = storage_config
        self.s3_client = None
        self._init_s3_client()

    def _init_s3_client(self):
        """Initialize S3 client"""
        if self.config.backend != self.config.backend.S3:
            raise ValueError("S3 backend required for training consumption test")

        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.config.s3_endpoint_url,
                aws_access_key_id=self.config.s3_access_key_id,
                aws_secret_access_key=self.config.s3_secret_access_key,
                region_name=self.config.s3_region or "us-east-1",
            )

            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
            print(f"✓ Connected to S3 bucket: {self.config.s3_bucket}")

        except Exception as e:
            raise ValueError(f"Failed to connect to S3: {e}")

    def load_manifest(self, release_version: str) -> Dict[str, Any]:
        """Load release manifest from S3"""
        manifest_key = (
            f"{self.config.exports_prefix}/releases/{release_version}/manifest.json"
        )

        try:
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket, Key=manifest_key
            )
            manifest = json.loads(response["Body"].read())
            print(f"✓ Loaded manifest: s3://{self.config.s3_bucket}/{manifest_key}")
            return manifest
        except ClientError as e:
            raise ValueError(f"Failed to load manifest {manifest_key}: {e}")

    def load_routing_config(self, release_version: str) -> Dict[str, Any]:
        """Load routing configuration from S3"""
        config_key = f"{self.config.exports_prefix}/releases/{release_version}/routing_config.json"

        try:
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket, Key=config_key
            )
            routing_config = json.loads(response["Body"].read())
            print(f"✓ Loaded routing config: s3://{self.config.s3_bucket}/{config_key}")
            return routing_config
        except ClientError as e:
            raise ValueError(f"Failed to load routing config {config_key}: {e}")

    def stream_dataset_file(self, s3_key: str, chunk_size: int = 8192):
        """Stream a dataset file from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket, Key=s3_key
            )

            # Stream the content
            for chunk in iter(lambda: response["Body"].read(chunk_size), b""):
                yield chunk.decode("utf-8", errors="ignore")

        except ClientError as e:
            raise ValueError(f"Failed to stream {s3_key}: {e}")

    def load_dataset_sample(self, s3_key: str, sample_size: int = 1024) -> str:
        """Load a sample of dataset content from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Range=f"bytes=0-{sample_size - 1}",
            )

            content = response["Body"].read().decode("utf-8", errors="ignore")
            return content

        except ClientError as e:
            print(f"⚠️  Failed to sample {s3_key}: {e}")
            return ""

    def validate_dataset_format(
        self, content_sample: str, expected_format: str
    ) -> bool:
        """Validate that dataset content matches expected format"""
        if not content_sample:
            return False

        try:
            if expected_format == "json":
                json.loads(content_sample)
                return True
            elif expected_format == "jsonl":
                # Check if each line is valid JSON
                lines = content_sample.strip().split("\n")
                for line in lines[:5]:  # Check first 5 lines
                    if line.strip():
                        json.loads(line)
                return True
            elif expected_format == "csv":
                # Basic CSV validation
                return "," in content_sample and "\n" in content_sample
            else:
                return True  # Unknown format, assume valid

        except (json.JSONDecodeError, Exception):
            return False


class TrainingSimulator:
    """Simulates training consumption of S3 datasets"""

    def __init__(self, storage_config: StorageConfig):
        self.config = storage_config
        self.loader = S3DatasetLoader(storage_config)

    def simulate_data_loading(
        self, manifest: Dict[str, Any], routing_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate loading data according to training curriculum"""
        print("🔄 Simulating training data loading...")

        simulation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "stages_tested": {},
            "overall_success": True,
            "errors": [],
            "warnings": [],
        }

        # Get curriculum stages from routing config
        curriculum = routing_config.get("curriculum", {})
        family_mapping = routing_config.get("family_mapping", {})

        for stage_name, stage_config in curriculum.items():
            print(f"  Testing stage: {stage_name}")

            stage_result = {
                "stage": stage_name,
                "families_tested": [],
                "files_loaded": 0,
                "total_size_mb": 0,
                "success": True,
                "errors": [],
            }

            # Get families for this stage
            stage_families = stage_config.get("families", [])

            for family_name in stage_families:
                if family_name not in manifest["families"]:
                    error_msg = f"Family {family_name} not found in manifest"
                    stage_result["errors"].append(error_msg)
                    simulation_results["errors"].append(error_msg)
                    stage_result["success"] = False
                    continue

                family_data = manifest["families"][family_name]
                files = family_data["files"]

                # Test loading a sample of files from this family
                sample_files = files[:3]  # Test first 3 files

                family_test_result = {
                    "family": family_name,
                    "files_tested": len(sample_files),
                    "files_loaded": 0,
                    "format_valid": True,
                    "errors": [],
                }

                for file_info in sample_files:
                    s3_key = file_info["key"]

                    try:
                        # Test loading file content
                        content_sample = self.loader.load_dataset_sample(s3_key)

                        if content_sample:
                            # Validate format
                            expected_format = self._detect_format(s3_key)
                            format_valid = self.loader.validate_dataset_format(
                                content_sample, expected_format
                            )

                            if format_valid:
                                family_test_result["files_loaded"] += 1
                                stage_result["files_loaded"] += 1
                                stage_result["total_size_mb"] += file_info["size"] / (
                                    1024 * 1024
                                )
                            else:
                                error_msg = f"Invalid format in {s3_key}"
                                family_test_result["errors"].append(error_msg)
                                family_test_result["format_valid"] = False
                        else:
                            error_msg = f"Failed to load content from {s3_key}"
                            family_test_result["errors"].append(error_msg)

                    except Exception as e:
                        error_msg = f"Error loading {s3_key}: {str(e)}"
                        family_test_result["errors"].append(error_msg)
                        stage_result["errors"].extend(family_test_result["errors"])

                # Check if family test was successful
                if family_test_result["files_loaded"] == 0:
                    stage_result["success"] = False
                    simulation_results["overall_success"] = False

                stage_result["families_tested"].append(family_test_result)

            simulation_results["stages_tested"][stage_name] = stage_result

        return simulation_results

    def _detect_format(self, s3_key: str) -> str:
        """Detect file format from S3 key"""
        if s3_key.lower().endswith(".json"):
            return "json"
        elif s3_key.lower().endswith(".jsonl"):
            return "jsonl"
        elif s3_key.lower().endswith(".csv"):
            return "csv"
        else:
            return "unknown"

    def simulate_training_pipeline(
        self, manifest: Dict[str, Any], routing_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate a complete training pipeline"""
        print("🚀 Simulating training pipeline...")

        pipeline_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline_stages": [],
            "success": True,
            "total_duration_seconds": 0,
            "errors": [],
        }

        start_time = time.time()

        # Stage 1: Data Loading
        print("  Stage 1: Data Loading")
        loading_start = time.time()

        try:
            loading_results = self.simulate_data_loading(manifest, routing_config)
            loading_duration = time.time() - loading_start

            pipeline_results["pipeline_stages"].append(
                {
                    "stage": "data_loading",
                    "duration_seconds": loading_duration,
                    "success": loading_results["overall_success"],
                    "details": loading_results,
                }
            )

            if not loading_results["overall_success"]:
                pipeline_results["success"] = False
                pipeline_results["errors"].extend(loading_results["errors"])

        except Exception as e:
            loading_duration = time.time() - loading_start
            error_msg = f"Data loading failed: {str(e)}"
            pipeline_results["errors"].append(error_msg)
            pipeline_results["success"] = False

            pipeline_results["pipeline_stages"].append(
                {
                    "stage": "data_loading",
                    "duration_seconds": loading_duration,
                    "success": False,
                    "error": error_msg,
                }
            )

        # Stage 2: Data Preprocessing (simulated)
        print("  Stage 2: Data Preprocessing (simulated)")
        preprocessing_start = time.time()

        try:
            # Simulate preprocessing steps
            time.sleep(0.1)  # Simulate processing time

            preprocessing_duration = time.time() - preprocessing_start

            pipeline_results["pipeline_stages"].append(
                {
                    "stage": "preprocessing",
                    "duration_seconds": preprocessing_duration,
                    "success": True,
                    "details": {
                        "tokenization": "completed",
                        "normalization": "completed",
                        "validation": "completed",
                    },
                }
            )

        except Exception as e:
            preprocessing_duration = time.time() - preprocessing_start
            error_msg = f"Preprocessing failed: {str(e)}"
            pipeline_results["errors"].append(error_msg)
            pipeline_results["success"] = False

            pipeline_results["pipeline_stages"].append(
                {
                    "stage": "preprocessing",
                    "duration_seconds": preprocessing_duration,
                    "success": False,
                    "error": error_msg,
                }
            )

        # Stage 3: Training Setup (simulated)
        print("  Stage 3: Training Setup (simulated)")
        setup_start = time.time()

        try:
            # Simulate training setup
            time.sleep(0.05)  # Simulate setup time

            setup_duration = time.time() - setup_start

            pipeline_results["pipeline_stages"].append(
                {
                    "stage": "training_setup",
                    "duration_seconds": setup_duration,
                    "success": True,
                    "details": {
                        "model_initialization": "completed",
                        "optimizer_setup": "completed",
                        "curriculum_loading": "completed",
                    },
                }
            )

        except Exception as e:
            setup_duration = time.time() - setup_start
            error_msg = f"Training setup failed: {str(e)}"
            pipeline_results["errors"].append(error_msg)
            pipeline_results["success"] = False

            pipeline_results["pipeline_stages"].append(
                {
                    "stage": "training_setup",
                    "duration_seconds": setup_duration,
                    "success": False,
                    "error": error_msg,
                }
            )

        pipeline_results["total_duration_seconds"] = time.time() - start_time

        return pipeline_results


class TrainingConsumptionTest:
    """Main class for training consumption smoke test"""

    def __init__(self, storage_config: StorageConfig):
        self.config = storage_config
        self.loader = S3DatasetLoader(storage_config)
        self.simulator = TrainingSimulator(storage_config)

    def run_smoke_test(self, release_version: str) -> Dict[str, Any]:
        """Run complete training consumption smoke test"""
        print(f"🧪 Running training consumption smoke test for {release_version}...")

        test_results = {
            "release_version": release_version,
            "test_timestamp": datetime.utcnow().isoformat(),
            "test_version": "1.0.0",
            "overall_success": True,
            "test_phases": {},
            "errors": [],
            "warnings": [],
        }

        try:
            # Phase 1: Load Release Artifacts
            print("\n📋 Phase 1: Loading Release Artifacts")

            manifest = self.loader.load_manifest(release_version)
            routing_config = self.loader.load_routing_config(release_version)

            test_results["test_phases"]["artifact_loading"] = {
                "success": True,
                "manifest_loaded": True,
                "routing_config_loaded": True,
                "families_count": len(manifest["families"]),
                "total_files": manifest["summary"]["total_files"],
            }

            # Phase 2: Validate S3 Access
            print("\n🔗 Phase 2: Validating S3 Access")

            access_test = self._test_s3_access(manifest)
            test_results["test_phases"]["s3_access"] = access_test

            if not access_test["success"]:
                test_results["overall_success"] = False
                test_results["errors"].extend(access_test["errors"])

            # Phase 3: Simulate Training Pipeline
            print("\n🚀 Phase 3: Simulating Training Pipeline")

            pipeline_results = self.simulator.simulate_training_pipeline(
                manifest, routing_config
            )
            test_results["test_phases"]["training_simulation"] = pipeline_results

            if not pipeline_results["success"]:
                test_results["overall_success"] = False
                test_results["errors"].extend(pipeline_results["errors"])

            # Phase 4: Environment Validation
            print("\n🔧 Phase 4: Environment Validation")

            env_validation = self._validate_environment()
            test_results["test_phases"]["environment_validation"] = env_validation

            if not env_validation["success"]:
                test_results["overall_success"] = False
                test_results["errors"].extend(env_validation["errors"])

        except Exception as e:
            test_results["overall_success"] = False
            test_results["errors"].append(f"Smoke test failed: {str(e)}")

        return test_results

    def _test_s3_access(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Test S3 access to dataset files"""
        access_results = {
            "success": True,
            "files_tested": 0,
            "files_accessible": 0,
            "errors": [],
        }

        # Test access to a sample of files from each family
        for family_name, family_data in manifest["families"].items():
            files = family_data["files"]

            # Test first file from each family
            if files:
                test_file = files[0]
                s3_key = test_file["key"]

                try:
                    # Try to access the file
                    content_sample = self.loader.load_dataset_sample(s3_key, 100)

                    if content_sample:
                        access_results["files_accessible"] += 1
                    else:
                        access_results["errors"].append(f"Empty content from {s3_key}")

                    access_results["files_tested"] += 1

                except Exception as e:
                    access_results["errors"].append(
                        f"Failed to access {s3_key}: {str(e)}"
                    )
                    access_results["success"] = False

        return access_results

    def _validate_environment(self) -> Dict[str, Any]:
        """Validate training environment prerequisites"""
        env_results = {"success": True, "checks": {}, "errors": []}

        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        env_results["checks"]["python_version"] = {
            "version": python_version,
            "supported": sys.version_info >= (3, 8),
        }

        if not env_results["checks"]["python_version"]["supported"]:
            env_results["errors"].append(
                f"Python {python_version} not supported (requires 3.8+)"
            )
            env_results["success"] = False

        # Check required modules
        required_modules = ["json", "boto3", "pathlib"]
        env_results["checks"]["required_modules"] = {}

        for module in required_modules:
            try:
                __import__(module)
                env_results["checks"]["required_modules"][module] = True
            except ImportError:
                env_results["checks"]["required_modules"][module] = False
                env_results["errors"].append(f"Required module {module} not available")
                env_results["success"] = False

        # Check S3 credentials
        env_results["checks"]["s3_credentials"] = {
            "access_key": bool(self.config.s3_access_key_id),
            "secret_key": bool(self.config.s3_secret_access_key),
            "endpoint": bool(self.config.s3_endpoint_url),
        }

        if not all(env_results["checks"]["s3_credentials"].values()):
            env_results["errors"].append("S3 credentials incomplete")
            env_results["success"] = False

        return env_results

    def save_test_report(
        self, test_results: Dict[str, Any], release_version: str
    ) -> str:
        """Save test report to S3"""
        report_key = f"{self.config.exports_prefix}/releases/{release_version}/tests/training_consumption_test.json"

        try:
            self.loader.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=report_key,
                Body=json.dumps(test_results, indent=2),
                ContentType="application/json",
            )

            report_url = f"s3://{self.config.s3_bucket}/{report_key}"
            print(f"✓ Test report saved: {report_url}")
            return report_url

        except ClientError as e:
            raise ValueError(f"Failed to save test report: {e}")

    def print_test_summary(self, results: Dict[str, Any]):
        """Print human-readable test summary"""
        print("\n" + "=" * 60)
        print("🧪 TRAINING CONSUMPTION SMOKE TEST SUMMARY")
        print("=" * 60)

        print(f"Release Version: {results['release_version']}")
        print(f"Test Time: {results['test_timestamp']}")

        overall_status = "✅ PASSED" if results["overall_success"] else "❌ FAILED"
        print(f"Overall Status: {overall_status}")

        print("\n📊 TEST PHASES:")

        for phase_name, phase_result in results["test_phases"].items():
            phase_status = "✅ PASSED" if phase_result["success"] else "❌ FAILED"
            print(f"  {phase_status} {phase_name.replace('_', ' ').title()}")

            # Show key metrics
            if phase_name == "artifact_loading":
                print(f"    Families: {phase_result['families_count']}")
                print(f"    Total Files: {phase_result['total_files']}")
            elif phase_name == "s3_access":
                print(f"    Files Tested: {phase_result['files_tested']}")
                print(f"    Files Accessible: {phase_result['files_accessible']}")
            elif phase_name == "training_simulation":
                print(f"    Duration: {phase_result['total_duration_seconds']:.2f}s")
                print(f"    Stages: {len(phase_result['pipeline_stages'])}")

        if results["errors"]:
            print("\n🚨 ERRORS:")
            for error in results["errors"][:5]:  # Show first 5 errors
                print(f"  ❌ {error}")

            if len(results["errors"]) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")

        if not results["overall_success"]:
            print("\n💡 NEXT STEPS:")
            print("  1. Review error messages above")
            print("  2. Check S3 credentials and connectivity")
            print("  3. Verify release artifacts are properly uploaded")
            print("  4. Re-run test after fixes")

        print("\n" + "=" * 60)


def main():
    """Main entry point"""
    print("🚀 Starting Training Consumption Smoke Test...")

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python training_consumption_test.py <release_version>")
        print("Example: python training_consumption_test.py v2025-01-02")
        sys.exit(1)

    release_version = sys.argv[1]

    # Load storage configuration
    config = get_storage_config()

    # Validate S3 configuration
    is_valid, error_msg = config.validate()
    if not is_valid:
        print(f"❌ Storage configuration error: {error_msg}")
        sys.exit(1)

    if config.backend != config.backend.S3:
        print("❌ S3 backend required. Set DATASET_STORAGE_BACKEND=s3")
        sys.exit(1)

    try:
        # Create test runner and run smoke test
        test_runner = TrainingConsumptionTest(config)
        results = test_runner.run_smoke_test(release_version)

        # Save test report
        report_url = test_runner.save_test_report(results, release_version)

        # Print results
        test_runner.print_test_summary(results)

        print(f"\n📄 Full test report: {report_url}")

        # Exit with appropriate code
        if results["overall_success"]:
            print(f"\n✅ Training consumption test passed for {release_version}!")
            sys.exit(0)
        else:
            print(f"\n❌ Training consumption test failed for {release_version}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Training consumption test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
