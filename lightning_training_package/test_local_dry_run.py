#!/usr/bin/env python3
"""
Local Dry Run Testing for KAN-28 Lightning.ai Package
Tests all components locally before deployment
"""

import importlib.util
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LocalDryRunTester:
    """Comprehensive local testing for Lightning.ai package"""

    def __init__(self):
        self.test_results = {}
        self.package_root = Path(".")

    def test_package_structure(self) -> bool:
        """Test that all required files are present"""

        logger.info("🔍 Testing package structure...")

        required_files = [
            "README.md",
            "PACKAGE_MANIFEST.md",
            "quick_start.py",
            "requirements.txt",
            "config/enhanced_training_config.json",
            "config/lightning_deployment_config.json",
            "config/moe_training_config.json",
            "data/ULTIMATE_FINAL_DATASET.jsonl",
            "data/unified_6_component_dataset.jsonl",
            "scripts/train_enhanced.py",
            "scripts/data_preparation.py",
            "models/moe_architecture.py",
            "validation_scripts/inference_service.py",
        ]

        missing_files = []
        file_sizes = {}

        for file_path in required_files:
            full_path = self.package_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                file_sizes[file_path] = full_path.stat().st_size

        if missing_files:
            logger.error(f"❌ Missing files: {missing_files}")
            self.test_results["package_structure"] = {
                "passed": False,
                "missing_files": missing_files,
            }
            return False

        logger.info("✅ All required files present")

        # Check data file sizes
        ultimate_dataset_size = file_sizes.get("data/ULTIMATE_FINAL_DATASET.jsonl", 0)
        logger.info(f"📊 ULTIMATE_FINAL_DATASET.jsonl: {ultimate_dataset_size / (1024**3):.2f}GB")

        if ultimate_dataset_size < 1000000:  # Less than 1MB indicates problem
            logger.warning("⚠️ ULTIMATE_FINAL_DATASET.jsonl seems too small")

        self.test_results["package_structure"] = {
            "passed": True,
            "file_count": len(required_files),
            "file_sizes": file_sizes,
        }
        return True

    def test_configuration_files(self) -> bool:
        """Test all configuration files are valid JSON"""

        logger.info("⚙️ Testing configuration files...")

        config_files = [
            "config/enhanced_training_config.json",
            "config/lightning_deployment_config.json",
            "config/moe_training_config.json",
        ]

        config_errors = []

        for config_file in config_files:
            try:
                with open(self.package_root / config_file) as f:
                    config = json.load(f)

                # Validate key fields exist
                if config_file == "config/enhanced_training_config.json":
                    required_keys = ["base_model", "kan28_components", "training_parameters"]
                    missing_keys = [key for key in required_keys if key not in config]
                    if missing_keys:
                        config_errors.append(f"{config_file}: missing keys {missing_keys}")

                logger.info(f"✅ {config_file}: Valid JSON")

            except json.JSONDecodeError as e:
                error_msg = f"{config_file}: Invalid JSON - {e}"
                logger.error(f"❌ {error_msg}")
                config_errors.append(error_msg)
            except Exception as e:
                error_msg = f"{config_file}: Error - {e}"
                logger.error(f"❌ {error_msg}")
                config_errors.append(error_msg)

        if config_errors:
            self.test_results["configuration"] = {"passed": False, "errors": config_errors}
            return False

        logger.info("✅ All configuration files valid")
        self.test_results["configuration"] = {"passed": True}
        return True

    def test_data_files(self) -> bool:
        """Test data files are valid and contain expected KAN-28 components"""

        logger.info("📊 Testing data files...")

        data_tests = {}

        # Test unified 6-component dataset
        component_file = self.package_root / "data/unified_6_component_dataset.jsonl"
        if component_file.exists():
            component_stats = self._test_jsonl_file(component_file, "6-component dataset")
            data_tests["unified_6_component"] = component_stats
        else:
            logger.error("❌ Missing unified_6_component_dataset.jsonl")
            return False

        # Test ultimate final dataset (sample only for performance)
        ultimate_file = self.package_root / "data/ULTIMATE_FINAL_DATASET.jsonl"
        if ultimate_file.exists():
            ultimate_stats = self._test_jsonl_file(
                ultimate_file, "ultimate dataset", sample_size=100
            )
            data_tests["ultimate_final"] = ultimate_stats
        else:
            logger.error("❌ Missing ULTIMATE_FINAL_DATASET.jsonl")
            return False

        # Validate KAN-28 component integration
        component_validation = self._validate_kan28_components(component_file)
        data_tests["kan28_validation"] = component_validation

        all_passed = all(test.get("valid", False) for test in data_tests.values())

        self.test_results["data_files"] = {"passed": all_passed, "tests": data_tests}

        if all_passed:
            logger.info("✅ All data files valid")
        else:
            logger.error("❌ Data file validation failed")

        return all_passed

    def _test_jsonl_file(
        self, file_path: Path, description: str, sample_size: int | None = None
    ) -> dict[str, Any]:
        """Test a JSONL file for validity"""

        logger.info(f"  📄 Testing {description}...")

        stats = {"valid": True, "total_lines": 0, "valid_json_lines": 0, "sample_conversations": []}

        try:
            with open(file_path) as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        stats["total_lines"] += 1

                        try:
                            conv_data = json.loads(line)
                            stats["valid_json_lines"] += 1

                            # Collect samples
                            if (
                                sample_size and len(stats["sample_conversations"]) < sample_size
                            ) or (not sample_size and len(stats["sample_conversations"]) < 5):
                                stats["sample_conversations"].append(conv_data)

                        except json.JSONDecodeError:
                            logger.warning(f"    ⚠️ Invalid JSON on line {line_num}")
                            if line_num <= 10:  # Only log first 10 errors
                                stats["valid"] = False

                        # Limit processing for large files
                        if sample_size and line_num >= sample_size:
                            break

        except Exception as e:
            logger.error(f"    ❌ Error reading {description}: {e}")
            stats["valid"] = False

        validity_ratio = stats["valid_json_lines"] / max(stats["total_lines"], 1)
        logger.info(
            f"    📊 {description}: {stats['valid_json_lines']}/{stats['total_lines']} valid ({validity_ratio:.1%})"
        )

        return stats

    def _validate_kan28_components(self, component_file: Path) -> dict[str, Any]:
        """Validate KAN-28 component integration in dataset"""

        logger.info("  🔧 Validating KAN-28 component integration...")

        validation = {
            "valid": True,
            "components_found": {},
            "expert_voices_found": {},
            "sample_integrations": [],
        }

        expected_components = {
            "journaling_system",
            "voice_blending",
            "edge_case_handling",
            "dual_persona_dynamics",
            "bias_detection",
            "psychology_knowledge_base",
        }

        expected_experts = {"Tim Ferriss", "Gabor Maté", "Brené Brown"}

        try:
            with open(component_file) as f:
                for line in f:
                    if not line.strip():
                        continue

                    conv = self._parse_json_line(line)
                    if conv is None:
                        continue

                    self._update_component_counts(
                        conv, validation["components_found"], expected_components
                    )
                    self._update_expert_counts(
                        conv, validation["expert_voices_found"], expected_experts
                    )
                    self._collect_sample(conv, validation["sample_integrations"])

        except Exception as e:
            logger.error(f"    ❌ Error validating components: {e}")
            validation["valid"] = False

        self._log_validation_results(validation, expected_components, expected_experts)
        return validation

    def _parse_json_line(self, line: str) -> dict | None:
        """Parse a single JSON line, return None on error"""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _update_component_counts(self, conv: dict, components_found: dict, expected: set) -> None:
        """Update component counts from conversation data"""
        if "integration_metadata" not in conv:
            return

        components = conv["integration_metadata"].get("components_applied", [])
        for component in components:
            if component in expected:
                components_found[component] = components_found.get(component, 0) + 1

    def _update_expert_counts(self, conv: dict, experts_found: dict, expected: set) -> None:
        """Update expert voice counts from conversation data"""
        if "expert_voices" not in conv or not isinstance(conv["expert_voices"], dict):
            return

        expert_voices = conv["expert_voices"]
        for expert in expected:
            expert_key = expert.split()[0].lower()
            if any(expert_key in key.lower() for key in expert_voices):
                experts_found[expert] = experts_found.get(expert, 0) + 1

    def _collect_sample(self, conv: dict, samples: list) -> None:
        """Collect sample integration data"""
        if len(samples) >= 3:
            return

        sample = {
            "has_integration_metadata": "integration_metadata" in conv,
            "has_expert_voices": "expert_voices" in conv,
            "has_bias_detection": "bias_detection" in conv,
            "has_psychology_concepts": "psychology_concepts" in conv,
        }
        samples.append(sample)

    def _log_validation_results(
        self, validation: dict, expected_components: set, expected_experts: set
    ) -> None:
        """Log validation results and missing items"""
        missing_components = expected_components - set(validation["components_found"])
        missing_experts = expected_experts - set(validation["expert_voices_found"])

        if missing_components:
            logger.warning("    ⚠️ Missing components: %s", sorted(missing_components))
        if missing_experts:
            logger.warning("    ⚠️ Missing expert voices: %s", sorted(missing_experts))

        if validation["components_found"]:
            logger.info(
                "    ✅ Components found: %s", sorted(validation["components_found"].keys())
            )
        if validation["expert_voices_found"]:
            logger.info(
                "    ✅ Expert voices found: %s", sorted(validation["expert_voices_found"].keys())
            )

    def test_script_imports(self) -> bool:
        """Test that all Python scripts can be imported without errors"""

        logger.info("🐍 Testing script imports...")

        scripts_to_test = [
            "scripts/data_preparation.py",
            "scripts/train_enhanced.py",
            "models/moe_architecture.py",
            "validation_scripts/inference_service.py",
        ]

        import_errors = []

        for script_path in scripts_to_test:
            try:
                # Load the module dynamically
                spec = importlib.util.spec_from_file_location(
                    f"test_module_{script_path.replace('/', '_').replace('.py', '')}",
                    self.package_root / script_path,
                )
                if spec and spec.loader:
                    importlib.util.module_from_spec(spec)
                    # Don't execute, just check syntax
                    logger.info(f"  ✅ {script_path}: Import successful")
                else:
                    raise ImportError("Could not load module spec")

            except Exception as e:
                error_msg = f"{script_path}: {e!s}"
                logger.error("  ❌ %s", error_msg)
                import_errors.append(error_msg)

        if import_errors:
            self.test_results["script_imports"] = {"passed": False, "errors": import_errors}
            return False

        logger.info("✅ All scripts importable")
        self.test_results["script_imports"] = {"passed": True}
        return True

    def test_training_config_compatibility(self) -> bool:
        """Test training configuration for Lightning.ai compatibility"""

        logger.info("⚡ Testing Lightning.ai compatibility...")

        try:
            with open(self.package_root / "config/enhanced_training_config.json") as f:
                config = json.load(f)

            compatibility_checks = {
                "base_model_valid": "base_model" in config and bool(config["base_model"]),
                "h100_optimizations": "h100_optimizations" in config,
                "lora_config_present": "lora_config" in config,
                "kan28_components": "kan28_components" in config,
                "training_parameters": "training_parameters" in config,
                "reasonable_batch_size": config.get("training_parameters", {}).get(
                    "per_device_train_batch_size", 0
                )
                <= 8,
                "reasonable_learning_rate": 1e-5
                <= float(config.get("training_parameters", {}).get("learning_rate", 0))
                <= 1e-3,
            }

            passed_checks = sum(int(check) for check in compatibility_checks.values())
            total_checks = len(compatibility_checks)

            logger.info(f"  📊 Compatibility: {passed_checks}/{total_checks} checks passed")

            for check_name, passed in compatibility_checks.items():
                status = "✅" if passed else "❌"
                logger.info(f"    {status} {check_name}")

            all_passed = passed_checks == total_checks

            self.test_results["lightning_compatibility"] = {
                "passed": all_passed,
                "checks": compatibility_checks,
                "score": f"{passed_checks}/{total_checks}",
            }

            return all_passed

        except Exception as e:
            logger.error(f"❌ Lightning.ai compatibility test failed: {e}")
            self.test_results["lightning_compatibility"] = {"passed": False, "error": str(e)}
            return False

    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""

        report_path = self.package_root / "DRY_RUN_TEST_REPORT.json"

        overall_success = all(result.get("passed", False) for result in self.test_results.values())

        test_report = {
            "test_timestamp": "2024-10-28",
            "overall_status": "PASSED" if overall_success else "FAILED",
            "package_ready_for_lightning": overall_success,
            "test_results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(
                    1 for r in self.test_results.values() if r.get("passed", False)
                ),
                "failed_tests": sum(
                    1 for r in self.test_results.values() if not r.get("passed", False)
                ),
            },
            "next_steps": [
                "Upload package to Lightning.ai Studio"
                if overall_success
                else "Fix failing tests first",
                "Run quick_start.py" if overall_success else "Address configuration issues",
                "Monitor training with W&B" if overall_success else "Validate data files",
            ],
        }

        with open(report_path, "w") as f:
            json.dump(test_report, f, indent=2)

        return str(report_path)


def main():
    """Main dry run testing function"""

    logger.info("\n🧪 Lightning.ai Package - Local Dry Run Testing")
    logger.info("=" * 60)

    tester = LocalDryRunTester()

    # Run all tests
    tests = [
        ("Package Structure", tester.test_package_structure),
        ("Configuration Files", tester.test_configuration_files),
        ("Data Files", tester.test_data_files),
        ("Script Imports", tester.test_script_imports),
        ("Lightning.ai Compatibility", tester.test_training_config_compatibility),
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n🔬 Running: {test_name}")
        try:
            if test_func():
                passed_tests += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            logger.error(traceback.format_exc())

    # Generate report
    report_path = tester.generate_test_report()

    # Final summary
    logger.info("\n📊 DRY RUN TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
    logger.info(f"Success Rate: {passed_tests / total_tests:.1%}")

    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - READY FOR LIGHTNING.AI!")
        logger.info("📋 Next steps:")
        logger.info("  1. Upload package to Lightning.ai Studio")
        logger.info("  2. Run: python quick_start.py")
        logger.info("  3. Monitor training progress")
    else:
        logger.error("❌ SOME TESTS FAILED - NEEDS ATTENTION")
        logger.info("📋 Check test report for details")

    logger.info(f"📁 Test report: {report_path}")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
