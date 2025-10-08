"""
CI/CD testing and smoke training system for Pixelated Empathy AI project.
Implements fast, low-resource tests for continuous integration.
"""

import unittest
import pytest
import tempfile
import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import time
import numpy as np
from datasets import Dataset

# Import our modules
from .training_manifest import TrainingManifest, create_default_manifest
from .training_runner import TrainingRunner
from .evaluation_system import ComprehensiveEvaluator
from .evaluation_gates import create_model_promotion_manager
from .traceability_system import create_traceability_manager
from .hyperparameter_sweeps import create_hyperparameter_system
from .resource_accounting import create_resource_manager, BudgetLimits


logger = logging.getLogger(__name__)


class SmokeTestDatasetGenerator:
    """Generates minimal datasets for smoke testing"""
    
    @staticmethod
    def generate_small_therapy_dataset(size: int = 10) -> Dict[str, List[Dict[str, str]]]:
        """Generate a small therapy conversation dataset for testing"""
        conversations = []
        
        for i in range(size):
            conversation = {
                "text": f"Therapist: How are you feeling today? Client: I'm feeling {['good', 'okay', 'a bit down', 'stressed', 'anxious'][i%5]} about things."
            }
            conversations.append(conversation)
        
        return {"conversations": conversations}
    
    @staticmethod
    def generate_tiny_dataset() -> Dict[str, List[Dict[str, str]]]:
        """Generate the tiniest possible dataset for ultra-fast testing"""
        return {
            "conversations": [
                {"text": "Hello, how are you?"},
                {"text": "I'm doing well, thank you."}
            ]
        }


class SmokeTrainingTest:
    """Test class for smoke training runs"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = logging.getLogger(__name__)
    
    def run_quick_training_test(self) -> Dict[str, Any]:
        """Run a quick training test with minimal resources"""
        start_time = time.time()
        
        try:
            # Generate tiny dataset
            dataset = SmokeTestDatasetGenerator.generate_tiny_dataset()
            dataset_path = os.path.join(self.temp_dir, "tiny_dataset.json")
            
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f)
            
            # Create minimal manifest
            manifest = create_default_manifest(dataset_path, "test_v1.0")
            # Reduce training parameters for quick test
            manifest.hyperparameters.num_train_epochs = 1
            manifest.hyperparameters.per_device_train_batch_size = 1
            manifest.hyperparameters.gradient_accumulation_steps = 1
            manifest.hyperparameters.logging_steps = 1
            manifest.hyperparameters.save_steps = 5
            manifest.output_dir = os.path.join(self.temp_dir, "test_output")
            manifest.log_dir = os.path.join(self.temp_dir, "test_logs")
            
            # Disable WandB for CI
            manifest.wandb_logging = False
            
            # Create and run training
            runner = TrainingRunner(manifest)
            result = runner.run_training()
            
            duration = time.time() - start_time
            
            return {
                "status": "passed",
                "duration_seconds": duration,
                "model_output_path": manifest.output_dir,
                "log_path": manifest.log_dir,
                "details": "Training completed successfully"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                "status": "failed", 
                "duration_seconds": duration,
                "error": str(e),
                "details": "Training failed"
            }
    
    def run_component_tests(self) -> Dict[str, Any]:
        """Test individual components work together"""
        start_time = time.time()
        
        try:
            # Test manifest creation
            dataset_path = os.path.join(self.temp_dir, "test_dataset.json")
            with open(dataset_path, 'w') as f:
                json.dump(SmokeTestDatasetGenerator.generate_tiny_dataset(), f)
            
            manifest = create_default_manifest(dataset_path, "test_v1.0")
            manifest.model_name = "distilgpt2"  # Use lightweight model
            manifest.wandb_logging = False
            
            # Test traceability system
            traceability_manager = create_traceability_manager()
            
            # Test evaluation system
            evaluator = ComprehensiveEvaluator()
            
            # Test promotion manager
            promotion_manager = create_model_promotion_manager()
            
            # Test hyperparameter system
            hp_system = create_hyperparameter_system()
            
            duration = time.time() - start_time
            
            return {
                "status": "passed",
                "duration_seconds": duration,
                "components_tested": 4,  # traceability, evaluation, promotion, hyperparameters
                "details": "All components imported and instantiated successfully"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                "status": "failed",
                "duration_seconds": duration,
                "error": str(e),
                "details": "Component integration failed"
            }
    
    def run_resource_monitoring_test(self) -> Dict[str, Any]:
        """Test resource monitoring system"""
        start_time = time.time()
        
        try:
            # Test resource accounting
            from .resource_accounting import ResourceManager, BudgetLimits
            
            rm = ResourceManager()
            budget_limits = BudgetLimits(
                max_cost_usd=1.0,  # Very low limit for testing
                max_runtime_hours=0.1,  # 6 minutes max
                max_gpu_memory_gb=1.0,
                max_system_memory_gb=2.0
            )
            
            run_id = "ci_test_run"
            rm.start_run_monitoring(run_id, budget_limits, interval=1.0)  # Check every second for testing
            
            # Simulate short monitoring period
            time.sleep(3)
            
            # Generate report
            report = rm.get_run_report(run_id)
            rm.stop_run_monitoring(run_id)
            
            duration = time.time() - start_time
            
            return {
                "status": "passed",
                "duration_seconds": duration,
                "has_report": report is not None,
                "details": "Resource monitoring working"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                "status": "failed",
                "duration_seconds": duration,
                "error": str(e),
                "details": "Resource monitoring failed"
            }


class IntegrationTestSuite:
    """Comprehensive integration tests"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = logging.getLogger(__name__)
    
    def test_full_pipeline(self) -> Dict[str, Any]:
        """Test the full training pipeline with minimal resources"""
        start_time = time.time()
        
        try:
            # Generate test dataset
            dataset_data = SmokeTestDatasetGenerator.generate_small_therapy_dataset(size=5)
            dataset_path = os.path.join(self.temp_dir, "integration_test_dataset.json")
            
            with open(dataset_path, 'w') as f:
                json.dump(dataset_data, f)
            
            # Create training manifest with minimal resources
            manifest = create_default_manifest(dataset_path, "integration_test_v1.0")
            manifest.model_name = "distilgpt2"  # Lightweight model
            manifest.hyperparameters.num_train_epochs = 1
            manifest.hyperparameters.learning_rate = 1e-4
            manifest.hyperparameters.per_device_train_batch_size = 1
            manifest.hyperparameters.gradient_accumulation_steps = 1
            manifest.hyperparameters.logging_steps = 1
            manifest.hyperparameters.save_steps = 2
            manifest.wandb_logging = False
            manifest.evaluation_enabled = False  # Disable for speed
            manifest.output_dir = os.path.join(self.temp_dir, "pipeline_output")
            
            # Initialize all systems
            traceability_manager = create_traceability_manager()
            evaluator = ComprehensiveEvaluator()
            promotion_manager = create_model_promotion_manager()
            hp_system = create_hyperparameter_system()
            resource_manager = create_resource_manager()
            
            # Setup resource monitoring for the run
            budget_limits = BudgetLimits(
                max_cost_usd=1.0,
                max_runtime_hours=0.1,
                max_gpu_memory_gb=1.0,
                max_system_memory_gb=2.0
            )
            
            run_id = f"integration_test_{int(time.time())}"
            resource_manager.start_run_monitoring(run_id, budget_limits, interval=5.0)
            
            # Create and run training
            runner = TrainingRunner(manifest)
            
            # Run training (this would be the slowest part)
            result = runner.run_training()
            
            # Stop resource monitoring
            resource_manager.stop_run_monitoring(run_id)
            
            # Generate resource report
            resource_report = resource_manager.get_run_report(run_id)
            
            # Test traceability
            if os.path.exists(manifest.output_dir):
                trace_record = traceability_manager.create_traceability_record(
                    manifest=manifest,
                    run_id=run_id,
                    model_artifact_path=manifest.output_dir,
                    model_id="test_model",
                    model_version="1.0.0"
                )
            else:
                trace_record = None
            
            duration = time.time() - start_time
            
            return {
                "status": "passed",
                "duration_seconds": duration,
                "has_traceability_record": trace_record is not None,
                "has_resource_report": resource_report is not None,
                "details": "Full pipeline completed successfully"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Integration test failed: {e}")
            return {
                "status": "failed",
                "duration_seconds": duration,
                "error": str(e),
                "details": "Full pipeline test failed"
            }
    
    def test_hyperparameter_sweep(self) -> Dict[str, Any]:
        """Test hyperparameter sweep functionality"""
        start_time = time.time()
        
        try:
            # Create a minimal sweep config
            from .hyperparameter_sweeps import SweepConfiguration, HyperparameterConfig
            
            sweep_config = SweepConfiguration(
                sweep_name="ci_test_sweep",
                sweep_description="Minimal sweep for CI testing",
                method="random",
                metric="eval_loss",
                goal="minimize",
                parameters=[
                    HyperparameterConfig(
                        name="learning_rate",
                        values=[1e-5, 2e-5],
                        type="categorical"
                    )
                ],
                total_trials=2
            )
            
            # Create system and run minimal sweep
            hp_system = create_hyperparameter_system()
            
            # Create a minimal manifest
            dataset_path = os.path.join(self.temp_dir, "sweep_test_dataset.json")
            with open(dataset_path, 'w') as f:
                json.dump(SmokeTestDatasetGenerator.generate_tiny_dataset(), f)
            
            base_manifest = create_default_manifest(dataset_path, "sweep_test_v1.0")
            base_manifest.model_name = "distilgpt2"
            base_manifest.wandb_logging = False
            base_manifest.hyperparameters.num_train_epochs = 1
            
            # Run a minimal sweep (with dummy functions to avoid full training)
            def dummy_training_fn(config):
                # Create a modified manifest with the config values
                updated_manifest = base_manifest
                if 'learning_rate' in config:
                    updated_manifest.hyperparameters.learning_rate = config['learning_rate']
                
                return {"learning_rate": config.get('learning_rate', 1e-5)}
            
            def dummy_evaluation_fn(result):
                # Return random metrics for testing
                return {"eval_loss": np.random.random()}
            
            # Use the sweeper directly to avoid WandB dependency
            result = hp_system.sweeper.sweep_hyperparameters(
                base_manifest,
                sweep_config,
                dummy_training_fn,
                dummy_evaluation_fn
            )
            
            duration = time.time() - start_time
            
            return {
                "status": "passed",
                "duration_seconds": duration,
                "completed_trials": result.completed_trials,
                "total_trials": len(result.trial_results),
                "details": "Hyperparameter sweep completed"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                "status": "failed",
                "duration_seconds": duration,
                "error": str(e),
                "details": "Hyperparameter sweep test failed"
            }


def run_smoke_tests() -> Dict[str, Any]:
    """Run all smoke tests and return results"""
    logger.info("Starting CI smoke tests...")
    
    start_time = time.time()
    
    smoke_tester = SmokeTrainingTest()
    integration_tester = IntegrationTestSuite()
    
    # Run all tests
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "tests": {},
        "summary": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "total_duration": 0
        }
    }
    
    # Quick training test
    results["tests"]["quick_training"] = smoke_tester.run_quick_training_test()
    
    # Component integration test
    results["tests"]["component_integration"] = smoke_tester.run_component_tests()
    
    # Resource monitoring test
    results["tests"]["resource_monitoring"] = smoke_tester.run_resource_monitoring_test()
    
    # Full pipeline test
    results["tests"]["full_pipeline"] = integration_tester.test_full_pipeline()
    
    # Hyperparameter sweep test
    results["tests"]["hyperparameter_sweep"] = integration_tester.test_hyperparameter_sweep()
    
    # Calculate summary
    total_tests = len(results["tests"])
    passed_tests = sum(1 for test in results["tests"].values() if test["status"] == "passed")
    failed_tests = total_tests - passed_tests
    total_duration = time.time() - start_time
    
    results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "total_duration": total_duration
    }
    
    # Print results
    print(f"\n=== CI Smoke Test Results ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Duration: {total_duration:.2f}s")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
    
    for test_name, test_result in results["tests"].items():
        status = "✅" if test_result["status"] == "passed" else "❌"
        print(f"  {status} {test_name}: {test_result['details']}")
    
    return results


def run_unit_tests():
    """Run unit tests for the training system"""
    # This would typically use pytest or unittest to run specific test files
    # For now, we'll create a basic unit test
    class TestTrainingComponents(unittest.TestCase):
        def test_manifest_creation(self):
            """Test that training manifests can be created"""
            dataset_path = os.path.join(tempfile.mkdtemp(), "test.json")
            with open(dataset_path, 'w') as f:
                json.dump({"conversations": [{"text": "test"}]}, f)
            
            manifest = create_default_manifest(dataset_path, "test_v1.0")
            self.assertIsNotNone(manifest)
            self.assertEqual(manifest.dataset.version, "test_v1.0")
        
        def test_hyperparameter_defaults(self):
            """Test that hyperparameters have reasonable defaults"""
            from .training_manifest import Hyperparameters
            hp = Hyperparameters()
            self.assertGreater(hp.learning_rate, 0)
            self.assertGreater(hp.num_train_epochs, 0)
            self.assertGreater(hp.per_device_train_batch_size, 0)
    
    # Run the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrainingComponents)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def generate_ci_report(results: Dict[str, Any], output_path: str = "./ci_test_report.json"):
    """Generate a CI report in JSON format"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"CI test report saved to {output_path}")
    
    # Generate a text summary
    summary_path = output_path.replace('.json', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CI Test Report Summary\n")
        f.write("=" * 20 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Total Tests: {results['summary']['total_tests']}\n")
        f.write(f"Passed: {results['summary']['passed_tests']}\n")
        f.write(f"Failed: {results['summary']['failed_tests']}\n")
        f.write(f"Duration: {results['summary']['total_duration']:.2f}s\n")
        f.write(f"Success Rate: {(results['summary']['passed_tests']/results['summary']['total_tests'])*100:.1f}%\n\n")
        
        f.write("Test Details:\n")
        for test_name, test_result in results['tests'].items():
            status = "PASS" if test_result['status'] == 'passed' else "FAIL"
            f.write(f"  {test_name}: {status} ({test_result['duration_seconds']:.2f}s)\n")
            f.write(f"    Details: {test_result['details']}\n")
            if test_result['status'] == 'failed':
                f.write(f"    Error: {test_result.get('error', 'Unknown')}\n")
            f.write("\n")
    
    logger.info(f"CI summary report saved to {summary_path}")


def run_ci_tests():
    """Main function to run all CI tests"""
    print("Running CI/CD smoke tests for Pixelated Empathy AI training pipeline...")
    
    # Run smoke tests
    smoke_results = run_smoke_tests()
    
    # Run unit tests
    print("\nRunning unit tests...")
    unit_test_result = run_unit_tests()
    
    # Combine results
    combined_results = smoke_results
    combined_results["unit_tests"] = {
        "total": unit_test_result.testsRun,
        "errors": len(unit_test_result.errors),
        "failures": len(unit_test_result.failures),
        "success": unit_test_result.wasSuccessful()
    }
    
    # Generate report
    generate_ci_report(combined_results)
    
    # Return success status
    total_passed = smoke_results["summary"]["passed_tests"]
    total_tests = smoke_results["summary"]["total_tests"] 
    unit_tests_passed = unit_test_result.testsRun - len(unit_test_result.errors) - len(unit_test_result.failures)
    unit_tests_total = unit_test_result.testsRun
    
    overall_success = (
        total_tests == 0 or total_passed == total_tests  # All smoke tests passed
        and unit_test_result.wasSuccessful()  # All unit tests passed
    )
    
    print(f"\nCI Test Summary:")
    print(f"  Smoke Tests: {total_passed}/{total_tests} passed")
    print(f"  Unit Tests: {unit_tests_passed}/{unit_tests_total} passed")
    print(f"  Overall Status: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    return overall_success


# Pytest-compatible test functions
def test_smoke_training():
    """Pytest-compatible smoke training test"""
    tester = SmokeTrainingTest()
    result = tester.run_quick_training_test()
    assert result["status"] == "passed", f"Smoke training failed: {result.get('error', 'Unknown error')}"


def test_component_integration():
    """Pytest-compatible component integration test"""
    tester = SmokeTrainingTest()
    result = tester.run_component_tests()
    assert result["status"] == "passed", f"Component integration failed: {result.get('error', 'Unknown error')}"


def test_resource_monitoring():
    """Pytest-compatible resource monitoring test"""
    tester = SmokeTrainingTest()
    result = tester.run_resource_monitoring_test()
    assert result["status"] == "passed", f"Resource monitoring failed: {result.get('error', 'Unknown error')}"


def test_full_pipeline_integration():
    """Pytest-compatible full pipeline test"""
    tester = IntegrationTestSuite()
    result = tester.test_full_pipeline()
    assert result["status"] == "passed", f"Full pipeline failed: {result.get('error', 'Unknown error')}"


if __name__ == "__main__":
    success = run_ci_tests()
    exit(0 if success else 1)