"""
Main Orchestrator for Pixelated Empathy AI Dataset Pipeline

This module orchestrates the complete dataset pipeline:
1. Unified preprocessing pipeline execution
2. Dataset composition and balancing
3. Training manifest creation
4. Final validation and reporting
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

# Import pipeline components
from ai.dataset_pipeline.unified_preprocessing_pipeline import (
    create_default_pipeline,
    run_pipeline as run_unified_pipeline
)
from ai.dataset_pipeline.dataset_composition_strategy import (
    create_default_composer,
    run_composition_strategy as run_composition
)
from ai.dataset_pipeline.training_manifest import (
    create_safety_aware_manifest,
    TrainingManifest
)
from ai.dataset_pipeline.youtube_rag_system import YouTubeRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPipelineOrchestrator:
    """Main orchestrator for the complete dataset pipeline"""

    def __init__(self):
        self.pipeline_results = {}
        self.composition_results = {}
        self.manifest = None

    def run_unified_preprocessing(self) -> str:
        """Run the unified preprocessing pipeline"""
        logger.info("Starting unified preprocessing pipeline...")

        try:
            # Run the unified preprocessing pipeline
            final_dataset_path = run_unified_pipeline()

            self.pipeline_results['unified_dataset_path'] = final_dataset_path
            logger.info(f"Unified preprocessing completed. Dataset saved to: {final_dataset_path}")

            return final_dataset_path
        except Exception as e:
            logger.error(f"Unified preprocessing failed: {str(e)}")
            raise

    def run_dataset_composition(self, input_dataset_path: str) -> Tuple[str, Dict[str, Any]]:
        """Run the dataset composition and balancing strategy"""
        logger.info("Starting dataset composition and balancing...")

        try:
            # Run the composition strategy
            balanced_dataset_path, composition_report = run_composition(input_dataset_path)

            self.composition_results['balanced_dataset_path'] = balanced_dataset_path
            self.composition_results['composition_report'] = composition_report

            logger.info(f"Dataset composition completed. Balanced dataset: {balanced_dataset_path}")
            return balanced_dataset_path, composition_report
        except Exception as e:
            logger.error(f"Dataset composition failed: {str(e)}")
            raise

    def create_training_manifest(self, dataset_path: str, composition_report_path: str = None) -> str:
        """Create the training manifest with safety protocols"""
        logger.info("Creating training manifest with safety protocols...")

        try:
            # Create safety-aware manifest
            manifest = create_safety_aware_manifest(dataset_path, "1.0")

            # Update with composition report if available
            if composition_report_path:
                manifest.metadata['composition_report_path'] = composition_report_path

            # Set appropriate compute target for H100 training
            manifest.compute_target = manifest.ComputeTarget.GPU_MULTI
            manifest.resources.min_gpu_memory_gb = 80.0  # H100 specs
            manifest.resources.cloud_provider = "lightning_ai"
            manifest.resources.instance_type = "h100"

            # Update hyperparameters for large model training
            manifest.hyperparameters.per_device_train_batch_size = 2
            manifest.hyperparameters.gradient_accumulation_steps = 32
            manifest.hyperparameters.bf16 = True
            manifest.hyperparameters.gradient_checkpointing = True

            # Save manifest
            output_dir = Path("ai/dataset_pipeline/final_output")
            output_dir.mkdir(exist_ok=True)
            manifest_path = output_dir / "training_manifest.json"
            manifest.save_to_file(str(manifest_path))

            self.manifest = manifest
            logger.info(f"Training manifest created: {manifest_path}")

            return str(manifest_path)
        except Exception as e:
            logger.error(f"Training manifest creation failed: {str(e)}")
            raise

    def validate_final_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Perform final validation of the integrated dataset"""
        logger.info("Performing final dataset validation...")

        validation_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_path": dataset_path,
            "validation_results": {}
        }

        try:
            # Basic validation - check file exists and is readable
            if not Path(dataset_path).exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

            # Count records
            record_count = 0
            with open(dataset_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            json.loads(line.strip())
                            record_count += 1
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON record at line {record_count + 1}")

            validation_report["validation_results"]["record_count"] = record_count
            validation_report["validation_results"]["file_readable"] = True

            # Check for required fields in sample records
            required_fields_found = {
                "messages": 0,
                "metadata": 0,
                "_source": 0,
                "_source_type": 0
            }

            sample_size = min(100, record_count) if record_count > 0 else 0
            sample_checked = 0

            if sample_size > 0:
                with open(dataset_path, 'r') as f:
                    for i, line in enumerate(f):
                        if sample_checked >= sample_size:
                            break
                        if line.strip():
                            try:
                                record = json.loads(line.strip())
                                sample_checked += 1

                                for field in required_fields_found:
                                    if field in record:
                                        required_fields_found[field] += 1
                            except json.JSONDecodeError:
                                continue

            # Calculate percentages
            for field, count in required_fields_found.items():
                percentage = (count / sample_size * 100) if sample_size > 0 else 0
                validation_report["validation_results"][f"{field}_coverage"] = f"{percentage:.1f}%"

            validation_report["validation_results"]["overall_validation"] = "PASSED"
            logger.info(f"Final dataset validation completed. Record count: {record_count}")

        except Exception as e:
            validation_report["validation_results"]["overall_validation"] = "FAILED"
            validation_report["validation_results"]["error"] = str(e)
            logger.error(f"Final dataset validation failed: {str(e)}")
            raise

        return validation_report

    def generate_final_report(self) -> str:
        """Generate the final comprehensive pipeline report"""
        logger.info("Generating final pipeline report...")

        final_report = {
            "pipeline_execution_report": {
                "timestamp": datetime.utcnow().isoformat(),
                "pipeline_version": "1.0",
                "components_executed": [
                    "unified_preprocessing_pipeline",
                    "dataset_composition_strategy",
                    "training_manifest_creation",
                    "final_validation"
                ]
            },
            "unified_preprocessing_results": self.pipeline_results,
            "dataset_composition_results": self.composition_results,
            "training_manifest": {
                "created": self.manifest is not None,
                "path": str(Path("ai/dataset_pipeline/final_output/training_manifest.json")) if self.manifest else None
            }
        }

        # Save final report
        output_dir = Path("ai/dataset_pipeline/final_output")
        output_dir.mkdir(exist_ok=True)
        report_path = output_dir / "final_pipeline_report.json"

        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)

        logger.info(f"Final pipeline report generated: {report_path}")
        return str(report_path)

    def execute_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete dataset pipeline from start to finish"""
        logger.info("Starting complete dataset pipeline execution...")

        results = {
            "success": False,
            "results": {},
            "error": None
        }

        try:
            # Step 1: Run unified preprocessing pipeline
            unified_dataset_path = self.run_unified_preprocessing()
            results["results"]["unified_dataset_path"] = unified_dataset_path

            # Step 2: Run dataset composition and balancing
            balanced_dataset_path, composition_report = self.run_dataset_composition(unified_dataset_path)
            results["results"]["balanced_dataset_path"] = balanced_dataset_path
            results["results"]["composition_report"] = composition_report

            # Step 3: Create training manifest
            composition_report_path = str(Path(balanced_dataset_path).with_name(
                Path(balanced_dataset_path).stem + "_composition_report.json"))
            manifest_path = self.create_training_manifest(balanced_dataset_path, composition_report_path)
            results["results"]["training_manifest_path"] = manifest_path

            # Step 4: Final validation
            validation_report = self.validate_final_dataset(balanced_dataset_path)
            results["results"]["validation_report"] = validation_report

            # Step 5: Generate final report
            final_report_path = self.generate_final_report()
            results["results"]["final_report_path"] = final_report_path

            results["success"] = True
            logger.info("Complete dataset pipeline execution successful!")

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Complete pipeline execution failed: {str(e)}")
            raise

        return results

def main():
    """Main entry point for the dataset pipeline orchestrator"""
    orchestrator = DatasetPipelineOrchestrator()

    try:
        logger.info("Pixelated Empathy AI Dataset Pipeline Orchestrator")
        logger.info("=" * 50)

        # Execute the complete pipeline
        results = orchestrator.execute_complete_pipeline()

        if results["success"]:
            print("\n🎉 Pipeline Execution Successful!")
            print("=" * 50)
            print(f"📊 Unified Dataset: {results['results']['unified_dataset_path']}")
            print(f"⚖️  Balanced Dataset: {results['results']['balanced_dataset_path']}")
            print(f"📋 Training Manifest: {results['results']['training_manifest_path']}")
            print(f"📄 Final Report: {results['results']['final_report_path']}")

            # Print composition summary
            composition_report = results['results']['composition_report']
            print(f"\n📈 Dataset Composition Summary:")
            print(f"   Total Records: {composition_report['final_dataset_stats']['total_records']}")
            if 'quality_scores' in composition_report['final_dataset_stats']:
                avg_quality = composition_report['final_dataset_stats']['quality_scores']['avg']
                print(f"   Average Quality Score: {avg_quality:.3f}")

            print(f"\n✅ Ready for Lightning.ai H100 training deployment!")
        else:
            print(f"\n❌ Pipeline Execution Failed: {results['error']}")
            return 1

    except Exception as e:
        logger.error(f"Pipeline orchestrator failed: {str(e)}")
        print(f"\n💥 Critical Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())