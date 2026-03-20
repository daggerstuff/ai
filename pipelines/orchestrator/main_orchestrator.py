"""
Main Orchestrator for Pixelated Empathy AI Dataset Pipeline

This module orchestrates the complete dataset pipeline:
1. Unified preprocessing pipeline execution
2. Dataset composition and balancing
3. Training manifest creation
4. Final validation and reporting
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.pipelines.orchestrator.data_splitter import DataSplitter
from ai.pipelines.orchestrator.dataset_composition_strategy import (
    run_composition_strategy as run_composition,
)
from ai.pipelines.orchestrator.generation_wrapper import GenerationWrapper
from ai.pipelines.orchestrator.logger import get_logger, setup_pipeline_logging
from ai.pipelines.orchestrator.storage_config import get_dataset_pipeline_output_root
from ai.pipelines.orchestrator.training_manifest import (
    create_safety_aware_manifest,
)

# Import pipeline components
from ai.pipelines.orchestrator.unified_preprocessing_pipeline import (
    run_pipeline as run_unified_pipeline,
)

# Configure logging
logger = get_logger(__name__)


class DatasetPipelineOrchestrator:
    """Main orchestrator for the complete dataset pipeline"""

    def __init__(self):
        # Configure file logging
        self.output_root = get_dataset_pipeline_output_root()
        setup_pipeline_logging(self.output_root / "logs")

        self.pipeline_results = {}
        self.composition_results = {}
        self.manifest = None

        # Initialize generation wrapper
        # Assuming workspace root is 3 levels up from orchestrator
        # (ai/pipelines/orchestrator)
        # /home/vivi/pixelated/ai/pipelines/orchestrator -> /home/vivi/pixelated
        self.workspace_root = Path(__file__).resolve().parents[3]
        self.generator = GenerationWrapper(self.workspace_root)
        self.splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    def _get_latest_artifact(self, search_dir: Path, pattern: str) -> Path | None:
        """Find the latest file matching pattern in search_dir"""
        if not search_dir.exists():
            return None

        if not (candidates := list(search_dir.glob(pattern))):
            return None

        # Return most recently modified file
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def ensure_data_completeness(self) -> dict[str, bool]:
        """
        Ensure all required synthetic and sourced data exists.
        Triggers generation if thresholds are not met.
        """
        logger.info("Verifying data completeness (NeMo, Nightmares, Edge Cases)...")
        results = {
            "nemo_synthetic": self.generator.ensure_nemo_synthetic(target_count=10000),
            "ultra_nightmares": self.generator.ensure_ultra_nightmares(
                count_per_category=5
            ),
            "edge_cases": self.generator.ensure_edge_cases(count=10000),
            "academic_sourcing": self.generator.ensure_academic_sourcing(
                limit_per_query=10
            ),
            "journal_research": self.generator.ensure_journal_research(),
            "books_extraction": self.generator.ensure_books_extraction(),
            "transcripts_extraction": self.generator.ensure_transcripts_extraction(),
        }

        logger.info(f"Data completeness check results: {results}")
        return results

    def run_unified_preprocessing(self, resume: bool = False) -> str:
        """Run the unified preprocessing pipeline"""
        logger.info("Starting unified preprocessing pipeline...")

        if resume:
            output_dir = self.output_root / "final_output"
            if existing := self._get_latest_artifact(
                output_dir, "unified_training_dataset_*.jsonl"
            ):
                logger.info(f"RESUME: Found existing unified dataset at {existing}")
                self.pipeline_results["unified_dataset_path"] = str(existing)
                return str(existing)

        try:
            # Run the unified preprocessing pipeline
            final_dataset_path = run_unified_pipeline()

            self.pipeline_results["unified_dataset_path"] = final_dataset_path
            logger.info(
                "Unified preprocessing completed. "
                f"Dataset saved to: {final_dataset_path}"
            )

            return final_dataset_path
        except Exception as e:
            logger.error(f"Unified preprocessing failed: {e!s}")
            raise

    def run_dataset_composition(
        self, input_dataset_path: str, resume: bool = False
    ) -> tuple[str, dict[str, Any]]:
        """Run the dataset composition and balancing strategy"""
        logger.info("Starting dataset composition and balancing...")

        if resume:
            # Look for likely output location in output_root
            balanced_dir = self.output_root
            existing_balanced = None
            # Helper search
            for p in balanced_dir.rglob("balanced_dataset_*.jsonl"):
                if (
                    not existing_balanced
                    or p.stat().st_mtime > existing_balanced.stat().st_mtime
                ):
                    existing_balanced = p

            if existing_balanced:
                # Try to find report next to it
                report_path = existing_balanced.parent / "composition_report.json"
                if not report_path.exists():
                    # try derived name
                    report_path = (
                        existing_balanced.parent
                        / f"{existing_balanced.stem}_composition_report.json"
                    )

                if report_path.exists():
                    logger.info(
                        "RESUME: Found existing balanced dataset at "
                        f"{existing_balanced}"
                    )
                    try:
                        with open(report_path) as f:
                            report = json.load(f)
                        self.composition_results["balanced_dataset_path"] = str(
                            existing_balanced
                        )
                        self.composition_results["composition_report"] = report
                        return str(existing_balanced), report
                    except Exception as e:
                        logger.warning(
                            f"Could not load existing composition report: {e}"
                        )

        try:
            # Run the composition strategy
            balanced_dataset_path, composition_report = run_composition(
                input_dataset_path
            )

            self.composition_results["balanced_dataset_path"] = balanced_dataset_path
            self.composition_results["composition_report"] = composition_report

            logger.info(
                "Dataset composition completed. "
                f"Balanced dataset: {balanced_dataset_path}"
            )
            return balanced_dataset_path, composition_report
        except Exception as e:
            logger.error(f"Dataset composition failed: {e!s}")
            raise

    def run_data_splitting(
        self, balanced_dataset_path: str, resume: bool = False
    ) -> dict[str, Any]:
        """Split the balanced dataset into train/val/test sets"""
        logger.info("Starting data splitting (70/15/15)...")

        output_dir = Path(balanced_dataset_path).parent

        if resume:
            # Check if all splits exist
            expected_splits = ["train", "val", "test"]
            existing_paths = {}
            all_exist = True
            for set_name in expected_splits:
                path = output_dir / f"{set_name}_dataset.jsonl"
                if path.exists():
                    existing_paths[set_name] = str(path)
                else:
                    all_exist = False
                    break

            if all_exist:
                logger.info(f"RESUME: Found existing data splits at {output_dir}")
                return existing_paths

        try:
            # Load balanced records
            records = []
            with open(balanced_dataset_path) as f:
                for line in f:
                    records.append(json.loads(line.strip()))

            # Perform split
            split_result = self.splitter.split(records)

            # Save split files
            output_dir = Path(balanced_dataset_path).parent
            split_paths = {}
            for set_name in ["train", "val", "test"]:
                subset = getattr(split_result, set_name)
                path = output_dir / f"{set_name}_dataset.jsonl"
                with open(path, "w") as f:
                    for record in subset:
                        f.write(json.dumps(record) + "\n")
                split_paths[set_name] = str(path)

            logger.info(f"Data splitting completed. Paths: {split_paths}")
            return split_paths
        except Exception as e:
            logger.error(f"Data splitting failed: {e!s}")
            raise

    def create_training_manifest(
        self,
        dataset_path: str,
        composition_report_path: str | None = None,
        resume: bool = False,
    ) -> str:
        """Create the training manifest with safety protocols"""
        logger.info("Creating training manifest with safety protocols...")

        output_dir = get_dataset_pipeline_output_root() / "final_output"
        manifest_path = output_dir / "training_manifest.json"

        if resume and manifest_path.exists():
            logger.info(f"RESUME: Found existing training manifest at {manifest_path}")
            return str(manifest_path)

        try:
            # Create safety-aware manifest
            manifest = create_safety_aware_manifest(dataset_path, "1.0")

            # Update with composition report if available
            if composition_report_path:
                manifest.metadata["composition_report_path"] = composition_report_path

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
            output_dir = get_dataset_pipeline_output_root() / "final_output"
            output_dir.mkdir(exist_ok=True)
            manifest_path = output_dir / "training_manifest.json"
            manifest.save_to_file(str(manifest_path))

            self.manifest = manifest
            logger.info(f"Training manifest created: {manifest_path}")

            return str(manifest_path)
        except Exception as e:
            logger.error(f"Training manifest creation failed: {e!s}")
            raise

    def validate_final_dataset(self, dataset_path: str) -> dict[str, Any]:
        """Perform final validation of the integrated dataset"""
        logger.info("Performing final dataset validation...")

        validation_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_path": dataset_path,
            "validation_results": {},
        }

        try:
            # Basic validation - check file exists and is readable
            if not Path(dataset_path).exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

            # Count records
            record_count = 0
            with open(dataset_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            json.loads(line.strip())
                            record_count += 1
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Invalid JSON record at line {record_count + 1}"
                            )

            validation_report["validation_results"]["record_count"] = record_count
            validation_report["validation_results"]["file_readable"] = True

            # Check for required fields in sample records
            required_fields_found = {
                "messages": 0,
                "metadata": 0,
                "_source": 0,
                "_source_type": 0,
            }

            sample_size = min(100, record_count) if record_count > 0 else 0
            sample_checked = 0

            if sample_size > 0:
                with open(dataset_path) as f:
                    for _line_idx, line in enumerate(f):
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
                validation_report["validation_results"][
                    f"{field}_coverage"
                ] = f"{percentage:.1f}%"

            validation_report["validation_results"]["overall_validation"] = "PASSED"
            logger.info(
                f"Final dataset validation completed. Record count: {record_count}"
            )

        except Exception as e:
            validation_report["validation_results"]["overall_validation"] = "FAILED"
            validation_report["validation_results"]["error"] = str(e)
            logger.error(f"Final dataset validation failed: {e!s}")
            raise

        return validation_report

    def generate_final_report(self) -> str:
        """Generate the final comprehensive pipeline report"""
        logger.info("Generating final pipeline report...")

        final_report = {
            "pipeline_execution_report": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": "1.0",
                "components_executed": [
                    "unified_preprocessing_pipeline",
                    "dataset_composition_strategy",
                    "training_manifest_creation",
                    "final_validation",
                ],
            },
            "unified_preprocessing_results": self.pipeline_results,
            "dataset_composition_results": self.composition_results,
            "training_manifest": {
                "created": self.manifest is not None,
                "path": (
                    str(
                        get_dataset_pipeline_output_root()
                        / "final_output"
                        / "training_manifest.json"
                    )
                    if self.manifest
                    else None
                ),
            },
        }

        # Save final report
        output_dir = get_dataset_pipeline_output_root() / "final_output"
        output_dir.mkdir(exist_ok=True)
        report_path = output_dir / "final_pipeline_report.json"

        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2)

        logger.info(f"Final pipeline report generated: {report_path}")
        return str(report_path)

    def execute_complete_pipeline(self, resume: bool = False) -> dict[str, Any]:
        """Execute the complete dataset pipeline from start to finish"""
        logger.info(
            f"Starting complete dataset pipeline execution (resume={resume})..."
        )

        results = {"success": False, "results": {}, "error": None}

        try:
            # Step 0: Ensure data completeness (New)
            generation_results = self.ensure_data_completeness()
            results["results"]["generation_completeness"] = generation_results

            # Step 1: Run unified preprocessing pipeline
            unified_dataset_path = self.run_unified_preprocessing(resume=resume)
            results["results"]["unified_dataset_path"] = unified_dataset_path

            # Step 2: Run dataset composition and balancing
            balanced_dataset_path, composition_report = self.run_dataset_composition(
                unified_dataset_path, resume=resume
            )
            results["results"]["balanced_dataset_path"] = balanced_dataset_path
            results["results"]["composition_report"] = composition_report

            # Step 2.5: Run data splitting
            split_paths = self.run_data_splitting(balanced_dataset_path, resume=resume)
            results["results"]["split_paths"] = split_paths

            # Step 3: Create training manifest
            # Use the train set for manifest
            manifest_dataset_path = split_paths["train"]
            composition_report_path = str(
                Path(balanced_dataset_path).with_name(
                    Path(balanced_dataset_path).stem + "_composition_report.json"
                )
            )
            manifest_path = self.create_training_manifest(
                manifest_dataset_path, composition_report_path, resume=resume
            )
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
            logger.error(f"Complete pipeline execution failed: {e!s}")
            raise

        return results


def _daemonize(output_dir: Path):
    """
    Detach from the controlling terminal and run in the background.
    Redirects stdout/stderr to a general daemon log in the output directory.
    """
    # First fork (detaches from parent)
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)  # Exit first parent
    except OSError as e:
        sys.stderr.write(f"fork #1 failed: {e}\n")
        sys.exit(1)

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Second fork (relinquish session leadership)
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)  # Exit second parent
    except OSError as e:
        sys.stderr.write(f"fork #2 failed: {e}\n")
        sys.exit(1)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "orchestrator_daemon.log"

    si = open(os.devnull, "r")
    so = open(log_path, "a+")
    se = open(log_path, "a+")

    os.dup2(si.fileno(), sys.stdin.fileno())
    os.dup2(so.fileno(), sys.stdout.fileno())
    os.dup2(se.fileno(), sys.stderr.fileno())

    # Write a marker to the log
    sys.stdout.write(f"\n--- Daemon started at {datetime.now(timezone.utc)} ---\n")
    sys.stdout.flush()


def main():
    """Main entry point for the dataset pipeline orchestrator"""
    parser = argparse.ArgumentParser(
        description="Pixelated Empathy AI Dataset Pipeline Orchestrator"
    )
    # Default is now resume=True. We provide a restart flag to disable it.
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Force a fresh start (ignore existing artifacts)",
    )
    # Legacy flag for compatibility, does nothing as it's default
    parser.add_argument(
        "--resume", action="store_true", help="Resume (now enabled by default)"
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in the foreground (do not daemonize)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without running heavy tasks",
    )
    args = parser.parse_args()

    # Determine execution mode
    run_resume = not args.restart
    run_background = not args.foreground and not args.dry_run

    if run_background:
        output_root = get_dataset_pipeline_output_root()
        print("🚀 Launching Orchestrator in BACKGROUND mode.")
        print(f"   Logs will be available at: {output_root}/logs")
        print(f"   Daemon output redirected to: {output_root}/orchestrator_daemon.log")
        _daemonize(output_root)

    # --- Daemon Context Starts Here (if backgrounded) ---

    orchestrator = DatasetPipelineOrchestrator()

    try:
        logger.info("Pixelated Empathy AI Dataset Pipeline Orchestrator")
        logger.info("=" * 50)
        if args.dry_run:
            logger.info("DRY RUN MODE ENABLED - No changes will be committed.")
        logger.info(
            f"Execution Mode: {'Background' if run_background else 'Foreground'}"
        )
        logger.info(f"Resume Enabled: {run_resume}")

        # Execute the complete pipeline
        if args.dry_run:
            logger.info("Performing dry run of dataset pipeline components...")
            # Simulate steps
            logger.info("Dry-run: Checking data completeness...")
            logger.info("Dry-run: Running unified preprocessing...")
            logger.info("Dry-run: Validating acoustic deduplication handles...")
            logger.info("Dry-run: Testing transcript quality Pass 2/3 availability...")
            results = {
                "success": True,
                "results": {"dry_run": True},
                "composition_report": {"final_dataset_stats": {"total_records": 0}},
            }
        else:
            results = orchestrator.execute_complete_pipeline(resume=run_resume)

        if results["success"]:
            logger.info("\n🎉 Pipeline Execution Successful!")
            logger.info("=" * 50)
            if not args.dry_run:
                logger.info(
                    f"📊 Unified Dataset: {results['results']['unified_dataset_path']}"
                )
                logger.info(
                    f"⚖️  Balanced Dataset: "
                    f"{results['results']['balanced_dataset_path']}"
                )
                logger.info(
                    f"📋 Training Manifest: "
                    f"{results['results']['training_manifest_path']}"
                )
                logger.info(
                    f"📄 Final Report: {results['results']['final_report_path']}"
                )

                # Print composition summary
                composition_report = results["results"]["composition_report"]
                logger.info("\n📈 Dataset Composition Summary:")
                logger.info(
                    f"   Total Records: "
                    f"{composition_report['final_dataset_stats']['total_records']}"
                )
                if "quality_scores" in composition_report["final_dataset_stats"]:
                    avg_quality = composition_report["final_dataset_stats"][
                        "quality_scores"
                    ]["avg"]
                    logger.info(f"   Average Quality Score: {avg_quality:.3f}")

            logger.info("\n✅ Ready for Lightning.ai H100 training deployment!")
        else:
            logger.error(f"\n❌ Pipeline Execution Failed: {results['error']}")
            return 1

    except Exception as e:
        logger.error(f"Pipeline orchestrator failed: {e!s}")
        logger.critical(f"\n💥 Critical Error: {e!s}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
