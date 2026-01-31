import logging
import subprocess
from pathlib import Path

# Import the direct class-based generators
try:
    from ai.training.ready_packages.scripts.generate_ultra_nightmares import UltraNightmareGenerator
except ImportError:
    UltraNightmareGenerator = None

try:
    from ai.pipelines.design.service import NeMoDataDesignerService
except ImportError:
    NeMoDataDesignerService = None

logger = logging.getLogger(__name__)


class GenerationWrapper:
    """
    Wrapper to safely invoke various data generation scripts and services
    from the central orchestrator.
    Enforces the 'Single Source of Truth' by wrapping both shell-based and
    import-based generation triggers.
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.scripts_dir = self.workspace_root / "ai/training/ready_packages/scripts"

    def ensure_ultra_nightmares(self, count_per_category: int = 5) -> bool:
        """
        Ensure 'Ultra Nightmare' scenarios are generated (Stage 3).
        Uses direct python import if available.
        """
        logger.info(
            f"Ensuring Ultra Nightmare scenarios (count_per_category={count_per_category})..."
        )

        if not UltraNightmareGenerator:
            logger.error("UltraNightmareGenerator could not be imported. Check paths.")
            return False

        try:
            generator = UltraNightmareGenerator()
            # This generates directly to ai/training_ready/data/generated/ultra_nightmares/
            generator.generate_all(count_per_category=count_per_category)
            return True
        except Exception as e:
            logger.error(f"Failed to generate Ultra Nightmares: {e}")
            return False

    def ensure_nemo_synthetic(self, target_count: int = 10000) -> bool:
        """
        Ensure NeMo synthetic data generation (Stage 1/2).
        Uses NeMoDataDesignerService.
        """
        logger.info(f"Ensuring NeMo synthetic data (target={target_count})...")

        if not NeMoDataDesignerService:
            # Check if we can mock it or just fail gracefully (it might be optional if env not set)
            logger.error("NeMoDataDesignerService could not be imported.")
            return False

        output_dir = self.workspace_root / "ai/training_ready/data/generated/nemo_synthetic"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "nemo_synthetic_dataset.jsonl"

        if output_file.exists():
            # Check count? For now simple existence check or we could assume it's good.
            # Ideally we check line count.
            logger.info(f"NeMo dataset exists at {output_file}. Skipping generation.")
            return True

        try:
            service = NeMoDataDesignerService()
            logger.info("Triggering NeMo Data Designer service...")
            result = service.generate_therapeutic_dataset(
                num_samples=target_count,
                include_demographics=True,
                include_symptoms=True,
                include_treatments=True,
                include_outcomes=True,
            )

            # Save to disk
            import json

            data = result.get("data", [])
            with open(output_file, "w") as f:
                if isinstance(data, list):
                    for record in data:
                        f.write(json.dumps(record) + "\n")
                else:
                    logger.error(f"Unexpected data format from NeMo: {type(data)}")
                    return False

            logger.info(f"Saved {len(data)} NeMo samples to {output_file}")
            return True
        except Exception as e:
            logger.error(f"NeMo generation failed: {e}")
            return False

    def ensure_edge_cases(self, count: int = 10000) -> bool:
        """
        Ensure general Edge Case synthetic data.
        Calls the script via subprocess as it assumes CLI usage.
        """
        logger.info(f"Ensuring Edge Case synthetic data (count={count})...")
        script_path = self.scripts_dir / "generate_edge_case_synthetic_dataset.py"

        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False

        cmd = [
            "uv",
            "run",
            "python",
            str(script_path),
            "--count",
            str(count),
            "--categories",
            "all",
        ]

        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.check_call(cmd, cwd=self.workspace_root)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Edge case generation failed check_call: {e}")
            return False
        except Exception as e:
            logger.error(f"Edge case generation failed: {e}")
            return False

    def ensure_long_running_extraction(self) -> bool:
        """
        Extract long-running therapy sessions.
        Run logic from run_phase1_production.sh -> extract_long_running_therapy.py.
        """
        logger.info("Ensuring Long Running Therapy extraction (Stage 5)...")
        script_path = self.scripts_dir / "extract_long_running_therapy.py"
        output_file = (
            self.workspace_root / "ai/training_ready/data/generated/long_running_therapy.jsonl"
        )

        # Check if already exists (skip if substantial size)
        if output_file.exists() and output_file.stat().st_size > 1024:
            logger.info(f"Long running therapy data exists at {output_file}. Skipping.")
            return True

        cmd = [
            "uv",
            "run",
            "python",
            str(script_path),
            "--min-turns",
            "20",
            # We don't necessarily upload to S3 here, local cache for pipeline is sufficient
            # But script supports --upload-s3 if needed
        ]

        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.check_call(cmd, cwd=self.workspace_root)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Long running extraction failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Long running extraction failed: {e}")
            return False

    def run_all_checks(self) -> None:
        """Run all generation checks in sequence."""
        self.ensure_nemo_synthetic(target_count=10000)
        self.ensure_ultra_nightmares(count_per_category=5)  # Start small for safety/speed
        self.ensure_edge_cases(count=10000)
        self.ensure_long_running_extraction()
