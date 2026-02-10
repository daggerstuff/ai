import contextlib
import json
import logging
import subprocess
from pathlib import Path

# Import the direct class-based generators
try:
    from ai.training.ready_packages.scripts.generate_ultra_nightmares import (
        UltraNightmareGenerator,
    )
except ImportError:
    UltraNightmareGenerator = None

try:
    from ai.pipelines.design.service import NeMoDataDesignerService
except ImportError:
    NeMoDataDesignerService = None

try:
    from ai.sourcing.academic.academic_sourcing import AcademicSourcingEngine
except ImportError:
    AcademicSourcingEngine = None

try:
    from ai.sourcing.journal.main import WorkflowExecutor
except ImportError as e:
    logging.getLogger(__name__).warning(f"Failed to import WorkflowExecutor: {e}")
    WorkflowExecutor = None

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
            # Configure storage to go into the new training ready area
            storage_path = (
                self.workspace_root
                / "ai/training/ready_packages/datasets/cache/local/nightmare_fuel"
            )
            storage_path.mkdir(parents=True, exist_ok=True)

            generator = UltraNightmareGenerator(output_dir=str(storage_path))
            # This generates directly to high-intensity edge cases area
            generator.generate_all(count_per_category=count_per_category)
            return True
        except Exception:
            logger.exception("Failed to generate Ultra Nightmares")
            return False

    def _ensure_nemo_service_running(self) -> bool:
        """Check if NeMo service is running, if not start it."""
        import os
        import socket
        import time

        host = "localhost"
        port = 8000

        # Check if port is open
        with contextlib.suppress(Exception):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((host, port))
                if result == 0:
                    logger.info("NeMo service is already running on port 8000.")
                    return True
        logger.info("NeMo service not found. Attempting to start via Docker Compose...")

        nemo_quickstart_dir = self.workspace_root / "ai/nemo"
        compose_file = (
            nemo_quickstart_dir / "docker-compose.yaml"
            if (nemo_quickstart_dir / "docker-compose.yaml").exists()
            else self.workspace_root / "docker/docker-compose.nemo-data-designer.yml"
        )

        if not compose_file.exists():
            logger.error(f"Docker Compose file not found: {compose_file}")
            return False

        try:
            env = os.environ.copy()
            if "NVIDIA_API_KEY" not in env:
                logger.warning("NVIDIA_API_KEY not set, NeMo service might fail to start.")
            else:
                # Setup required NeMo env vars
                env["NIM_API_KEY"] = env["NVIDIA_API_KEY"]
                env["NGC_API_KEY"] = env["NVIDIA_API_KEY"]
                env["NEMO_MICROSERVICES_IMAGE_REGISTRY"] = "nvcr.io/nvidia/nemo-microservices"
                env["NEMO_MICROSERVICES_IMAGE_TAG"] = "25.12"

                # Attempt docker login for NGC
                try:
                    logger.info("Attempting Docker login to nvcr.io...")
                    login_cmd = [
                        "docker",
                        "login",
                        "nvcr.io",
                        "-u",
                        "$oauthtoken",
                        "--password-stdin",
                    ]
                    # Pass API key as password via stdin
                    subprocess.run(
                        login_cmd,
                        input=env["NVIDIA_API_KEY"].encode(),
                        check=True,
                        capture_output=True,
                        shell=False,
                    )
                    logger.info("✅ Docker login to nvcr.io successful")
                except subprocess.CalledProcessError as e:
                    # Log warning but proceed, maybe we are already logged in or using
                    # cached image
                    logger.warning(
                        f"⚠️ Docker login failed (this might differ if already logged in): {e}"
                    )

            # Force recreate to ensure fresh state if needed
            # Use profile if using the quickstart dir
            cmd = ["docker", "compose", "-f", str(compose_file)]
            cwd = self.workspace_root

            if str(nemo_quickstart_dir) in str(compose_file):
                cmd.extend(["--profile", "data-designer"])
                cwd = nemo_quickstart_dir

            cmd.extend(["up", "-d"])

            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd, cwd=cwd, env=env, shell=False)

            # Wait for service to be ready
            logger.info("Waiting for NeMo service to become ready...")
            for _ in range(120):
                with contextlib.suppress(Exception):
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1)
                        if s.connect_ex((host, port)) == 0:
                            logger.info("NeMo service port is open. Waiting for app init...")
                            time.sleep(10)  # Give app time to initialize
                            return True
                time.sleep(2)

            logger.error("NeMo service failed to become ready within timeout.")
            return False

        except subprocess.CalledProcessError:
            logger.exception("Failed to start NeMo service")
            return False
        except Exception:
            logger.exception("Unexpected error starting NeMo service")
            return False

    def ensure_nemo_synthetic(self, target_count: int = 10000) -> bool:
        """
        Ensure NeMo synthetic data generation (Stage 1/2).
        Uses NeMoDataDesignerService.
        """
        logger.info(f"Ensuring NeMo synthetic data (target={target_count})...")

        if not NeMoDataDesignerService:
            # Check if we can mock it or just fail gracefully (it might be optional
            # if env not set)
            logger.error("NeMoDataDesignerService could not be imported.")
            return False

        # Ensure the service infrastructure is up
        if not self._ensure_nemo_service_running():
            logger.warning("Could not start NeMo service. Skipping generation.")
            return False

        output_dir = (
            self.workspace_root / "ai/training/ready_packages/datasets/cache/local/nemo_synthetic"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "nemo_synthetic_dataset.jsonl"

        if output_file.exists() and output_file.stat().st_size > 1024:
            # Check count? For now simple existence check or we could assume it's good.
            # Ideally we check line count.
            logger.info(f"NeMo dataset exists at {output_file}. Skipping generation.")
            return True

        try:
            return self._trigger_nemo_generation(target_count, output_file)
        except Exception:
            logger.exception("NeMo generation failed")
            return False

    def _trigger_nemo_generation(self, target_count, output_file):
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

        output_dir = (
            self.workspace_root
            / "ai/training/ready_packages/datasets/cache/local/edge_case_synthetic"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv",
            "run",
            "python",
            str(script_path),
            "--count",
            str(count),
            "--categories",
            "all",
            "--output-dir",
            str(output_dir),
        ]

        try:
            return self._run_shell_command(cmd)
        except subprocess.CalledProcessError:
            logger.exception("Edge case generation failed check_call")
            return False
        except Exception:
            logger.exception("Edge case generation failed")
            return False

    def ensure_long_running_extraction(self) -> bool:
        """
        Extract long-running therapy sessions.
        Run logic from run_phase1_production.sh -> extract_long_running_therapy.py.
        """
        logger.info("Ensuring Long Running Therapy extraction (Stage 5)...")
        script_path = self.scripts_dir / "extract_long_running_therapy.py"
        output_dir = (
            self.workspace_root
            / "ai/training/ready_packages/datasets/cache/local/long_running_therapy"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "long_running_therapy.jsonl"

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
            "--output-dir",
            str(output_dir),
            # We don't necessarily upload to S3 here, local cache for pipeline is
            # sufficient. But script supports --upload-s3 if needed
        ]

        try:
            return self._run_shell_command(cmd)
        except subprocess.CalledProcessError:
            logger.exception("Long running extraction failed")
            return False

    def ensure_hydration(self) -> bool:
        """Run the hydration script for unused material."""
        logger.info("Ensuring Unused Material Hydration...")
        script_path = self.scripts_dir / "hydrate_unused_material.py"
        if not script_path.exists():
            logger.error(f"Hydration script not found: {script_path}")
            return False

        cmd = ["uv", "run", "python", str(script_path)]
        try:
            return self._run_shell_command(cmd)
        except Exception:
            logger.exception("Hydration failed")
            return False

    def ensure_academic_sourcing(self, limit_per_query: int = 10) -> bool:
        """
        Ensure academic findings are sourced (PubMed/Scholar).
        Uses AcademicSourcingEngine.
        """
        logger.info(f"Ensuring Academic findings (limit_per_query={limit_per_query})...")

        if not AcademicSourcingEngine:
            logger.error("AcademicSourcingEngine could not be imported.")
            return False

        try:
            # We use the default output path defined in the engine, but resolve it
            # absolutely
            output_path = self.workspace_root / "ai/training/ready_packages/datasets"
            engine = AcademicSourcingEngine(output_base_path=str(output_path))
            engine.run_sourcing_pipeline(limit_per_query=limit_per_query)
            return True
        except Exception:
            logger.exception("Academic sourcing failed")
            return False

    def ensure_journal_research(self, keywords: list[str] | None = None) -> bool:
        """
        Ensure journal research findings are sourced.
        Uses WorkflowExecutor from journal sourcing.
        """
        if keywords is None:
            keywords = [
                "trauma informed care",
                "therapeutic dialogue",
                "empathetic response",
                "clinical psychology conversation",
            ]

        logger.info(f"Ensuring Journal research (keywords={keywords})...")

        if not WorkflowExecutor:
            logger.error("WorkflowExecutor could not be imported from journal sourcing.")
            return False

        try:
            return self._run_journal_workflow(keywords)
        except Exception:
            logger.exception("Journal research workflow failed")
            return False

    def _run_journal_workflow(self, keywords):
        # Configure storage to go into the training ready area
        storage_path = (
            self.workspace_root / "ai/training/ready_packages/datasets/cache/local/journal_research"
        )
        storage_path.mkdir(parents=True, exist_ok=True)
        config = {"acquisition": {"storage_base_path": str(storage_path)}}

        executor = WorkflowExecutor(config=config, dry_run=False, interactive=False)
        search_keywords = {
            "therapeutic": keywords,
            "dataset": keywords,
        }
        # This triggers discovery, evaluation, acquisition, and integration
        executor.execute_workflow(
            search_keywords=search_keywords,
            target_sources=["pubmed", "doaj"],
        )
        return True

    def ensure_books_extraction(self) -> bool:
        """
        Ensure therapeutic book content is extracted (Stage 2).
        """
        logger.info("Ensuring Book Content extraction (Stage 2)...")
        script_path = self.scripts_dir / "extract_all_books_to_training.py"

        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False

        cmd = [
            "uv",
            "run",
            "python",
            str(script_path),
            "--all",
        ]

        try:
            return self._run_shell_command(cmd)
        except subprocess.CalledProcessError:
            logger.exception("Book extraction failed")
            return False
        except Exception:
            logger.exception("Book extraction failed")
            return False

    def ensure_transcripts_extraction(self) -> bool:
        """
        Ensure YouTube transcripts are extracted (Stage 4).
        """
        logger.info("Ensuring YouTube Transcripts extraction (Stage 4)...")
        script_path = self.scripts_dir / "extract_all_youtube_transcripts.py"

        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False

        cmd = [
            "uv",
            "run",
            "python",
            str(script_path),
            "--all",
        ]

        try:
            return self._run_shell_command(cmd)
        except subprocess.CalledProcessError:
            logger.exception("Transcript extraction failed")
            return False
        except Exception:
            logger.exception("Transcript extraction failed")
            return False

    def _run_shell_command(self, cmd):
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.check_call(cmd, cwd=self.workspace_root)
        return True

    def run_all_checks(self) -> None:
        """Run all generation checks in sequence."""
        self.ensure_hydration()
        self.ensure_nemo_synthetic(target_count=10000)
        self.ensure_ultra_nightmares(count_per_category=5)
        self.ensure_edge_cases(count=10000)
        self.ensure_long_running_extraction()
        self.ensure_academic_sourcing()
        self.ensure_journal_research()
        self.ensure_books_extraction()
        self.ensure_transcripts_extraction()
