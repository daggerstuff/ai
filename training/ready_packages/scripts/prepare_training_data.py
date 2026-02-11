#!/usr/bin/env python3
"""
End-to-End Data Preparation Orchestration

Orchestrates complete data preparation workflow:
source → process → filter → format → assemble

Provides CLI interface with options for each stage and checkpoint support.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DataPreparationOrchestrator:
    """Orchestrates complete data preparation pipeline"""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.checkpoint_file = base_path / "ai" / "training_ready" / "scripts" / "output" / "checkpoint.json"
        self.checkpoint = self._load_checkpoint()

        # Script paths
        self.scripts = {
            "source": base_path / "ai" / "training_ready" / "tools" / "data_preparation" / "source_datasets.py",
            "process": base_path / "ai" / "training_ready" / "pipelines" / "integrated" / "process_all_datasets.py",
            "filter": base_path / "ai" / "training_ready" / "tools" / "data_preparation" / "filter_and_clean.py",
            "format": base_path / "ai" / "training_ready" / "tools" / "data_preparation" / "format_for_training.py",
            "assemble": base_path / "ai" / "training_ready" / "pipelines" / "integrated" / "assemble_final_dataset.py",
        }

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint state"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return {
            "completed_stages": [],
            "last_stage": None,
            "timestamp": None,
        }

    def _save_checkpoint(self, stage: str):
        """Save checkpoint state"""
        self.checkpoint["completed_stages"].append(stage)
        self.checkpoint["last_stage"] = stage
        self.checkpoint["timestamp"] = datetime.now().isoformat()

        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.checkpoint, f, indent=2)

    def _run_script(self, stage: str, script_path: Path, args: Optional[list] = None) -> bool:
        """Run a pipeline script"""
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False

        import subprocess
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)

        logger.info(f"🚀 Running {stage} stage...")
        logger.info(f"   Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {stage} stage failed:")
            logger.error(e.stdout)
            logger.error(e.stderr)
            return False

    def run_stage(self, stage: str, force: bool = False) -> bool:
        """Run a specific stage"""
        if stage in self.checkpoint["completed_stages"] and not force:
            logger.info(f"⏭️  Skipping {stage} (already completed). Use --force to rerun.")
            return True

        script_path = self.scripts.get(stage)
        if not script_path:
            logger.error(f"Unknown stage: {stage}")
            return False

        success = self._run_script(stage, script_path)
        if success:
            self._save_checkpoint(stage)

        return success

    def run_all(self, start_from: Optional[str] = None, force: bool = False) -> bool:
        """Run all stages in sequence"""
        stages = ["source", "process", "filter", "format", "assemble"]

        if start_from:
            try:
                start_idx = stages.index(start_from)
                stages = stages[start_idx:]
            except ValueError:
                logger.error(f"Unknown stage: {start_from}")
                return False

        logger.info("=" * 60)
        logger.info("🚀 Starting End-to-End Data Preparation Pipeline")
        logger.info("=" * 60)

        for stage in stages:
            if not self.run_stage(stage, force=force):
                logger.error(f"❌ Pipeline failed at {stage} stage")
                return False
            logger.info("")

        logger.info("=" * 60)
        logger.info("✅ Data preparation pipeline completed successfully!")
        logger.info("=" * 60)

        return True

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive preparation report"""
        reports = {}
        report_files = {
            "sourcing": self.base_path / "ai" / "training_ready" / "scripts" / "output" / "sourcing_report.json",
            "processing": self.base_path / "ai" / "training_ready" / "scripts" / "output" / "processing_report.json",
            "filtering": self.base_path / "ai" / "training_ready" / "scripts" / "output" / "filtering_report.json",
            "formatting": self.base_path / "ai" / "training_ready" / "scripts" / "output" / "formatting_report.json",
            "assembly": self.base_path / "ai" / "training_ready" / "scripts" / "output" / "assembly_report.json",
        }

        for name, path in report_files.items():
            if path.exists():
                try:
                    with open(path, "r") as f:
                        reports[name] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load {name} report: {e}")

        return {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": self.checkpoint,
            "reports": reports,
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="End-to-end data preparation pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages
  python prepare_training_data.py --all

  # Run specific stage
  python prepare_training_data.py --stage filter

  # Resume from checkpoint
  python prepare_training_data.py --all --start-from filter

  # Force rerun completed stages
  python prepare_training_data.py --all --force
        """
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all stages in sequence"
    )
    parser.add_argument(
        "--stage",
        choices=["source", "process", "filter", "format", "assemble"],
        help="Run a specific stage"
    )
    parser.add_argument(
        "--start-from",
        choices=["source", "process", "filter", "format", "assemble"],
        help="Start from a specific stage (resume from checkpoint)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if stage is already completed"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive preparation report"
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=100000,
        help="Target total conversations for assembly (default: 100000)"
    )

    args = parser.parse_args()

    base_path = Path.cwd()
    orchestrator = DataPreparationOrchestrator(base_path)

    if args.report:
        report = orchestrator.generate_report()
        report_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "preparation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Preparation report saved to: {report_path}")
        return 0

    if args.all:
        success = orchestrator.run_all(start_from=args.start_from, force=args.force)
        return 0 if success else 1
    elif args.stage:
        success = orchestrator.run_stage(args.stage, force=args.force)
        return 0 if success else 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

