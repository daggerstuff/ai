#!/usr/bin/env python3
"""
Master orchestration script that runs all dataset registry enhancement
and maintenance operations in sequence.
"""

from datetime import datetime, timezone

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class DatasetRegistryOrchestrator:
    """Orchestrates all dataset registry maintenance operations."""

    def __init__(self, registry_path: Path, scripts_dir: Path):
        self.registry_path = registry_path
        self.scripts_dir = scripts_dir
        self.results = {}

    def run_script(self, script_name: str, args: list = None) -> dict[str, Any]:
        """
        Run a Python script and capture results.

        Args:
            script_name: Name of the script to run
            args: Additional arguments for the script

        Returns:
            Dictionary with success status and output
        """
        script_path = self.scripts_dir / script_name

        if not script_path.exists():
            return {"success": False, "error": f"Script not found: {script_path}"}

        cmd = ["uv", "run", "python", str(script_path)]
        if args:
            cmd.extend(args)

        print(f"\n{'=' * 80}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'=' * 80}\n")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.scripts_dir.parent,
                timeout=300,  # 5 minute timeout
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Script timed out after 5 minutes"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def enhance_registry(self, limit: int = None) -> bool:
        """Enhance registry with new fields."""
        print("\n### STEP 1: Enhancing Registry Structure ###")

        args = []
        if limit:
            args.extend(["--limit", str(limit)])

        result = self.run_script("enhance_dataset_registry.py", args)
        self.results["enhancement"] = result

        if result["success"]:
            print(result["stdout"])
        else:
            print(
                f"ERROR: {result.get('error', result.get('stderr', 'Unknown error'))}"
            )

        return result["success"]

    def validate_datasets(self, limit: int = None, dry_run: bool = False) -> bool:
        """Validate all datasets using rclone."""
        print("\n### STEP 2: Validating Datasets (rclone) ###")

        args = []
        if limit:
            args.extend(["--limit", str(limit)])
        if dry_run:
            args.append("--dry-run")

        result = self.run_script("dataset_validation_rclone.py", args)
        self.results["validation"] = result

        if result["success"]:
            print(result["stdout"])
        else:
            print(
                f"ERROR: {result.get('error', result.get('stderr', 'Unknown error'))}"
            )

        return result["success"]

    def verify_sync(self, limit: int = None) -> bool:
        """Verify dataset sync using rclone."""
        print("\n### STEP 3: Verifying Dataset Sync (rclone) ###")

        args = []
        if limit:
            args.extend(["--limit", str(limit)])

        result = self.run_script("dataset_sync_verification_rclone.py", args)
        self.results["sync_verification"] = result

        if result["success"]:
            print(result["stdout"])
        else:
            print(
                f"ERROR: {result.get('error', result.get('stderr', 'Unknown error'))}"
            )

        return result["success"]

    def score_quality(self, limit: int = None) -> bool:
        """Score dataset quality using rclone."""
        print("\n### STEP 5: Scoring Dataset Quality (rclone) ###")

        args = []
        if limit:
            args.extend(["--limit", str(limit)])

        result = self.run_script("dataset_quality_scorer_rclone.py", args)
        self.results["quality_scoring"] = result

        if result["success"]:
            print(result["stdout"])
        else:
            print(
                f"ERROR: {result.get('error', result.get('stderr', 'Unknown error'))}"
            )

        return result["success"]

    def deduplicate_datasets(self, limit: int = None) -> bool:
        """Deduplicate datasets using rclone."""
        print("\n### STEP 6: Deduplicating Datasets (rclone) ###")

        args = []
        if limit:
            args.extend(["--limit", str(limit)])

        result = self.run_script("dataset_deduplication_rclone.py", args)
        self.results["deduplication"] = result

        if result["success"]:
            print(result["stdout"])
        else:
            print(
                f"ERROR: {result.get('error', result.get('stderr', 'Unknown error'))}"
            )

        return result["success"]

    def update_usage_metrics(self, limit: int = None) -> bool:
        """Update usage analytics."""
        print("\n### STEP 4: Updating Usage Metrics ###")

        args = ["--action", "update"]
        if limit:
            args.extend(["--limit", str(limit)])

        result = self.run_script("dataset_usage_tracker.py", args)
        self.results["usage_metrics"] = result

        if result["success"]:
            print(result["stdout"])
        else:
            print(
                f"ERROR: {result.get('error', result.get('stderr', 'Unknown error'))}"
            )

        return result["success"]

    def score_quality(self, limit: int = None) -> bool:
        """Score dataset quality."""
        print("\n### STEP 5: Scoring Dataset Quality ###")

        args = []
        if limit:
            args.extend(["--limit", str(limit)])

        result = self.run_script("dataset_quality_scorer.py", args)
        self.results["quality_scoring"] = result

        if result["success"]:
            print(result["stdout"])
        else:
            print(
                f"ERROR: {result.get('error', result.get('stderr', 'Unknown error'))}"
            )

        return result["success"]

        args = ["--action", "dedupe"]
        if limit:
            args.extend(["--limit", str(limit)])

        result = self.run_script("dataset_deduplication.py", args)
        self.results["deduplication"] = result

        if result["success"]:
            print(result["stdout"])
        else:
            print(
                f"ERROR: {result.get('error', result.get('stderr', 'Unknown error'))}"
            )

        return result["success"]

    def generate_report(self) -> dict[str, Any]:
        """Generate summary report of all operations."""
        print("\n### GENERATING FINAL REPORT ###\n")

        # Load final registry state
        with open(self.registry_path) as f:
            registry = json.load(f)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "registry_version": registry.get("schema_version", "1.0.0"),
            "enhancements_applied": registry.get("registry_metadata", {}).get(
                "enhancements_applied", []
            ),
            "statistics": registry.get("registry_statistics", {}),
            "operation_results": {
                name: {
                    "success": result.get("success", False),
                    "has_output": bool(result.get("stdout")),
                }
                for name, result in self.results.items()
            },
        }

        # Print report
        print("=" * 80)
        print("DATASET REGISTRY ENHANCEMENT REPORT")
        print("=" * 80)
        print(f"\nTimestamp: {report['timestamp']}")
        print(f"Registry Version: {report['registry_version']}")
        print("\nEnhancements Applied:")
        for enhancement in report["enhancements_applied"]:
            print(f"  ✓ {enhancement}")

        print("\nStatistics:")
        stats = report["statistics"]
        print(f"  Total Datasets: {stats.get('total_datasets', 'N/A')}")

        if "datasets_by_stage" in stats:
            print("\n  By Stage:")
            for stage, count in stats["datasets_by_stage"].items():
                print(f"    {stage}: {count}")

        if "datasets_by_quality" in stats:
            print("\n  By Quality:")
            for tier, count in stats["datasets_by_quality"].items():
                print(f"    {tier}: {count}")

        if "validation_summary" in stats:
            print("\n  Validation Summary:")
            for status, count in stats["validation_summary"].items():
                print(f"    {status}: {count}")

        if "sync_summary" in stats:
            print("\n  Sync Summary:")
            for status, count in stats["sync_summary"].items():
                print(f"    {status}: {count}")

        print("\nOperation Results:")
        all_success = True
        for operation, result in report["operation_results"].items():
            status = "✓" if result["success"] else "✗"
            print(f"  {status} {operation}")
            if not result["success"]:
                all_success = False

        print(f"\n{'=' * 80}")
        if all_success:
            print("✓ ALL OPERATIONS COMPLETED SUCCESSFULLY")
        else:
            print("✗ SOME OPERATIONS FAILED - CHECK LOGS ABOVE")
        print(f"{'=' * 80}\n")

        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Orchestrate all dataset registry enhancements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all enhancements
  python orchestrate_registry_enhancements.py

  # Run with limit for testing
  python orchestrate_registry_enhancements.py --limit 5

  # Skip validation (faster)
  python orchestrate_registry_enhancements.py --skip-validation

  # Generate report only
  python orchestrate_registry_enhancements.py --report-only
        """,
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/home/vivi/pixelated/ai/config/dataset_registry.json"),
        help="Path to dataset registry",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of datasets to process per operation",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step (faster but less thorough)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report from current registry state",
    )

    args = parser.parse_args()

    scripts_dir = Path(__file__).parent
    orchestrator = DatasetRegistryOrchestrator(args.registry, scripts_dir)

    if args.report_only:
        orchestrator.generate_report()
        return 0

    print("=" * 80)
    print("DATASET REGISTRY ENHANCEMENT ORCHESTRATION")
    print("=" * 80)
    print(f"\nRegistry: {args.registry}")
    print(f"Limit: {args.limit or 'None (all datasets)'}")
    print(f"Skip Validation: {args.skip_validation}")
    print()

    success = True

    # Run all operations in sequence
    if not orchestrator.enhance_registry(limit=args.limit):
        success = False
        print("\n⚠ Warning: Enhancement step had issues, continuing...")

    if not args.skip_validation:
        if not orchestrator.validate_datasets(limit=args.limit):
            success = False
            print("\n⚠ Warning: Validation step had issues, continuing...")

    if not orchestrator.verify_sync(limit=args.limit):
        success = False
        print("\n⚠ Warning: Sync verification had issues, continuing...")

    if not orchestrator.update_usage_metrics(limit=args.limit):
        success = False
        print("\n⚠ Warning: Usage metrics update had issues, continuing...")

    if not orchestrator.score_quality(limit=args.limit):
        success = False
        print("\n⚠ Warning: Quality scoring had issues, continuing...")

    if not orchestrator.deduplicate_datasets(limit=args.limit):
        success = False
        print("\n⚠ Warning: Deduplication had issues, continuing...")

    # Generate final report
    report = orchestrator.generate_report()

    # Save report
    report_path = args.registry.parent / "enhancement_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
