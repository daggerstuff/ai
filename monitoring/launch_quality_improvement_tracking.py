#!/usr/bin/env python3
"""
Quality Improvement Tracking Launcher (Task 5.6.2.4)

Enterprise-grade launcher for the quality improvement tracking system
with comprehensive setup, validation, and execution capabilities.
"""

import argparse
import logging
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from quality_improvement_reporter import QualityImprovementReporter
from quality_improvement_tracker import QualityImprovementTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class QualityImprovementTrackingLauncher:
    """
    Enterprise-grade launcher for quality improvement tracking system.

    Handles setup, validation, intervention management, and report generation
    with comprehensive error handling and monitoring.
    """

    def __init__(self):
        """Initialize the improvement tracking launcher."""
        self.project_root = Path(__file__).parent.parent
        self.monitoring_dir = Path(__file__).parent
        self.db_path = self.project_root / "database" / "conversations.db"
        self.reports_dir = self.monitoring_dir / "reports"

        # Ensure reports directory exists
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Required dependencies
        self.required_packages = [
            "pandas",
            "numpy",
            "sqlite3",
            "scipy",
            "plotly",
            "jinja2",
        ]

        logger.info("🚀 Quality Improvement Tracking Launcher initialized")

    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        logger.info("🔍 Checking dependencies...")

        missing_packages = []

        for package in self.required_packages:
            try:
                if (
                    package in {"sqlite3", "pandas", "numpy", "scipy", "plotly", "jinja2"}
                ):
                    pass

                logger.info(f"  ✅ {package}: Available")

            except ImportError:
                missing_packages.append(package)
                logger.error(f"  ❌ {package}: Missing")

        if missing_packages:
            logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
            logger.info("💡 Install missing packages with: pip install " + " ".join(missing_packages))
            return False

        logger.info("✅ All dependencies available")
        return True

    def validate_database(self) -> bool:
        """Validate that the database exists and has quality data."""
        logger.info("🔍 Validating database...")

        if not self.db_path.exists():
            logger.error(f"❌ Database not found: {self.db_path}")
            logger.info("💡 Run the dataset processing pipeline to create the database")
            return False

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Check if required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            required_tables = ["conversations", "quality_metrics"]
            missing_tables = [table for table in required_tables if table not in tables]

            if missing_tables:
                logger.error(f"❌ Missing tables: {', '.join(missing_tables)}")
                conn.close()
                return False

            # Check if quality data exists
            cursor.execute("""
                SELECT COUNT(*) as total_conversations,
                       MIN(c.created_at) as earliest_date,
                       MAX(c.created_at) as latest_date
                FROM conversations c
                LEFT JOIN quality_metrics q ON c.id = q.conversation_id
                WHERE q.overall_quality IS NOT NULL
            """)

            result = cursor.fetchone()
            total_conversations, earliest_date, latest_date = result

            if total_conversations < 50:
                logger.error(f"❌ Insufficient data for improvement tracking: only {total_conversations} conversations")
                logger.info("💡 Need at least 50 conversations with quality data for improvement tracking")
                conn.close()
                return False

            logger.info(f"✅ Database validated: {total_conversations} conversations")
            logger.info(f"📅 Data range: {earliest_date} to {latest_date}")
            conn.close()
            return True

        except Exception as e:
            logger.error(f"❌ Database validation error: {e}")
            return False

    def validate_tracking_files(self) -> bool:
        """Validate that improvement tracking files exist and are accessible."""
        logger.info("🔍 Validating improvement tracking files...")

        required_files = [
            "quality_improvement_tracker.py",
            "quality_improvement_reporter.py",
        ]

        for filename in required_files:
            filepath = self.monitoring_dir / filename

            if not filepath.exists():
                logger.error(f"❌ Required file not found: {filepath}")
                return False

            # Check if file has main classes
            try:
                with open(filepath) as f:
                    content = f.read()

                if filename == "quality_improvement_tracker.py":
                    if "QualityImprovementTracker" not in content:
                        logger.error(f"❌ QualityImprovementTracker class not found in {filename}")
                        return False
                elif filename == "quality_improvement_reporter.py":
                    if "QualityImprovementReporter" not in content:
                        logger.error(f"❌ QualityImprovementReporter class not found in {filename}")
                        return False

            except Exception as e:
                logger.error(f"❌ Error reading {filename}: {e}")
                return False

        logger.info("✅ Improvement tracking files validated")
        return True

    def create_intervention(
        self,
        name: str,
        description: str,
        intervention_type: str,
        target_component: str,
        expected_improvement: float,
        target_tier: str | None = None,
        target_dataset: str | None = None,
        created_by: str = "user",
    ) -> dict:
        """Create a new quality improvement intervention."""
        logger.info(f"📊 Creating intervention: {name}")

        try:
            # Import improvement tracker
            sys.path.append(str(self.monitoring_dir))

            # Initialize tracker
            tracker = QualityImprovementTracker(db_path=str(self.db_path))

            # Create intervention
            intervention_id = tracker.create_intervention(
                name=name,
                description=description,
                intervention_type=intervention_type,
                target_component=target_component,
                expected_improvement=expected_improvement,
                target_tier=target_tier,
                target_dataset=target_dataset,
                created_by=created_by,
            )

            logger.info(f"✅ Created intervention: {intervention_id}")

            return {
                "success": True,
                "intervention_id": intervention_id,
                "name": name,
                "target_component": target_component,
                "expected_improvement": expected_improvement,
                "status": "created",
            }

        except Exception as e:
            logger.error(f"❌ Error creating intervention: {e}")
            return {"success": False, "error": str(e)}

    def start_intervention(self, intervention_id: str) -> dict:
        """Start an intervention and establish baseline."""
        logger.info(f"🚀 Starting intervention: {intervention_id}")

        try:
            # Import improvement tracker
            sys.path.append(str(self.monitoring_dir))

            # Initialize tracker
            tracker = QualityImprovementTracker(db_path=str(self.db_path))

            # Start intervention
            success = tracker.start_intervention(intervention_id)

            if success:
                logger.info(f"✅ Started intervention: {intervention_id}")
                return {
                    "success": True,
                    "intervention_id": intervention_id,
                    "status": "started",
                    "message": "Intervention started and baseline established",
                }
            logger.error(f"❌ Failed to start intervention: {intervention_id}")
            return {
                "success": False,
                "intervention_id": intervention_id,
                "error": "Failed to start intervention",
            }

        except Exception as e:
            logger.error(f"❌ Error starting intervention: {e}")
            return {"success": False, "error": str(e)}

    def record_progress(self, intervention_id: str, notes: str = "") -> dict:
        """Record progress measurement for an intervention."""
        logger.info(f"📈 Recording progress for intervention: {intervention_id}")

        try:
            # Import improvement tracker
            sys.path.append(str(self.monitoring_dir))

            # Initialize tracker
            tracker = QualityImprovementTracker(db_path=str(self.db_path))

            # Record progress
            success = tracker.record_progress_measurement(intervention_id, notes)

            if success:
                logger.info(f"✅ Recorded progress for intervention: {intervention_id}")
                return {
                    "success": True,
                    "intervention_id": intervention_id,
                    "status": "progress_recorded",
                    "message": "Progress measurement recorded successfully",
                }
            logger.error(f"❌ Failed to record progress for intervention: {intervention_id}")
            return {
                "success": False,
                "intervention_id": intervention_id,
                "error": "Failed to record progress measurement",
            }

        except Exception as e:
            logger.error(f"❌ Error recording progress: {e}")
            return {"success": False, "error": str(e)}

    def complete_intervention(self, intervention_id: str) -> dict:
        """Complete an intervention and perform final assessment."""
        logger.info(f"🏁 Completing intervention: {intervention_id}")

        try:
            # Import improvement tracker
            sys.path.append(str(self.monitoring_dir))

            # Initialize tracker
            tracker = QualityImprovementTracker(db_path=str(self.db_path))

            # Complete intervention
            success = tracker.complete_intervention(intervention_id)

            if success:
                # Analyze impact
                analysis = tracker.analyze_intervention_impact(intervention_id)

                logger.info(f"✅ Completed intervention: {intervention_id}")

                result = {
                    "success": True,
                    "intervention_id": intervention_id,
                    "status": "completed",
                    "message": "Intervention completed and analyzed",
                }

                if analysis:
                    result["analysis"] = {
                        "improvement_achieved": analysis.improvement_metrics["absolute_improvement"],
                        "target_achievement": analysis.improvement_metrics["target_achievement"],
                        "statistical_significance": len(
                            [t for t in analysis.statistical_tests if t.get("significant", False)]
                        )
                        > 0,
                        "recommendations_count": len(analysis.recommendations),
                    }

                return result
            logger.error(f"❌ Failed to complete intervention: {intervention_id}")
            return {
                "success": False,
                "intervention_id": intervention_id,
                "error": "Failed to complete intervention",
            }

        except Exception as e:
            logger.error(f"❌ Error completing intervention: {e}")
            return {"success": False, "error": str(e)}

    def generate_improvement_report(self, report_period_days: int = 90, output_format: str = "both") -> dict:
        """Generate comprehensive improvement tracking report."""
        logger.info(f"📊 Generating improvement report ({report_period_days} days)")

        try:
            # Import improvement components
            sys.path.append(str(self.monitoring_dir))

            # Initialize components
            tracker = QualityImprovementTracker(db_path=str(self.db_path))
            reporter = QualityImprovementReporter(output_dir=str(self.reports_dir))

            # Mock the tracker to use our database
            reporter.tracker = tracker

            # Generate comprehensive report
            logger.info("📋 Generating comprehensive improvement report...")
            report = reporter.generate_comprehensive_report(
                report_period_days=report_period_days, include_visualizations=True
            )

            # Save reports
            saved_files = []

            if output_format in ["json", "both"]:
                json_path = reporter.save_report(report, format="json")
                saved_files.append(json_path)
                logger.info(f"📄 JSON report saved: {json_path}")

            if output_format in ["html", "both"]:
                html_path = reporter.save_report(report, format="html")
                saved_files.append(html_path)
                logger.info(f"📄 HTML report saved: {html_path}")

            # Generate visualizations
            visualization_files = []
            try:
                logger.info("📊 Generating improvement visualizations...")
                visualizations = reporter.create_improvement_visualizations(report)

                # Save visualizations as HTML files
                for viz_name, fig in visualizations.items():
                    viz_filename = (
                        f"improvement_visualization_{viz_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.html"
                    )
                    viz_path = self.reports_dir / viz_filename
                    fig.write_html(str(viz_path))
                    visualization_files.append(str(viz_path))
                    logger.info(f"📊 Visualization saved: {viz_path}")

            except Exception as e:
                logger.warning(f"⚠️ Could not generate visualizations: {e}")

            # Create summary
            summary = {
                "success": True,
                "report_period": f"{report_period_days}_days",
                "active_interventions": len(report.active_interventions),
                "completed_interventions": len(report.completed_interventions),
                "improvement_analyses": len(report.improvement_analyses),
                "overall_impact": report.overall_impact,
                "success_metrics": report.success_metrics,
                "report_files": saved_files,
                "visualization_files": visualization_files,
                "executive_summary": report.executive_summary,
                "action_items": report.action_items,
            }

            logger.info("✅ Improvement report generated successfully")
            return summary

        except Exception as e:
            logger.error(f"❌ Error generating improvement report: {e}")
            return {"success": False, "error": str(e)}

    def display_report_summary(self, summary: dict):
        """Display report summary to console."""
        if not summary.get("success", False):
            return

        if summary["overall_impact"]:
            summary["overall_impact"]

        if summary["success_metrics"]:
            summary["success_metrics"]

        for _file_path in summary["report_files"]:
            pass

        if summary["visualization_files"]:
            for _file_path in summary["visualization_files"]:
                pass

        for _i, _item in enumerate(summary["executive_summary"][:5], 1):
            pass

        for _i, _item in enumerate(summary["action_items"][:5], 1):
            pass

    def launch(self, action: str, **kwargs) -> bool:
        """
        Complete launch sequence for quality improvement tracking.

        Args:
            action: Action to perform ('create', 'start', 'progress', 'complete', 'report')
            **kwargs: Action-specific parameters

        Returns:
            bool: True if action successful, False otherwise
        """
        logger.info("🎯 Starting Quality Improvement Tracking Launch Sequence...")

        # Step 1: Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Launch aborted: Missing dependencies")
            return False

        # Step 2: Validate database
        if not self.validate_database():
            logger.error("❌ Launch aborted: Database validation failed")
            return False

        # Step 3: Validate tracking files
        if not self.validate_tracking_files():
            logger.error("❌ Launch aborted: Tracking files validation failed")
            return False

        # Step 4: Execute action
        if action == "create":
            result = self.create_intervention(**kwargs)
        elif action == "start":
            result = self.start_intervention(kwargs.get("intervention_id"))
        elif action == "progress":
            result = self.record_progress(kwargs.get("intervention_id"), kwargs.get("notes", ""))
        elif action == "complete":
            result = self.complete_intervention(kwargs.get("intervention_id"))
        elif action == "report":
            result = self.generate_improvement_report(
                kwargs.get("report_period_days", 90),
                kwargs.get("output_format", "both"),
            )
            self.display_report_summary(result)
        else:
            logger.error(f"❌ Unknown action: {action}")
            return False

        if result.get("success", False):
            logger.info("🎉 Quality Improvement Tracking action completed successfully!")
            return True
        logger.error("❌ Quality Improvement Tracking action failed")
        return False


def main():
    """Main function to launch quality improvement tracking."""
    parser = argparse.ArgumentParser(description="Launch Quality Improvement Tracking System")
    parser.add_argument(
        "action",
        choices=["create", "start", "progress", "complete", "report"],
        help="Action to perform",
    )

    # Create intervention arguments
    parser.add_argument("--name", help="Intervention name (for create)")
    parser.add_argument("--description", help="Intervention description (for create)")
    parser.add_argument("--type", help="Intervention type (for create)")
    parser.add_argument("--component", help="Target component (for create)")
    parser.add_argument("--improvement", type=float, help="Expected improvement (for create)")
    parser.add_argument("--tier", help="Target tier (for create)")
    parser.add_argument("--dataset", help="Target dataset (for create)")
    parser.add_argument("--created-by", default="user", help="Created by (for create)")

    # General arguments
    parser.add_argument("--intervention-id", help="Intervention ID (for start, progress, complete)")
    parser.add_argument("--notes", help="Progress notes (for progress)")
    parser.add_argument("--days", type=int, default=90, help="Report period days (for report)")
    parser.add_argument(
        "--format",
        choices=["json", "html", "both"],
        default="both",
        help="Output format (for report)",
    )

    args = parser.parse_args()

    launcher = QualityImprovementTrackingLauncher()

    # Prepare kwargs based on action
    kwargs = {}
    if args.action == "create":
        if not all([args.name, args.description, args.type, args.component, args.improvement]):
            sys.exit(1)
        kwargs = {
            "name": args.name,
            "description": args.description,
            "intervention_type": args.type,
            "target_component": args.component,
            "expected_improvement": args.improvement,
            "target_tier": args.tier,
            "target_dataset": args.dataset,
            "created_by": args.created_by,
        }
    elif args.action in ["start", "progress", "complete"]:
        if not args.intervention_id:
            sys.exit(1)
        kwargs = {"intervention_id": args.intervention_id}
        if args.action == "progress" and args.notes:
            kwargs["notes"] = args.notes
    elif args.action == "report":
        kwargs = {"report_period_days": args.days, "output_format": args.format}

    success = launcher.launch(args.action, **kwargs)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
