"""
Tracking Document Updater

Automatically updates JOURNAL_RESEARCH_TARGETS.md with current progress metrics,
completed tasks, and status summaries.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ai.journal_dataset_research.models.dataset_models import (
    ResearchProgress,
    ResearchSession,
    WeeklyReport,
)

MARKER_PATTERNS = {
    "progress_section": r"<!-- PROGRESS_METRICS_START -->(.*?)<!-- PROGRESS_METRICS_END -->",
    "completed_tasks": r"<!-- COMPLETED_TASKS_START -->(.*?)<!-- COMPLETED_TASKS_END -->",
    "status_summary": r"<!-- STATUS_SUMMARY_START -->(.*?)<!-- STATUS_SUMMARY_END -->",
    "weekly_targets": r"<!-- WEEKLY_TARGETS_START -->(.*?)<!-- WEEKLY_TARGETS_END -->",
}


class TrackingDocumentUpdater:
    """
    Updates JOURNAL_RESEARCH_TARGETS.md with current research progress.

    Automatically updates progress sections, marks completed tasks with timestamps,
    and generates status summaries based on current research state.
    """

    def __init__(self, tracking_document_path: str = "JOURNAL_RESEARCH_TARGETS.md"):
        """
        Initialize the tracking document updater.

        Args:
            tracking_document_path: Path to the tracking document
        """
        self.tracking_document_path = Path(tracking_document_path)
        self.tracking_document_path.parent.mkdir(parents=True, exist_ok=True)

    def update_progress_section(
        self,
        progress: ResearchProgress,
        session: Optional[ResearchSession] = None,
    ) -> None:
        """
        Update the progress metrics section in the tracking document.

        Args:
            progress: Current research progress metrics
            session: Optional research session for additional context
        """
        content = self._read_document()

        progress_markdown = self._generate_progress_markdown(progress, session)
        updated_content = self._replace_section(
            content, "progress_section", progress_markdown
        )

        self._write_document(updated_content)

    def mark_task_completed(
        self, task_id: str, task_description: str, completion_date: Optional[datetime] = None
    ) -> None:
        """
        Mark a task as completed in the tracking document.

        Args:
            task_id: Task identifier
            task_description: Description of the completed task
            completion_date: Optional completion date (defaults to now)
        """
        if completion_date is None:
            completion_date = datetime.now()

        content = self._read_document()

        task_entry = f"- [x] {task_id}: {task_description} (Completed: {completion_date.strftime('%Y-%m-%d %H:%M:%S')})\n"
        updated_content = self._append_to_section(content, "completed_tasks", task_entry)

        self._write_document(updated_content)

    def update_status_summary(
        self,
        progress: ResearchProgress,
        session: Optional[ResearchSession] = None,
        weekly_report: Optional[WeeklyReport] = None,
    ) -> None:
        """
        Update the status summary section in the tracking document.

        Args:
            progress: Current research progress
            session: Optional research session
            weekly_report: Optional weekly report for additional context
        """
        content = self._read_document()

        status_markdown = self._generate_status_summary_markdown(
            progress, session, weekly_report
        )
        updated_content = self._replace_section(
            content, "status_summary", status_markdown
        )

        self._write_document(updated_content)

    def update_weekly_targets(
        self, session: ResearchSession, progress: ResearchProgress
    ) -> None:
        """
        Update the weekly targets section with current progress.

        Args:
            session: Research session with weekly targets
            progress: Current research progress
        """
        content = self._read_document()

        targets_markdown = self._generate_weekly_targets_markdown(session, progress)
        updated_content = self._replace_section(
            content, "weekly_targets", targets_markdown
        )

        self._write_document(updated_content)

    def create_tracking_document_template(self) -> None:
        """Create a template tracking document if it doesn't exist."""
        if self.tracking_document_path.exists():
            return

        template = """# Journal Dataset Research Targets

## Overview

This document tracks progress towards researching and acquiring therapeutic datasets from academic sources.

Last Updated: {timestamp}

## Progress Metrics

<!-- PROGRESS_METRICS_START -->
<!-- Progress metrics will be automatically updated here -->
<!-- PROGRESS_METRICS_END -->

## Weekly Targets

<!-- WEEKLY_TARGETS_START -->
<!-- Weekly targets and progress will be automatically updated here -->
<!-- WEEKLY_TARGETS_END -->

## Status Summary

<!-- STATUS_SUMMARY_START -->
<!-- Status summary will be automatically updated here -->
<!-- STATUS_SUMMARY_END -->

## Completed Tasks

<!-- COMPLETED_TASKS_START -->
<!-- Completed tasks will be automatically listed here -->
<!-- COMPLETED_TASKS_END -->

## Research Tasks

### Phase 1: Discovery
- [ ] Search PubMed Central for therapeutic datasets
- [ ] Investigate DOAJ psychology journals
- [ ] Search repository APIs (Dryad, Zenodo, ClinicalTrials.gov)

### Phase 2: Evaluation
- [ ] Evaluate dataset therapeutic relevance
- [ ] Assess data structure quality
- [ ] Evaluate training integration potential
- [ ] Assess ethical accessibility

### Phase 3: Acquisition
- [ ] Submit access requests
- [ ] Download available datasets
- [ ] Organize and store acquired datasets

### Phase 4: Integration
- [ ] Create integration plans
- [ ] Generate preprocessing scripts
- [ ] Integrate datasets into training pipeline

## Notes

Add research notes and findings here.
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        self._write_document(template)

    def _read_document(self) -> str:
        """Read the tracking document content."""
        if not self.tracking_document_path.exists():
            self.create_tracking_document_template()

        try:
            return self.tracking_document_path.read_text(encoding="utf-8")
        except IOError as e:
            raise IOError(f"Failed to read tracking document: {e}") from e

    def _write_document(self, content: str) -> None:
        """Write content to the tracking document."""
        # Update last updated timestamp
        timestamp_pattern = r"Last Updated: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        new_timestamp = f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        content = re.sub(timestamp_pattern, new_timestamp, content)

        try:
            self.tracking_document_path.write_text(content, encoding="utf-8")
        except IOError as e:
            raise IOError(f"Failed to write tracking document: {e}") from e

    def _replace_section(
        self, content: str, section_name: str, new_content: str
    ) -> str:
        """Replace a marked section in the document."""
        pattern = MARKER_PATTERNS.get(section_name)
        if not pattern:
            raise ValueError(f"Unknown section: {section_name}")

        # Map section names to their actual marker names
        marker_map = {
            "progress_section": "PROGRESS_METRICS",
            "completed_tasks": "COMPLETED_TASKS",
            "status_summary": "STATUS_SUMMARY",
            "weekly_targets": "WEEKLY_TARGETS",
        }
        marker_name = marker_map.get(section_name, section_name.upper())
        start_marker = f"<!-- {marker_name}_START -->"
        end_marker = f"<!-- {marker_name}_END -->"

        replacement = f"{start_marker}\n{new_content}\n{end_marker}"

        if re.search(pattern, content, re.DOTALL):
            return re.sub(pattern, replacement, content, flags=re.DOTALL)
        else:
            # Section doesn't exist, append it before the closing of the document
            # or add it to a suitable location
            return content + f"\n\n{replacement}\n"

    def _append_to_section(self, content: str, section_name: str, new_content: str) -> str:
        """Append content to a marked section."""
        pattern = MARKER_PATTERNS.get(section_name)
        if not pattern:
            raise ValueError(f"Unknown section: {section_name}")

        # Map section names to their actual marker names
        marker_map = {
            "progress_section": "PROGRESS_METRICS",
            "completed_tasks": "COMPLETED_TASKS",
            "status_summary": "STATUS_SUMMARY",
            "weekly_targets": "WEEKLY_TARGETS",
        }
        marker_name = marker_map.get(section_name, section_name.upper())
        start_marker = f"<!-- {marker_name}_START -->"
        end_marker = f"<!-- {marker_name}_END -->"

        match = re.search(pattern, content, re.DOTALL)
        if match:
            existing_content = match.group(1).strip()
            updated_content = existing_content + "\n" + new_content
            replacement = f"{start_marker}\n{updated_content}\n{end_marker}"
            return re.sub(pattern, replacement, content, flags=re.DOTALL)
        else:
            # Section doesn't exist, create it
            replacement = f"{start_marker}\n{new_content}\n{end_marker}"
            return content + f"\n\n{replacement}\n"

    def _generate_progress_markdown(
        self,
        progress: ResearchProgress,
        session: Optional[ResearchSession] = None,
    ) -> str:
        """Generate markdown for progress metrics section."""
        lines = [
            "### Current Progress",
            "",
            f"- **Sources Identified**: {progress.sources_identified}",
            f"- **Datasets Evaluated**: {progress.datasets_evaluated}",
            f"- **Access Established**: {progress.access_established}",
            f"- **Datasets Acquired**: {progress.datasets_acquired}",
            f"- **Integration Plans Created**: {progress.integration_plans_created}",
            "",
        ]

        if progress.last_updated:
            lines.append(
                f"**Last Updated**: {progress.last_updated.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        if session:
            lines.extend(["", "### Session Information", ""])
            lines.append(f"- **Session ID**: {session.session_id}")
            lines.append(f"- **Current Phase**: {session.current_phase.title()}")
            lines.append(
                f"- **Session Start**: {session.start_date.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        return "\n".join(lines)

    def _generate_status_summary_markdown(
        self,
        progress: ResearchProgress,
        session: Optional[ResearchSession] = None,
        weekly_report: Optional[WeeklyReport] = None,
    ) -> str:
        """Generate markdown for status summary section."""
        lines = ["### Research Status", ""]

        # Overall progress
        total_activities = (
            progress.sources_identified
            + progress.datasets_evaluated
            + progress.access_established
            + progress.datasets_acquired
            + progress.integration_plans_created
        )

        if total_activities > 0:
            lines.append(f"**Total Activities**: {total_activities}")
            lines.append("")
            lines.append("**Progress Breakdown**:")
            lines.append(
                f"- Discovery: {progress.sources_identified} sources identified"
            )
            lines.append(
                f"- Evaluation: {progress.datasets_evaluated} datasets evaluated"
            )
            lines.append(
                f"- Acquisition: {progress.access_established} access established, {progress.datasets_acquired} datasets acquired"
            )
            lines.append(
                f"- Integration: {progress.integration_plans_created} plans created"
            )
        else:
            lines.append("**Status**: Research has not yet started.")
            lines.append("")

        # Weekly report summary
        if weekly_report:
            lines.extend(["", "### Weekly Summary", ""])
            lines.append(f"**Week {weekly_report.week_number}** ({weekly_report.start_date.strftime('%Y-%m-%d')} to {weekly_report.end_date.strftime('%Y-%m-%d')})")
            lines.append("")

            if weekly_report.key_findings:
                lines.append("**Key Findings**:")
                for finding in weekly_report.key_findings:
                    lines.append(f"- {finding}")
                lines.append("")

            if weekly_report.challenges:
                lines.append("**Challenges**:")
                for challenge in weekly_report.challenges:
                    lines.append(f"- {challenge}")
                lines.append("")

            if weekly_report.next_week_priorities:
                lines.append("**Next Week Priorities**:")
                for priority in weekly_report.next_week_priorities:
                    lines.append(f"- {priority}")
                lines.append("")

        return "\n".join(lines)

    def _generate_weekly_targets_markdown(
        self, session: ResearchSession, progress: ResearchProgress
    ) -> str:
        """Generate markdown for weekly targets section."""
        if not session.weekly_targets:
            return "No weekly targets set."

        lines = ["### Weekly Targets Progress", ""]

        for target_key, target_value in session.weekly_targets.items():
            achieved = getattr(progress, target_key, 0)
            percentage = (achieved / target_value * 100) if target_value > 0 else 0
            status_icon = "✅" if achieved >= target_value else "🔄"

            lines.append(
                f"{status_icon} **{target_key.replace('_', ' ').title()}**: {achieved}/{target_value} ({percentage:.1f}%)"
            )

        lines.append("")

        # Overall progress
        total_achieved = sum(
            getattr(progress, key, 0) for key in session.weekly_targets.keys()
        )
        total_targets = sum(session.weekly_targets.values())
        overall_percentage = (
            (total_achieved / total_targets * 100) if total_targets > 0 else 0
        )

        lines.append(
            f"**Overall Progress**: {total_achieved}/{total_targets} ({overall_percentage:.1f}%)"
        )

        return "\n".join(lines)

