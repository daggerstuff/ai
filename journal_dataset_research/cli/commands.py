"""
Command implementations for the CLI.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from ai.journal_dataset_research.cli.config import load_config
from ai.journal_dataset_research.cli.interactive import (
    display_progress,
    prompt_for_acquisition_approval,
    prompt_for_dataset_review,
    prompt_for_integration_approval,
    prompt_for_manual_evaluation_override,
)
from ai.journal_dataset_research.discovery import DiscoveryService
from ai.journal_dataset_research.models.dataset_models import (
    DatasetEvaluation,
    DatasetSource,
)
from ai.journal_dataset_research.orchestrator.research_orchestrator import (
    ResearchOrchestrator,
)
from ai.journal_dataset_research.orchestrator.types import OrchestratorConfig

console = Console()
logger = logging.getLogger(__name__)


class CommandHandler:
    """Handles CLI commands for the research system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, dry_run: bool = False):
        """Initialize command handler with configuration."""
        self.config = config or load_config()
        self.dry_run = dry_run
        self.orchestrator: Optional[ResearchOrchestrator] = None

    def _get_orchestrator(self) -> ResearchOrchestrator:
        """Get or create orchestrator instance."""
        if self.orchestrator is None:
            orchestrator_config = OrchestratorConfig(
                **self.config.get("orchestrator", {})
            )
            # Initialize discovery service
            discovery_service = DiscoveryService(config=self.config)
            self.orchestrator = ResearchOrchestrator(
                config=orchestrator_config,
                discovery_service=discovery_service,
            )
        return self.orchestrator

    def search(
        self,
        keywords: List[str],
        sources: List[str],
        session_id: Optional[str] = None,
        interactive: bool = False,
    ) -> Dict[str, Any]:
        """Search for dataset sources."""
        console.print("[bold blue]Searching for dataset sources...[/bold blue]\n")

        if self.dry_run:
            console.print("[yellow]DRY RUN: Would search for sources[/yellow]")
            return {"sources": [], "session_id": session_id or "dry-run-session"}

        orchestrator = self._get_orchestrator()

        # Create search keywords dict
        search_keywords = {
            "therapeutic": keywords,
            "dataset": keywords,  # Use same keywords for dataset search
        }

        # Create session if needed
        if not session_id:
            session = orchestrator.start_research_session(
                target_sources=sources,
                search_keywords=search_keywords,
            )
            session_id = session.session_id
        else:
            try:
                orchestrator.load_session_state(session_id)
            except FileNotFoundError:
                session = orchestrator.start_research_session(
                    target_sources=sources,
                    search_keywords=search_keywords,
                    session_id=session_id,
                )
                session_id = session.session_id

        # Run discovery phase
        session = orchestrator.sessions[session_id]
        state = orchestrator.get_session_state(session_id)

        if orchestrator.discovery_service:
            try:
                sources_list = orchestrator.discovery_service.discover_sources(session)
                state.sources = sources_list
                orchestrator.update_progress(
                    session_id, {"sources_identified": len(state.sources)}
                )
                console.print(
                    f"[green]Found {len(sources_list)} dataset sources[/green]"
                )
            except Exception as e:
                console.print(f"[red]Error during discovery: {e}[/red]")
                logger.exception("Discovery error")
                sources_list = []
        else:
            console.print("[yellow]Warning: No discovery service configured[/yellow]")
            sources_list = []

        # Save session
        orchestrator.save_session_state(session_id)

        return {
            "sources": [self._source_to_dict(s) for s in sources_list],
            "session_id": session_id,
        }

    def evaluate(
        self,
        session_id: Optional[str] = None,
        source_id: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
        interactive: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate dataset sources."""
        console.print("[bold blue]Evaluating dataset sources...[/bold blue]\n")

        # Handle single source_id parameter
        if source_id and not source_ids:
            source_ids = [source_id]

        # If no session_id provided, create a default one
        if not session_id:
            session_id = "default-session"

        if self.dry_run:
            console.print("[yellow]DRY RUN: Would evaluate sources[/yellow]")
            return {"evaluations": [], "session_id": session_id}

        orchestrator = self._get_orchestrator()
        try:
            orchestrator.load_session_state(session_id)
        except FileNotFoundError:
            # Create a new session if it doesn't exist
            session = orchestrator.start_research_session(
                target_sources=[],
                search_keywords={},
                session_id=session_id,
            )
            session_id = session.session_id

        state = orchestrator.get_session_state(session_id)
        sources_to_evaluate = (
            [s for s in state.sources if s.source_id in source_ids]
            if source_ids
            else state.sources
        )

        if not sources_to_evaluate:
            console.print("[yellow]No sources to evaluate[/yellow]")
            return {"evaluations": [], "session_id": session_id}

        evaluations: List[DatasetEvaluation] = []

        if orchestrator.evaluation_engine:
            for source in sources_to_evaluate:
                if interactive:
                    # Show source info
                    console.print(f"\n[cyan]Evaluating: {source.title}[/cyan]")
                    if not prompt_for_dataset_review(source.source_id):
                        continue

                try:
                    evaluation = orchestrator.evaluation_engine.evaluate_dataset(
                        source, evaluator="system"
                    )

                    if interactive:
                        # Allow manual override
                        override = prompt_for_manual_evaluation_override(source.source_id)
                        if override:
                            for key, value in override.items():
                                setattr(evaluation, key, value)
                            # Recalculate overall score if needed
                            evaluation.overall_score = (
                                evaluation.therapeutic_relevance * 0.35
                                + evaluation.data_structure_quality * 0.25
                                + evaluation.training_integration * 0.20
                                + evaluation.ethical_accessibility * 0.20
                            )

                    evaluations.append(evaluation)
                    console.print(f"[green]Evaluated: {source.source_id}[/green]")
                except Exception as e:
                    console.print(
                        f"[red]Error evaluating {source.source_id}: {e}[/red]"
                    )
                    logger.exception("Evaluation error")
        else:
            console.print("[yellow]Warning: No evaluation engine configured[/yellow]")

        state.evaluations.extend(evaluations)
        orchestrator.update_progress(
            session_id, {"datasets_evaluated": len(state.evaluations)}
        )
        orchestrator.save_session_state(session_id)

        return {
            "evaluations": [self._evaluation_to_dict(e) for e in evaluations],
            "session_id": session_id,
        }

    def acquire(
        self,
        session_id: Optional[str] = None,
        source_id: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
        interactive: bool = False,
    ) -> Dict[str, Any]:
        """Acquire datasets."""
        console.print("[bold blue]Acquiring datasets...[/bold blue]\n")

        # Handle single source_id parameter
        if source_id and not source_ids:
            source_ids = [source_id]

        # If no session_id provided, create a default one
        if not session_id:
            session_id = "default-session"

        if self.dry_run:
            console.print("[yellow]DRY RUN: Would acquire datasets[/yellow]")
            return {"acquired": [], "session_id": session_id}

        orchestrator = self._get_orchestrator()
        try:
            orchestrator.load_session_state(session_id)
        except FileNotFoundError:
            # Create a new session if it doesn't exist
            session = orchestrator.start_research_session(
                target_sources=[],
                search_keywords={},
                session_id=session_id,
            )
            session_id = session.session_id

        # Ensure session_id is a string (type narrowing)
        assert session_id is not None, "session_id must be set"
        state = orchestrator.get_session_state(session_id)

        # Filter sources if source_ids provided
        sources_to_acquire = (
            [s for s in state.sources if s.source_id in source_ids]
            if source_ids
            else state.sources
        )

        if not sources_to_acquire:
            console.print("[yellow]No sources to acquire[/yellow]")
            return {"acquired": [], "session_id": session_id}

        acquired_count = 0

        if orchestrator.acquisition_manager:
            for source in sources_to_acquire:
                if interactive:
                    if not prompt_for_acquisition_approval(source.source_id):
                        continue

                try:
                    # Submit access request
                    access_request = orchestrator.acquisition_manager.submit_access_request(
                        source
                    )
                    state.access_requests.append(access_request)

                    # Download dataset
                    acquired_dataset = orchestrator.acquisition_manager.download_dataset(
                        source, access_request
                    )
                    state.acquired_datasets.append(acquired_dataset)
                    acquired_count += 1

                    console.print(
                        f"[green]Acquired: {source.source_id}[/green]"
                    )
                except Exception as e:
                    console.print(
                        f"[red]Error acquiring {source.source_id}: {e}[/red]"
                    )
                    logger.exception("Acquisition error")
        else:
            console.print(
                "[yellow]Warning: No acquisition manager configured[/yellow]"
            )

        orchestrator.update_progress(
            session_id, {"datasets_acquired": len(state.acquired_datasets)}
        )
        orchestrator.save_session_state(session_id)

        return {
            "acquired": [d.source_id for d in state.acquired_datasets],
            "session_id": session_id,
        }

    def integrate(
        self,
        session_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
        target_format: str = "chatml",
        interactive: bool = False,
    ) -> Dict[str, Any]:
        """Create integration plans for acquired datasets."""
        console.print("[bold blue]Creating integration plans...[/bold blue]\n")

        # Handle dataset_id parameter (alias for source_ids)
        if dataset_id and not source_ids:
            source_ids = [dataset_id]

        # If no session_id provided, create a default one
        if not session_id:
            session_id = "default-session"

        if self.dry_run:
            console.print("[yellow]DRY RUN: Would create integration plans[/yellow]")
            return {"plans": [], "session_id": session_id}

        orchestrator = self._get_orchestrator()
        try:
            orchestrator.load_session_state(session_id)
        except FileNotFoundError:
            # Create a new session if it doesn't exist
            session = orchestrator.start_research_session(
                target_sources=[],
                search_keywords={},
                session_id=session_id,
            )
            session_id = session.session_id

        # Ensure session_id is a string (type narrowing)
        assert session_id is not None, "session_id must be set"
        state = orchestrator.get_session_state(session_id)

        # Filter datasets if source_ids provided
        datasets_to_integrate = (
            [d for d in state.acquired_datasets if d.source_id in source_ids]
            if source_ids
            else state.acquired_datasets
        )

        if not datasets_to_integrate:
            console.print("[yellow]No datasets to integrate[/yellow]")
            return {"plans": [], "session_id": session_id}

        plans_count = 0

        if orchestrator.integration_engine:
            for dataset in datasets_to_integrate:
                if interactive:
                    # Show dataset info
                    console.print(f"\n[cyan]Creating plan for: {dataset.source_id}[/cyan]")

                try:
                    plan = orchestrator.integration_engine.create_integration_plan(
                        dataset, target_format
                    )

                    if interactive:
                        if not prompt_for_integration_approval(
                            dataset.source_id, self._plan_to_dict(plan)
                        ):
                            continue

                    state.integration_plans.append(plan)
                    plans_count += 1

                    console.print(
                        f"[green]Created plan: {dataset.source_id}[/green]"
                    )
                except Exception as e:
                    console.print(
                        f"[red]Error creating plan for {dataset.source_id}: {e}[/red]"
                    )
                    logger.exception("Integration planning error")
        else:
            console.print(
                "[yellow]Warning: No integration engine configured[/yellow]"
            )

        orchestrator.update_progress(
            session_id, {"integration_plans_created": len(state.integration_plans)}
        )
        orchestrator.save_session_state(session_id)

        return {
            "plans": [p.source_id for p in state.integration_plans],
            "session_id": session_id,
        }

    def status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get research session status."""
        orchestrator = self._get_orchestrator()

        if session_id:
            try:
                orchestrator.load_session_state(session_id)
                session = orchestrator.sessions[session_id]
                state = orchestrator.get_session_state(session_id)
                progress = orchestrator.progress_states[session_id]

                # Display status
                display_progress(
                    session_id,
                    session.current_phase,
                    session.progress_metrics,
                    session.weekly_targets,
                )

                # Show state summary
                table = Table(title="Session State")
                table.add_column("Item", style="cyan")
                table.add_column("Count", style="green")

                table.add_row("Sources", str(len(state.sources)))
                table.add_row("Evaluations", str(len(state.evaluations)))
                table.add_row("Access Requests", str(len(state.access_requests)))
                table.add_row("Acquired Datasets", str(len(state.acquired_datasets)))
                table.add_row("Integration Plans", str(len(state.integration_plans)))

                console.print(table)

                return {
                    "session_id": session_id,
                    "phase": session.current_phase,
                    "metrics": session.progress_metrics,
                    "state": {
                        "sources": len(state.sources),
                        "evaluations": len(state.evaluations),
                        "access_requests": len(state.access_requests),
                        "acquired_datasets": len(state.acquired_datasets),
                        "integration_plans": len(state.integration_plans),
                    },
                }
            except FileNotFoundError:
                console.print(f"[red]Session not found: {session_id}[/red]")
                return {}
        else:
            # List all sessions
            sessions_dir = Path(orchestrator._session_storage_path)
            if sessions_dir.exists():
                session_files = list(sessions_dir.glob("*.json"))
                console.print(f"\n[bold blue]Found {len(session_files)} sessions:[/bold blue]\n")

                table = Table(title="Research Sessions")
                table.add_column("Session ID", style="cyan")
                table.add_column("File", style="white")

                for session_file in session_files:
                    session_id_from_file = session_file.stem
                    table.add_row(session_id_from_file, str(session_file))

                console.print(table)

                return {"sessions": [f.stem for f in session_files]}
            else:
                console.print("[yellow]No sessions found[/yellow]")
                return {"sessions": []}

    def report(
        self,
        session_id: str,
        output_path: Optional[Path] = None,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Generate research report."""
        console.print("[bold blue]Generating report...[/bold blue]\n")

        orchestrator = self._get_orchestrator()
        orchestrator.load_session_state(session_id)

        session = orchestrator.sessions[session_id]
        state = orchestrator.get_session_state(session_id)

        # Generate report data
        report_data = {
            "session_id": session_id,
            "start_date": session.start_date.isoformat(),
            "current_phase": session.current_phase,
            "progress_metrics": session.progress_metrics,
            "weekly_targets": session.weekly_targets,
            "sources": [self._source_to_dict(s) for s in state.sources],
            "evaluations": [
                self._evaluation_to_dict(e) for e in state.evaluations
            ],
            "acquired_datasets": [
                {
                    "source_id": d.source_id,
                    "acquisition_date": d.acquisition_date.isoformat(),
                    "storage_path": d.storage_path,
                    "file_size_mb": d.file_size_mb,
                }
                for d in state.acquired_datasets
            ],
            "integration_plans": [
                self._plan_to_dict(p) for p in state.integration_plans
            ],
        }

        # Save report
        if output_path:
            if format == "json":
                with open(output_path, "w") as f:
                    json.dump(report_data, f, indent=2)
                console.print(f"[green]Report saved to {output_path}[/green]")
            else:
                # For other formats, you could use the report generator
                console.print(f"[yellow]Format {format} not yet implemented[/yellow]")

        return report_data

    def _source_to_dict(self, source: DatasetSource) -> Dict[str, Any]:
        """Convert DatasetSource to dictionary."""
        return {
            "source_id": source.source_id,
            "title": source.title,
            "authors": source.authors,
            "publication_date": source.publication_date.isoformat(),
            "source_type": source.source_type,
            "url": source.url,
            "doi": source.doi,
            "open_access": source.open_access,
            "data_availability": source.data_availability,
        }

    def _evaluation_to_dict(self, evaluation: DatasetEvaluation) -> Dict[str, Any]:
        """Convert DatasetEvaluation to dictionary."""
        return {
            "source_id": evaluation.source_id,
            "therapeutic_relevance": evaluation.therapeutic_relevance,
            "data_structure_quality": evaluation.data_structure_quality,
            "training_integration": evaluation.training_integration,
            "ethical_accessibility": evaluation.ethical_accessibility,
            "overall_score": evaluation.overall_score,
            "priority_tier": evaluation.priority_tier,
            "evaluation_date": evaluation.evaluation_date.isoformat(),
        }

    def _plan_to_dict(self, plan: Any) -> Dict[str, Any]:
        """Convert IntegrationPlan to dictionary."""
        return {
            "source_id": plan.source_id,
            "dataset_format": plan.dataset_format,
            "complexity": plan.complexity,
            "estimated_effort_hours": plan.estimated_effort_hours,
            "required_transformations": plan.required_transformations,
        }

