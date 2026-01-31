#!/usr/bin/env python3
"""
Main execution script for journal dataset research system.

This script provides automated workflow execution with phase-by-phase execution,
checkpointing, resume capability, and dry-run mode.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ai.sourcing.journal.cli.commands import CommandHandler
from ai.sourcing.journal.cli.config import load_config
from ai.sourcing.journal.cli.interactive import (
    display_progress,
    prompt_for_phase_transition,
)
from ai.sourcing.journal.discovery import DiscoveryService
from ai.sourcing.journal.models.dataset_models import ResearchSession
from ai.sourcing.journal.orchestrator.research_orchestrator import (
    ResearchOrchestrator,
)
from ai.sourcing.journal.orchestrator.types import OrchestratorConfig

console = Console()
logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Executes the research workflow with checkpointing and resume capability."""

    PHASES = ["discovery", "evaluation", "acquisition", "integration"]

    def __init__(
        self,
        config: Optional[Dict] = None,
        dry_run: bool = False,
        interactive: bool = False,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Initialize workflow executor."""
        self.config = config or load_config()
        self.dry_run = dry_run
        self.interactive = interactive
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.command_handler = CommandHandler(self.config, dry_run)

    def execute_workflow(
        self,
        session_id: Optional[str] = None,
        target_sources: Optional[List[str]] = None,
        search_keywords: Optional[Dict[str, List[str]]] = None,
        weekly_targets: Optional[Dict[str, int]] = None,
        start_phase: str = "discovery",
        resume: bool = False,
    ) -> Dict:
        """Execute the complete research workflow."""
        console.print("[bold blue]Starting Research Workflow[/bold blue]\n")

        # Initialize orchestrator with discovery service
        orchestrator_config = OrchestratorConfig(**self.config.get("orchestrator", {}))
        discovery_service = DiscoveryService(config=self.config)
        orchestrator = ResearchOrchestrator(
            config=orchestrator_config,
            discovery_service=discovery_service,
        )

        # Load or create session
        if resume and session_id:
            try:
                console.print(f"[cyan]Resuming session: {session_id}[/cyan]")
                session = orchestrator.load_session_state(session_id)
                state = orchestrator.get_session_state(session_id)
                current_phase = session.current_phase
            except FileNotFoundError:
                console.print(f"[yellow]Session not found, creating new session[/yellow]")
                session = self._create_session(
                    orchestrator, session_id, target_sources, search_keywords, weekly_targets
                )
                current_phase = start_phase
        else:
            session = self._create_session(
                orchestrator, session_id, target_sources, search_keywords, weekly_targets
            )
            current_phase = start_phase

        session_id = session.session_id
        console.print(f"[green]Session ID: {session_id}[/green]\n")

        # Determine which phases to run
        phase_index = self.PHASES.index(current_phase) if current_phase in self.PHASES else 0
        phases_to_run = self.PHASES[phase_index:]

        # Execute phases
        for phase in phases_to_run:
            if self.interactive:
                if phase != phases_to_run[0]:
                    if not prompt_for_phase_transition(
                        session.current_phase, phase
                    ):
                        console.print("[yellow]Workflow paused by user[/yellow]")
                        break

            console.print(f"\n[bold cyan]Phase: {phase.upper()}[/bold cyan]\n")

            try:
                self._execute_phase(orchestrator, session_id, phase)
                self._save_checkpoint(orchestrator, session_id)

                # Display progress
                session = orchestrator.sessions[session_id]
                display_progress(
                    session_id,
                    session.current_phase,
                    session.progress_metrics,
                    session.weekly_targets,
                )

            except KeyboardInterrupt:
                console.print("\n[yellow]Workflow interrupted by user[/yellow]")
                self._save_checkpoint(orchestrator, session_id)
                sys.exit(1)
            except Exception as e:
                console.print(f"\n[red]Error in phase {phase}: {e}[/red]")
                logger.exception("Phase execution error")
                if not self.config.get("orchestrator", {}).get("fallback_on_failure", True):
                    raise
                console.print("[yellow]Continuing with next phase...[/yellow]")

        # Final status
        console.print("\n[bold green]Workflow completed![/bold green]\n")
        self.command_handler.status(session_id=session_id)

        return {
            "session_id": session_id,
            "status": "completed",
            "final_phase": session.current_phase,
        }

    def _create_session(
        self,
        orchestrator: ResearchOrchestrator,
        session_id: Optional[str],
        target_sources: Optional[List[str]],
        search_keywords: Optional[Dict[str, List[str]]],
        weekly_targets: Optional[Dict[str, int]],
    ) -> ResearchSession:
        """Create a new research session."""
        if not target_sources:
            target_sources = ["pubmed", "doaj"]
        if not search_keywords:
            search_keywords = {
                "therapeutic": ["therapy", "counseling", "psychotherapy"],
                "dataset": ["dataset", "conversation", "transcript"],
            }

        session = orchestrator.start_research_session(
            target_sources=target_sources,
            search_keywords=search_keywords,
            weekly_targets=weekly_targets,
            session_id=session_id,
        )

        return session

    def _execute_phase(
        self, orchestrator: ResearchOrchestrator, session_id: str, phase: str
    ) -> None:
        """Execute a single phase of the workflow."""
        session = orchestrator.sessions[session_id]
        state = orchestrator.get_session_state(session_id)

        if self.dry_run:
            console.print(f"[yellow]DRY RUN: Would execute {phase} phase[/yellow]")
            return

        if phase == "discovery":
            self._execute_discovery_phase(orchestrator, session_id, session, state)
        elif phase == "evaluation":
            self._execute_evaluation_phase(orchestrator, session_id, state)
        elif phase == "acquisition":
            self._execute_acquisition_phase(orchestrator, session_id, state)
        elif phase == "integration":
            self._execute_integration_phase(orchestrator, session_id, state)

    def _execute_discovery_phase(
        self, orchestrator: ResearchOrchestrator, session_id: str, session: ResearchSession, state
    ) -> None:
        """Execute discovery phase."""
        console.print("[cyan]Discovering dataset sources...[/cyan]")

        if orchestrator.discovery_service:
            sources = orchestrator.discovery_service.discover_sources(session)
            state.sources = sources
            orchestrator.update_progress(
                session_id, {"sources_identified": len(state.sources)}
            )
            console.print(f"[green]Found {len(sources)} sources[/green]")
        else:
            console.print("[yellow]No discovery service configured[/yellow]")

        orchestrator.advance_phase(session_id)

    def _execute_evaluation_phase(
        self, orchestrator: ResearchOrchestrator, session_id: str, state
    ) -> None:
        """Execute evaluation phase."""
        console.print("[cyan]Evaluating dataset sources...[/cyan]")

        if not state.sources:
            console.print("[yellow]No sources to evaluate[/yellow]")
            return

        if orchestrator.evaluation_engine:
            evaluations = []
            for source in state.sources:
                try:
                    evaluation = orchestrator.evaluation_engine.evaluate_dataset(
                        source, evaluator="system"
                    )
                    evaluations.append(evaluation)
                except Exception as e:
                    console.print(f"[red]Error evaluating {source.source_id}: {e}[/red]")
                    logger.exception("Evaluation error")

            state.evaluations.extend(evaluations)
            orchestrator.update_progress(
                session_id, {"datasets_evaluated": len(state.evaluations)}
            )
            console.print(f"[green]Evaluated {len(evaluations)} datasets[/green]")
        else:
            console.print("[yellow]No evaluation engine configured[/yellow]")

        orchestrator.advance_phase(session_id)

    def _execute_acquisition_phase(
        self, orchestrator: ResearchOrchestrator, session_id: str, state
    ) -> None:
        """Execute acquisition phase."""
        console.print("[cyan]Acquiring datasets...[/cyan]")

        if not state.sources:
            console.print("[yellow]No sources to acquire[/yellow]")
            return

        if orchestrator.acquisition_manager:
            acquired_count = 0
            for source in state.sources:
                try:
                    access_request = orchestrator.acquisition_manager.submit_access_request(
                        source
                    )
                    state.access_requests.append(access_request)

                    acquired_dataset = orchestrator.acquisition_manager.download_dataset(
                        source, access_request
                    )
                    state.acquired_datasets.append(acquired_dataset)
                    acquired_count += 1
                except Exception as e:
                    console.print(f"[red]Error acquiring {source.source_id}: {e}[/red]")
                    logger.exception("Acquisition error")

            orchestrator.update_progress(
                session_id, {"datasets_acquired": len(state.acquired_datasets)}
            )
            console.print(f"[green]Acquired {acquired_count} datasets[/green]")
        else:
            console.print("[yellow]No acquisition manager configured[/yellow]")

        orchestrator.advance_phase(session_id)

    def _execute_integration_phase(
        self, orchestrator: ResearchOrchestrator, session_id: str, state
    ) -> None:
        """Execute integration phase."""
        console.print("[cyan]Creating integration plans...[/cyan]")

        if not state.acquired_datasets:
            console.print("[yellow]No datasets to integrate[/yellow]")
            return

        target_format = self.config.get("integration", {}).get("target_format", "chatml")

        if orchestrator.integration_engine:
            plans_count = 0
            for dataset in state.acquired_datasets:
                try:
                    plan = orchestrator.integration_engine.create_integration_plan(
                        dataset, target_format
                    )
                    state.integration_plans.append(plan)
                    plans_count += 1
                except Exception as e:
                    console.print(f"[red]Error creating plan for {dataset.source_id}: {e}[/red]")
                    logger.exception("Integration planning error")

            orchestrator.update_progress(
                session_id, {"integration_plans_created": len(state.integration_plans)}
            )
            console.print(f"[green]Created {plans_count} integration plans[/green]")
        else:
            console.print("[yellow]No integration engine configured[/yellow]")

    def _save_checkpoint(self, orchestrator: ResearchOrchestrator, session_id: str) -> None:
        """Save workflow checkpoint."""
        try:
            checkpoint_path = orchestrator.save_session_state(
                session_id, directory=self.checkpoint_dir
            )
            console.print(f"[dim]Checkpoint saved: {checkpoint_path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save checkpoint: {e}[/yellow]")
            logger.warning("Checkpoint save failed", exc_info=True)


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Journal Dataset Research System - Main Execution Script"
    )
    parser.add_argument(
        "--session-id",
        help="Session ID (for resume or new session)",
    )
    parser.add_argument(
        "--target-sources",
        nargs="+",
        default=["pubmed", "doaj"],
        help="Target sources for discovery",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["therapy", "counseling", "psychotherapy"],
        help="Search keywords",
    )
    parser.add_argument(
        "--start-phase",
        choices=["discovery", "evaluation", "acquisition", "integration"],
        default="discovery",
        help="Phase to start from",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing session",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual changes)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode with manual approvals",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config = load_config(args.config)

    # Create workflow executor
    executor = WorkflowExecutor(
        config=config,
        dry_run=args.dry_run,
        interactive=args.interactive,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Prepare search keywords
    search_keywords = {
        "therapeutic": args.keywords,
        "dataset": args.keywords,
    }

    # Execute workflow
    try:
        result = executor.execute_workflow(
            session_id=args.session_id,
            target_sources=args.target_sources,
            search_keywords=search_keywords,
            start_phase=args.start_phase,
            resume=args.resume,
        )

        console.print(f"\n[bold green]Workflow completed successfully![/bold green]")
        console.print(f"Session ID: {result['session_id']}")

    except Exception as e:
        console.print(f"\n[bold red]Workflow failed: {e}[/bold red]")
        logger.exception("Workflow execution failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

