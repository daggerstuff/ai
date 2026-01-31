"""
Interactive mode for manual oversight and decisions.
"""

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()


def prompt_for_session_config() -> Dict[str, Any]:
    """Prompt user for research session configuration."""
    console.print("\n[bold blue]Research Session Configuration[/bold blue]\n")

    # Target sources
    console.print("Enter target sources (comma-separated):")
    console.print("Options: pubmed, doaj, dryad, zenodo, clinical_trials")
    sources_input = Prompt.ask("Target sources", default="pubmed,doaj")
    target_sources = [s.strip() for s in sources_input.split(",")]

    # Search keywords
    search_keywords: Dict[str, List[str]] = {}
    console.print("\nEnter search keywords for each category:")
    keywords_input = Prompt.ask(
        "Therapeutic keywords (comma-separated)",
        default="therapy,counseling,psychotherapy,mental health",
    )
    search_keywords["therapeutic"] = [k.strip() for k in keywords_input.split(",")]

    keywords_input = Prompt.ask(
        "Dataset keywords (comma-separated)",
        default="dataset,conversation,transcript,dialogue",
    )
    search_keywords["dataset"] = [k.strip() for k in keywords_input.split(",")]

    # Weekly targets
    weekly_targets: Dict[str, int] = {}
    console.print("\nEnter weekly targets (optional, press Enter to skip):")
    for metric in [
        "sources_identified",
        "datasets_evaluated",
        "datasets_acquired",
        "integration_plans_created",
    ]:
        target = Prompt.ask(f"{metric.replace('_', ' ').title()}", default="")
        if target and target.isdigit():
            weekly_targets[metric] = int(target)

    # Session ID
    session_id = Prompt.ask("Session ID (optional)", default="")

    return {
        "target_sources": target_sources,
        "search_keywords": search_keywords,
        "weekly_targets": weekly_targets,
        "session_id": session_id if session_id else None,
    }


def prompt_for_dataset_review(
    source_id: str, evaluation: Optional[Dict[str, Any]] = None
) -> bool:
    """Prompt user to review and approve a dataset."""
    console.print(f"\n[bold yellow]Reviewing Dataset: {source_id}[/bold yellow]\n")

    if evaluation:
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Notes", style="white")

        table.add_row(
            "Therapeutic Relevance",
            str(evaluation.get("therapeutic_relevance", "N/A")),
            evaluation.get("therapeutic_relevance_notes", "")[:50],
        )
        table.add_row(
            "Data Structure Quality",
            str(evaluation.get("data_structure_quality", "N/A")),
            evaluation.get("data_structure_notes", "")[:50],
        )
        table.add_row(
            "Training Integration",
            str(evaluation.get("training_integration", "N/A")),
            evaluation.get("integration_notes", "")[:50],
        )
        table.add_row(
            "Ethical Accessibility",
            str(evaluation.get("ethical_accessibility", "N/A")),
            evaluation.get("ethical_notes", "")[:50],
        )
        table.add_row(
            "Overall Score",
            str(evaluation.get("overall_score", "N/A")),
            evaluation.get("priority_tier", "N/A"),
        )

        console.print(table)

    return Confirm.ask("Approve this dataset for acquisition?", default=True)


def prompt_for_acquisition_approval(
    source_id: str, access_request: Optional[Dict[str, Any]] = None
) -> bool:
    """Prompt user to approve dataset acquisition."""
    console.print(f"\n[bold yellow]Acquisition Request: {source_id}[/bold yellow]\n")

    if access_request:
        console.print(f"Access Method: {access_request.get('access_method', 'N/A')}")
        console.print(f"Status: {access_request.get('status', 'N/A')}")
        if access_request.get("notes"):
            console.print(f"Notes: {access_request.get('notes')}")

    return Confirm.ask("Proceed with acquisition?", default=True)


def prompt_for_integration_approval(
    source_id: str, integration_plan: Optional[Dict[str, Any]] = None
) -> bool:
    """Prompt user to approve integration plan."""
    console.print(f"\n[bold yellow]Integration Plan: {source_id}[/bold yellow]\n")

    if integration_plan:
        console.print(f"Complexity: {integration_plan.get('complexity', 'N/A')}")
        console.print(
            f"Estimated Effort: {integration_plan.get('estimated_effort_hours', 'N/A')} hours"
        )
        if integration_plan.get("required_transformations"):
            console.print(
                f"Transformations: {', '.join(integration_plan.get('required_transformations', []))}"
            )

    return Confirm.ask("Approve this integration plan?", default=True)


def prompt_for_phase_transition(current_phase: str, next_phase: str) -> bool:
    """Prompt user to approve phase transition."""
    console.print(
        f"\n[bold cyan]Phase Transition: {current_phase} → {next_phase}[/bold cyan]\n"
    )
    return Confirm.ask(f"Proceed to {next_phase} phase?", default=True)


def prompt_for_manual_evaluation_override(
    source_id: str,
) -> Optional[Dict[str, Any]]:
    """Prompt user for manual evaluation override."""
    console.print(f"\n[bold yellow]Manual Evaluation Override: {source_id}[/bold yellow]\n")

    override = {}
    if Confirm.ask("Override therapeutic relevance score?", default=False):
        score = Prompt.ask("Therapeutic relevance (1-10)", default="5")
        if score.isdigit():
            override["therapeutic_relevance"] = int(score)
        notes = Prompt.ask("Notes (optional)", default="")
        if notes:
            override["therapeutic_relevance_notes"] = notes

    if Confirm.ask("Override data structure quality score?", default=False):
        score = Prompt.ask("Data structure quality (1-10)", default="5")
        if score.isdigit():
            override["data_structure_quality"] = int(score)
        notes = Prompt.ask("Notes (optional)", default="")
        if notes:
            override["data_structure_notes"] = notes

    if Confirm.ask("Override training integration score?", default=False):
        score = Prompt.ask("Training integration (1-10)", default="5")
        if score.isdigit():
            override["training_integration"] = int(score)
        notes = Prompt.ask("Notes (optional)", default="")
        if notes:
            override["integration_notes"] = notes

    if Confirm.ask("Override ethical accessibility score?", default=False):
        score = Prompt.ask("Ethical accessibility (1-10)", default="5")
        if score.isdigit():
            override["ethical_accessibility"] = int(score)
        notes = Prompt.ask("Notes (optional)", default="")
        if notes:
            override["ethical_notes"] = notes

    return override if override else None


def display_progress(
    session_id: str,
    current_phase: str,
    metrics: Dict[str, int],
    targets: Optional[Dict[str, int]] = None,
) -> None:
    """Display research progress in an interactive format."""
    console.print(f"\n[bold green]Session: {session_id}[/bold green]")
    console.print(f"[bold cyan]Current Phase: {current_phase}[/bold cyan]\n")

    table = Table(title="Progress Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Current", style="green")
    if targets:
        table.add_column("Target", style="yellow")
        table.add_column("Progress", style="magenta")

    for key, value in metrics.items():
        row = [key.replace("_", " ").title(), str(value)]
        if targets and key in targets:
            target = targets[key]
            progress = (value / target * 100) if target > 0 else 0
            row.extend([str(target), f"{progress:.1f}%"])
        else:
            if targets:
                row.extend(["N/A", "N/A"])
        table.add_row(*row)

    console.print(table)


def prompt_for_action(actions: List[str]) -> str:
    """Prompt user to select an action from a list."""
    console.print("\n[bold blue]Available Actions:[/bold blue]")
    for i, action in enumerate(actions, 1):
        console.print(f"  {i}. {action}")

    while True:
        choice = Prompt.ask("Select action (number)", default="1")
        if choice.isdigit() and 1 <= int(choice) <= len(actions):
            return actions[int(choice) - 1]
        console.print("[red]Invalid choice. Please try again.[/red]")

