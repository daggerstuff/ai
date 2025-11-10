"""
Main CLI interface for journal dataset research system.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console

from ai.journal_dataset_research.cli.commands import CommandHandler
from ai.journal_dataset_research.cli.config import (
    get_config_value,
    load_config,
    save_config,
)
from ai.journal_dataset_research.cli.interactive import prompt_for_session_config

console = Console()


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option("--dry-run", is_flag=True, help="Run in dry-run mode (no actual changes)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-file", type=click.Path(path_type=Path), help="Log file path")
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], dry_run: bool, verbose: bool, log_file: Optional[Path]) -> None:
    """Journal Dataset Research System CLI."""
    ctx.ensure_object(dict)

    # Load configuration
    cfg = load_config(config)
    ctx.obj["config"] = cfg
    ctx.obj["dry_run"] = dry_run

    # Setup logging
    log_level = "DEBUG" if verbose else cfg.get("logging", {}).get("level", "INFO")
    setup_logging(log_level, log_file)

    if dry_run:
        console.print("[yellow]Running in DRY-RUN mode[/yellow]")


@cli.command()
@click.option("--keywords", "-k", multiple=True, help="Search keywords")
@click.option("--sources", "-s", multiple=True, help="Target sources (pubmed, doaj, etc.)")
@click.option("--session-id", help="Session ID (optional)")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.pass_context
def search(
    ctx: click.Context,
    keywords: tuple,
    sources: tuple,
    session_id: Optional[str],
    interactive: bool,
) -> None:
    """Search for dataset sources."""
    handler = CommandHandler(ctx.obj["config"], ctx.obj["dry_run"])

    if interactive and not keywords:
        # Prompt for configuration
        config = prompt_for_session_config()
        keywords_list = config["search_keywords"].get("therapeutic", [])
        sources_list = config["target_sources"]
        session_id = config.get("session_id") or session_id
    else:
        keywords_list = list(keywords) or ["therapy", "counseling", "psychotherapy"]
        sources_list = list(sources) or ["pubmed", "doaj"]

    result = handler.search(
        keywords=keywords_list,
        sources=sources_list,
        session_id=session_id,
        interactive=interactive,
    )

    console.print(f"\n[green]Search completed. Session ID: {result['session_id']}[/green]")
    console.print(f"Found {len(result['sources'])} sources")


@cli.command()
@click.option("--session-id", required=True, help="Session ID")
@click.option("--source-ids", multiple=True, help="Source IDs to evaluate (optional)")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.pass_context
def evaluate(
    ctx: click.Context,
    session_id: str,
    source_ids: tuple,
    interactive: bool,
) -> None:
    """Evaluate dataset sources."""
    handler = CommandHandler(ctx.obj["config"], ctx.obj["dry_run"])

    source_ids_list = list(source_ids) if source_ids else None

    result = handler.evaluate(
        session_id=session_id,
        source_ids=source_ids_list,
        interactive=interactive,
    )

    console.print(f"\n[green]Evaluation completed[/green]")
    console.print(f"Evaluated {len(result['evaluations'])} datasets")


@cli.command()
@click.option("--session-id", required=True, help="Session ID")
@click.option("--source-ids", multiple=True, help="Source IDs to acquire (optional)")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.pass_context
def acquire(
    ctx: click.Context,
    session_id: str,
    source_ids: tuple,
    interactive: bool,
) -> None:
    """Acquire datasets."""
    handler = CommandHandler(ctx.obj["config"], ctx.obj["dry_run"])

    source_ids_list = list(source_ids) if source_ids else None

    result = handler.acquire(
        session_id=session_id,
        source_ids=source_ids_list,
        interactive=interactive,
    )

    console.print(f"\n[green]Acquisition completed[/green]")
    console.print(f"Acquired {len(result['acquired'])} datasets")


@cli.command()
@click.option("--session-id", required=True, help="Session ID")
@click.option("--source-ids", multiple=True, help="Source IDs to integrate (optional)")
@click.option("--target-format", default="chatml", help="Target format for integration")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.pass_context
def integrate(
    ctx: click.Context,
    session_id: str,
    source_ids: tuple,
    target_format: str,
    interactive: bool,
) -> None:
    """Create integration plans for acquired datasets."""
    handler = CommandHandler(ctx.obj["config"], ctx.obj["dry_run"])

    source_ids_list = list(source_ids) if source_ids else None

    result = handler.integrate(
        session_id=session_id,
        source_ids=source_ids_list,
        target_format=target_format,
        interactive=interactive,
    )

    console.print(f"\n[green]Integration planning completed[/green]")
    console.print(f"Created {len(result['plans'])} integration plans")


@cli.command()
@click.option("--session-id", help="Session ID (optional, lists all if not provided)")
@click.pass_context
def status(ctx: click.Context, session_id: Optional[str]) -> None:
    """Get research session status."""
    handler = CommandHandler(ctx.obj["config"], ctx.obj["dry_run"])
    handler.status(session_id=session_id)


@cli.command()
@click.option("--session-id", required=True, help="Session ID")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file path")
@click.option("--format", "report_format", default="json", help="Report format (json, markdown)")
@click.pass_context
def report(
    ctx: click.Context,
    session_id: str,
    output: Optional[Path],
    report_format: str,
) -> None:
    """Generate research report."""
    handler = CommandHandler(ctx.obj["config"], ctx.obj["dry_run"])

    if not output:
        output = Path(f"report_{session_id}.{report_format}")

    result = handler.report(
        session_id=session_id,
        output_path=output,
        format=report_format,
    )

    console.print(f"\n[green]Report generated: {output}[/green]")


@cli.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command("show")
@click.option("--key", help="Configuration key (dot-separated path)")
@click.pass_context
def config_show(ctx: click.Context, key: Optional[str]) -> None:
    """Show configuration."""
    config_data = ctx.obj["config"]

    if key:
        value = get_config_value(key)
        console.print(f"{key}: {value}")
    else:
        import json
        console.print(json.dumps(config_data, indent=2, default=str))


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
    """Set configuration value."""
    config_data = ctx.obj["config"]

    # Try to parse value as appropriate type
    if value.lower() == "true":
        parsed_value = True
    elif value.lower() == "false":
        parsed_value = False
    elif value.isdigit():
        parsed_value = int(value)
    else:
        try:
            parsed_value = float(value)
        except ValueError:
            parsed_value = value

    # Set value
    keys = key.split(".")
    target = config_data
    for k in keys[:-1]:
        if k not in target:
            target[k] = {}
        target = target[k]
    target[keys[-1]] = parsed_value

    save_config(config_data)
    console.print(f"[green]Set {key} = {parsed_value}[/green]")


@config.command("get")
@click.argument("key")
@click.pass_context
def config_get(ctx: click.Context, key: str) -> None:
    """Get configuration value."""
    value = get_config_value(key)
    console.print(f"{key}: {value}")


if __name__ == "__main__":
    cli()

