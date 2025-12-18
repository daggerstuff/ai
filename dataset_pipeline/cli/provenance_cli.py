#!/usr/bin/env python3
"""
Provenance CLI Commands

Command-line interface for managing dataset provenance metadata.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.dataset_pipeline.schemas.provenance_schema import (
    ProvenanceRecord,
)
from ai.dataset_pipeline.services.provenance_service import ProvenanceService

console = Console()
logger = logging.getLogger(__name__)


@click.group()
def provenance():
    """Provenance metadata management commands."""
    pass


@provenance.command()
@click.option("--dataset-id", required=True, help="Dataset ID")
@click.option(
    "--file", type=click.Path(exists=True), help="JSON file with provenance data"
)
@click.option("--changed-by", default="cli", help="User/system creating the record")
def create(dataset_id: str, file: Optional[str], changed_by: str):
    """Create a new provenance record."""
    asyncio.run(_create_provenance(dataset_id, file, changed_by))


async def _create_provenance(dataset_id: str, file: Optional[str], changed_by: str):
    """Create provenance record."""
    try:
        service = ProvenanceService()
        await service.connect()
        await service.ensure_schema()

        if file:
            # Load from file
            with open(file, "r") as f:
                data = json.load(f)
            provenance = ProvenanceRecord.from_dict(data)
        else:
            console.print(
                "[yellow]Creating minimal provenance record. Use --file for full record."
            )
            # Create minimal record (would need more params in real CLI)
            console.print("[red]Error: Full provenance creation requires --file option")
            return

        prov_id = await service.create_provenance(provenance, changed_by)
        console.print(f"[green]✅ Created provenance record: {prov_id}")

    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}")
        raise click.Abort()


@provenance.command()
@click.option("--dataset-id", required=True, help="Dataset ID")
def get(dataset_id: str):
    """Get provenance record by dataset ID."""
    asyncio.run(_get_provenance(dataset_id))


async def _get_provenance(dataset_id: str):
    """Get provenance record."""
    try:
        service = ProvenanceService()
        await service.connect()

        provenance = await service.get_provenance(dataset_id)

        if not provenance:
            console.print(f"[yellow]No provenance record found for: {dataset_id}")
            return

        # Display as JSON
        console.print_json(provenance.to_json())

    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}")
        raise click.Abort()


@provenance.command()
@click.option("--source-id", help="Filter by source ID")
@click.option("--license-type", help="Filter by license type")
@click.option("--quality-tier", help="Filter by quality tier")
@click.option("--limit", default=20, help="Maximum results")
def list(
    source_id: Optional[str],
    license_type: Optional[str],
    quality_tier: Optional[str],
    limit: int,
):
    """List provenance records with optional filters."""
    asyncio.run(_list_provenance(source_id, license_type, quality_tier, limit))


async def _list_provenance(
    source_id: Optional[str],
    license_type: Optional[str],
    quality_tier: Optional[str],
    limit: int,
):
    """List provenance records."""
    try:
        service = ProvenanceService()
        await service.connect()

        records = await service.query_provenance(
            source_id=source_id,
            license_type=license_type,
            quality_tier=quality_tier,
            limit=limit,
        )

        if not records:
            console.print("[yellow]No provenance records found")
            return

        # Create table
        table = Table(title="Provenance Records")
        table.add_column("Dataset ID", style="cyan")
        table.add_column("Dataset Name", style="green")
        table.add_column("Source", style="yellow")
        table.add_column("License", style="magenta")
        table.add_column("Quality Tier", style="blue")
        table.add_column("Created", style="white")

        for record in records:
            table.add_row(
                record.dataset_id,
                record.dataset_name[:40] + "..."
                if len(record.dataset_name) > 40
                else record.dataset_name,
                record.source.source_name[:30] + "..."
                if len(record.source.source_name) > 30
                else record.source.source_name,
                record.license.license_type.value,
                record.metadata.quality_tier.value
                if record.metadata.quality_tier
                else "N/A",
                record.timestamps.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}")
        raise click.Abort()


@provenance.command()
@click.option("--dataset-id", help="Filter by dataset ID")
@click.option("--provenance-id", help="Filter by provenance ID")
@click.option("--action", help="Filter by action type")
@click.option("--limit", default=50, help="Maximum results")
def audit_log(
    dataset_id: Optional[str],
    provenance_id: Optional[str],
    action: Optional[str],
    limit: int,
):
    """View audit log entries."""
    asyncio.run(_view_audit_log(dataset_id, provenance_id, action, limit))


async def _view_audit_log(
    dataset_id: Optional[str],
    provenance_id: Optional[str],
    action: Optional[str],
    limit: int,
):
    """View audit log."""
    try:
        service = ProvenanceService()
        await service.connect()

        entries = await service.get_audit_log(
            dataset_id=dataset_id,
            provenance_id=provenance_id,
            action=action,
            limit=limit,
        )

        if not entries:
            console.print("[yellow]No audit log entries found")
            return

        table = Table(title="Audit Log")
        table.add_column("Date", style="cyan")
        table.add_column("Dataset ID", style="green")
        table.add_column("Action", style="yellow")
        table.add_column("Changed By", style="magenta")
        table.add_column("Reason", style="white")

        for entry in entries:
            changed_at = entry.get("changed_at")
            if isinstance(changed_at, str):
                date_str = changed_at[:19]  # Extract date part
            else:
                date_str = str(changed_at)[:19]

            table.add_row(
                date_str,
                entry.get("dataset_id", "N/A")[:20],
                entry.get("action", "N/A"),
                entry.get("changed_by", "N/A")[:20],
                (entry.get("change_reason") or "N/A")[:40],
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}")
        raise click.Abort()


@provenance.command()
def init_schema():
    """Initialize database schema for provenance tables."""
    asyncio.run(_init_schema())


async def _init_schema():
    """Initialize schema."""
    try:
        service = ProvenanceService()
        await service.connect()
        await service.ensure_schema()
        console.print("[green]✅ Schema initialized successfully")

    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}")
        raise click.Abort()


if __name__ == "__main__":
    provenance()
