"""Apply explicit, hash-bound assistant-target edits to a new research artifact.

Run: PYTHONPATH=ai uv run python -m training.apply_research_target_repairs SOURCE PLAN OUT
Frozen releases are never overwritten. The plan is an agent research judgment,
not clinical adjudication; this tool makes no inference or external requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import BaseModel, ConfigDict, Field, JsonValue

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Repair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)
    row_number: int = Field(ge=1)
    row_sha256: Digest
    assistant_turn: int = Field(ge=0)
    original_target_sha256: Digest
    replacement: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    sources: tuple[str, ...] = Field(min_length=1)


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)
    source_train_sha256: Digest
    review_state: Literal["agent_research_not_clinically_adjudicated"]
    clinical_reviewed: Literal[False] = False
    repairs: tuple[Repair, ...] = Field(min_length=1)


class Payload(BaseModel):
    """Preserve all JSON fields and original numeric weights in edited rows."""

    model_config = ConfigDict(frozen=True, extra="allow", hide_input_in_errors=True)
    messages: tuple[dict[str, JsonValue], ...]


class Applied(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_sha256: str
    output_sha256: str
    rows: int
    changed_rows: int
    changed_targets: int
    review_state: Literal["agent_research_not_clinically_adjudicated"]


class RepairInputError(RuntimeError):
    """The source or explicit target identity differs from the repair plan."""


class RepairGroups(BaseModel):
    model_config = ConfigDict(frozen=True)
    rows: dict[int, tuple[Repair, ...]]
    target_count: int


def group_repairs(plan: Plan) -> RepairGroups:
    """Reject duplicate target identities before either source or token edits."""
    rows: dict[int, list[Repair]] = {}
    identities: set[tuple[int, int]] = set()
    for repair in plan.repairs:
        identity = (repair.row_number, repair.assistant_turn)
        if identity in identities:
            raise RepairInputError(f"duplicate target repair: {identity}")
        identities.add(identity)
        rows.setdefault(repair.row_number, []).append(repair)
    return RepairGroups(rows={number: tuple(items) for number, items in rows.items()}, target_count=len(identities))


def rewrite(payload: Payload, repair: Repair) -> Payload:
    """Replace exactly the identified weighted assistant content."""
    if repair.assistant_turn >= len(payload.messages):
        raise RepairInputError(f"target index outside row {repair.row_number}")
    target = payload.messages[repair.assistant_turn]
    content = target.get("content")
    if target.get("role") != "assistant" or target.get("weight") != 1 or not isinstance(content, str):
        raise RepairInputError(f"target is not a weighted assistant turn in row {repair.row_number}")
    if hashlib.sha256(content.encode()).hexdigest() != repair.original_target_sha256:
        raise RepairInputError(f"target hash mismatch in row {repair.row_number}")
    if not repair.replacement.strip() or repair.replacement == content:
        raise RepairInputError(f"repair must change the target in row {repair.row_number}")
    messages = tuple(
        {**message, "content": repair.replacement} if index == repair.assistant_turn else message
        for index, message in enumerate(payload.messages)
    )
    return payload.model_copy(update={"messages": messages})


def apply(source: Path, out: Path, plan: Plan) -> Applied:
    """Publish a new artifact only after every row, target and file hash agrees."""
    if out.exists():
        raise FileExistsError(out)
    groups = group_repairs(plan)
    repairs = groups.rows
    source_digest, output_digest = hashlib.sha256(), hashlib.sha256()
    seen: set[int] = set()
    row_count = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with source.open("rb") as stream:
                for row_count, raw in enumerate(stream, start=1):
                    source_digest.update(raw)
                    written = raw
                    if row_count in repairs:
                        payload = Payload.model_validate_json(raw)
                        for repair in repairs[row_count]:
                            if hashlib.sha256(raw).hexdigest() != repair.row_sha256:
                                raise RepairInputError(f"row hash mismatch at {row_count}")
                            payload = rewrite(payload, repair)
                        written = (payload.model_dump_json(exclude_unset=True) + "\n").encode()
                        seen.add(row_count)
                    handle.write(written)
                    output_digest.update(written)
            if source_digest.hexdigest() != plan.source_train_sha256:
                raise RepairInputError("source train SHA-256 differs from the plan")
            if seen != set(repairs):
                raise RepairInputError("one or more planned source rows are missing")
            handle.flush()
            # Hard-link publication refuses an existing output even if another
            # process created it after the initial check. Both paths share a disk.
            os.link(temporary, out)
        finally:
            temporary.unlink()
    return Applied(
        source_sha256=source_digest.hexdigest(),
        output_sha256=output_digest.hexdigest(),
        rows=row_count,
        changed_rows=len(seen),
        changed_targets=groups.target_count,
        review_state=plan.review_state,
    )


def main(source: Path, plan: Path, out: Path) -> None:
    specification = Plan.model_validate_json(plan.read_bytes())
    result = apply(source, out, specification)
    receipt = {**result.model_dump(), "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest()}
    out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    typer.echo(f"{result.changed_targets} assistant targets edited; {result.rows} rows preserved")


if __name__ == "__main__":
    typer.run(main)
