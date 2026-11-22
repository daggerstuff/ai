"""Restore the redacted source instruction on the retained Reddit classification rows.

The v30 projection dropped the system instruction from the 21 retained
``reddit_mental_nlp`` rows. Their assistant targets are disorder labels, not
therapist replies. This tool builds a separately bound research-comparison
artifact: the same train corpus with the source instruction prepended to those
rows, plus a retokenized export that keeps every unrelated datum byte-exact.
It never edits the frozen release, the wave-repaired corpus, or any export.

Run:
  PYTHONPATH=ai python -m training.restore_classification_context apply SOURCE PLAN OUT
  PYTHONPATH=ai python -m training.restore_classification_context export EXPERIMENT OUT
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

import typer
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from training.apply_research_target_repairs import Payload, RepairInputError
from training.export_research_target_repairs import (
    Receipt,
    Retokenized,
    render,
)
from training.export_tinker_release import Datum, _sha256, build_tokens

Digest = str


class RestoreRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    row_number: int = Field(ge=1)
    row_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    selected_line: int = Field(ge=1)
    candidate_row: int = Field(ge=1)


class RestorePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    schema_version: Literal["classification-context-restore-v1"]
    source_train_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    system_instruction: str = Field(min_length=1)
    system_instruction_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    review_state: Literal["agent_research_not_clinically_adjudicated"]
    clinical_reviewed: Literal[False] = False
    rows: tuple[RestoreRow, ...] = Field(min_length=1)

    def checked_instruction(self) -> str:
        if hashlib.sha256(self.system_instruction.encode()).hexdigest() != self.system_instruction_sha256:
            raise RepairInputError("system instruction does not match its recorded sha256")
        return self.system_instruction


class Experiment(BaseModel):
    """Bound inputs for the restored-context research comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    original_train: Path
    restored_train: Path
    source_receipt: Path
    source_receipt_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    restore_plan: Path
    restore_plan_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    release_id: str = Field(min_length=1)


class Restored(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_sha256: str
    output_sha256: str
    rows: int
    changed_rows: int
    review_state: Literal["agent_research_not_clinically_adjudicated"]


def restore_messages(messages: list[dict], instruction: str) -> tuple[dict, ...]:
    """Prepend the source system instruction; refuse rows that already carry one."""
    if any(message.get("role") == "system" for message in messages):
        raise RepairInputError("row already carries a system message; refusing to double-restore")
    return ({"role": "system", "content": instruction, "weight": 0}, *messages)


def apply(source: Path, out: Path, plan: RestorePlan) -> Restored:
    """Publish a restored corpus only after every bound row checks out."""
    if out.exists():
        raise FileExistsError(out)
    instruction = plan.checked_instruction()
    repairs = {row.row_number: row for row in plan.rows}
    if len(repairs) != len(plan.rows):
        raise RepairInputError("duplicate row_number in restore plan")
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
                        row = repairs[row_count]
                        if hashlib.sha256(raw).hexdigest() != row.row_sha256:
                            raise RepairInputError(f"row hash mismatch at {row_count}")
                        payload = Payload.model_validate_json(raw)
                        messages = list(payload.messages)
                        restored = restore_messages(messages, instruction)
                        written = (
                            payload.model_copy(update={"messages": restored}).model_dump_json(exclude_unset=True) + "\n"
                        ).encode()
                        seen.add(row_count)
                    handle.write(written)
                    output_digest.update(written)
            if source_digest.hexdigest() != plan.source_train_sha256:
                raise RepairInputError("source train SHA-256 differs from the plan")
            if seen != set(repairs):
                missing = sorted(set(repairs) - seen)
                raise RepairInputError(f"planned rows absent from source: {missing}")
            handle.flush()
            # Hard-link publication refuses an existing output even if another
            # process created it after the initial check. Both paths share a disk.
            os.link(temporary, out)
        finally:
            temporary.unlink()
    return Restored(
        source_sha256=source_digest.hexdigest(),
        output_sha256=output_digest.hexdigest(),
        rows=row_count,
        changed_rows=len(seen),
        review_state=plan.review_state,
    )


def load_experiment(experiment: Experiment) -> tuple[RestorePlan, Receipt]:
    if experiment.source_receipt.exists() is False:
        raise RepairInputError("source export receipt is missing")
    if _sha256(experiment.source_receipt) != experiment.source_receipt_sha256:
        raise RepairInputError("source export receipt differs from the experiment")
    if _sha256(experiment.restore_plan) != experiment.restore_plan_sha256:
        raise RepairInputError("restore plan differs from the experiment")
    if not experiment.original_train.exists():
        raise RepairInputError("original train split is missing")
    if not experiment.restored_train.exists():
        raise RepairInputError("restored train split is missing")
    plan = RestorePlan.model_validate_json(experiment.restore_plan.read_bytes())
    receipt = Receipt.model_validate_json(experiment.source_receipt.read_bytes())
    if receipt.source_train_sha256 != plan.source_train_sha256:
        raise RepairInputError("the source export and the restore plan bind different training corpora")
    if _sha256(receipt.path) != receipt.sha256:
        raise RepairInputError("source export differs from its receipt")
    return plan, receipt


@lru_cache(maxsize=1)
def _tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name, fast=True, local_files_only=True)


def _datum_for(messages: list[dict], receipt: Receipt) -> Datum:
    plain = messages
    if plain and plain[0].get("role") == "assistant":
        plain = plain[1:]
    ids, weights = build_tokens(
        _tokenizer(receipt.tokenizer), plain, end_token=receipt.end_token, template_kwargs=receipt.template_kwargs
    )
    if len(ids) > receipt.max_tokens or not any(weights[1:]):
        raise RepairInputError("restored row would fall outside the export token budget")
    return Datum(input_ids=ids[:-1], target_ids=ids[1:], weights=weights[1:])


def retokenize(experiment: Experiment, plan: RestorePlan, receipt: Receipt) -> Retokenized:
    """Rebuild only the restored rows; carry every other datum over byte-exact."""
    repairs = {row.row_number: row for row in plan.rows}
    replacements: dict[str, bytes] = {}
    source_hash, restored_hash = hashlib.sha256(), hashlib.sha256()
    def line_count(path: Path) -> int:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)

    original_lines, restored_lines = line_count(experiment.original_train), line_count(experiment.restored_train)
    if original_lines != restored_lines:
        raise RepairInputError(f"restored corpus has {restored_lines} rows; the source has {original_lines}")
    with experiment.original_train.open("rb") as src, experiment.restored_train.open("rb") as restored_stream:
        for number, (original_raw, restored_raw) in enumerate(
            zip(src, restored_stream, strict=True), start=1
        ):
            source_hash.update(original_raw)
            restored_hash.update(restored_raw)
            if number not in repairs:
                if original_raw != restored_raw:
                    raise RepairInputError(f"restored corpus changes row {number} outside the plan")
                continue
            row = repairs[number]
            if hashlib.sha256(original_raw).hexdigest() != row.row_sha256:
                raise RepairInputError(f"source row differs from plan at {number}")
            original_payload = Payload.model_validate_json(original_raw)
            restored_payload = Payload.model_validate_json(restored_raw)
            if list(restored_payload.messages)[1:] != [dict(m) for m in original_payload.messages]:
                raise RepairInputError(f"restored row {number} changes more than the prepended system message")
            before = _datum_for(list(original_payload.messages), receipt)
            after = _datum_for(list(restored_payload.messages), receipt)
            old_hash = hashlib.sha256(before.line().encode()).hexdigest()
            if old_hash in replacements or before.line() == after.line():
                raise RepairInputError("ambiguous or token-identical restore")
            replacements[old_hash] = after.line().encode() + b"\n"
    if source_hash.hexdigest() != plan.source_train_sha256:
        raise RepairInputError("original train differs from the restore plan")
    if restored_hash.hexdigest() != _sha256(experiment.restored_train):
        raise RepairInputError("restored corpus hash drifted during verification")
    if len(replacements) != len(repairs):
        raise RepairInputError("one or more planned rows are absent from the source")
    return Retokenized(replacements=replacements, repaired_train_sha256=restored_hash.hexdigest())


def export(experiment_path: Path, out: Path):
    """Publish the restored-context export after every identity check passes."""
    if out.exists():
        raise FileExistsError(out)
    experiment = Experiment.model_validate_json(experiment_path.read_bytes())
    plan, receipt = load_experiment(experiment)
    repaired = retokenize(experiment, plan, receipt)
    rendered = render(receipt, repaired)
    path = out / "tinker-sft-train.jsonl"
    record = receipt.model_copy(
        update={
            "path": path.resolve(),
            "sha256": hashlib.sha256(rendered.payload).hexdigest(),
            "bytes": len(rendered.payload),
            "source_manifest": str(experiment_path.resolve()),
            "source_manifest_sha256": _sha256(experiment_path),
            "source_release_id": experiment.release_id,
            "source_train_sha256": repaired.repaired_train_sha256,
            "base_export_sha256": receipt.sha256,
            "restore_plan_sha256": experiment.restore_plan_sha256,
            "selection_mode": "exact_original_export_population",
            "changed_datums": len(repaired.replacements),
            "clinical_reviewed": False,
            "review_state": plan.review_state,
            "counts": {
                **receipt.counts,
                "total_tokens": rendered.total_tokens,
                "trained_tokens": rendered.trained_tokens,
            },
        }
    )
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    path.write_bytes(rendered.payload)
    path.chmod(0o600)
    (out / "tinker-dataset.json").write_text(record.model_dump_json(indent=2) + "\n")
    return record


def main() -> None:
    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.command("apply")
    def apply_cmd(source: Path, plan: Path, out: Path):
        """Write the restored corpus after hash verification."""
        specification = RestorePlan.model_validate_json(plan.read_bytes())
        result = apply(source, out, specification)
        receipt = {**result.model_dump(), "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest()}
        out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        typer.echo(f"{result.changed_rows} rows restored; {result.rows} rows preserved")

    @app.command("export")
    def export_cmd(experiment: Path, out: Path):
        """Retokenize the restored rows into a separately bound export."""
        receipt = export(experiment, out)
        typer.echo(f"{receipt.datums} datums retained in original order; {receipt.changed_datums} datums restored")

    app()


if __name__ == "__main__":
    main()
