"""Retokenize explicit repairs inside a frozen training-export population.

Run: PYTHONPATH=ai uv run python -m training.export_research_target_repairs EXPERIMENT OUT
Every unchanged datum retains its exact bytes and position. No sampling runs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import typer
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer

from training.apply_research_target_repairs import Digest, Payload, Plan, RepairInputError, group_repairs, rewrite
from training.crisis_target_audit import Dataset
from training.export_tinker_release import Datum, _sha256, build_tokens


class Experiment(BaseModel):
    """Bound source artifacts for one controlled target-repair comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)
    original_train: Path
    repaired_train: Path
    source_receipt: Path
    source_receipt_sha256: Digest
    repair_plan: Path
    repair_plan_sha256: Digest
    release_id: str


class Receipt(Dataset):
    model_config = ConfigDict(frozen=True, extra="allow", hide_input_in_errors=True)
    datums: int
    source_train_sha256: Digest
    counts: dict[str, int]


class Retokenized(BaseModel):
    model_config = ConfigDict(frozen=True)
    replacements: dict[str, bytes]
    repaired_train_sha256: Digest


class Rendered(BaseModel):
    model_config = ConfigDict(frozen=True)
    payload: bytes
    total_tokens: int
    trained_tokens: int


def load_inputs(experiment: Experiment) -> tuple[Plan, Receipt]:
    """Bind the repair plan and source export before tokenization."""
    if _sha256(experiment.source_receipt) != experiment.source_receipt_sha256:
        raise RepairInputError("source export receipt differs from the experiment")
    if _sha256(experiment.repair_plan) != experiment.repair_plan_sha256:
        raise RepairInputError("repair plan differs from the experiment")
    plan = Plan.model_validate_json(experiment.repair_plan.read_bytes())
    receipt = Receipt.model_validate_json(experiment.source_receipt.read_bytes())
    if receipt.source_train_sha256 != plan.source_train_sha256:
        raise RepairInputError("the export and repair plan have different source corpora")
    if _sha256(receipt.path) != receipt.sha256:
        raise RepairInputError("source export differs from its receipt")
    return plan, receipt


def retokenize(experiment: Experiment, plan: Plan, receipt: Receipt) -> Retokenized:
    """Check the edited corpus and rebuild only the explicitly repaired rows."""
    repairs = group_repairs(plan).rows
    tokenizer = AutoTokenizer.from_pretrained(receipt.tokenizer, fast=True, local_files_only=True)
    replacements: dict[str, bytes] = {}
    source_hash, repaired_hash = hashlib.sha256(), hashlib.sha256()
    with experiment.original_train.open("rb") as stream:
        for number, raw in enumerate(stream, start=1):
            source_hash.update(raw)
            repaired = raw
            if number in repairs:
                original = Payload.model_validate_json(raw)
                changed = original
                for repair in repairs[number]:
                    if hashlib.sha256(raw).hexdigest() != repair.row_sha256:
                        raise RepairInputError(f"source row differs from plan at {number}")
                    changed = rewrite(changed, repair)
                repaired = (changed.model_dump_json(exclude_unset=True) + "\n").encode()
                datums: list[Datum] = []
                for payload in (original, changed):
                    messages = list(payload.messages)
                    if messages[0].get("role") == "assistant":
                        messages = messages[1:]
                    ids, weights = build_tokens(
                        tokenizer, messages, end_token=receipt.end_token, template_kwargs=receipt.template_kwargs
                    )
                    if len(ids) > receipt.max_tokens or not any(weights[1:]):
                        raise RepairInputError(f"repair would exclude a datum at row {number}")
                    datums.append(Datum(input_ids=ids[:-1], target_ids=ids[1:], weights=weights[1:]))
                before, after = datums
                old_hash = hashlib.sha256(before.line().encode()).hexdigest()
                if old_hash in replacements or before.line() == after.line():
                    raise RepairInputError("ambiguous or token-identical repair")
                replacements[old_hash] = after.line().encode() + b"\n"
            repaired_hash.update(repaired)
    if source_hash.hexdigest() != plan.source_train_sha256:
        raise RepairInputError("original source train SHA-256 differs from the plan")
    if repaired_hash.hexdigest() != _sha256(experiment.repaired_train):
        raise RepairInputError("repaired corpus has changes outside the explicit plan")
    if len(replacements) != len(repairs):
        raise RepairInputError("one or more planned source rows are absent")
    return Retokenized(replacements=replacements, repaired_train_sha256=repaired_hash.hexdigest())


def render(receipt: Receipt, repaired: Retokenized) -> Rendered:
    """Retain each original datum's position and bytes unless explicitly edited."""
    replacements = repaired.replacements
    original_lines = receipt.path.read_bytes().splitlines(keepends=True)
    occurrences: Counter[str] = Counter()
    rendered: list[bytes] = []
    for raw in original_lines:
        key = hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest()
        if key in replacements:
            occurrences[key] += 1
        rendered.append(replacements.get(key, raw))
    if len(original_lines) != receipt.datums or any(occurrences[key] != 1 for key in replacements):
        raise RepairInputError("every repaired source row must match exactly one existing export datum")
    total_tokens = trained_tokens = 0
    for raw in rendered:
        datum = json.loads(raw)
        total_tokens += len(datum["input_ids"])
        trained_tokens += sum(datum["weights"])
    return Rendered(payload=b"".join(rendered), total_tokens=total_tokens, trained_tokens=trained_tokens)


def export(experiment_path: Path, out: Path) -> Receipt:
    """Publish the controlled export after every identity and inclusion check."""
    if out.exists():
        raise FileExistsError(out)
    experiment = Experiment.model_validate_json(experiment_path.read_bytes())
    plan, receipt = load_inputs(experiment)
    repaired = retokenize(experiment, plan, receipt)
    rendered = render(receipt, repaired)
    payload = rendered.payload
    path = out / "tinker-sft-train.jsonl"
    record = receipt.model_copy(
        update={
            "path": path.resolve(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "source_manifest": str(experiment_path.resolve()),
            "source_manifest_sha256": _sha256(experiment_path),
            "source_release_id": experiment.release_id,
            "source_train_sha256": repaired.repaired_train_sha256,
            "base_export_sha256": receipt.sha256,
            "repair_plan_sha256": experiment.repair_plan_sha256,
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
    path.write_bytes(payload)
    path.chmod(0o600)
    (out / "tinker-dataset.json").write_text(record.model_dump_json(indent=2) + "\n")
    return record


def main(experiment: Path, out: Path) -> None:
    receipt = export(experiment, out)
    typer.echo(f"{receipt.datums} datums retained in original order; explicit repairs retokenized")


if __name__ == "__main__":
    typer.run(main)
