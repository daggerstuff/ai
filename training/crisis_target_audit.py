"""Screen message payloads for crisis vocabulary followed by un-escalated targets.

Run: PYTHONPATH=ai uv run python -m training.crisis_target_audit INPUT REPORT

Findings are review candidates, never clinical labels or automatic exclusions.
Negation, history, medication use and third-party disclosures need review. The
report contains locations and hashes rather than conversation text.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Final, Literal

import typer
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer

from training.export_tinker_release import (
    MIN_DATUM_TOKENS,
    Datum,
    TokenLimits,
    trim_to_budget,
)
from training.strict_crisis_gate import ESCALATION_TERMS

GROUPS: Final = {
    "self_harm": ("cut myself", "cutting myself", "cut again", "self harm", "self-harm", "hurt myself", "razor"),
    "suicide": (
        "kill myself",
        "end my life",
        "want to die",
        "better off without me",
        "better off dead",
        "suicidal",
        "suicide",
        "not exist",
        "don't want to be here",
    ),
    "overdose": ("overdose", "overdosed", "took too many", "took a handful", "pills", "paracetamol", "insulin"),
    "means": ("rope", "noose", "ligature", "gun", "pistol", "firearm", "bridge", "strangled"),
    "psychosis": ("voices", "hearing things", "paranoid", "psychosis", "not real"),
    "medical": ("chest pain", "can't breathe", "cannot breathe", "seizure", "fainted", "passed out", "bleeding"),
}


def mentions(text: str, term: str) -> bool:
    """Match whole terms after whitespace and apostrophe normalization."""
    normalized = " ".join(text.casefold().replace("\u2019", "'").split())
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", normalized) is not None


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = ""
    weight: float = 0


class Conversation(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    messages: tuple[Message, ...]


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)
    line: int
    assistant_turn: int
    row_sha256: str
    groups: tuple[str, ...]


class Audit(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Literal["crisis-target-review-v1"] = "crisis-target-review-v1"
    scope: Literal["input message payloads before tokenization"] = "input message payloads before tokenization"
    input_sha256: str
    rows: int
    assistant_targets: int
    candidate_rows: int
    candidates_by_group: dict[str, int]
    findings: tuple[Finding, ...]


class Dataset(BaseModel):
    """Fields consumed from the exporter's tinker-dataset.json receipt."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    path: Path
    sha256: str
    tokenizer: str
    max_tokens: int
    end_token: str
    template_kwargs: dict[str, str | bool | int | float] = {}
    max_final_exchange_tokens: int | None = None


class ExportReview(BaseModel):
    """Candidate rows whose exact datums occur in the verified training file."""

    model_config = ConfigDict(frozen=True)
    dataset_sha256: str
    candidate_selected_rows: int
    present_selected_lines: tuple[int, ...]
    trimmed_target_lines: tuple[int, ...]
    excluded_by_length: int
    unmatched_selected_lines: tuple[int, ...]


def audit_export(source: Path, receipt: Path, report: Audit) -> ExportReview:
    """Tokenize candidate rows and match their exact bytes to a pinned export.

    This does not equate source selection with training exposure. Only rows
    whose complete datum matches are counted as present in the actual export.
    """
    dataset = Dataset.model_validate_json(receipt.read_bytes())
    payload = dataset.path.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != dataset.sha256:
        raise ExportIdentityError(dataset.path)
    exported = {hashlib.sha256(line).hexdigest() for line in payload.splitlines() if line.strip()}
    tokenizer = AutoTokenizer.from_pretrained(dataset.tokenizer, fast=True, local_files_only=True)
    wanted = {finding.line for finding in report.findings}
    findings_by_line: dict[int, list[Finding]] = {}
    for finding in report.findings:
        findings_by_line.setdefault(finding.line, []).append(finding)
    present: list[int] = []
    unmatched: list[int] = []
    trimmed_targets: list[int] = []
    excluded = 0
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if line_number not in wanted:
                continue
            conversation = Conversation.model_validate_json(raw)
            messages = [message.model_dump() for message in conversation.messages]
            dropped = 0
            if messages[0]["role"] == "assistant":
                messages = messages[1:]
                dropped = 1
            retained = len(messages)
            messages, ids, weights, trimmed = trim_to_budget(
                tokenizer,
                messages,
                dataset.max_tokens,
                end_token=dataset.end_token,
                template_kwargs=dataset.template_kwargs,
            )
            dropped += retained - len(messages)
            flagged = findings_by_line.get(line_number, [])
            if flagged and all(finding.assistant_turn < dropped for finding in flagged):
                trimmed_targets.append(line_number)
                continue
            targets = weights[1:]
            if len(ids) < MIN_DATUM_TOKENS or not any(targets):
                unmatched.append(line_number)
                continue
            limits = TokenLimits(
                max_tokens=dataset.max_tokens,
                final_exchange_tokens=dataset.max_final_exchange_tokens or dataset.max_tokens,
            )
            if len(ids) > limits.budget(trimmed=trimmed, message_count=len(messages)):
                excluded += 1
                continue
            datum = Datum(input_ids=ids[:-1], target_ids=ids[1:], weights=targets)
            if hashlib.sha256(datum.line().encode()).hexdigest() in exported:
                present.append(line_number)
            else:
                unmatched.append(line_number)
    return ExportReview(
        dataset_sha256=checksum,
        candidate_selected_rows=len(wanted),
        present_selected_lines=tuple(present),
        trimmed_target_lines=tuple(trimmed_targets),
        excluded_by_length=excluded,
        unmatched_selected_lines=tuple(unmatched),
    )


class ExportIdentityError(RuntimeError):
    """The training file differs from its receipt."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"export SHA-256 does not match receipt: {path}")


def audit(path: Path) -> Audit:
    """Pair each weighted assistant target with its preceding user turn.

    An escalation in another turn cannot license a later target. Tool results
    clear the pairing because tool continuations need their own audit contract.
    """
    digest = hashlib.sha256()
    rows = targets = 0
    findings: list[Finding] = []
    counts: Counter[str] = Counter()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if not raw.strip():
                continue
            conversation = Conversation.model_validate_json(raw)
            rows += 1
            user = ""
            for turn, message in enumerate(conversation.messages):
                content = message.content or ""
                if message.role == "user":
                    user = content
                if message.role in {"tool", "system"}:
                    user = ""
                if message.role != "assistant" or message.weight <= 0 or not content.strip():
                    continue
                targets += 1
                groups = tuple(name for name, terms in GROUPS.items() if any(mentions(user, term) for term in terms))
                if groups and not any(mentions(content, term) for term in ESCALATION_TERMS):
                    findings.append(
                        Finding(
                            line=line_number,
                            assistant_turn=turn,
                            row_sha256=hashlib.sha256(raw).hexdigest(),
                            groups=groups,
                        )
                    )
                    counts.update(groups)
    return Audit(
        input_sha256=digest.hexdigest(),
        rows=rows,
        assistant_targets=targets,
        candidate_rows=len({item.line for item in findings}),
        candidates_by_group=dict(counts),
        findings=tuple(findings),
    )


def main(source: Path, out: Path, dataset: Path | None = None) -> None:
    """Write a text-free review index; print only aggregate counts."""
    report = audit(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2) + "\n")
    typer.echo(f"{report.candidate_rows}/{report.rows} rows need review ({len(report.findings)} target turns)")
    if dataset is not None:
        presence = audit_export(source, dataset, report)
        out.with_suffix(".export.json").write_text(presence.model_dump_json(indent=2) + "\n")
        typer.echo(
            f"{len(presence.present_selected_lines)} candidate rows matched the exact export; "
            f"{len(presence.trimmed_target_lines)} lost their flagged reply to trimming; "
            f"{presence.excluded_by_length} exceed length; {len(presence.unmatched_selected_lines)} unmatched"
        )


if __name__ == "__main__":
    typer.run(main)
