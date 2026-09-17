"""Quadit dataset gate — audit training-data records before they enter a corpus.

A thin adapter over :func:`run_quadit_audit` that accepts raw dataset
records (dicts) and fails the gate when the quadit flags critical findings
or any judge falls below the pass threshold. QA pipelines and corpus
builders call :func:`audit_dataset_records` (or the record-normalizing
:func:`audit_dataset_records`) and refuse records/corpora that do not pass.

Deterministic mode is the default for CI gates: it is a pattern scan, no
LLM required. LLM mode reuses the same judge prompts as the pe adapter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ai.research.quadit.models import AuditItem, QuadAuditReport
from ai.research.quadit.personas import AuditorDescriptor, load_auditor_descriptor
from ai.research.quadit.review import QuaditLLMClient, run_quadit_audit

DEFAULT_AUDITORS: tuple[str, ...] = ("brene_brown",)

_TEXT_KEYS: tuple[str, ...] = (
    "text",
    "content",
    "response",
    "completion",
    "assistant_response",
    "output",
)
_ID_KEYS: tuple[str, ...] = ("id", "record_id", "uuid", "sample_id", "index")


class DatasetGateError(ValueError):
    """A dataset record cannot be converted to an audit item."""


def _build_item(
    record: Mapping[str, Any],
    position: int,
    text: str,
    kind: str,
    excluded: tuple[str, ...],
) -> AuditItem:
    """Shared ``AuditItem`` assembly (id / author_role / context) for the record adapters."""
    item_id: str | None = None
    for key in _ID_KEYS:
        value = record.get(key)
        if value is not None:
            item_id = str(value)
            break
    if item_id is None:
        item_id = f"record-{position}"

    author_role = record.get("author_role", "dataset-record")
    if not isinstance(author_role, str):
        author_role = "dataset-record"

    context: dict[str, str] = {str(k): str(v) for k, v in record.items() if k not in excluded and v is not None}

    return AuditItem(
        id=item_id,
        kind=kind,
        author_role=author_role,
        content=text,
        context=context,
    )


def record_to_audit_item(record: Mapping[str, Any], position: int = 0) -> AuditItem:
    """Convert one raw dataset record to an ``AuditItem``.

    Recognizes common text/id field names; the first present key wins.
    Raises :class:`DatasetGateError` when no text field is found or the
    text is empty — a gate should fail loudly on malformed records, not
    skip them.
    """
    text: str | None = None
    for key in _TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            text = value
            break
    if text is None:
        raise DatasetGateError(
            f"Record at position {position} has no non-empty text field (looked for: {', '.join(_TEXT_KEYS)})."
        )

    return _build_item(
        record,
        position,
        text,
        str(record.get("kind", "dataset_record")),
        _TEXT_KEYS + _ID_KEYS,
    )


def chatml_to_audit_item(record: Mapping[str, Any], position: int = 0) -> AuditItem:
    """Convert one ChatML dialogue record (``{"messages": [{"role", "content"}, ...]}``).

    The full dialogue is serialized as ``role: content`` lines so judges also
    see the client turns — a banned phrase or PHI leak in a user turn poisons
    the training signal just as much as one in the assistant turn. Raises
    :class:`DatasetGateError` when no usable dialogue turns exist.
    """
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise DatasetGateError(f"Record at position {position} has no 'messages' list.")
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role and content:
            lines.append(f"{role}: {content}")
    if not lines:
        raise DatasetGateError(f"Record at position {position} has no non-empty messages.")

    return _build_item(
        record,
        position,
        "\n".join(lines),
        str(record.get("kind", "chatml_record")),
        _TEXT_KEYS + _ID_KEYS + ("messages",),
    )


def audit_dataset_records(
    records: Iterable[Mapping[str, Any]],
    *,
    client: QuaditLLMClient | None = None,
    auditors: tuple[str, ...] | None = None,
) -> QuadAuditReport:
    """Run the quadit over raw dataset records.

    ``records`` is any iterable of mappings — JSONL rows, HF dataset dicts,
    or pipeline records. Deterministic mode (``client=None``) by design;
    pass a client for full LLM judging.
    """
    resolved: tuple[str, ...] = DEFAULT_AUDITORS if auditors is None else auditors
    loaded: tuple[AuditorDescriptor, ...] = tuple(load_auditor_descriptor(name) for name in resolved)
    items = [record_to_audit_item(record, position=position) for position, record in enumerate(records)]
    return run_quadit_audit(items, client=client, auditors=loaded)


def gate_should_block(report: QuadAuditReport) -> bool:
    """True when the dataset (or record batch) must not enter the corpus."""
    return not report.passed
