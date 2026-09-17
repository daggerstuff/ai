"""Tests for the quadit dataset gate adapter."""

from __future__ import annotations

import pytest

from ai.research.quadit import (
    AuditItem,
    DatasetGateError,
    audit_dataset_records,
    gate_should_block,
    record_to_audit_item,
)
from ai.research.quadit.review import DEFAULT_JUDGES


def test_record_to_audit_item_common_fields() -> None:
    item = record_to_audit_item({"id": "r-1", "text": "hello", "author_role": "persona"})
    assert item.id == "r-1"
    assert item.content == "hello"
    assert item.author_role == "persona"
    assert item.kind == "dataset_record"


def test_record_to_audit_item_alternate_keys() -> None:
    item = record_to_audit_item({"record_id": "x9", "response": "I hear you.", "source": "eval-run"})
    assert item.id == "x9"
    assert item.content == "I hear you."
    assert item.context == {"source": "eval-run"}


def test_record_to_audit_item_synthesizes_id() -> None:
    item = record_to_audit_item({"content": "plain"}, position=7)
    assert item.id == "record-7"


def test_record_to_audit_item_rejects_missing_text() -> None:
    with pytest.raises(DatasetGateError, match="no non-empty text field"):
        record_to_audit_item({"id": "r"})


def test_record_to_audit_item_rejects_empty_text() -> None:
    with pytest.raises(DatasetGateError, match="no non-empty text field"):
        record_to_audit_item({"id": "r", "text": "   "})


def test_audit_dataset_records_clean_corpus_passes() -> None:
    records = [
        {"id": "a", "text": "We practiced reflective listening in session today."},
        {"id": "b", "text": "The care plan was updated after the supervision meeting."},
    ]
    n_records = 2
    report = audit_dataset_records(records)
    assert report.passed
    assert report.item_count == n_records
    assert not gate_should_block(report)


def test_audit_dataset_records_platitude_corpus_flags() -> None:
    records = [
        {"id": "good", "text": "We reviewed the session recording together."},
        {
            "id": "plat",
            "text": "I'm here for you. It's ok to not be ok. Treat yourself kindly.",
        },
    ]
    report = audit_dataset_records(records)
    brene = next(v for v in report.verdicts if v.persona == "Brené Brown")
    assert "plat" in brene.flagged_ids
    # Warning-only findings do not block the corpus.
    assert not gate_should_block(report)


def test_audit_dataset_records_banned_phrase_blocks() -> None:
    records = [
        {"id": "corp", "text": "Let's circle back on the treatment plan next sprint."},
    ]
    report = audit_dataset_records(records)
    assert gate_should_block(report)


def test_audit_dataset_records_rejects_malformed_record() -> None:
    records = [{"id": "ok", "text": "fine"}, {"id": "bad", "note": "no text"}]
    with pytest.raises(DatasetGateError):
        audit_dataset_records(records)


def test_audit_dataset_records_works_with_generators() -> None:
    records = ({"id": str(i), "text": "Session notes reviewed."} for i in range(3))
    n_records = 3
    report = audit_dataset_records(records)
    assert report.item_count == n_records
    assert report.passed


def test_gate_is_content_agnostic_with_core_items() -> None:
    """The dataset gate shares the core verdicts, not a separate rulebook."""
    records = [{"id": "r1", "text": "I understand this is a challenging time for you."}]
    report = audit_dataset_records(records)
    assert all(isinstance(v.score, float) for v in report.verdicts)
    assert any(v.persona == "Brené Brown" for v in report.verdicts)


def test_audit_dataset_records_explicit_auditors_override() -> None:
    records = [{"id": "r1", "text": "I understand this is a challenging time for you."}]
    report = audit_dataset_records(records, auditors=())
    assert all(v.persona != "Brené Brown" for v in report.verdicts)
    assert len(report.verdicts) == len(DEFAULT_JUDGES)


def test_record_roundtrip_matches_core_item() -> None:
    item = record_to_audit_item({"id": "r1", "text": "hello"})
    core = AuditItem(id="r1", kind="dataset_record", author_role="dataset-record", content="hello")
    assert item.model_dump() == core.model_dump()
