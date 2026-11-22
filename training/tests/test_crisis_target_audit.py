"""Review candidates must preserve turn alignment and avoid substring hits."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from transformers import AutoTokenizer

from training.crisis_target_audit import ExportIdentityError, audit, audit_export
from training.export_tinker_release import Datum, build_tokens
from training.tests.test_export_tinker_release import StubTokenizer


@pytest.mark.parametrize(
    ("disclosure", "reply", "flagged"),
    [
        ("I want to die.", "What makes you feel that way?", True),
        ("I want to die.", "Call a crisis line now.", False),
        ("I fluffed the pillow properly.", "Tell me more.", False),
        ("I don\u2019t want to be here.", "Tell me more.", True),
    ],
)
def test_audit_pairs_disclosures_with_targets(tmp_path: Path, disclosure: str, reply: str, flagged: bool) -> None:
    source = tmp_path / "train.jsonl"
    raw = (
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": disclosure},
                    {"role": "assistant", "weight": 1, "content": reply},
                ]
            }
        )
        + "\n"
    )
    source.write_text(raw)

    report = audit(source)

    assert report.candidate_rows == int(flagged)
    assert report.input_sha256 == hashlib.sha256(raw.encode()).hexdigest()


def test_audit_does_not_combine_escalations_across_turns(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "I want to die."},
                    {"role": "assistant", "weight": 1, "content": "Call a crisis line."},
                    {"role": "user", "content": "I still want to die."},
                    {"role": "assistant", "weight": 1, "content": "What would happen if you didn't?"},
                    {"role": "user", "content": "I want to die."},
                    {"role": "assistant", "content": "Hidden target.", "weight": 0},
                    {"role": "tool", "content": "result"},
                    {"role": "assistant", "weight": 1, "content": "Tool continuation."},
                ]
            }
        )
        + "\n"
    )

    report = audit(source)

    assert [(item.line, item.assistant_turn) for item in report.findings] == [(1, 3)]
    expected_weighted_targets = 3
    assert report.assistant_targets == expected_weighted_targets
    assert "Hidden target" not in report.model_dump_json()


@pytest.mark.parametrize(
    "case",
    [(True, 100, (1,), 0, ()), (False, 100, (), 0, (1,)), (True, 1, (), 1, ())],
)
def test_export_review_checks_actual_datum_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[bool, int, tuple[int, ...], int, tuple[int, ...]],
) -> None:
    included, max_tokens, present, excluded, unmatched = case
    source = tmp_path / "selected.jsonl"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "I want to die."},
        {"role": "assistant", "weight": 1, "content": "Tell me more."},
    ]
    source.write_text(json.dumps({"messages": messages}) + "\n")
    tokenizer = StubTokenizer()
    ids, weights = build_tokens(tokenizer, messages)
    datum = Datum(input_ids=ids[:-1], target_ids=ids[1:], weights=weights[1:])
    exported = tmp_path / "datums.jsonl"
    exported.write_text(datum.line() + "\n" if included else "")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "path": str(exported),
                "sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
                "tokenizer": "test",
                "max_tokens": max_tokens,
                "end_token": "<|im_end|>",
            }
        )
    )

    def load_tokenizer(_name: str, **_options: bool) -> StubTokenizer:
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    result = audit_export(source, receipt, audit(source))

    assert result.present_selected_lines == present
    assert result.excluded_by_length == excluded
    assert result.unmatched_selected_lines == unmatched


def test_export_review_matches_a_row_the_export_trimmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "selected.jsonl"
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "older prompt " + "x " * 40},
        {"role": "assistant", "weight": 1, "content": "older reply"},
    ]
    final: list[dict[str, Any]] = [
        {"role": "user", "content": "I want to die."},
        {"role": "assistant", "weight": 1, "content": "Tell me more."},
    ]
    source.write_text(json.dumps({"messages": history + final}) + "\n")
    tokenizer = StubTokenizer()
    ids, weights = build_tokens(tokenizer, final)
    datum = Datum(input_ids=ids[:-1], target_ids=ids[1:], weights=weights[1:])
    exported = tmp_path / "datums.jsonl"
    exported.write_text(datum.line() + "\n")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "path": str(exported),
                "sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
                "tokenizer": "test",
                "max_tokens": len(ids) - 1,
                "max_final_exchange_tokens": 1024,
                "end_token": "<|im_end|>",
            }
        )
    )

    def load_tokenizer(_name: str, **_options: bool) -> StubTokenizer:
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    result = audit_export(source, receipt, audit(source))

    assert result.present_selected_lines == (1,)
    assert result.excluded_by_length == 0


def test_export_review_honors_a_zero_final_exchange_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "selected.jsonl"
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "older prompt " + "x " * 40},
        {"role": "assistant", "weight": 1, "content": "older reply"},
    ]
    final: list[dict[str, Any]] = [
        {"role": "user", "content": "I want to die."},
        {"role": "assistant", "weight": 1, "content": "Tell me more."},
    ]
    source.write_text(json.dumps({"messages": history + final}) + "\n")
    tokenizer = StubTokenizer()
    ids, _ = build_tokens(tokenizer, final)
    exported = tmp_path / "datums.jsonl"
    exported.write_text("")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "path": str(exported),
                "sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
                "tokenizer": "test",
                "max_tokens": len(ids) - 1,
                "max_final_exchange_tokens": 0,
                "end_token": "<|im_end|>",
            }
        )
    )

    def load_tokenizer(_name: str, **_options: bool) -> StubTokenizer:
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    result = audit_export(source, receipt, audit(source))

    assert result.excluded_by_length == 1
    assert result.present_selected_lines == ()


def _flagged_trim_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    exported: bool,
    final_exchange_tokens: int,
) -> Any:
    """Review a row whose only flagged reply is removed by trimming."""
    source = tmp_path / "selected.jsonl"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "I want to die. " + "x " * 40},
        {"role": "assistant", "weight": 1, "content": "Tell me more."},
        {"role": "user", "content": "I am okay now."},
        {"role": "assistant", "weight": 1, "content": "Call 988 now."},
    ]
    source.write_text(json.dumps({"messages": messages}) + "\n")
    tokenizer = StubTokenizer()
    final = messages[2:]
    ids, weights = build_tokens(tokenizer, final)
    datum = Datum(input_ids=ids[:-1], target_ids=ids[1:], weights=weights[1:])
    exported_path = tmp_path / "datums.jsonl"
    exported_path.write_text(datum.line() + "\n" if exported else "")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "path": str(exported_path),
                "sha256": hashlib.sha256(exported_path.read_bytes()).hexdigest(),
                "tokenizer": "test",
                "max_tokens": len(ids) - 1,
                "max_final_exchange_tokens": final_exchange_tokens,
                "end_token": "<|im_end|>",
            }
        )
    )

    def load_tokenizer(_name: str, **_options: bool) -> StubTokenizer:
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    return audit_export(source, receipt, audit(source))


def test_export_review_reports_a_flagged_target_lost_to_trimming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _flagged_trim_review(tmp_path, monkeypatch, exported=True, final_exchange_tokens=1024)

    assert result.trimmed_target_lines == (1,)
    assert result.present_selected_lines == ()
    assert result.unmatched_selected_lines == ()
    assert result.excluded_by_length == 0


def test_export_review_still_counts_a_trimmed_row_dropped_by_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _flagged_trim_review(tmp_path, monkeypatch, exported=True, final_exchange_tokens=1)

    assert result.trimmed_target_lines == ()
    assert result.excluded_by_length == 1


def test_export_review_still_counts_a_trimmed_row_missing_from_the_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _flagged_trim_review(tmp_path, monkeypatch, exported=False, final_exchange_tokens=1024)

    assert result.trimmed_target_lines == ()
    assert result.unmatched_selected_lines == (1,)


def test_export_review_rejects_a_changed_training_file(tmp_path: Path) -> None:
    source = tmp_path / "selected.jsonl"
    source.write_text("")
    exported = tmp_path / "datums.jsonl"
    exported.write_text("changed\n")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "path": str(exported),
                "sha256": hashlib.sha256(b"original\n").hexdigest(),
                "tokenizer": "test",
                "max_tokens": 512,
                "end_token": "<|im_end|>",
            }
        )
    )

    with pytest.raises(ExportIdentityError):
        audit_export(source, receipt, audit(source))
