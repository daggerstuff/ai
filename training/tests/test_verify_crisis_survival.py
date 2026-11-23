"""Tests for the crisis-family survival audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from training import verify_crisis_survival as survival
from training.build_agent_tool_sft_v12 import collect_rows as v12_rows
from training.export_tinker_release import CRISIS_SOURCE_PREFIX, Datum, TokenLimits
from training.verify_crisis_survival import audit, crisis_row_numbers


def _crisis_row() -> dict[str, Any]:
    """Return a real crisis-family row so the manifest surface resolves."""
    return next(iter(v12_rows().values()))[0]


def _datum() -> Datum:
    return Datum(input_ids=[1, 2, 3], target_ids=[2, 3, 4], weights=[0, 1, 1])


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path, *, export_lines: list[str] | None = None) -> dict[str, Path]:
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        [
            _crisis_row(),
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "an unrelated filler reply"},
                ]
            },
        ],
    )
    export = tmp_path / "export.jsonl"
    lines = [_datum().line() + "\n"] if export_lines is None else export_lines
    export.write_text("".join(lines), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(export.read_bytes()).hexdigest(),
                "datums": 1,
                "max_tokens": 512,
                "end_token": "<|im_end|>",
                "tokenizer": "fake/tokenizer",
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "release_id": "test-release",
                "scope": {"appended_rows_by_source": {f"{CRISIS_SOURCE_PREFIX}v12": 1}},
            }
        ),
        encoding="utf-8",
    )
    return {"train": train, "export": export, "receipt": receipt, "manifest": manifest}


def _audit(paths: Mapping[str, Path], tokenize: Any = None) -> dict[str, Any]:
    return audit(
        train_path=paths["train"],
        export_path=paths["export"],
        receipt_path=paths["receipt"],
        manifest_path=paths["manifest"],
        tokenize=tokenize or (lambda _row: _datum()),
    )


def test_audit_passes_when_every_crisis_row_survives(tmp_path: Path) -> None:
    report = _audit(_fixture(tmp_path))

    assert report["result"] == "pass"
    assert report["crisis_rows_in_train"] == 1
    assert report["crisis_datums_expected"] == 1
    assert report["crisis_datums_present"] == 1
    assert report["crisis_supervised_tokens"] == sum(_datum().weights)
    assert report["dropped_rows"] == []
    assert report["zero_weight_rows"] == []
    assert report["missing_datum_hashes"] == []


def test_audit_reports_a_crisis_row_missing_from_the_export(tmp_path: Path) -> None:
    report = _audit(_fixture(tmp_path, export_lines=[]))

    assert report["result"] == "fail"
    assert report["crisis_datums_present"] == 0
    assert report["missing_datum_hashes"] == [hashlib.sha256(_datum().line().encode()).hexdigest()[:12]]


def test_audit_reports_rows_dropped_by_the_token_budget(tmp_path: Path) -> None:
    report = _audit(_fixture(tmp_path), tokenize=lambda _row: None)

    assert report["result"] == "fail"
    assert report["dropped_rows"] == [1]
    assert report["crisis_datums_expected"] == 0


def _stub_trim(trimmed: bool) -> Any:
    def trim(_tokenizer: Any, messages: list[dict[str, Any]], _max_tokens: int, **_: Any) -> tuple[Any, ...]:
        return messages, list(range(20)), [0, 1] + [1] * 18, trimmed

    return trim


@pytest.mark.parametrize(
    ("trimmed", "max_tokens", "final_exchange_tokens", "kept"),
    [
        (True, 10, 1000, True),
        (True, 1000, 10, False),
        (False, 10, 1000, True),
        (False, 1000, 10, True),
        (False, 10, 10, False),
    ],
)
def test_default_tokenize_uses_the_shared_budget_rule(
    monkeypatch: pytest.MonkeyPatch,
    trimmed: bool,
    max_tokens: int,
    final_exchange_tokens: int,
    kept: bool,
) -> None:
    monkeypatch.setattr(survival, "trim_to_budget", _stub_trim(trimmed))
    row = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}

    datum = survival.default_tokenize(
        None,
        row,
        TokenLimits(max_tokens=max_tokens, final_exchange_tokens=final_exchange_tokens),
        end_token="<|im_end|>",
        template_kwargs={},
    )

    assert (datum is not None) is kept


@pytest.mark.parametrize(("stored", "expected"), [("absent", 512), ("null", 512), (777, 777)])
def test_audit_reads_the_final_exchange_budget_from_the_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stored: int | str, expected: int
) -> None:
    paths = _fixture(tmp_path)
    receipt = json.loads(paths["receipt"].read_text())
    if stored == "absent":
        receipt.pop("max_final_exchange_tokens", None)
    elif stored == "null":
        receipt["max_final_exchange_tokens"] = None
    else:
        receipt["max_final_exchange_tokens"] = stored
    paths["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(survival.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: None)

    def fake_tokenize(_tokenizer: Any, _row: Mapping[str, Any], limits: Any, **kwargs: Any) -> Datum:
        seen.update(kwargs)
        seen["limits"] = limits
        return _datum()

    monkeypatch.setattr(survival, "default_tokenize", fake_tokenize)

    report = audit(
        train_path=paths["train"],
        export_path=paths["export"],
        receipt_path=paths["receipt"],
        manifest_path=paths["manifest"],
    )

    assert report["result"] == "pass"
    assert seen["limits"].final_exchange_tokens == expected


def test_audit_rejects_an_export_that_does_not_match_its_receipt(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["export"].write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match receipt"):
        _audit(paths)


def test_crisis_row_numbers_selects_only_assistant_matches(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        [
            {"messages": [{"role": "assistant", "content": "in surface"}]},
            {"messages": [{"role": "assistant", "content": "not in surface"}]},
            {"messages": [{"role": "user", "content": "in surface"}]},
            {"messages": [{"role": "assistant", "content": "in surface"}]},
        ],
    )

    assert crisis_row_numbers(train, frozenset({"in surface"})) == [1, 4]
