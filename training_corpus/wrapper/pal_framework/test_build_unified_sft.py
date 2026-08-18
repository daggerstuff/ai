"""Tests for ``build_unified_sft`` (Phase 2.3 — Dry-Run SFT Validation).

Cover: ChatML validation, JSON-leakage rejection, token-bound rejection,
task-type tagging, interleaving ratios, 10K cap, seed determinism,
replacement-fill when one source is short, no-JSON-leakage invariant,
and CLI ``main`` exit codes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build_unified_sft as bus
import pytest

VALID_SYSTEM = "You are a clinical persona classifier."
VALID_USER = "Dialogue:\nhi\n\nCandidate personas:\n1. A\n\nWhich persona (1-1) best matches this dialogue? Respond with only the number."
VALID_ASSISTANT = "1"


def _valid_record() -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": VALID_SYSTEM},
            {"role": "user", "content": VALID_USER},
            {"role": "assistant", "content": VALID_ASSISTANT},
        ],
        "metadata": {"correct_option": 1, "n_options": 1},
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------- is_chatml_compliant ----------------------


def test_is_chatml_compliant_valid() -> None:
    assert bus.is_chatml_compliant(_valid_record()["messages"])


def test_is_chatml_compliant_rejects_empty() -> None:
    assert not bus.is_chatml_compliant([])


def test_is_chatml_compliant_rejects_non_list() -> None:
    assert not bus.is_chatml_compliant("not a list")  # type: ignore[arg-type]
    assert not bus.is_chatml_compliant(None)  # type: ignore[arg-type]


def test_is_chatml_compliant_rejects_bad_role() -> None:
    msgs = _valid_record()["messages"]
    msgs = [dict(m) for m in msgs]
    msgs[0] = {"role": "narrator", "content": "x"}
    assert not bus.is_chatml_compliant(msgs)


def test_is_chatml_compliant_rejects_non_string_content() -> None:
    msgs = _valid_record()["messages"]
    msgs = [dict(m) for m in msgs]
    msgs[0] = {"role": "system", "content": 123}
    assert not bus.is_chatml_compliant(msgs)


def test_is_chatml_compliant_rejects_empty_content() -> None:
    msgs = _valid_record()["messages"]
    msgs = [dict(m) for m in msgs]
    msgs[1] = {"role": "user", "content": ""}
    assert not bus.is_chatml_compliant(msgs)


def test_is_chatml_compliant_rejects_non_system_first() -> None:
    msgs = _valid_record()["messages"]
    msgs = [dict(m) for m in msgs]
    # Swap system to position 2
    msgs[0], msgs[1] = msgs[1], msgs[0]
    assert not bus.is_chatml_compliant(msgs)


# ---------------------- JSON leakage ----------------------


def test_has_json_leakage_detects_braces_and_quotes() -> None:
    assert bus._has_json_leakage('{"a": 1}')
    assert bus._has_json_leakage('say "hi"')
    # Single quotes are legitimate natural-language punctuation (apostrophes,
    # possessives) and are NOT flagged as JSON leakage.
    assert not bus._has_json_leakage("patient's history")
    assert not bus._has_json_leakage("plain text without json chars")


def test_messages_have_json_leakage_positive() -> None:
    msgs = _valid_record()["messages"]
    msgs = [dict(m) for m in msgs]
    msgs[1] = {"role": "user", "content": "leaked {demographics: ...}"}
    assert bus._messages_have_json_leakage(msgs)


def test_messages_have_json_leakage_negative() -> None:
    assert not bus._messages_have_json_leakage(_valid_record()["messages"])


# ---------------------- validate_record ----------------------


def test_validate_record_valid() -> None:
    assert bus.validate_record(_valid_record())


def test_validate_record_rejects_non_dict() -> None:
    assert not bus.validate_record(["not a dict"])  # type: ignore[arg-type]


def test_validate_record_rejects_missing_messages() -> None:
    rec = _valid_record()
    del rec["messages"]
    assert not bus.validate_record(rec)


def test_validate_record_rejects_non_list_messages() -> None:
    rec = _valid_record()
    rec["messages"] = "not a list"
    assert not bus.validate_record(rec)


def test_validate_record_rejects_json_leakage() -> None:
    rec = _valid_record()
    rec = json.loads(json.dumps(rec))
    rec["messages"][1]["content"] = 'bad {"persona": ...}'
    assert not bus.validate_record(rec)


def test_validate_record_rejects_token_overflow() -> None:
    rec = _valid_record()
    rec = json.loads(json.dumps(rec))
    # ~4097 chars → ~1025 tokens > DEFAULT_MAX_TOKENS=1024
    rec["messages"][1]["content"] = "x" * 4097
    assert not bus.validate_record(rec)


def test_validate_record_respects_custom_max_tokens() -> None:
    rec = _valid_record()
    assert bus.validate_record(rec, max_tokens=1) is False
    # Total chars ~180 → ~45 tokens; passes at 50.
    assert bus.validate_record(rec, max_tokens=50)


# ---------------------- load_records ----------------------


def test_load_records_tags_task_type(tmp_path: Path) -> None:
    p = tmp_path / "sel.jsonl"
    _write_jsonl(p, [_valid_record()])
    out = bus.load_records(p, bus.TASK_SELECTION)
    assert len(out) == 1
    assert out[0]["metadata"]["task_type"] == bus.TASK_SELECTION
    assert "estimated_tokens" in out[0]["metadata"]


def test_load_records_invalid_task_type_raises(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.touch()
    with pytest.raises(ValueError):
        bus.load_records(p, "garbage")


def test_load_records_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bus.load_records(tmp_path / "missing.jsonl", bus.TASK_SELECTION)


def test_load_records_drops_invalid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "mixed.jsonl"
    records = [_valid_record(), {"messages": [], "metadata": {}}]  # 2nd invalid
    _write_jsonl(p, records)
    out = bus.load_records(p, bus.TASK_SELECTION)
    assert len(out) == 1
    captured = capsys.readouterr()
    assert "dropped 1 invalid" in captured.err


def test_load_records_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "blanks.jsonl"
    p.write_text(
        json.dumps(_valid_record(), ensure_ascii=False) + "\n\n\n",
        encoding="utf-8",
    )
    out = bus.load_records(p, bus.TASK_SELECTION)
    assert len(out) == 1


# ---------------------- build_unified_dataset ----------------------


def _make_selection_jsonl(path: Path, n: int) -> None:
    records = []
    for i in range(n):
        rec = _valid_record()
        rec = json.loads(json.dumps(rec))
        rec["messages"][2]["content"] = str((i % 5) + 1)
        records.append(rec)
    _write_jsonl(path, records)


def _make_dialogue_jsonl(path: Path, n: int) -> None:
    records = []
    for i in range(n):
        rec = {
            "messages": [
                {"role": "system", "content": "You are roleplaying a patient."},
                {
                    "role": "user",
                    "content": f"Given this persona: P{i}\n\nDialogue history:\n\nGenerate the next response.",
                },
                {"role": "assistant", "content": f"Response {i}"},
            ],
            "metadata": {"persona_string": f"P{i}", "n_dialogue_turns": 0, "estimated_tokens": 50},
        }
        records.append(rec)
    _write_jsonl(path, records)


def test_build_unified_dataset_roundtrip(tmp_path: Path) -> None:
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out = tmp_path / "unified.jsonl"
    _make_selection_jsonl(sel, 6)
    _make_dialogue_jsonl(dia, 6)
    stats = bus.build_unified_dataset(sel, dia, out, target_records=10, seed=42)
    assert stats.total == 10
    assert stats.selection + stats.dialogue == 10
    # Interleaving should give roughly 50/50 when both pools are full.
    assert stats.selection == 5
    assert stats.dialogue == 5
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 10
    for line in lines:
        rec = json.loads(line)
        assert rec["metadata"]["task_type"] in bus.VALID_TASK_TYPES


def test_build_unified_dataset_target_cap(tmp_path: Path) -> None:
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out = tmp_path / "unified.jsonl"
    _make_selection_jsonl(sel, 100)
    _make_dialogue_jsonl(dia, 100)
    stats = bus.build_unified_dataset(sel, dia, out, target_records=10, seed=1)
    assert stats.total == 10


def test_build_unified_dataset_10k_ac(tmp_path: Path) -> None:
    """PIX-4070 AC: 10,000 records in the unified file."""
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out = tmp_path / "unified.jsonl"
    _make_selection_jsonl(sel, 6000)
    _make_dialogue_jsonl(dia, 6000)
    stats = bus.build_unified_dataset(sel, dia, out, target_records=10_000, seed=7)
    assert stats.total == 10_000
    assert stats.selection + stats.dialogue == 10_000


def test_build_unified_dataset_seed_determinism(tmp_path: Path) -> None:
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out1 = tmp_path / "u1.jsonl"
    out2 = tmp_path / "u2.jsonl"
    _make_selection_jsonl(sel, 20)
    _make_dialogue_jsonl(dia, 20)
    s1 = bus.build_unified_dataset(sel, dia, out1, target_records=20, seed=99)
    s2 = bus.build_unified_dataset(sel, dia, out2, target_records=20, seed=99)
    assert s1 == s2
    assert out1.read_bytes() == out2.read_bytes()


def test_build_unified_dataset_unequal_sources(tmp_path: Path) -> None:
    """When one side is smaller than half the target, the other fills in."""
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out = tmp_path / "unified.jsonl"
    _make_selection_jsonl(sel, 2)
    _make_dialogue_jsonl(dia, 20)
    stats = bus.build_unified_dataset(sel, dia, out, target_records=10, seed=3)
    assert stats.total == 10
    assert stats.selection <= 2  # only 2 available
    # dialogue fills the rest
    assert stats.dialogue == 10 - stats.selection


def test_build_unified_dataset_empty_sources(tmp_path: Path) -> None:
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out = tmp_path / "unified.jsonl"
    sel.touch()
    dia.touch()
    stats = bus.build_unified_dataset(sel, dia, out, target_records=5, seed=1)
    assert stats.total == 0
    assert stats.selection == 0
    assert stats.dialogue == 0


def test_build_unified_dataset_rejects_zero_target(tmp_path: Path) -> None:
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out = tmp_path / "unified.jsonl"
    sel.touch()
    dia.touch()
    with pytest.raises(ValueError):
        bus.build_unified_dataset(sel, dia, out, target_records=0)


def test_build_unified_dataset_no_json_leakage_in_output(tmp_path: Path) -> None:
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out = tmp_path / "unified.jsonl"
    _make_selection_jsonl(sel, 5)
    _make_dialogue_jsonl(dia, 5)
    bus.build_unified_dataset(sel, dia, out, target_records=10, seed=1)
    for line in out.read_text(encoding="utf-8").strip().split("\n"):
        rec = json.loads(line)
        for m in rec["messages"]:
            assert not bus._has_json_leakage(m["content"]), f"JSON leakage detected in content: {m['content']!r}"


def test_build_unified_dataset_drops_json_leakage_inputs(tmp_path: Path) -> None:
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out = tmp_path / "unified.jsonl"
    bad = _valid_record()
    bad = json.loads(json.dumps(bad))
    bad["messages"][1]["content"] = 'leaked {"persona": ...}'
    _write_jsonl(sel, [_valid_record(), bad])
    _make_dialogue_jsonl(dia, 4)
    stats = bus.build_unified_dataset(sel, dia, out, target_records=4, seed=1)
    assert stats.total == 4
    # The leaked record should not appear
    for line in out.read_text(encoding="utf-8").strip().split("\n"):
        rec = json.loads(line)
        for m in rec["messages"]:
            assert "{" not in m["content"]


# ---------------------- main / CLI ----------------------


def test_main_missing_input_returns_1(tmp_path: Path) -> None:
    rc = bus.main(
        [
            str(tmp_path / "no_sel.jsonl"),
            str(tmp_path / "no_dia.jsonl"),
            str(tmp_path / "out.jsonl"),
        ]
    )
    assert rc == 1


def test_main_under_target_returns_2(tmp_path: Path) -> None:
    """When both sources are empty, target can't be hit → exit 2 with warning."""
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out = tmp_path / "unified.jsonl"
    sel.touch()
    dia.touch()
    rc = bus.main(
        [
            str(sel),
            str(dia),
            str(out),
            "--target-records",
            "5",
            "--seed",
            "1",
        ]
    )
    assert rc == 2
    assert out.exists()


def test_main_success(tmp_path: Path) -> None:
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out = tmp_path / "unified.jsonl"
    _make_selection_jsonl(sel, 10)
    _make_dialogue_jsonl(dia, 10)
    rc = bus.main(
        [
            str(sel),
            str(dia),
            str(out),
            "--target-records",
            "10",
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 10


def test_main_preserves_unicode(tmp_path: Path) -> None:
    sel = tmp_path / "sel.jsonl"
    dia = tmp_path / "dia.jsonl"
    out = tmp_path / "unified.jsonl"
    rec = _valid_record()
    rec = json.loads(json.dumps(rec))
    rec["messages"][2]["content"] = "1 — Hanoi 🇻🇳"
    _write_jsonl(sel, [rec])
    _make_dialogue_jsonl(dia, 1)
    rc = bus.main(
        [
            str(sel),
            str(dia),
            str(out),
            "--target-records",
            "2",
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "🇻🇳" in text
