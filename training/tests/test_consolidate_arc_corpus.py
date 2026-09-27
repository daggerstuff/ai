"""Tests for consolidate_arc_corpus (pure, tmp-dir fixtures, no network)."""

import json
from pathlib import Path
from typing import Any

import pytest

from pipelines.ingestion_deduplication import compute_primary_hash
from training.cliche_gate import reject_reason_for_record
from training.consolidate_arc_corpus import (
    FAMILY,
    TASK_TYPE,
    _gold_record,
    _session_messages,
    consolidate,
)

LEDGER = {
    "dx": "adjustment disorder with anxious mood",
    "def": "client under sustained workplace strain",
    "soma": "jaw tension, shallow sleep",
    "risk": "none",
    "hx": "no prior episodes",
    "onset": "six weeks ago",
    "track": "stable",
    "tx": "cbt referral",
    "tl": "present",
}

_TURNS_CLEAN: list[dict[str, Any]] = [
    {"role": "client", "content": "I keep cancelling on people."},
    {"role": "therapist", "content": "What happened the last time you cancelled?", "ledger": LEDGER},
]


def _turn(role: str, content: str, ledger: dict[str, str] | None = None) -> dict[str, Any]:
    turn: dict[str, Any] = {"role": role, "content": content}
    if ledger is not None:
        turn["ledger"] = ledger
    return turn


def _session(n: int, turns: list[dict[str, Any]]) -> dict[str, Any]:
    return {"n": n, "turns": turns}


def _arc(
    arc_id: str,
    sessions: list[dict[str, Any]],
    verdict: str = "accept",
    plan_path: str = "training/arc_plans/__absent__.json",
) -> dict[str, Any]:
    return {
        "arc_id": arc_id,
        "audit": {"verdict": verdict, "flags_final": [], "revisions": 0},
        "auditor_model": "moonshotai/kimi-k3",
        "beats_planned": 3,
        "metrics": {},
        "plan_path": plan_path,
        "sessions": sessions,
        "spec_version": "1.0",
        "timeline_final": [],
        "writer_model": "deepseek/deepseek-v4.1-flash",
    }


# --- _session_messages: writer turns -> strict alternating ChatML ---


def test_session_messages_converts_with_ledger_prefix() -> None:
    session = _session(
        1,
        [
            _turn("client", "I can't sleep before shifts."),
            _turn("therapist", "When did the sleep trouble start?", ledger=LEDGER),
        ],
    )
    messages = _session_messages(session)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "I can't sleep before shifts."
    head, reply = messages[1]["content"].split("\n", 1)
    assert json.loads(head) == LEDGER  # spec §9: ledger stays IN the assistant content
    assert reply == "When did the sleep trouble start?"


def test_session_messages_merges_consecutive_client_turns() -> None:
    session = _session(
        1,
        [
            _turn("client", "There's a person, they—"),
            _turn("client", "Forget it."),
            _turn("therapist", "We can go wherever you want to start.", ledger=LEDGER),
        ],
    )
    messages = _session_messages(session)
    n_messages = 2
    assert len(messages) == n_messages
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "There's a person, they—\n\nForget it."


def test_session_messages_drops_trailing_client_turn() -> None:
    session = _session(
        1,
        [
            _turn("client", "Fine. I'll come back next week."),
            _turn("therapist", "We'll pick it up then.", ledger=LEDGER),
            _turn("client", "Okay."),
        ],
    )
    messages = _session_messages(session)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert "Okay." not in messages[0]["content"]


def test_session_messages_rejects_missing_ledger() -> None:
    session = _session(1, [_turn("client", "hi"), _turn("therapist", "hello")])
    with pytest.raises(ValueError, match="ledger"):
        _session_messages(session)


def test_session_messages_rejects_consecutive_therapist() -> None:
    session = _session(
        1,
        [
            _turn("client", "hi"),
            _turn("therapist", "first", ledger=LEDGER),
            _turn("therapist", "second", ledger=LEDGER),
        ],
    )
    with pytest.raises(ValueError, match="consecutive therapist"):
        _session_messages(session)


def test_session_messages_rejects_unknown_role() -> None:
    session = _session(1, [_turn("narrator", "meanwhile")])
    with pytest.raises(ValueError, match="unknown role"):
        _session_messages(session)


def test_session_messages_rejects_client_only_session() -> None:
    # A client-only session drops to empty (trailing clients) and must fail loudly.
    session = _session(1, [_turn("client", "hi")])
    with pytest.raises(ValueError, match="start with client"):
        _session_messages(session)


# --- _gold_record: metadata mapping ---


def test_gold_record_metadata_from_plan() -> None:
    arc = _arc("pilot_07", [], plan_path="training/arc_plans/pilot_07.json")
    arc["spec_version"] = "2.0"
    session = _session(1, [])
    messages = [
        {"role": "user", "content": "I blew up at my sister again."},
        {"role": "assistant", "content": "{}\nTake whatever time you need."},
    ]
    plan = {"seed": {"domain": "boundary_testing", "severity": "severe"}}

    rec = _gold_record(arc, session, messages, plan)

    assert rec["source"] == "arc_corpus_pilot_07"
    assert rec["task_type"] == TASK_TYPE
    assert TASK_TYPE == "long_session_arc"
    assert rec["tier"] == "T1_GOLD"
    assert rec["family"] == FAMILY
    assert FAMILY == "arc_corpus"
    assert rec["diagnostic_tag"] == "boundary testing"
    assert rec["difficulty"] == "severe"
    assert rec["consolidated_at"]
    prov = rec["provenance"]
    assert prov["type"] == FAMILY
    assert prov["arc_id"] == "pilot_07"
    assert prov["session_n"] == 1
    assert prov["plan_path"] == "training/arc_plans/pilot_07.json"
    assert prov["writer_model"] == "deepseek/deepseek-v4.1-flash"
    assert prov["auditor_model"] == "moonshotai/kimi-k3"
    assert prov["verdict"] == "accept"
    assert prov["spec_version"] == "2.0"
    assert prov["domain"] == "boundary_testing"


def test_gold_record_degrades_without_plan() -> None:
    rec = _gold_record(_arc("arc_0001", []), _session(1, []), [], None)
    assert rec["source"] == "arc_corpus_arc_0001"
    assert rec["diagnostic_tag"] == ""
    assert rec["difficulty"] == ""
    assert rec["provenance"]["domain"] == ""


def test_gold_record_hash_is_metadata_independent() -> None:
    a = _gold_record(_arc("pilot_01", []), _session(1, []), [], None)
    b = _gold_record(_arc("pilot_01b", []), _session(2, []), [], None)
    assert compute_primary_hash(a) == compute_primary_hash(b)


# --- consolidate: end-to-end gate pipeline over arc records ---


def test_consolidate_end_to_end(tmp_path: Path) -> None:
    gold = tmp_path / "train_master_gold.jsonl"
    inputs = tmp_path / "arc_records.jsonl"
    plan_file = tmp_path / "pilot_01.json"
    plan_file.write_text(
        json.dumps({"arc_id": "pilot_01", "seed": {"domain": "boundary_testing", "severity": "severe"}}),
        encoding="utf-8",
    )
    accept = _arc("pilot_01", [_session(1, _TURNS_CLEAN)], plan_path=str(plan_file))
    hr = _arc("arc_0004", [_session(1, _TURNS_CLEAN)], verdict="hr")
    cliche = _arc(
        "arc_0012",
        [
            _session(
                1,
                [
                    _turn("client", "I think I messed up with my sister."),
                    _turn("therapist", "You're right, it does sound hard.", ledger=LEDGER),
                ],
            )
        ],
    )
    duplicate = _arc("pilot_01b", [_session(1, _TURNS_CLEAN)])
    invalid = _arc(
        "arc_0043",
        [_session(1, [_turn("client", "Where do we start."), _turn("therapist", "Wherever you want.")])],
    )
    inputs.write_text(
        "\n".join(json.dumps(a) for a in (accept, hr, cliche, duplicate, invalid)) + "\n",
        encoding="utf-8",
    )

    summary = consolidate([inputs], gold_path=gold, reject_path=tmp_path / "rejections.jsonl")

    n_scanned_arcs, n_skipped_verdict, n_scanned_sessions = 5, 1, 4
    assert summary["scanned_arcs"] == n_scanned_arcs
    assert summary["skipped_verdict"] == n_skipped_verdict  # hr arc skipped before its sessions are scanned
    assert summary["scanned_sessions"] == n_scanned_sessions
    assert summary["emitted_gold"] == 1
    assert summary["duplicates"] == 1
    assert summary["rejected"] == 1
    assert summary["rejected_invalid"] == 1
    assert summary["by_arc"] == {"pilot_01": 1}

    gold_lines = [ln for ln in gold.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(gold_lines) == 1
    record = json.loads(gold_lines[0])
    assert record["source"] == "arc_corpus_pilot_01"
    assert record["diagnostic_tag"] == "boundary testing"  # plan seed metadata applied
    assert record["difficulty"] == "severe"
    assert record["provenance"]["plan_path"] == str(plan_file)
    head = record["messages"][1]["content"].split("\n", 1)[0]
    assert json.loads(head) == LEDGER

    reject_lines = (tmp_path / "rejections.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("caving_phrase_detected" in ln for ln in reject_lines)
    assert any("invalid_session" in ln for ln in reject_lines)


def test_consolidate_idempotent(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    inputs = tmp_path / "arc_records.jsonl"
    inputs.write_text(json.dumps(_arc("pilot_01", [_session(1, _TURNS_CLEAN)])) + "\n", encoding="utf-8")

    first = consolidate([inputs], gold_path=gold, reject_path=tmp_path / "r1.jsonl")
    second = consolidate([inputs], gold_path=gold, reject_path=tmp_path / "r2.jsonl")

    assert first["emitted_gold"] == 1
    assert second["emitted_gold"] == 0
    assert second["duplicates"] == 1
    gold_lines = [ln for ln in gold.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(gold_lines) == 1


def test_consolidate_rejects_quadit_banned_phrase(tmp_path: Path) -> None:
    # "circle back" slips past the cliché gate but is a quadit banned phrase.
    gold = tmp_path / "gold.jsonl"
    inputs = tmp_path / "arc_records.jsonl"
    arc = _arc(
        "pilot_02",
        [
            _session(
                1,
                [
                    _turn("client", "What do we do about next week?"),
                    _turn("therapist", "Let's circle back on the safety plan next sprint.", ledger=LEDGER),
                ],
            )
        ],
    )
    inputs.write_text(json.dumps(arc) + "\n", encoding="utf-8")

    # Cross-check: the cliché gate stays quiet on this content — quadit does the work.
    session = arc["sessions"][0]
    record = _gold_record(arc, session, _session_messages(session), None)
    assert reject_reason_for_record(record, family=FAMILY) is None

    summary = consolidate([inputs], gold_path=gold, reject_path=tmp_path / "rejections.jsonl")

    assert summary["rejected"] == 1
    assert summary["emitted_gold"] == 0
    assert "circle back" not in (gold.read_text(encoding="utf-8") if gold.exists() else "")
    reject_lines = (tmp_path / "rejections.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("quadit" in ln for ln in reject_lines)


def test_consolidate_dry_run_writes_nothing(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    reject = tmp_path / "rejections.jsonl"
    inputs = tmp_path / "arc_records.jsonl"
    inputs.write_text(json.dumps(_arc("pilot_01", [_session(1, _TURNS_CLEAN)])) + "\n", encoding="utf-8")

    summary = consolidate([inputs], gold_path=gold, reject_path=reject, dry_run=True)

    assert summary["emitted_gold"] == 1
    assert not gold.exists()
    assert not reject.exists()
