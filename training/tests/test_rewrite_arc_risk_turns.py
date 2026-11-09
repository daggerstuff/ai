"""Contract tests for the arc risk-turn rewrite pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import rewrite_arc_risk_turns as rewrite

LEDGER = {
    "dx": "MDD",
    "risk": "passive SI, no plan",
    "tx": "weekly",
}


def make_selection(original: str, level: int = rewrite.LEVEL_PASSIVE) -> rewrite.Selection:
    return rewrite.Selection(
        source="arc_corpus_arc_0001",
        row_sha256="a" * 64,
        msg_index=3,
        level=level,
        risk=LEDGER["risk"],
        original_spoken=f": {original}",
        ledger_sha256="b" * 64,
        context="Client: I keep thinking there is no point.",
    )


def message(role: str, content: str) -> dict[str, object]:
    return {"role": role, "content": content}


def therapist(ledger: dict[str, object], spoken: str) -> dict[str, object]:
    return message("assistant", f"{json.dumps(ledger)}\n: {spoken}")


class TestRiskLevel:
    def test_active_and_passive_are_distinguished(self) -> None:
        assert rewrite.risk_level("active SI, plan") == rewrite.LEVEL_ACTIVE
        assert rewrite.risk_level("Passive ideation") == rewrite.LEVEL_PASSIVE

    def test_absent_or_denied_risk_is_none(self) -> None:
        assert rewrite.risk_level("") == rewrite.LEVEL_NONE
        assert rewrite.risk_level(None) == rewrite.LEVEL_NONE
        assert rewrite.risk_level("denied") == rewrite.LEVEL_NONE


class TestSplitContent:
    def test_ledger_is_preserved_byte_for_byte(self) -> None:
        content = f"{json.dumps(LEDGER)}\n: Call 988 tonight."
        ledger_prefix, ledger, spoken = rewrite.split_content(content)
        assert ledger == LEDGER
        assert ledger_prefix == json.dumps(LEDGER)
        assert rewrite.spoken_body(spoken) == "Call 988 tonight."

    def test_a_non_ledger_turn_is_returned_whole(self) -> None:
        ledger_prefix, ledger, spoken = rewrite.split_content("Client: hello")
        assert ledger_prefix == ""
        assert ledger is None
        assert spoken == "Client: hello"


class TestSelection:
    def test_first_engagement_escalation_and_closing_turn(self) -> None:
        messages = [
            message("user", "Hello."),
            therapist({"risk": ""}, "What has the week been like?"),
            message("user", "Empty."),
            therapist({"risk": "passive"}, "Empty how?"),
            message("user", "Like nothing."),
            therapist({"risk": "passive"}, "Still nothing?"),
            message("user", "Worse."),
            therapist({"risk": "active"}, "How bad?"),
        ]
        selected = rewrite.select_risk_turns(messages, "c" * 64)
        assert [turn.msg_index for turn in selected] == [3, 7]

    def test_a_session_without_risk_yields_nothing(self) -> None:
        messages = [
            message("user", "Hello."),
            therapist({"risk": ""}, "What has the week been like?"),
        ]
        assert rewrite.select_risk_turns(messages, "d" * 64) == []


class TestValidateRewrite:
    def test_a_minimal_edit_passes(self) -> None:
        selection = make_selection("Read me the food column. Just what you wrote down.")
        text = "Call 988 before we go on. Read me the food column. Just what you wrote down."
        assert rewrite.validate_rewrite(selection, text) == text

    def test_a_missing_routing_term_is_rejected(self) -> None:
        selection = make_selection("Read me the food column.")
        with pytest.raises(rewrite.RewriteRejectedError, match="no routing term"):
            rewrite.validate_rewrite(selection, "Read me the food column.")

    def test_routing_must_lead_the_turn(self) -> None:
        selection = make_selection("Read me the food column.")
        with pytest.raises(rewrite.RewriteRejectedError, match="first sentence"):
            rewrite.validate_rewrite(selection, "Read me the food column. Call 988 if the thought comes back.")

    def test_dropping_a_question_is_rejected(self) -> None:
        selection = make_selection("Do you want to disappear? What stops you?")
        with pytest.raises(rewrite.RewriteRejectedError, match="dropped a question"):
            rewrite.validate_rewrite(selection, "Call 988 tonight. What stops you?")

    def test_added_softening_is_rejected(self) -> None:
        selection = make_selection("Read me the food column.")
        with pytest.raises(rewrite.RewriteRejectedError, match="softening filler"):
            rewrite.validate_rewrite(selection, "Call 988 tonight. I hear you about the food column.")

    def test_a_softener_the_original_already_carried_is_tolerated(self) -> None:
        selection = make_selection("Let's go through the food column.")
        text = "Call 988 before we finish. Let's go through the food column."
        assert rewrite.validate_rewrite(selection, text) == text

    def test_replacing_the_turn_instead_of_routing_it_is_rejected(self) -> None:
        selection = make_selection("Read me the food column. Just what you wrote down. All of it.")
        with pytest.raises(rewrite.RewriteRejectedError, match="instead of routing"):
            rewrite.validate_rewrite(selection, "Call 988 tonight and tell them everything.")

    def test_an_overlong_turn_is_rejected(self) -> None:
        selection = make_selection("Read me the food column.")
        text = "Call 988. " + " ".join(["word"] * rewrite.MAX_SPOKEN_WORDS)
        with pytest.raises(rewrite.RewriteRejectedError, match="words exceeds"):
            rewrite.validate_rewrite(selection, text)


class TestResumeLedger:
    def test_records_round_trip_through_the_output_file(self, tmp_path: Path) -> None:
        output = tmp_path / "rewrites.jsonl"
        record = {
            "row_sha256": "e" * 64,
            "msg_index": 7,
            "new_spoken": "Call 988 tonight.",
        }
        output.write_text(json.dumps(record) + "\n")
        done = rewrite.load_done(output)
        assert list(done) == [f"{'e' * 64}#7"]
        assert done[f"{'e' * 64}#7"]["new_spoken"] == "Call 988 tonight."

    def test_a_missing_file_is_an_empty_ledger(self, tmp_path: Path) -> None:
        assert rewrite.load_done(tmp_path / "absent.jsonl") == {}
