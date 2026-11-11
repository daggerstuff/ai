"""Tests for the v21 release: v20's pipeline plus the train register repair.

v20 fixed the ratio and still lost crisis cases to validation openers ("I'm not
going to pretend that's not serious. Let's get you to..."). v21 deletes exactly
those sentences from train chat rows. The strip is the risky part — it mutates
carried text — so the sentence splitter, the arc-ledger guard and the "never
empty a target" rule are asserted here rather than discovered in a build log.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import training.build_agent_tool_sft_v13 as v13
import training.build_agent_tool_sft_v20 as v20
import training.build_agent_tool_sft_v21 as v21
import training.build_v14_release as v14
import training.build_v15_release as v15
import training.build_v16_release as v16
import training.build_v17_release as v17
import training.build_v19_release as v19
import training.build_v20_release as prev
import training.build_v21_release as release

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V20_ROWS = 1760
V21_ROWS = 2352
MESSAGES_PER_ROW = 2
SENTENCES_IN_ROW = 2
PRETEND = "I'm not going to pretend that's not serious."
SLOW_DOWN = "Let's slow down."


def test_the_release_identity_names_v21() -> None:
    assert release.RELEASE_ID.endswith("-v21")
    assert release.RELEASE_ID in str(release.OUTPUT)
    assert release.RELEASE_ID in str(release.MANIFEST)
    assert release.SHUFFLE_SEED not in {
        prev.SHUFFLE_SEED,
        v19.SHUFFLE_SEED,
        v17.SHUFFLE_SEED,
        v16.SHUFFLE_SEED,
    }


def test_limitations_drop_the_stale_v20_family_claims() -> None:
    """v20 described its own family's size, register and placement; v21 replaces it."""
    inherited = prev._limitations()
    for stale in release._STALE_V20_LIMITATIONS:
        assert any(text.startswith(stale) for text in inherited), stale
        assert not any(text.startswith(stale) for text in release._limitations()), stale
    assert len(release._limitations()) == (
        len(inherited) - len(release._STALE_V20_LIMITATIONS) + len(release.LIMITATIONS_V21)
    )


def test_limitations_keep_every_other_v20_entry() -> None:
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(release._STALE_V20_LIMITATIONS))
    assert release._limitations()[: len(kept)] == kept


def test_limitations_record_the_size_the_register_and_the_repair() -> None:
    added = " ".join(release.LIMITATIONS_V21)
    assert "2,352 rows" in added
    assert "576" in added
    assert "2.8% of train" in added
    assert "not reviewed by a clinician" in added
    assert "deletes whole sentences" in added


def test_the_pipeline_is_v20s_with_the_family_and_the_strip_added() -> None:
    assert issubclass(release._Release, prev._Release)
    assert release._Release.carry is prev._Release.carry
    assert release._Release.append_addenda is not prev._Release.append_addenda
    assert release._Release.verify_totals is not prev._Release.verify_totals
    assert release._Release.record_retained is not prev._Release.record_retained


def test_the_addendum_is_v21_and_v20_is_gone() -> None:
    assert release.V21_SOURCE_LABEL == "crisis_direct_v21"
    assert release.V21_SOURCE_LABEL != prev.V20_SOURCE_LABEL
    assert release.EXPECTED_V21_TRAIN_ROWS == V21_ROWS
    assert release.EXPECTED_V21_TRAIN_ROWS > prev.EXPECTED_V20_TRAIN_ROWS


def test_v21_candidate_rows_are_clean_chat_rows() -> None:
    rows = release.v21_candidate_rows()
    assert len(rows) == V21_ROWS
    for candidate, payload in rows:
        assert candidate["source"] == release.V21_SOURCE_LABEL
        assert not v14.is_agent_tool(candidate)
        assert len(candidate["messages"]) == MESSAGES_PER_ROW
        assert candidate["messages"][0]["role"] == "user"
        assert candidate["messages"][1]["role"] == "assistant"
        assert payload, "a v21 row must project to an assistant target"


def test_v21_keeps_the_v20_register_and_grows_the_surface() -> None:
    """The action table is still v13's; only volume and surface move."""
    assert v21.ACTIONS.keys() == v13.ACTIONS.keys()
    surface = v21.paraphrases()
    assert sum(len(items) for items in surface.values()) > sum(len(items) for items in v20.paraphrases().values())


def test_strip_drops_a_sentence_carrying_the_phrase() -> None:
    text, dropped, matched = release.strip_register_sentences(f"{PRETEND} Call 911.")
    assert text == "Call 911."
    assert dropped == 1
    assert matched == ["not going to pretend"]


def test_strip_drops_the_slow_down_sentence() -> None:
    text, dropped, _ = release.strip_register_sentences(f"{SLOW_DOWN} Are you alone right now?")
    assert text == "Are you alone right now?"
    assert dropped == 1


def test_strip_matches_the_curly_apostrophe() -> None:
    text, dropped, _ = release.strip_register_sentences("Let\u2019s slow down. Call 911.")
    assert dropped == 1
    assert "slow down" not in text


def test_strip_leaves_clean_text_untouched() -> None:
    clean = "Put the pills in the kitchen and stay on the line."
    text, dropped, matched = release.strip_register_sentences(clean)
    assert (text, dropped, matched) == (clean, 0, [])


def test_strip_never_empties_a_target() -> None:
    """A turn whose only sentence is banned is left alone: no empty targets."""
    text, dropped, matched = release.strip_register_sentences(PRETEND)
    assert (text, dropped, matched) == (PRETEND, 0, [])


def test_strip_preserves_the_arc_ledger_prefix() -> None:
    ledger = '{"dx":"mdd","risk":"acute"}'
    text, dropped, _ = release.strip_register_sentences(f"{ledger}\n{PRETEND} Call 911.")
    assert text == f"{ledger}\nCall 911."
    assert dropped == 1


def test_split_ledger_ignores_braces_inside_strings() -> None:
    head, tail = release.split_ledger('{"note":"a } b"}\nCall 911.')
    assert head == '{"note":"a } b"}'
    assert tail == "\nCall 911."


def test_split_ledger_leaves_plain_text_alone() -> None:
    assert release.split_ledger("Call 911.") == ("", "Call 911.")


def test_split_ledger_returns_the_whole_string_when_unbalanced() -> None:
    assert release.split_ledger('{"note":"unterminated') == ("", '{"note":"unterminated')


def chat_line(assistant: str) -> str:
    payload = [
        {"role": "user", "content": "I cut again last night.", "weight": 0},
        {"role": "assistant", "content": assistant, "weight": 1},
    ]
    return v14.payload_line(payload, None)


def release_fixture(tmp_path: Path) -> release._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return release._Release(indexes, tmp_path, dry_run=True)


def test_strip_train_register_rewrites_the_last_row_only(tmp_path: Path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.payload_lines["train"] = [chat_line("Call 911."), chat_line(f"{PRETEND} Call 911.")]

    fixture.strip_train_register()

    body = json.loads(fixture.payload_lines["train"][-1])
    assert body["messages"][-1]["content"] == "Call 911."
    assert body["messages"][-1]["weight"] == 1
    assert json.loads(fixture.payload_lines["train"][0])["messages"][-1]["content"] == "Call 911."
    assert fixture.register_stripped_rows == 1
    assert fixture.register_stripped_turns == 1
    assert fixture.register_stripped_sentences == 1
    assert fixture.register_stripped_phrases == {"not going to pretend": 1}


def test_strip_train_register_counts_nothing_for_a_clean_row(tmp_path: Path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.payload_lines["train"] = [chat_line("Call 911.")]

    fixture.strip_train_register()

    assert fixture.register_stripped_rows == 0
    assert fixture.register_stripped_sentences == 0


def sample_manifest() -> dict[str, Any]:
    """The v20-shaped manifest the v21 stamping has to correct."""
    return {
        "release_id": prev.RELEASE_ID,
        "created_at_utc": prev.CREATED_AT_UTC,
        "scope": {
            "appended_rows": V9_ROWS + V10_ROWS + V11_ROWS + V20_ROWS,
            "appended_rows_by_source": {
                v14.V9_SOURCE_LABEL: V9_ROWS,
                v15.V10_SOURCE_LABEL: V10_ROWS,
                v16.V11_SOURCE_LABEL: V11_ROWS,
                prev.V20_SOURCE_LABEL: V20_ROWS,
            },
        },
        "payload_contract": {
            "shuffle": {"train": True, "validation": False, "test": False, "seed": prev.SHUFFLE_SEED},
            "agent_tool_note": f"2,807 v13 agent_tool rows plus {V9_ROWS} v9 rows",
        },
        "limitations": list(prev._limitations()),
        "owner_direction": "User-directed v20 after the v19 paired eval.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v21_rows=V21_ROWS
    )
    assert stamped["release_id"] == release.RELEASE_ID
    assert stamped["created_at_utc"] == release.CREATED_AT_UTC
    assert stamped["payload_contract"]["shuffle"] == {
        "train": True,
        "validation": False,
        "test": False,
        "seed": release.SHUFFLE_SEED,
    }
    assert stamped["limitations"] == list(release._limitations())
    assert stamped["owner_direction"] != sample_manifest()["owner_direction"]


def test_stamping_replaces_the_addenda_counts_it_is_given() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v21_rows=V21_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        release.V21_SOURCE_LABEL: V21_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V21_ROWS
    assert prev.V20_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]


def test_the_register_repair_record_describes_the_strip(tmp_path: Path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.payload_lines["train"] = [chat_line(f"{PRETEND} {SLOW_DOWN} Call 911.")]
    fixture.strip_train_register()

    repairs = release.register_repair_record(
        rows_changed=fixture.register_stripped_rows,
        turns_changed=fixture.register_stripped_turns,
        sentences_dropped=fixture.register_stripped_sentences,
        phrases_dropped=fixture.register_stripped_phrases,
    )

    assert repairs["phrases"] == list(release.STRIP_PHRASES)
    assert repairs["splits"] == ["train"]
    assert repairs["rows_changed"] == 1
    assert repairs["turns_changed"] == 1
    assert repairs["sentences_dropped"] == SENTENCES_IN_ROW
    assert repairs["phrases_dropped"] == {"not going to pretend": 1, "let's slow down": 1}
    assert "content_group_sha256" in repairs["applied_after"]


def test_summary_reports_the_v21_release_and_the_repair(tmp_path: Path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.v21_rows = release.v21_candidate_rows()
    fixture.register_stripped_rows = 1

    summary = fixture.summary()

    assert summary["release_id"] == release.RELEASE_ID
    assert summary["appended_v21_rows"] == V21_ROWS
    assert summary["register_stripped_rows"] == 1
    assert "appended_v20_rows" not in summary
