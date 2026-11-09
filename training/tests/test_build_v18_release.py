"""Tests for the v18 release: v17's pipeline with the arc risk turns rewritten in place."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import training.build_v14_release as v14
import training.build_v18_release as v18
import training.rewrite_arc_risk_turns as rw

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V13_ROWS = 708

LEDGER: dict[str, Any] = {"dx": "PTSD", "risk": "passive ideation", "tl": ["week 3: sleep"]}
LEDGER_BYTES = json.dumps(LEDGER, ensure_ascii=False)
SPOKEN = ": How long has the sleep been this broken?"
ORIGINAL = f"{LEDGER_BYTES}\n{SPOKEN}"
REWRITE = "Put 988 in your phone tonight. How long has the sleep been this broken?"
CAP = v18.EVALUATION_DUPLICATE_TARGET_CAP
REWRITTEN_ROWS = 541
REWRITTEN_TURNS = 1_125
SAMPLE_ROWS = 2
SAMPLE_TURNS = 3
REWRITE_COUNTS = {
    "rows": REWRITTEN_ROWS,
    "turns": REWRITTEN_TURNS,
    "header_rows": 1_368,
    "header_turns": 1_389,
    "dropped_rows": 0,
}
PLAIN = "The week has been heavy and the sleep is broken."
HEADER_PREFIX = "Call 988 now."


def message(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


def row_with_turn(rewritten: bool = False) -> dict[str, Any]:
    """One ledger turn as the corpus stores it: ledger JSON, newline, spoken text."""
    spoken = REWRITE if rewritten else SPOKEN
    return {
        "messages": [
            message("user", "I am not sleeping and it is getting worse."),
            message("assistant", f"{LEDGER_BYTES}\n{spoken}"),
        ],
        "source": "arc_corpus_arc_0001",
    }


def plain_row() -> dict[str, Any]:
    """A chat row whose assistant reply is a bare sentence: the header target."""
    return {
        "messages": [
            message("user", "I cannot keep going like this."),
            message("assistant", PLAIN),
        ],
        "source": "reddit_mental_health_posts",
    }


def think_row() -> dict[str, Any]:
    """A row whose final assistant turn opens with a reasoning preamble."""
    return {
        "messages": [
            message("user", "I am not sleeping and it is getting worse."),
            message("assistant", f'<think time="7s">weighing the risk</think>{ORIGINAL}'),
        ],
        "source": "arc_corpus_arc_0002",
    }


def header_record(content: str = PLAIN, msg_index: int = 1, **overrides: Any) -> dict[str, Any]:
    """A header record: a routing prefix prepended to the reply it claims."""
    record: dict[str, Any] = {
        "row_sha256": "0" * 64,
        "msg_index": msg_index,
        "kind": v18.hdr.KIND,
        "new_spoken": f"{HEADER_PREFIX} {content}",
        "original_spoken": content,
        "error": None,
    }
    record.update(overrides)
    return record


def rewrite_record(content: str = ORIGINAL, msg_index: int = 1, **overrides: Any) -> dict[str, Any]:
    """A rewrite record describing the ledger turn in the fixture row."""
    _prefix, ledger, spoken = rw.split_content(content)
    assert ledger is not None
    record: dict[str, Any] = {
        "row_sha256": "0" * 64,
        "msg_index": msg_index,
        "new_spoken": REWRITE,
        "original_spoken": rw.spoken_body(spoken),
        "ledger_sha256": v18._ledger_digest(ledger),
        "risk": "passive ideation",
        "level": "passive",
        "error": None,
        "model": rw.WRITER_MODEL,
    }
    record.update(overrides)
    return record


def test_rewritten_content_keeps_the_ledger_bytes() -> None:
    """The ledger is the label the turn supervises, so it cannot move by a byte."""
    rebuilt = v18.rewritten_content(ORIGINAL, REWRITE)
    assert rebuilt.startswith(LEDGER_BYTES)
    assert rebuilt.endswith(REWRITE)
    _, ledger, spoken = rw.split_content(rebuilt)
    assert ledger == LEDGER
    assert rw.spoken_body(spoken) == REWRITE


def test_rewritten_content_prefixes_a_plain_reply() -> None:
    """A header turn carries no ledger, so the replacement cannot require one."""
    header = f"{HEADER_PREFIX} {PLAIN}"

    rebuilt = v18.rewritten_content(PLAIN, header)

    assert rebuilt == header
    assert rebuilt.endswith(PLAIN)


def test_apply_rewrites_touches_only_the_named_turn() -> None:
    row = row_with_turn()
    updated = v18.apply_rewrites(row, {1: REWRITE})
    assert updated["messages"][0] == row["messages"][0]
    assert updated["messages"][1]["content"] == v18.rewritten_content(ORIGINAL, REWRITE)
    # The caller's row is left alone: carry hashes the original bytes afterwards.
    assert row["messages"][1]["content"] == ORIGINAL
    assert updated["source"] == row["source"]


def test_apply_rewrites_rejects_a_non_assistant_turn() -> None:
    with pytest.raises(ValueError, match="not an assistant turn"):
        v18.apply_rewrites(row_with_turn(), {0: REWRITE})


def test_apply_rewrites_rejects_an_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        v18.apply_rewrites(row_with_turn(), {9: REWRITE})


def test_ledger_digest_is_the_digest_the_rewriter_records() -> None:
    selection = rw.select_risk_turns(row_with_turn()["messages"], "0" * 64)
    assert len(selection) == 1
    assert v18._ledger_digest(LEDGER) == selection[0].ledger_sha256


def test_validate_rewrite_accepts_the_turn_it_describes() -> None:
    v18._validate_rewrite(rewrite_record(), row_with_turn(), 1)


def test_validate_rewrite_rejects_different_spoken_text() -> None:
    record = rewrite_record(original_spoken="Something else entirely.")
    with pytest.raises(ValueError, match="different spoken text"):
        v18._validate_rewrite(record, row_with_turn(), 1)


def test_validate_rewrite_rejects_a_different_ledger() -> None:
    record = rewrite_record(ledger_sha256=hashlib.sha256(b"{}").hexdigest())
    with pytest.raises(ValueError, match="different ledger"):
        v18._validate_rewrite(record, row_with_turn(), 1)


def test_validate_rewrite_rejects_a_mislabelled_risk_level() -> None:
    record = rewrite_record(level="active")
    with pytest.raises(ValueError, match="claims risk level"):
        v18._validate_rewrite(record, row_with_turn(), 1)


def test_validate_header_accepts_a_routing_prefix_on_a_plain_reply() -> None:
    v18._validate_header(header_record(), plain_row(), 2)


def test_validate_header_accepts_a_ledger_turn_with_its_digest() -> None:
    """A ledger turn the writer never rewrote is routed by the deterministic floor."""
    record = header_record(
        content=rw.spoken_body(SPOKEN),
        ledger_sha256=v18._ledger_digest(LEDGER),
    )

    v18._validate_header(record, row_with_turn(), 1)


def test_validate_header_rejects_a_ledger_turn_written_against_another_ledger() -> None:
    record = header_record(content=rw.spoken_body(SPOKEN), ledger_sha256="f" * 64)
    with pytest.raises(ValueError, match="written against a different ledger"):
        v18._validate_header(record, row_with_turn(), 1)


def test_validate_header_rejects_a_prefix_that_drops_the_reply() -> None:
    record = header_record(new_spoken=HEADER_PREFIX)
    with pytest.raises(ValueError, match="does not keep the original reply"):
        v18._validate_header(record, plain_row(), 2)


def test_validate_header_rejects_a_reply_that_already_routes() -> None:
    routing = "Call 988 now and stay on the line with them."
    record = header_record(content=routing)
    row = {**plain_row(), "messages": [message("user", "help"), message("assistant", routing)]}
    with pytest.raises(ValueError, match="already routes"):
        v18._validate_header(record, row, 2)


def test_load_rewrite_records_keeps_only_usable_rewrites(tmp_path: Path) -> None:
    path = tmp_path / "rewrites.jsonl"
    good = rewrite_record()
    failed = rewrite_record(msg_index=3, new_spoken="", error="rejected: no routing term")
    duplicate = rewrite_record()
    with path.open("w", encoding="utf-8") as handle:
        for record in (good, failed, duplicate):
            handle.write(json.dumps(record) + "\n")

    records = v18.load_rewrite_records(path)

    assert records == [good]


def test_load_rewrite_records_rejects_conflicting_rewrites(tmp_path: Path) -> None:
    """Two different rewrites of one turn means the file was built twice over itself."""
    path = tmp_path / "rewrites.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(rewrite_record()) + "\n")
        handle.write(json.dumps(rewrite_record(new_spoken="Call 911 now.")) + "\n")

    with pytest.raises(ValueError, match="conflicting rewrites"):
        v18.load_rewrite_records(path)


def test_build_overlay_matches_the_candidate_rows_it_came_from(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.jsonl"
    row = row_with_turn()
    other = {"messages": [message("user", "hello"), message("assistant", "hi")], "source": "reddit_x"}
    raw = _json_bytes(row)
    with candidate.open("wb") as stream:
        stream.write(raw)
        stream.write(_json_bytes(other))
    record = rewrite_record(row_sha256=v14.line_sha(raw))

    overlay, counts = v18.build_overlay(candidate, [record])

    assert overlay == {record["row_sha256"]: {1: REWRITE}}
    assert counts == {
        "rows": 1,
        "turns": 1,
        "model_rows": 1,
        "model_turns": 1,
        "header_rows": 0,
        "header_turns": 0,
    }


def test_build_overlay_counts_headers_apart_from_rewrites(tmp_path: Path) -> None:
    """The manifest reports the two mechanisms separately, so the counts must not merge."""
    candidate = tmp_path / "candidate.jsonl"
    ledger_row = row_with_turn()
    reply_row = plain_row()
    ledger_raw = _json_bytes(ledger_row)
    reply_raw = _json_bytes(reply_row)
    with candidate.open("wb") as stream:
        stream.write(ledger_raw)
        stream.write(reply_raw)

    overlay, counts = v18.build_overlay(
        candidate,
        [rewrite_record(row_sha256=v14.line_sha(ledger_raw))],
        [header_record(row_sha256=v14.line_sha(reply_raw))],
    )

    assert overlay[v14.line_sha(reply_raw)] == {1: f"{HEADER_PREFIX} {PLAIN}"}
    assert counts == {
        "rows": 2,
        "turns": 2,
        "model_rows": 1,
        "model_turns": 1,
        "header_rows": 1,
        "header_turns": 1,
    }


def test_a_model_rewrite_outranks_the_header_for_the_same_turn(tmp_path: Path) -> None:
    """The header is the floor: where the writer succeeded, its sentence is the one kept."""
    candidate = tmp_path / "candidate.jsonl"
    raw = _json_bytes(row_with_turn())
    with candidate.open("wb") as stream:
        stream.write(raw)
    row_sha = v14.line_sha(raw)
    header = header_record(
        content=rw.spoken_body(SPOKEN),
        row_sha256=row_sha,
        ledger_sha256=v18._ledger_digest(LEDGER),
    )

    overlay, counts = v18.build_overlay(candidate, [rewrite_record(row_sha256=row_sha)], [header])

    assert overlay[row_sha] == {1: REWRITE}
    assert counts["model_turns"] == 1
    assert counts["header_turns"] == 0


def test_build_overlay_fails_when_a_rewrite_matches_nothing(tmp_path: Path) -> None:
    """A rewrite file from another release must fail the build, not vanish."""
    candidate = tmp_path / "candidate.jsonl"
    with candidate.open("wb") as stream:
        stream.write(_json_bytes(row_with_turn()))
    record = rewrite_record(row_sha256="f" * 64)

    with pytest.raises(ValueError, match="do not match any row"):
        v18.build_overlay(candidate, [record])


def test_the_release_identity_names_v18() -> None:
    assert v18.RELEASE_ID.endswith("-v18")
    assert v18.RELEASE_ID in str(v18.OUTPUT)
    assert v18.RELEASE_ID in str(v18.MANIFEST)
    assert v18.SHUFFLE_SEED not in {
        v18.v17.SHUFFLE_SEED,
        v18.v17.v16.SHUFFLE_SEED,
        v18.v17.v15.SHUFFLE_SEED,
    }


def test_limitations_extend_v17_with_the_rewrite_claims() -> None:
    inherited = v18.v17._limitations()
    limitations = v18._limitations()
    assert limitations[: len(inherited)] == inherited
    assert len(limitations) == len(inherited) + len(v18.LIMITATIONS_V18)
    added = " ".join(limitations[len(inherited) :])
    assert "1,125" in added
    assert "clinician" in added
    assert "re-bucketed" in added


def sample_manifest() -> dict[str, Any]:
    """The v17-shaped manifest the v18 stamping has to correct."""
    return {
        "release_id": v18.v17.RELEASE_ID,
        "created_at_utc": v18.v17.CREATED_AT_UTC,
        "scope": {
            "appended_rows": 2_052,
            "appended_rows_by_source": {
                v14.V9_SOURCE_LABEL: V9_ROWS,
                v18.v17.v15.V10_SOURCE_LABEL: V10_ROWS,
                v18.v17.v16.V11_SOURCE_LABEL: V11_ROWS,
                v18.v17.V13_SOURCE_LABEL: V13_ROWS,
            },
        },
        "payload_contract": {
            "shuffle": {"train": True, "validation": False, "test": False, "seed": v18.v17.SHUFFLE_SEED},
            "agent_tool_note": "2,807 v13 agent_tool rows plus 624 v9 rows plus 480 v10 rows",
        },
        "limitations": list(v18.v17._limitations()),
        "owner_direction": "User-directed v17 after the v16 paired eval.",
    }


def test_stamping_records_the_addenda_and_the_rewrites() -> None:
    stamped = v18.stamp_release_identity(
        sample_manifest(),
        addenda={
            v14.V9_SOURCE_LABEL: V9_ROWS,
            v18.v17.v15.V10_SOURCE_LABEL: V10_ROWS,
            v18.v17.v16.V11_SOURCE_LABEL: V11_ROWS,
            v18.v17.V13_SOURCE_LABEL: V13_ROWS,
        },
        rewrites=REWRITE_COUNTS,
    )
    assert stamped["release_id"] == v18.RELEASE_ID
    assert stamped["created_at_utc"] == v18.CREATED_AT_UTC
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V13_ROWS
    assert stamped["scope"]["rewritten_rows"] == REWRITTEN_ROWS
    assert stamped["scope"]["rewritten_turns"] == REWRITTEN_TURNS
    assert stamped["scope"]["header_rows"] == REWRITE_COUNTS["header_rows"]
    assert stamped["scope"]["header_turns"] == REWRITE_COUNTS["header_turns"]
    assert stamped["scope"]["overlay_rows_dropped_with_purged_rows"] == REWRITE_COUNTS["dropped_rows"]
    assert stamped["payload_contract"]["shuffle"]["seed"] == v18.SHUFFLE_SEED
    assert stamped["limitations"] == list(v18._limitations())
    assert stamped["owner_direction"] != sample_manifest()["owner_direction"]


def test_stamping_keeps_v13_out_of_the_agent_tool_note() -> None:
    """v13 is chat-shaped, so the agent-tool sentence must not claim its rows."""
    stamped = v18.stamp_release_identity(
        sample_manifest(),
        addenda={
            v14.V9_SOURCE_LABEL: V9_ROWS,
            v18.v17.v15.V10_SOURCE_LABEL: V10_ROWS,
            v18.v17.v16.V11_SOURCE_LABEL: V11_ROWS,
            v18.v17.V13_SOURCE_LABEL: V13_ROWS,
        },
        rewrites=REWRITE_COUNTS,
    )
    note = stamped["payload_contract"]["agent_tool_note"]
    assert f"{V11_ROWS} v11 rows" in note
    assert f"{V13_ROWS} v13 rows" not in note


def plan_purge(
    monkeypatch: Any,
    tmp_path: Path,
    rows: list[dict[str, Any]],
    splits: dict[int, str],
    overlay: dict[str, dict[int, str]] | None = None,
) -> tuple[dict[int, str], dict[str, dict[str, Any]], set[str]]:
    """Run the purge planner against a throwaway candidate directory."""
    with (tmp_path / "candidate.jsonl").open("wb") as stream:
        for row in rows:
            stream.write(_json_bytes(row))
    monkeypatch.setattr(v14, "V13_DIR", tmp_path)
    indexes = v14._Indexes(
        references={number: {"provisional_split": split} for number, split in splits.items()},
        internal_source={},
        heldout_source={},
        repairs={},
    )
    return v18.plan_evaluation_purge(indexes, overlay or {})


def test_purge_buckets_an_unrewritten_row_by_its_provisional_split(monkeypatch: Any, tmp_path: Path) -> None:
    rows = [{"messages": [message("user", f"client {index}"), message("assistant", "BPD")]} for index in range(3)]
    splits = dict.fromkeys(range(1, 4), "validation")

    purge_plan, stats, _purged = plan_purge(monkeypatch, tmp_path, rows, splits)

    assert purge_plan == {}
    assert stats["validation"]["rows_before"] == len(rows)
    assert stats["validation"]["rows_after"] == len(rows)


def test_purge_buckets_a_rewritten_row_by_its_new_content_group(monkeypatch: Any, tmp_path: Path) -> None:
    """A rewrite changes the content group, so the cap has to see the new split."""
    rewritten = row_with_turn()
    plain = [{"messages": [message("user", f"client {index}"), message("assistant", "BPD")]} for index in range(2)]
    rows = [plain[0], rewritten, plain[1]]
    splits = dict.fromkeys(range(1, 4), "train")
    overlay = {v14.line_sha(_json_bytes(rewritten)): {1: REWRITE}}
    monkeypatch.setattr(v18.v15, "assign_split", lambda _group_hash: "validation")

    _, stats, _purged = plan_purge(monkeypatch, tmp_path, rows, splits, overlay)

    assert stats["validation"]["rows_before"] == 1
    assert stats["validation"]["rows_after"] == 1
    assert stats["test"]["rows_before"] == 0


def test_purge_ignores_train_rows(monkeypatch: Any, tmp_path: Path) -> None:
    over = CAP + 5
    rows = [{"messages": [message("user", f"client {index}"), message("assistant", "BPD")]} for index in range(over)]
    splits = dict.fromkeys(range(1, over + 1), "train")

    purge_plan, stats, _purged = plan_purge(monkeypatch, tmp_path, rows, splits)

    assert purge_plan == {}
    assert stats["validation"]["rows_before"] == 0
    assert stats["test"]["rows_before"] == 0


def guard_release(indexes: v14._Indexes, output: Path) -> v18._Release:
    """A dry-run release with only the rewrite guards under test armed."""
    release = v18._Release(indexes, output, dry_run=True)
    release.purge_plan = {}
    return release


def indexes_fixture(**overrides: Any) -> v14._Indexes:
    return v14._Indexes(
        references={1: {"provisional_split": "train"}},
        internal_source=overrides.get("internal_source", {}),
        heldout_source=overrides.get("heldout_source", {}),
        repairs=overrides.get("repairs", {}),
    )


def test_carry_refuses_a_rewritten_row_that_is_purged(tmp_path: Path) -> None:
    release = guard_release(indexes_fixture(), tmp_path)
    raw = _json_bytes(row_with_turn())
    release.overlay = {v14.line_sha(raw): {1: REWRITE}}
    release.purge_plan = {1: "validation"}

    with pytest.raises(ValueError, match="purged as a held-out duplicate"):
        release.carry(1, raw)


def test_carry_refuses_a_rewritten_row_in_the_heldout_reserve(tmp_path: Path) -> None:
    release = guard_release(indexes_fixture(heldout_source={1: {"group": "x"}}), tmp_path)
    raw = _json_bytes(row_with_turn())
    release.overlay = {v14.line_sha(raw): {1: REWRITE}}

    with pytest.raises(ValueError, match="held-out reserve"):
        release.carry(1, raw)


def test_carry_refuses_a_rewritten_row_with_a_projection_repair(tmp_path: Path) -> None:
    """A projection-time repair replaces the message list and would drop the rewrite."""
    release = guard_release(indexes_fixture(repairs={1: {"messages": []}}), tmp_path)
    raw = _json_bytes(row_with_turn())
    release.overlay = {v14.line_sha(raw): {1: REWRITE}}

    with pytest.raises(ValueError, match="projection-time repair"):
        release.carry(1, raw)


def test_the_overlay_key_is_the_line_the_build_admits() -> None:
    """Repairs run before the overlay lookup, so a repaired row still matches its record."""
    raw = _json_bytes(think_row())
    repaired_row, kind, line = v18.carried_line(raw, json.loads(raw))

    assert kind == v18.v15.REPAIR_STRIPPED
    assert v14.line_sha(line) != v14.line_sha(raw)
    rewritten = v18.apply_rewrites(repaired_row, {1: REWRITE})
    assert rewritten["messages"][1]["content"] == v18.rewritten_content(ORIGINAL, REWRITE)


def test_carry_rewrites_a_row_whose_preamble_was_stripped(tmp_path: Path) -> None:
    """The rewrite survives the projection repair, and the repair is still counted."""
    release = guard_release(indexes_fixture(internal_source={1: "arc_corpus_arc_0002"}), tmp_path)
    raw = _json_bytes(think_row())
    _repaired, _kind, line = v18.carried_line(raw, json.loads(raw))
    release.overlay = {v14.line_sha(line): {1: REWRITE}}

    release.carry(1, raw)

    emitted = "".join("".join(lines) for lines in release.payload_lines.values())
    assert REWRITE in emitted
    assert "think" not in emitted
    assert release.rewritten_rows == 1
    assert release.rewritten_turns == 1
    assert len(release.rewritten_hashes) == 1
    assert release.repair_counts[v18.v15.REPAIR_STRIPPED] == 1


def _json_bytes(row: dict[str, Any]) -> bytes:
    return v18.v15.json_bytes(row)


def test_the_rewrite_source_is_the_v17_candidate() -> None:
    """The rewrite hashes are v17 candidate line hashes, so the paths must agree."""
    assert v18.REWRITE_SOURCE_CANDIDATE.parent.name == v18.v17.RELEASE_ID
    assert v18.REWRITE_PATH.name == "arc_risk_rewrites.jsonl"
    assert set(v18.EXPECTED_EVALUATION_PURGED) == set(v18.EVALUATION_SPLITS)


def test_summary_reports_the_rewrite_coverage(tmp_path: Path) -> None:
    release = guard_release(indexes_fixture(), tmp_path)
    release.overlay = {"a": {1: REWRITE, 5: REWRITE}, "b": {3: REWRITE}}
    release.overlay_counts = {"rows": 2, "turns": 3}
    release.rewritten_rows = SAMPLE_ROWS
    release.rewritten_turns = SAMPLE_TURNS
    release.rewritten_by_split = Counter({"train": SAMPLE_ROWS})

    data = release.summary()

    assert data["release_id"] == v18.RELEASE_ID
    assert data["rewritten_rows"] == SAMPLE_ROWS
    assert data["rewritten_turns"] == SAMPLE_TURNS
    assert data["rewritten_by_split"] == {"train": SAMPLE_ROWS}
