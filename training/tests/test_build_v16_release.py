"""Tests for the v16 release: v15's pipeline plus the v11 and v12 addenda."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import training.build_agent_tool_sft_v11 as v11
import training.build_agent_tool_sft_v12 as v12
import training.build_v16_release as v16


def arguments_of(row: dict[str, Any]) -> dict[str, Any]:
    """The tool-call arguments of the supervised assistant turn."""
    messages = row["messages"]
    call = messages[2]["tool_calls"][0]["function"]
    assert call["name"] == v11.TOOL_NAME
    parsed: dict[str, Any] = json.loads(call["arguments"])
    return parsed


def test_the_addendum_is_the_v11_train_split() -> None:
    total = sum(len(rows) for rows in v11.collect_rows().values())
    assert len(v16.v11_candidate_rows()) == v16.EXPECTED_V11_TRAIN_ROWS
    assert total > v16.EXPECTED_V11_TRAIN_ROWS


def test_v11_candidate_rows_are_agent_tool_rows() -> None:
    for candidate, payload in v16.v11_candidate_rows():
        assert v16.v14.is_agent_tool(candidate)
        assert candidate["tools"] == v11.TOOLS
        assert payload == v16.v14.project_agent(candidate["messages"])
        assert payload[-1]["role"] == "assistant"


def test_the_tool_argument_carries_the_bare_value() -> None:
    for candidate, _ in v16.v11_candidate_rows():
        arguments = arguments_of(candidate)
        assert set(arguments) == {"key", "value"}
        assert arguments["value"] in v11.VALUES
        assert not arguments["value"].startswith(("a ", "an "))
        assert len(arguments["value"].split()) >= v11.MIN_VALUE_WORDS
        assert arguments["value"] == arguments["value"].strip()


def test_the_key_is_the_snake_case_label_the_ask_names() -> None:
    for candidate, _ in v16.v11_candidate_rows():
        arguments = arguments_of(candidate)
        assert arguments["key"] in {v11.v4.snake(label) for label in v11.KEYS}
        ask = candidate["messages"][1]["content"]
        assert arguments["value"] in ask
        assert arguments["key"] in ask


def test_the_pool_never_reuses_the_eval_fixtures() -> None:
    for candidate, _ in v16.v11_candidate_rows():
        arguments = arguments_of(candidate)
        assert arguments["key"] not in v11.FORBIDDEN_KEYS
        assert arguments["value"] not in v11.FORBIDDEN_VALUES


def test_the_v12_addendum_is_the_v12_train_split() -> None:
    rows = v12.collect_rows()
    assert len(v16.v12_candidate_rows()) == v16.EXPECTED_V12_TRAIN_ROWS
    assert sum(len(split) for split in rows.values()) == v16.EXPECTED_V12_TRAIN_ROWS
    assert set(rows) == {"train"}


def test_v12_candidate_rows_are_chat_rows() -> None:
    for candidate, payload in v16.v12_candidate_rows():
        assert not v16.v14.is_agent_tool(candidate)
        assert "tools" not in candidate
        assert candidate["source"] == v16.V12_SOURCE_LABEL
        assert payload[-1]["role"] == "assistant"
        assert payload == v16.v14.project_chat(candidate["messages"], Counter())


def test_v12_rows_project_without_redaction() -> None:
    """A redacted v12 row would reach train mangled, so the addendum fails instead."""
    for _candidate, payload in v16.v12_candidate_rows():
        for message in payload:
            content = message.get("content")
            if isinstance(content, str):
                assert "REDACTED" not in content


def test_the_two_addenda_are_labelled_apart() -> None:
    assert v16.V11_SOURCE_LABEL != v16.V12_SOURCE_LABEL
    assert v16.V12_SOURCE_LABEL not in {
        v16.v14.V9_SOURCE_LABEL,
        v16.v15.V10_SOURCE_LABEL,
        v16.V11_SOURCE_LABEL,
    }


def test_the_release_identity_names_v16() -> None:
    assert v16.RELEASE_ID.endswith("-v16")
    assert v16.RELEASE_ID in str(v16.OUTPUT)
    assert v16.RELEASE_ID in str(v16.MANIFEST)
    assert v16.SHUFFLE_SEED not in {v16.v15.SHUFFLE_SEED, v16.v14.SHUFFLE_SEED}


def test_limitations_drop_the_stale_v14_evaluation_claim() -> None:
    """v16 caps validation and test, so v14's "never filtered" sentence must go."""
    assert any(text.startswith(v16._STALE_EVALUATION_LIMITATION) for text in v16.v15.LIMITATIONS)
    limitations = v16._limitations()
    assert not any(text.startswith(v16._STALE_EVALUATION_LIMITATION) for text in limitations)
    assert len(limitations) == len(v16.v15.LIMITATIONS) - 1 + len(v16.LIMITATIONS_V16)


def test_limitations_keep_every_other_v15_entry() -> None:
    kept = tuple(text for text in v16.v15.LIMITATIONS if not text.startswith(v16._STALE_EVALUATION_LIMITATION))
    assert v16._limitations()[: len(kept)] == kept
    added = " ".join(v16.LIMITATIONS_V16)
    assert "v11" in added
    assert "v12" in added
    assert "per exact final assistant target" in added


def sample_manifest() -> dict[str, Any]:
    """The v15-shaped manifest the v16 stamping has to correct."""
    return {
        "release_id": v16.v15.RELEASE_ID,
        "created_at_utc": v16.v15.CREATED_AT_UTC,
        "scope": {
            "appended_rows": 1_104,
            "appended_rows_by_source": {
                v16.v14.V9_SOURCE_LABEL: 624,
                v16.v15.V10_SOURCE_LABEL: 480,
            },
        },
        "payload_contract": {
            "shuffle": {"train": True, "validation": False, "test": False, "seed": v16.v15.SHUFFLE_SEED},
            "agent_tool_note": "2,807 v13 agent_tool rows plus 624 v9 rows plus 480 v10 rows",
        },
        "limitations": list(v16.v15.LIMITATIONS),
        "owner_direction": "User-directed v15 after the v14 paired eval.",
    }


def test_stamping_rewrites_every_addenda_dependent_field() -> None:
    stamped = v16.stamp_release_identity(sample_manifest(), v9_rows=624, v10_rows=480, v11_rows=240, v12_rows=672)
    assert stamped["release_id"] == v16.RELEASE_ID
    assert stamped["created_at_utc"] == v16.CREATED_AT_UTC
    assert stamped["scope"]["appended_rows_by_source"] == {
        v16.v14.V9_SOURCE_LABEL: 624,
        v16.v15.V10_SOURCE_LABEL: 480,
        v16.V11_SOURCE_LABEL: 240,
        v16.V12_SOURCE_LABEL: 672,
    }
    assert stamped["scope"]["appended_rows"] == sum(stamped["scope"]["appended_rows_by_source"].values())
    assert stamped["payload_contract"]["shuffle"]["seed"] == v16.SHUFFLE_SEED
    assert "240 v11 rows" in stamped["payload_contract"]["agent_tool_note"]
    # v12 is chat-shaped, so the agent-tool note must not claim its rows.
    assert "672 v12 rows" not in stamped["payload_contract"]["agent_tool_note"]
    assert stamped["limitations"] == list(v16._limitations())
    assert stamped["owner_direction"] != sample_manifest()["owner_direction"]


def test_stamping_keeps_the_counts_it_is_given() -> None:
    counts = {
        v16.v14.V9_SOURCE_LABEL: 1,
        v16.v15.V10_SOURCE_LABEL: 2,
        v16.V11_SOURCE_LABEL: 3,
        v16.V12_SOURCE_LABEL: 4,
    }
    stamped = v16.stamp_release_identity(sample_manifest(), v9_rows=1, v10_rows=2, v11_rows=3, v12_rows=4)
    assert stamped["scope"]["appended_rows_by_source"] == counts
    assert stamped["scope"]["appended_rows"] == sum(counts.values())


def test_stamping_replaces_the_limitations_rather_than_extending_them() -> None:
    manifest = sample_manifest()
    manifest["limitations"] = ["stale claim from a previous release"]
    stamped = v16.stamp_release_identity(manifest, v9_rows=624, v10_rows=480, v11_rows=240, v12_rows=672)
    assert stamped["limitations"] == list(v16._limitations())
    assert "stale claim from a previous release" not in stamped["limitations"]


def candidate_row(
    prompt: str,
    target: str,
    *,
    tools: list[Any] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """One v13-shaped candidate row: a single exchange, optionally agent-tool shaped."""
    row: dict[str, Any] = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}]}
    if tools is not None:
        row["tools"] = tools
    if source is not None:
        row["source"] = source
    return row


def project(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The payload messages the purge keys its buckets on."""
    return v16.v14.project_chat(row["messages"], Counter())


def plan_purge(
    monkeypatch: Any,
    tmp_path: Path,
    rows: list[dict[str, Any]],
    splits: dict[int, str],
    cap: int = v16.EVALUATION_DUPLICATE_TARGET_CAP,
) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    """Run the purge planner against a throwaway candidate directory."""
    with (tmp_path / "candidate.jsonl").open("wb") as stream:
        for row in rows:
            stream.write(v16.v15.json_bytes(row))
    monkeypatch.setattr(v16.v14, "V13_DIR", tmp_path)
    indexes = v16.v14._Indexes(
        references={number: {"provisional_split": split} for number, split in splits.items()},
        internal_source={},
        heldout_source={},
        repairs={},
    )
    return v16.plan_evaluation_purge(indexes, cap)


CAP = v16.EVALUATION_DUPLICATE_TARGET_CAP


def test_the_evaluation_cap_is_the_train_admission_rule() -> None:
    assert CAP == v16.v14.DUPLICATE_TARGET_CAP
    assert set(v16.EVALUATION_SPLITS) == {"validation", "test"}
    assert set(v16.EXPECTED_EVALUATION_PURGED) == set(v16.EVALUATION_SPLITS)
    assert v16.EVALUATION_PURGE_REASON not in {v16.v14.PURGE_REASON}


def test_purge_keeps_the_lowest_hash_rows_per_target(monkeypatch: Any, tmp_path: Path) -> None:
    over = CAP + 10
    rows = [candidate_row(f"client {index}", "BPD") for index in range(over)]
    splits = dict.fromkeys(range(1, over + 1), "validation")

    purge_plan, stats = plan_purge(monkeypatch, tmp_path, rows, splits)

    hashes = {number: v16.v14.canonical_hash(project(rows[number - 1])) for number in range(1, over + 1)}
    ordered = sorted(hashes, key=hashes.__getitem__)
    assert set(purge_plan) == set(ordered[CAP:])
    assert set(purge_plan.values()) == {"validation"}
    validation = stats["validation"]
    assert validation["rows_before"] == over
    assert validation["rows_after"] == CAP
    assert validation["purged_rows"] == over - CAP
    assert validation["purged_targets"] == 1
    assert validation["purged_by_target"] == {"BPD": over - CAP}
    assert validation["max_rows_per_final_target_before"] == over
    assert validation["targets_over_cap"] == 1


def test_purge_leaves_an_under_cap_split_untouched(monkeypatch: Any, tmp_path: Path) -> None:
    rows = [candidate_row(f"client {index}", "BPD") for index in range(CAP)]
    splits = dict.fromkeys(range(1, CAP + 1), "validation")

    purge_plan, stats = plan_purge(monkeypatch, tmp_path, rows, splits)

    assert purge_plan == {}
    assert stats["validation"]["purged_rows"] == 0
    assert stats["validation"]["targets_over_cap"] == 0
    assert stats["test"]["rows_before"] == 0


def test_purge_exempts_agent_tool_rows(monkeypatch: Any, tmp_path: Path) -> None:
    over = CAP + 10
    rows = [candidate_row(f"client {index}", "BPD", tools=[]) for index in range(over)]
    splits = dict.fromkeys(range(1, over + 1), "validation")

    purge_plan, stats = plan_purge(monkeypatch, tmp_path, rows, splits)

    assert purge_plan == {}
    assert stats["validation"]["rows_before"] == 0


def test_a_repaired_row_is_bucketed_by_its_new_content_group(monkeypatch: Any, tmp_path: Path) -> None:
    """v15 re-buckets repaired chat rows, so the purge must follow the new split."""
    rows = [candidate_row(f"client {index}", "BPD") for index in range(3)]
    splits = dict.fromkeys(range(1, 4), "train")
    monkeypatch.setattr(v16.v15, "repair_row", lambda row: (row, "reasoning_preamble_stripped"))
    monkeypatch.setattr(v16.v15, "assign_split", lambda _group_hash: "validation")

    _, stats = plan_purge(monkeypatch, tmp_path, rows, splits)

    assert stats["validation"]["rows_before"] == len(rows)
    assert stats["test"]["rows_before"] == 0


def test_purge_ignores_train_and_held_out_rows(monkeypatch: Any, tmp_path: Path) -> None:
    over = CAP + 5
    rows = [candidate_row(f"client {index}", "BPD") for index in range(over)]
    splits = dict.fromkeys(range(1, over + 1), "train")

    purge_plan, stats = plan_purge(monkeypatch, tmp_path, rows, splits)

    assert purge_plan == {}
    assert {name: stats[name]["rows_before"] for name in v16.EVALUATION_SPLITS} == {
        "validation": 0,
        "test": 0,
    }


def test_purge_skips_rows_without_a_reference(monkeypatch: Any, tmp_path: Path) -> None:
    over = CAP + 5
    rows = [candidate_row(f"client {index}", "BPD") for index in range(over)]

    purge_plan, stats = plan_purge(monkeypatch, tmp_path, rows, {})

    assert purge_plan == {}
    assert stats["validation"]["rows_before"] == 0


def test_the_purge_is_a_partition_of_each_split(monkeypatch: Any, tmp_path: Path) -> None:
    targets = ("BPD", "Anxiety", "depression")
    per_target = 40
    small_cap = 10
    rows = [candidate_row(f"client {target} {index}", target) for target in targets for index in range(per_target)]
    splits = dict.fromkeys(range(1, len(rows) + 1), "validation")

    purge_plan, stats = plan_purge(monkeypatch, tmp_path, rows, splits, cap=small_cap)

    validation = stats["validation"]
    assert validation["purged_rows"] == len(purge_plan) == len(targets) * (per_target - small_cap)
    assert validation["rows_before"] == validation["rows_after"] + validation["purged_rows"]
    assert validation["rows_after"] == len(targets) * small_cap
    assert set(validation["purged_by_target"]) == set(targets)


def test_the_purge_does_not_depend_on_row_order(monkeypatch: Any, tmp_path: Path) -> None:
    over = CAP + 10
    rows = [candidate_row(f"client {index}", "BPD") for index in range(over)]
    # Row numbers follow position, so the same split map describes either order.
    splits = dict.fromkeys(range(1, over + 1), "validation")

    purged_forward, _ = plan_purge(monkeypatch, tmp_path, rows, splits)
    purged_reversed, _ = plan_purge(monkeypatch, tmp_path, list(reversed(rows)), splits)

    def surviving(lines: list[dict[str, Any]], purged: dict[int, str]) -> set[str]:
        return {
            v16.v14.canonical_hash(project(row)) for number, row in enumerate(lines, start=1) if number not in purged
        }

    assert surviving(rows, purged_forward) == surviving(list(reversed(rows)), purged_reversed)
    assert len(surviving(rows, purged_forward)) == CAP


def test_carry_drops_a_purged_row_from_every_ledger(tmp_path: Path) -> None:
    """A purged row must not reach the payload, the ledgers, or the group sets."""
    empty = v16.v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    release = v16._Release(empty, tmp_path, dry_run=True)
    release.purge_plan = {7: "validation"}
    row = candidate_row("client 7", "BPD")

    release.carry(7, v16.v15.json_bytes(row))

    assert release.payload_lines == {name: [] for name in v16.SPLITS}
    assert release.internal_records == []
    assert release.rowref_records == []
    assert release.group_sets == {name: set() for name in v16.SPLITS}
    assert release.candidate_stream is None
    assert release.evaluation_purged["validation"] == 1
    assert release.evaluation_purged["test"] == 0


def _totals_ready_release(tmp_path: Path) -> v16._Release:
    """A release whose inherited totals all match, so only the purge can drift."""
    empty = v16.v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    release = v16._Release(empty, tmp_path, dry_run=True)
    release.purged = v16.v15.EXPECTED_PURGED_ROWS
    release.repair_counts = Counter(v16.v15.EXPECTED_REPAIR_COUNTS)
    return release


def _seed_emitted_splits(release: v16._Release, rows_after: int = 2) -> None:
    """Give the release the emitted split payloads and purge stats a real build ends with."""
    for split in v16.EVALUATION_SPLITS:
        release.evaluation_purge_stats[split] = {"rows_after": rows_after}
        release.payload_lines[split] = ["{}"] * rows_after


def test_verify_totals_accepts_the_verified_purge(tmp_path: Path) -> None:
    release = _totals_ready_release(tmp_path)
    release.evaluation_purged.update(v16.EXPECTED_EVALUATION_PURGED)
    _seed_emitted_splits(release)

    release.verify_totals()


def test_verify_totals_rejects_a_drifted_purge(tmp_path: Path) -> None:
    release = _totals_ready_release(tmp_path)
    release.evaluation_purged.update({"validation": 1, "test": 0})
    _seed_emitted_splits(release)

    with pytest.raises(ValueError, match="held-out purge drifted"):
        release.verify_totals()


def test_verify_totals_rejects_an_emitted_split_the_plan_does_not_expect(tmp_path: Path) -> None:
    release = _totals_ready_release(tmp_path)
    release.evaluation_purged.update(v16.EXPECTED_EVALUATION_PURGED)
    _seed_emitted_splits(release)
    release.payload_lines["validation"].append("{}")

    with pytest.raises(ValueError, match="validation: emitted 3 rows"):
        release.verify_totals()


def test_summary_reports_the_purge_per_split(tmp_path: Path) -> None:
    release = _totals_ready_release(tmp_path)
    release.evaluation_purged.update({"validation": 3, "test": 2})

    summary = release.summary()

    assert summary["evaluation_purged_rows"] == {"validation": 3, "test": 2}
    assert summary["release_id"] == v16.RELEASE_ID
