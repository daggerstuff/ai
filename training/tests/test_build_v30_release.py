"""Tests for the v30 release: the family escalates in its first sentence.

v29 scored 11/12 on both the dev split and a genuinely held-out sealed split, and
its two failures were different prompts on the same trajectory — reflect on the
disclosure, then ask a question, with no escalation anywhere. That is v26's three
reflective frames wrapping the imperative as ``"{reflection} {action}"``: 2,214
crisis rows train a reflection-first opener, and 1,448 of the 10,834 datums in
the training file teach "reflect, then question, no escalation".

The tests here pin the release side: the pipeline must be v29's unchanged except
for the addendum, the family must be the v30 builder's rows verbatim, the
prompt surface must be v29's unchanged, every shipped reply must escalate in its
first sentence, and the manifest must record the measured register defect and the
reordering that answers it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import training.build_agent_tool_sft_v29 as v29
import training.build_agent_tool_sft_v30 as v30
import training.build_v14_release as v14
import training.build_v15_release as v15
import training.build_v16_release as v16
import training.build_v17_release as v17
import training.build_v19_release as v19
import training.build_v20_release as v20
import training.build_v21_release as v21
import training.build_v22_release as v22
import training.build_v23_release as v23_release
import training.build_v24_release as v24_release
import training.build_v25_release as v25_release
import training.build_v26_release as v26_release
import training.build_v27_release as v27_release
import training.build_v28_release as v28_release
import training.build_v29_release as prev
import training.build_v30_release as release

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V30_ROWS = 6504
MESSAGES_PER_ROW = 2
# The reflective rows the reorder moves behind the imperative, and the rows the
# escalation-first invariant must therefore cover.
EXPECTED_REFLECTIVE_ROWS = 2214
EXPECTED_REPLIES = 1072
EXPECTED_DISCLOSURES = 838


def test_the_release_identity_names_v30() -> None:
    assert release.RELEASE_ID.endswith("-v30")
    assert release.RELEASE_ID in str(release.OUTPUT)
    assert release.RELEASE_ID in str(release.MANIFEST)
    assert release.SHUFFLE_SEED not in {
        prev.SHUFFLE_SEED,
        v28_release.SHUFFLE_SEED,
        v27_release.SHUFFLE_SEED,
        v26_release.SHUFFLE_SEED,
        v25_release.SHUFFLE_SEED,
        v24_release.SHUFFLE_SEED,
        v23_release.SHUFFLE_SEED,
        v22.SHUFFLE_SEED,
        v21.SHUFFLE_SEED,
        v20.SHUFFLE_SEED,
        v19.SHUFFLE_SEED,
        v17.SHUFFLE_SEED,
        v16.SHUFFLE_SEED,
    }


def test_the_source_label_is_the_v30_family() -> None:
    assert release.V30_SOURCE_LABEL == "crisis_direct_v30"
    assert release.EXPECTED_V30_TRAIN_ROWS == V30_ROWS
    assert release.V30_SOURCE_LABEL != prev.V29_SOURCE_LABEL


def test_limitations_drop_the_stale_v29_family_claims() -> None:
    """v29 described its own family's size, ratio and placement; v30 replaces it."""
    inherited = prev._limitations()
    for stale in release._STALE_V29_LIMITATIONS:
        assert any(text.startswith(stale) for text in inherited), stale
        assert not any(text.startswith(stale) for text in release._limitations()), stale
    assert len(release._limitations()) == (
        len(inherited) - len(release._STALE_V29_LIMITATIONS) + len(release.LIMITATIONS_V30)
    )


def test_limitations_keep_every_other_v29_entry() -> None:
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(release._STALE_V29_LIMITATIONS))
    assert release._limitations()[: len(kept)] == kept


def test_limitations_record_the_size_the_ratio_and_the_kept_revert() -> None:
    added = " ".join(release.LIMITATIONS_V30)
    assert "6,504 rows" in added
    assert "7.9% of train" in added
    assert "838" in added
    assert "9.8 escalations per ordinary reply" in added
    assert "not reviewed by a clinician" in added
    assert "carries no corpus repair" in added


def test_limitations_record_the_reorder_as_the_mechanism() -> None:
    added = " ".join(release.LIMITATIONS_V30)
    assert "3 reflective frames" in added or "three v26 reflective frames" in added
    assert "2,214 rows" in added
    assert "first sentence" in added
    assert "escalation-first invariant" in added


def test_limitations_record_the_competing_register_it_does_not_remove() -> None:
    added = " ".join(release.LIMITATIONS_V30)
    assert "1,448" in added
    assert "13.4%" in added
    assert "312 datums" in added
    assert "purge" in added.lower() or "purging" in added.lower()


def test_limitations_record_that_the_opener_is_now_uniform() -> None:
    added = " ".join(release.LIMITATIONS_V30)
    assert "140 crisis actions" in added
    assert "uniformity at the opener" in added
    assert "second sentence" in added


def test_limitations_keep_the_closed_lists_and_the_caps() -> None:
    added = " ".join(release.LIMITATIONS_V30)
    assert "closed lists" in added
    assert "37 abstract psychosis" in added
    assert "1 abstract means_access" in added
    assert "appended to train only" in added


def test_the_pipeline_is_v29s_with_only_the_family_swapped() -> None:
    assert issubclass(release._Release, prev._Release)
    # The pipeline itself is inherited untouched; only the addendum and the two
    # counters that name the family change.
    for name in ("run", "write", "append_chat_row", "append_agent_row"):
        assert getattr(release._Release, name) is getattr(prev._Release, name), name
    for name in ("append_addenda", "verify_totals", "summary", "manifest"):
        assert getattr(release._Release, name) is not getattr(prev._Release, name), name


def test_the_addendum_is_v30_and_the_older_families_are_gone() -> None:
    assert release.V30_SOURCE_LABEL not in {prev.V29_SOURCE_LABEL, v14.V9_SOURCE_LABEL}
    assert not hasattr(release, "V29_SOURCE_LABEL")


def test_v30_candidate_rows_are_clean_chat_rows() -> None:
    rows = release.v30_candidate_rows()
    assert len(rows) == V30_ROWS
    for candidate, payload in rows:
        assert set(candidate) == {"messages", "source"}
        assert candidate["source"] == release.V30_SOURCE_LABEL
        assert not v14.is_agent_tool(candidate)
        assert len(candidate["messages"]) == MESSAGES_PER_ROW
        assert candidate["messages"][0]["role"] == "user"
        assert candidate["messages"][1]["role"] == "assistant"
        assert payload, "a v30 row must project to an assistant target"


def test_the_family_is_the_v30_builder_rows_verbatim() -> None:
    """The release must ship what the builder built, not a re-derived copy."""
    built = v30.collect_rows()["train"]
    shipped = [candidate for candidate, _ in release.v30_candidate_rows()]
    assert len(shipped) == len(built) == V30_ROWS
    for candidate, item in zip(shipped, built, strict=True):
        assert candidate["messages"] == item["messages"]


def test_the_family_is_built_by_the_v30_builder() -> None:
    """Guard against a copy-paste that leaves the v29 builder wired in."""
    assert release.v30 is v30
    assert release.v30 is not v29
    assert release.v30.FAMILY == v29.FAMILY
    assert release.RELEASE_ID != v30.RELEASE_ID


def test_the_prompt_surface_is_v29s_unchanged() -> None:
    """Only the reply order moves: the disclosure the model sees must not."""
    v29_rows = prev.v29_candidate_rows()
    v30_rows = release.v30_candidate_rows()
    assert len(v29_rows) == len(v30_rows)
    for (v29_candidate, _), (v30_candidate, _) in zip(v29_rows, v30_rows, strict=True):
        assert v29_candidate["messages"][0] == v30_candidate["messages"][0]


def test_every_shipped_crisis_reply_escalates_in_its_first_sentence() -> None:
    control_replies = {
        item["messages"][1]["content"] for item in v30.collect_rows()["train"] if item["category"] == "control"
    }
    crisis = [
        candidate
        for candidate, _ in release.v30_candidate_rows()
        if candidate["messages"][1]["content"] not in control_replies
    ]
    assert len(crisis) == V30_ROWS - 600
    for candidate in crisis:
        reply = candidate["messages"][1]["content"]
        assert v30.escalates_in_first_sentence(reply), reply


def test_no_shipped_reply_opens_on_a_reflection() -> None:
    for candidate, _ in release.v30_candidate_rows():
        reply = candidate["messages"][1]["content"]
        assert not reply.startswith(v30.REFLECTIVE_TAILS), reply


def test_the_reflective_rows_carry_the_reflection_after_the_imperative() -> None:
    replies = [candidate["messages"][1]["content"] for candidate, _ in release.v30_candidate_rows()]
    reflective = [reply for reply in replies if reply.endswith(v30.REFLECTIVE_TAILS)]
    assert len(reflective) == EXPECTED_REFLECTIVE_ROWS
    for reply in reflective:
        assert v30.escalates_in_first_sentence(reply), reply


def test_the_shipped_surface_keeps_v29s_reply_count_and_disclosures() -> None:
    candidates = [candidate for candidate, _ in release.v30_candidate_rows()]
    replies = {candidate["messages"][1]["content"] for candidate in candidates}
    disclosures = {candidate["messages"][0]["content"] for candidate in candidates}
    assert len(replies) == EXPECTED_REPLIES
    assert len(disclosures) == EXPECTED_DISCLOSURES


def test_the_builder_reports_the_escalation_first_counts() -> None:
    summary = v30.summarize(v30.collect_rows())
    assert summary["reflection_first_crisis_rows"] == 0
    assert summary["escalation_first_crisis_rows"] == summary["crisis_rows"]


def sample_manifest() -> dict[str, Any]:
    return {
        "scope": {"appended_rows": 0, "appended_rows_by_source": {}},
        "payload_contract": {"shuffle": {"train": False, "validation": False, "test": False, "seed": 0}},
        "limitations": list(prev._limitations()),
        "owner_direction": "User-directed v29 after the v28 eval.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v30_rows=V30_ROWS
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
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v30_rows=V30_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        release.V30_SOURCE_LABEL: V30_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V30_ROWS
    assert prev.V29_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]
    assert v28_release.V28_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]


def test_the_owner_direction_records_the_measured_cause_and_the_fix() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v30_rows=V30_ROWS
    )
    direction = stamped["owner_direction"]
    assert "Neither escalates" in direction
    assert "2,294 of the 6,000 crisis rows" in direction
    assert "2,214 rows training a reflection-first opener" in direction
    assert "1,448 datums (13.4%)" in direction
    assert "312 datums disclose an active crisis" in direction
    assert "moves those three reflections behind the imperative" in direction
    assert "escalation_first_crisis_rows" in direction
    assert "reflection_first_crisis_rows at 0" in direction


def release_fixture(tmp_path: Path) -> release._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return release._Release(indexes, tmp_path, dry_run=True)


def test_summary_reports_the_v30_release(tmp_path: Path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.v30_rows = release.v30_candidate_rows()

    summary = fixture.summary()

    assert summary["release_id"] == release.RELEASE_ID
    assert summary["appended_v30_rows"] == V30_ROWS
    assert "appended_v29_rows" not in summary
