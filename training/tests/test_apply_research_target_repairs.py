"""Explicit repairs must not mutate prompts, weights, metadata, or other rows."""

import hashlib
import json
from pathlib import Path
from typing import Literal, assert_never

import pytest

from training.apply_research_target_repairs import Plan, Repair, RepairInputError, apply


def inputs(tmp_path: Path) -> tuple[Path, Plan, bytes]:
    row = {
        "messages": [
            {"role": "user", "content": "Fictional prompt.", "weight": 0},
            {"role": "assistant", "content": "Old target.", "weight": 1},
        ],
        "extra": {"kept": True},
    }
    first = (json.dumps(row) + "\n").encode()
    untouched = b'{"messages": [], "other": "keep these exact bytes"}\n'
    source = tmp_path / "train.jsonl"
    source.write_bytes(first + untouched)
    plan = Plan(
        source_train_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        review_state="agent_research_not_clinically_adjudicated",
        repairs=(
            Repair(
                row_number=1,
                row_sha256=hashlib.sha256(first).hexdigest(),
                assistant_turn=1,
                original_target_sha256=hashlib.sha256(b"Old target.").hexdigest(),
                replacement="A revised fictional target.",
                rationale="Synthetic test revision.",
                sources=("https://example.test/reference",),
            ),
        ),
    )
    return source, plan, untouched


def test_repairs_only_change_the_bound_assistant_target(tmp_path: Path) -> None:
    source, plan, untouched = inputs(tmp_path)
    before = source.read_bytes()
    out = tmp_path / "new.jsonl"

    result = apply(source, out, plan)

    assert source.read_bytes() == before
    edited, preserved = out.read_bytes().splitlines(keepends=True)
    assert preserved == untouched
    old_row, new_row = json.loads(before.splitlines()[0]), json.loads(edited)
    old_row["messages"][1]["content"] = "A revised fictional target."
    assert new_row == old_row
    assert result.changed_targets == 1
    assert result.output_sha256 == hashlib.sha256(out.read_bytes()).hexdigest()


@pytest.mark.parametrize("mismatch", ["source", "row", "target", "role", "duplicate"])
def test_identity_mismatches_never_publish_an_output(
    tmp_path: Path, mismatch: Literal["source", "row", "target", "role", "duplicate"]
) -> None:
    source, plan, _ = inputs(tmp_path)
    repair = plan.repairs[0]
    match mismatch:
        case "source":
            plan = plan.model_copy(update={"source_train_sha256": "0" * 64})
        case "row":
            repair = repair.model_copy(update={"row_sha256": "0" * 64})
            plan = plan.model_copy(update={"repairs": (repair,)})
        case "target":
            repair = repair.model_copy(update={"original_target_sha256": "0" * 64})
            plan = plan.model_copy(update={"repairs": (repair,)})
        case "role":
            repair = repair.model_copy(update={"assistant_turn": 0})
            plan = plan.model_copy(update={"repairs": (repair,)})
        case "duplicate":
            plan = plan.model_copy(update={"repairs": (repair, repair)})
        case unreachable:
            assert_never(unreachable)
    out = tmp_path / "new.jsonl"

    with pytest.raises(RepairInputError):
        apply(source, out, plan)

    assert not out.exists()


def test_an_existing_output_is_preserved(tmp_path: Path) -> None:
    source, plan, _ = inputs(tmp_path)
    out = tmp_path / "new.jsonl"
    out.write_bytes(b"existing artifact")

    with pytest.raises(FileExistsError):
        apply(source, out, plan)

    assert out.read_bytes() == b"existing artifact"
