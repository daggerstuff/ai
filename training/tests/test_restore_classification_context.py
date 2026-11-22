"""The classification-context restore must prepend the source instruction only."""

import hashlib
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from training.apply_research_target_repairs import RepairInputError
from training.export_tinker_release import Datum, build_tokens
from training.restore_classification_context import (
    Experiment,
    RestorePlan,
    RestoreRow,
    apply,
    export,
    restore_messages,
)
from training.tests.test_export_tinker_release import StubTokenizer

INSTRUCTION = "You are a clinical classification assistant. Classify the symptoms."
EXPECTED_ROWS = 3
ROWS = [
    {
        "messages": [
            {"role": "user", "content": f"Prompt {index}.", "weight": 0},
            {"role": "assistant", "content": f"Label {index}", "weight": 1},
        ]
    }
    for index in range(3)
]


def _raw_rows() -> list[bytes]:
    return [(json.dumps(row) + "\n").encode() for row in ROWS]


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Experiment, bytes, list[bytes]]:
    raw = _raw_rows()
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(raw))
    plan = RestorePlan(
        schema_version="classification-context-restore-v1",
        source_train_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        system_instruction=INSTRUCTION,
        system_instruction_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        review_state="agent_research_not_clinically_adjudicated",
        clinical_reviewed=False,
        rows=(
            RestoreRow(row_number=2, row_sha256=hashlib.sha256(raw[1]).hexdigest(), selected_line=2, candidate_row=99),
        ),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json())
    restored = tmp_path / "restored.jsonl"
    apply(source, restored, plan)

    tokenizer = StubTokenizer()
    datums = []
    for row in ROWS:
        ids, weights = build_tokens(tokenizer, row["messages"])
        datums.append(Datum(input_ids=ids[:-1], target_ids=ids[1:], weights=weights[1:]))
    dataset = tmp_path / "base-datums.jsonl"
    payload = b"".join((datum.line() + "\n").encode() for datum in datums)
    dataset.write_bytes(payload)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "path": str(dataset),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_train_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "tokenizer": "stub",
                "max_tokens": 512,
                "end_token": "<|im_end|>",
                "datums": len(datums),
                "source_rows": len(ROWS),
                "counts": {"datums": len(datums), "trained_tokens": sum(sum(d.weights) for d in datums)},
            }
        )
    )
    experiment = Experiment(
        original_train=source,
        restored_train=restored,
        source_receipt=receipt,
        source_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        restore_plan=plan_path,
        restore_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        release_id="test-restore",
    )
    specification = tmp_path / "experiment.json"
    specification.write_text(experiment.model_dump_json())

    def load_tokenizer(_name: str, **_options: object) -> StubTokenizer:
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    return specification, experiment, payload, raw


def test_apply_prepends_the_instruction_and_preserves_every_other_byte(tmp_path: Path) -> None:
    raw = _raw_rows()
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(raw))
    plan = RestorePlan(
        schema_version="classification-context-restore-v1",
        source_train_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        system_instruction=INSTRUCTION,
        system_instruction_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        review_state="agent_research_not_clinically_adjudicated",
        clinical_reviewed=False,
        rows=(
            RestoreRow(
                row_number=2, row_sha256=hashlib.sha256(raw[1]).hexdigest(), selected_line=2, candidate_row=99
            ),
        ),
    )

    out = tmp_path / "restored.jsonl"
    result = apply(source, out, plan)

    lines = out.read_bytes().splitlines(keepends=True)
    assert lines[0] == raw[0]
    assert lines[2] == raw[2]
    restored = json.loads(lines[1])
    assert restored["messages"][0] == {"role": "system", "content": INSTRUCTION, "weight": 0}
    assert [m["role"] for m in restored["messages"]] == ["system", "user", "assistant"]
    assert result.changed_rows == 1
    assert result.rows == EXPECTED_ROWS
    assert hashlib.sha256(out.read_bytes()).hexdigest() == result.output_sha256


def test_apply_refuses_to_double_restore(tmp_path: Path) -> None:
    raw = _raw_rows()
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(raw))
    plan = RestorePlan(
        schema_version="classification-context-restore-v1",
        source_train_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        system_instruction=INSTRUCTION,
        system_instruction_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        review_state="agent_research_not_clinically_adjudicated",
        clinical_reviewed=False,
        rows=(
            RestoreRow(
                row_number=1, row_sha256=hashlib.sha256(raw[0]).hexdigest(), selected_line=1, candidate_row=1
            ),
        ),
    )
    first = tmp_path / "restored.jsonl"
    apply(source, first, plan)
    restored_line = first.read_bytes().splitlines(keepends=True)[0]
    # rebuild a plan bound to the restored corpus; the row now carries a system message
    double = plan.model_copy(update={
        "source_train_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        "rows": (plan.rows[0].model_copy(update={"row_sha256": hashlib.sha256(restored_line).hexdigest()}),),
    })
    with pytest.raises(RepairInputError, match="already carries a system message"):
        apply(first, tmp_path / "twice.jsonl", double)


def test_restore_messages_keeps_weight_zero() -> None:
    messages = [
        {"role": "user", "content": "Hi", "weight": 0},
        {"role": "assistant", "content": "Depression", "weight": 1},
    ]
    restored = restore_messages(messages, INSTRUCTION)
    assert restored[0] == {"role": "system", "content": INSTRUCTION, "weight": 0}
    assert list(restored[1:]) == messages


def test_export_retokenizes_only_the_restored_datum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    specification, experiment, payload, _ = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"

    receipt = export(specification, out)

    before = payload.splitlines()
    after = (out / "tinker-sft-train.jsonl").read_bytes().splitlines()
    assert len(after) == len(before) == EXPECTED_ROWS
    changed = [i for i, (old, new) in enumerate(zip(before, after, strict=True)) if old != new]
    assert changed == [1]
    assert json.loads(after[0]) == json.loads(before[0])
    assert json.loads(after[2]) == json.loads(before[2])
    restored = json.loads(after[1])
    original = json.loads(before[1])
    assert sum(restored["weights"]) == sum(original["weights"])
    assert (receipt.model_extra or {})["restore_plan_sha256"]
    assert (receipt.model_extra or {})["changed_datums"] == 1
    assert receipt.source_train_sha256 == hashlib.sha256(experiment.restored_train.read_bytes()).hexdigest()
    assert receipt.counts["trained_tokens"] == sum(sum(json.loads(line)["weights"]) for line in after)


def test_export_rejects_a_drifted_restored_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    specification, experiment, _, _ = fixture(tmp_path, monkeypatch)
    path = experiment.restored_train
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(RepairInputError):
        export(specification, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_export_refuses_to_overwrite_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    specification, _, _, _ = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(FileExistsError):
        export(specification, out)
