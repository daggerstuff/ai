"""Research edits must preserve the selected datum population and its order."""

import hashlib
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from training.apply_research_target_repairs import Plan, Repair, RepairInputError, apply
from training.export_research_target_repairs import Experiment, export
from training.export_tinker_release import Datum, build_tokens
from training.tests.test_export_tinker_release import StubTokenizer


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Experiment, bytes]:
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Prompt {index}.", "weight": 0},
                {"role": "assistant", "content": target, "weight": 1},
            ]
        }
        for index, target in enumerate(("Not selected.", "Original target.", "Untouched target."))
    ]
    raw = [(json.dumps(row) + "\n").encode() for row in rows]
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(raw))
    plan = Plan(
        source_train_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        review_state="agent_research_not_clinically_adjudicated",
        repairs=(
            Repair(
                row_number=2,
                row_sha256=hashlib.sha256(raw[1]).hexdigest(),
                assistant_turn=1,
                original_target_sha256=hashlib.sha256(b"Original target.").hexdigest(),
                replacement="An explicitly revised fictional research reply.",
                rationale="Test edit.",
                sources=("https://example.test/reference",),
            ),
        ),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json())
    repaired = tmp_path / "repaired.jsonl"
    apply(source, repaired, plan)
    tokenizer = StubTokenizer()
    datums = []
    for row in rows[1:]:
        ids, weights = build_tokens(tokenizer, row["messages"])
        datums.append(Datum(input_ids=ids[:-1], target_ids=ids[1:], weights=weights[1:]))
    dataset = tmp_path / "original-datums.jsonl"
    payload = b"".join((datum.line() + "\n").encode() for datum in datums)
    dataset.write_bytes(payload)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "path": str(dataset),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_train_sha256": plan.source_train_sha256,
                "tokenizer": "stub",
                "max_tokens": 512,
                "end_token": "<|im_end|>",
                "datums": len(datums),
                "source_rows": 3,
                "counts": {"datums": len(datums)},
            }
        )
    )
    experiment = Experiment(
        original_train=source,
        repaired_train=repaired,
        source_receipt=receipt,
        source_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        repair_plan=plan_path,
        repair_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        release_id="test-repair",
    )
    specification = tmp_path / "experiment.json"
    specification.write_text(experiment.model_dump_json())

    def load_tokenizer(_name: str, **_options: bool) -> StubTokenizer:
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    return specification, experiment, payload


def test_export_keeps_the_original_population_and_untouched_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specification, experiment, original = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"

    receipt = export(specification, out)

    changed, untouched = receipt.path.read_bytes().splitlines(keepends=True)
    before, preserved = original.splitlines(keepends=True)
    assert changed != before
    assert untouched == preserved
    assert receipt.datums == len(original.splitlines())
    assert receipt.source_train_sha256 == hashlib.sha256(experiment.repaired_train.read_bytes()).hexdigest()
    assert receipt.model_extra["selection_mode"] == "exact_original_export_population"
    assert receipt.counts["trained_tokens"] == sum(
        sum(json.loads(line)["weights"]) for line in receipt.path.read_bytes().splitlines()
    )


@pytest.mark.parametrize("artifact", ["original_train", "repaired_train", "source_receipt", "repair_plan"])
def test_export_rejects_changed_inputs_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    specification, experiment, _ = fixture(tmp_path, monkeypatch)
    path = getattr(experiment, artifact)
    path.write_bytes(path.read_bytes() + b"\n")
    out = tmp_path / "out"

    with pytest.raises(RepairInputError):
        export(specification, out)

    assert not out.exists()


def test_ambiguous_export_matches_cannot_duplicate_a_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    specification, experiment, original = fixture(tmp_path, monkeypatch)
    receipt = json.loads(experiment.source_receipt.read_text())
    dataset = Path(receipt["path"])
    payload = original.splitlines(keepends=True)[0] + original
    dataset.write_bytes(payload)
    receipt["sha256"] = hashlib.sha256(payload).hexdigest()
    receipt["datums"] = len(payload.splitlines())
    experiment.source_receipt.write_text(json.dumps(receipt))
    experiment = experiment.model_copy(
        update={"source_receipt_sha256": hashlib.sha256(experiment.source_receipt.read_bytes()).hexdigest()}
    )
    specification.write_text(experiment.model_dump_json())

    with pytest.raises(RepairInputError, match="exactly one existing export datum"):
        export(specification, tmp_path / "out")


def test_a_longer_repair_cannot_silently_remove_a_training_datum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specification, experiment, original = fixture(tmp_path, monkeypatch)
    receipt = json.loads(experiment.source_receipt.read_text())
    receipt["max_tokens"] = max(len(json.loads(line)["input_ids"]) + 1 for line in original.splitlines())
    experiment.source_receipt.write_text(json.dumps(receipt))
    experiment = experiment.model_copy(
        update={"source_receipt_sha256": hashlib.sha256(experiment.source_receipt.read_bytes()).hexdigest()}
    )
    specification.write_text(experiment.model_dump_json())

    with pytest.raises(RepairInputError, match="would exclude a datum"):
        export(specification, tmp_path / "out")
