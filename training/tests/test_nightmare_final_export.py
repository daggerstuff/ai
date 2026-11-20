"""Only final-judge survivors may reach the master research corpus."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from training import nightmare_fuel_generator as generator, run_nightmare_fuel_distillation as exporter
from training.nightmare_records import GateAcceptedSession


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", [True, False])
async def test_export_uses_final_judge_survivors_and_clears_stale_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accept: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NF_USE_ASYNC_JUDGE", "0")
    messages = [
        {"role": "user", "content": "A fictional prompt."},
        {"role": "assistant", "content": "A fictional reply."},
    ]
    checkpoint_only = {"id": "checkpoint-only", "scenario": "checkpoint-only", "messages": messages}
    final_accepted = {"id": "accepted", "scenario": "final-accepted", "messages": messages}
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    (checkpoint / "records.jsonl").write_text(json.dumps(checkpoint_only) + "\n")
    master = tmp_path / "master.jsonl"
    master.write_text('{"source":"existing"}\n')
    artifact = tmp_path / "ai/training/output/nightmare_fuel/synthetic_chatml.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(checkpoint_only) + "\n")
    monkeypatch.setattr(exporter, "CHECKPOINT_DIR", str(checkpoint))
    monkeypatch.setattr(exporter, "OUT_MASTER", str(master))
    monkeypatch.setattr(generator, "generate_cases_async", AsyncMock(return_value=[checkpoint_only, final_accepted]))
    frame = generator.pd.DataFrame([final_accepted] if accept else [])
    monkeypatch.setattr(generator, "_run_clinical_gate_async", AsyncMock(return_value=frame))

    await exporter.run()

    records = [json.loads(line) for line in master.read_text().splitlines()]
    assert records[0] == {"source": "existing"}
    assert len(records) == 1 + int(accept)
    if accept:
        assert records[-1]["provenance"]["scenario"] == "final-accepted"
        assert records[-1]["clinical_reviewed"] is False
        assert records[-1]["mi_quality"] == "unassessed"
        assert "dual_judge_validated" not in records[-1]["provenance"]
    else:
        assert artifact.read_text() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("trailing_newline", [True, False])
async def test_repeated_export_is_idempotent_and_preserves_jsonl_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trailing_newline: bool
) -> None:
    master = tmp_path / "master.jsonl"
    master.write_text('{"source":"existing"}' + ("\n" if trailing_newline else ""))
    session = GateAcceptedSession.model_validate(
        {
            "scenario": "Fictional scenario",
            "messages": [
                {"role": "user", "content": "Fictional prompt."},
                {"role": "assistant", "content": "Fictional response."},
            ],
        }
    )
    monkeypatch.setattr(exporter, "OUT_MASTER", str(master))
    monkeypatch.setattr(exporter, "main_async", AsyncMock(return_value=(session, session)))

    for _ in range(2):
        await exporter.run()

    existing, generated = [json.loads(line) for line in master.read_text().splitlines()]
    assert existing == {"source": "existing"}
    assert generated["messages"] == [message.model_dump() for message in session.messages]
    assert generated["clinical_reviewed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("async_judge", [True, False])
async def test_completed_checkpoint_candidates_reach_final_judge_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, async_judge: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NF_USE_ASYNC_JUDGE", str(int(async_judge)))
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    record = {
        "id": "generated-before-interruption",
        "case_index": 0,
        "requested_cases": 1,
        "scenario": "Fictional scenario",
        "messages": [
            {"role": "user", "content": "Fictional prompt."},
            {"role": "assistant", "content": "Fictional response."},
        ],
        "raw_content": "Fictional session for the judge.",
    }
    (checkpoint / "records.jsonl").write_text(json.dumps(record) + "\n")
    remote_generation = AsyncMock(side_effect=AssertionError("completed generation must not repeat"))
    monkeypatch.setattr(generator, "_generate_case", remote_generation)
    final_judge = AsyncMock(return_value=generator.pd.DataFrame([record]))
    judge_name = "_judge_cases_async" if async_judge else "_run_clinical_gate_async"
    monkeypatch.setattr(generator, judge_name, final_judge)

    accepted = await generator.main_async(num_cases=1, checkpoint_dir=str(checkpoint))
    replayed = await generator.main_async(num_cases=1, checkpoint_dir=str(checkpoint))

    remote_generation.assert_not_awaited()
    final_judge.assert_awaited_once_with([record])
    assert len(accepted) == 1
    assert accepted[0].scenario == record["scenario"]
    assert replayed == accepted
