"""Resume decides pending slots before generation and never invents legacy identity."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from training import nightmare_fuel_generator as generator
from training.nightmare_recovery import CheckpointIdentity, ResumeIdentityError, pending_indices


def test_pending_indices_preserve_completed_slots() -> None:
    records = (CheckpointIdentity(id="finished", case_index=1, requested_cases=3),)

    assert pending_indices(3, records, frozenset({"finished"})) == (0, 2)


@pytest.mark.parametrize(
    "record",
    [
        CheckpointIdentity(id="old"),
        CheckpointIdentity(id="old", case_index=1, requested_cases=20),
    ],
)
def test_ambiguous_checkpoint_identity_is_rejected(record: CheckpointIdentity) -> None:
    with pytest.raises(ResumeIdentityError):
        pending_indices(3, (record,), frozenset({"old"}))


@pytest.mark.asyncio
async def test_completed_slots_are_skipped_before_remote_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = generator.CheckpointManager(tmp_path)
    requested_total = 3
    messages = [
        {"role": "user", "content": "Fictional client."},
        {"role": "assistant", "content": "Fictional response."},
    ]
    old = {
        "id": "finished",
        "case_index": 0,
        "requested_cases": requested_total,
        "scenario": "fiction",
        "messages": messages,
    }
    checkpoint.records_path.write_text(json.dumps(old) + "\n")
    scenario_api = AsyncMock(return_value="Fictional generated scenario")
    session_api = AsyncMock(return_value={"messages": messages})
    monkeypatch.setattr(generator, "generate_nightmare_scenario_async", scenario_api)
    monkeypatch.setattr(generator, "simulate_therapy_session_async", session_api)
    monkeypatch.setattr(generator, "reject_reason_for_record", lambda *_args, **_kwargs: None)

    generated = await generator.generate_cases_async(
        num_cases=requested_total, concurrency=1, checkpoint=checkpoint, skip_ids={"finished"}
    )

    assert [row["case_index"] for row in generated] == [1, 2]
    assert all(row["requested_cases"] == requested_total for row in generated)
    persisted = [json.loads(line) for line in checkpoint.records_path.read_text().splitlines()]
    assert [row["case_index"] for row in persisted] == [0, 1, 2]
    assert scenario_api.await_count == len(generated)


@pytest.mark.asyncio
async def test_legacy_resume_fails_before_any_remote_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = generator.CheckpointManager(tmp_path)
    checkpoint.records_path.write_text('{"id":"legacy-without-slot"}\n')
    remote = AsyncMock(side_effect=AssertionError("a model request must not run"))
    monkeypatch.setattr(generator, "_generate_case", remote)

    with pytest.raises(ResumeIdentityError, match="legacy checkpoint"):
        await generator.generate_cases_async(num_cases=3, checkpoint=checkpoint, skip_ids={"legacy-without-slot"})

    remote.assert_not_awaited()


@pytest.mark.asyncio
async def test_fully_completed_generation_makes_no_remote_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = generator.CheckpointManager(tmp_path)
    records = [{"id": f"finished-{index}", "case_index": index, "requested_cases": 3} for index in range(3)]
    checkpoint.records_path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    remote = AsyncMock(side_effect=AssertionError("completed slots must not be regenerated"))
    monkeypatch.setattr(generator, "_generate_case", remote)

    result = await generator.generate_cases_async(
        num_cases=3, checkpoint=checkpoint, skip_ids={row["id"] for row in records}
    )

    assert result == []
    remote.assert_not_awaited()
