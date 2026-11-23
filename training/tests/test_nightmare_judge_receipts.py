import json
import stat
from pathlib import Path
from unittest.mock import AsyncMock, call

import pytest
from pydantic import ValidationError

from training import nightmare_fuel_generator as generator
from training.clinical_validity_judge_async import AsyncClinicalJudge
from training.nightmare_judge_receipts import (
    Candidate,
    JudgeEvidenceError,
    JudgeReceipts,
    current_recipe,
    judge_with_receipts,
)
from training.nightmare_records import GeneratedMessage


@pytest.fixture
def candidate() -> Candidate:
    return Candidate(
        id="fictional-case",
        scenario="Fictional scenario",
        raw_content="Private fictional session payload",
        messages=(
            GeneratedMessage(role="user", content="Fictional question."),
            GeneratedMessage(role="assistant", content="Fictional response."),
        ),
    )


@pytest.fixture
def receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JudgeReceipts:
    monkeypatch.setenv("NF_BACKEND", "cloudflare")
    return JudgeReceipts(tmp_path, current_recipe(async_judge=False, prompt="Frozen rubric"))


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", [True, False])
async def test_completed_accept_and_reject_decisions_skip_rejudging(
    candidate: Candidate, receipts: JudgeReceipts, accept: bool
) -> None:
    remote = AsyncMock(return_value=(candidate.id,) if accept else ())

    first = await judge_with_receipts((candidate,), receipts, remote)
    replay = await judge_with_receipts((candidate,), receipts, remote)

    assert first == replay == ((candidate,) if accept else ())
    remote.assert_awaited_once_with((candidate,))
    artifact = next(receipts.root.glob("*.json"))
    assert candidate.raw_content not in artifact.read_text()
    assert "messages" not in artifact.read_text()
    assert stat.S_IMODE(artifact.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR
    assert stat.S_IMODE(receipts.root.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR


@pytest.mark.asyncio
async def test_new_candidate_and_recipe_changes_invalidate_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: Candidate, receipts: JudgeReceipts
) -> None:
    remote = AsyncMock(return_value=())
    await judge_with_receipts((candidate,), receipts, remote)
    changed = candidate.model_copy(update={"raw_content": "Different fictional session"})
    await judge_with_receipts((changed,), receipts, remote)
    monkeypatch.setenv("NF_MODEL", "different-model")
    different_recipe = JudgeReceipts(tmp_path, current_recipe(async_judge=False, prompt="Frozen rubric"))
    await judge_with_receipts((candidate,), different_recipe, remote)

    assert remote.await_args_list == [call((candidate,)), call((changed,)), call((candidate,))]
    assert receipts.root != different_recipe.root


@pytest.mark.asyncio
async def test_only_unjudged_candidates_are_sent_on_extension(candidate: Candidate, receipts: JudgeReceipts) -> None:
    remote = AsyncMock(return_value=())
    await judge_with_receipts((candidate,), receipts, remote)
    added = candidate.model_copy(update={"id": "added", "raw_content": "New fictional case"})

    await judge_with_receipts((candidate, added), receipts, remote)

    remote.assert_awaited_with((added,))


@pytest.mark.asyncio
async def test_failed_batch_does_not_publish_rejection_receipts(candidate: Candidate, receipts: JudgeReceipts) -> None:
    remote = AsyncMock(side_effect=JudgeEvidenceError("judge unavailable"))
    with pytest.raises(JudgeEvidenceError):
        await judge_with_receipts((candidate,), receipts, remote)
    assert list(receipts.root.glob("*.json")) == []

    remote.side_effect = None
    remote.return_value = (candidate.id,)
    assert await judge_with_receipts((candidate,), receipts, remote) == (candidate,)
    assert remote.await_args_list == [call((candidate,)), call((candidate,))]


@pytest.mark.asyncio
async def test_corrupt_receipt_stops_before_remote_request(candidate: Candidate, receipts: JudgeReceipts) -> None:
    receipts.store(candidate, accepted=True)
    artifact = next(receipts.root.glob("*.json"))
    record = json.loads(artifact.read_text())
    record["candidate_sha256"] = "0" * 64
    artifact.write_text(json.dumps(record))
    remote = AsyncMock()

    with pytest.raises(JudgeEvidenceError):
        await judge_with_receipts((candidate,), receipts, remote)
    remote.assert_not_awaited()

    artifact.write_text('{"accepted":"false"}')
    with pytest.raises(ValidationError):
        await judge_with_receipts((candidate,), receipts, remote)
    remote.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["14 (not a score)", "Score: 5", "5 and 1", ""])
async def test_direct_judge_rejects_ambiguous_score_responses(
    monkeypatch: pytest.MonkeyPatch, candidate: Candidate, response: str
) -> None:
    remote = AsyncMock(return_value=response)
    monkeypatch.setattr(generator, "_chat_completion", remote)

    with pytest.raises(JudgeEvidenceError, match="invalid score"):
        await generator._run_clinical_gate_async([candidate.model_dump()])


@pytest.mark.asyncio
@pytest.mark.parametrize(("score", "accepted"), [("3 (below the threshold)", False), ("4 (meets the threshold)", True)])
async def test_direct_judge_preserves_valid_score_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: Candidate, score: str, accepted: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai/training/output/nightmare_fuel").mkdir(parents=True)
    monkeypatch.setattr(generator, "_chat_completion", AsyncMock(return_value=score))

    survivors = await generator._run_clinical_gate_async([candidate.model_dump()])

    assert len(survivors) == int(accepted)


@pytest.mark.asyncio
async def test_direct_judge_outage_is_not_a_low_score(monkeypatch: pytest.MonkeyPatch, candidate: Candidate) -> None:
    monkeypatch.setattr(generator, "_chat_completion", AsyncMock(side_effect=TimeoutError))

    with pytest.raises(TimeoutError):
        await generator._run_clinical_gate_async([candidate.model_dump()])


@pytest.mark.asyncio
async def test_async_judge_outage_is_not_an_empty_survivor_set(
    monkeypatch: pytest.MonkeyPatch, candidate: Candidate
) -> None:
    monkeypatch.setattr(generator, "_build_nemo_config_from_env", lambda: None)
    monkeypatch.setattr(AsyncClinicalJudge, "evaluate", AsyncMock(side_effect=RuntimeError("unavailable")))

    with pytest.raises(JudgeEvidenceError, match="did not evaluate"):
        await generator._judge_cases_async([candidate.model_dump()])
