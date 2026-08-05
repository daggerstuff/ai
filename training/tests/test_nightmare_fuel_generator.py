"""Tests for nightmare_fuel_generator async rewrite + PIX-4235 checkpointing."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from training import nightmare_fuel_generator as nfg
from training.nightmare_fuel_generator import CheckpointManager, GenerationState


def _mock_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = AsyncMock(return_value=payload)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return response


class TestParseTranscript:
    def test_parse_transcript_splits_roles(self):
        transcript = (
            "Patient: I don't want to talk.\n"
            "Therapist: I hear that this feels difficult.\n"
            "Patient: You don't understand me."
        )
        parsed = nfg._parse_transcript(transcript, "scenario")
        assert parsed["scenario"] == "scenario"
        assert parsed["messages"] == [
            {"role": "user", "content": "I don't want to talk."},
            {"role": "assistant", "content": "I hear that this feels difficult."},
            {"role": "user", "content": "You don't understand me."},
        ]


class TestAsyncGeneration:
    @pytest.mark.asyncio
    async def test_generate_nightmare_scenario_async(self):
        session = MagicMock()
        session.post = MagicMock(
            return_value=_mock_response(
                {"choices": [{"message": {"content": "A resistant patient with comorbidities."}}]}
            )
        )

        scenario = await nfg.generate_nightmare_scenario_async(session)
        assert scenario == "A resistant patient with comorbidities."
        session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_simulate_therapy_session_async(self):
        session = MagicMock()
        session.post = MagicMock(
            return_value=_mock_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Patient: Leave me alone.\n"
                                    "Therapist: I can stay with you while we figure out next steps."
                                )
                            }
                        }
                    ]
                }
            )
        )

        result = await nfg.simulate_therapy_session_async(session, "scenario")
        assert result["scenario"] == "scenario"
        assert len(result["messages"]) == 2

    @pytest.mark.asyncio
    async def test_generate_cases_async_runs_concurrently(self):
        call_count = 0
        in_flight = 0
        max_in_flight = 0

        async def fake_scenario(*_args, **_kwargs):
            nonlocal call_count, in_flight, max_in_flight
            call_count += 1
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return f"scenario-{call_count}"

        async def fake_session(_session, scenario):
            return {
                "scenario": scenario,
                "messages": [
                    {"role": "user", "content": "Patient line"},
                    {"role": "assistant", "content": "Therapist line"},
                ],
            }

        with (
            patch.object(nfg, "generate_nightmare_scenario_async", side_effect=fake_scenario),
            patch.object(nfg, "simulate_therapy_session_async", side_effect=fake_session),
        ):
            cases = await nfg.generate_cases_async(num_cases=3, concurrency=3)

        assert len(cases) == 3
        assert call_count == 3
        assert max_in_flight > 1, "concurrency not achieved: tasks ran sequentially"


class TestSyncWrappers:
    def test_generate_nightmare_scenario_sync_wrapper(self):
        with patch.object(
            nfg,
            "generate_nightmare_scenario_async",
            new=AsyncMock(return_value="sync scenario"),
        ):
            assert nfg.generate_nightmare_scenario() == "sync scenario"

    def test_main_runs_async_pipeline(self, tmp_path):
        """main() should call main_async which drives the pipeline.

        We patch sys.argv to provide argparse defaults and patch the async
        entrypoint so we do not exercise the network or the disk pipeline.
        """

        with (
            patch("sys.argv", ["nightmare_fuel_generator"]),
            patch.object(nfg, "main_async", new=AsyncMock()) as main_async_mock,
        ):
            nfg.main()

        main_async_mock.assert_called_once()


class TestGenerationState:
    def test_state_round_trip(self):
        state = GenerationState(current_category="anxiety")
        state.total_attempted = 10
        state.total_validated = 7
        state.total_rejected = 3
        dumped = state.to_dict()
        loaded = GenerationState.from_dict(dumped)
        assert loaded.batch_id == state.batch_id
        assert loaded.current_category == "anxiety"
        assert loaded.total_attempted == 10
        assert loaded.total_validated == 7
        assert loaded.total_rejected == 3

    def test_state_from_dict_ignores_unknown_fields(self):
        state = GenerationState.from_dict({"batch_id": "abc", "unknown_field": "ignored"})
        assert state.batch_id == "abc"

    def test_state_defaults(self):
        state = GenerationState()
        assert state.current_category == "default"
        assert state.total_attempted == 0
        assert state.total_validated == 0
        assert state.total_rejected == 0


class TestCheckpointManager:
    @pytest.mark.asyncio
    async def test_flush_writes_records_and_state(self, tmp_path):
        manager = CheckpointManager(tmp_path, interval_records=1, interval_seconds=0)
        record = {"id": "rec-1", "scenario": "s", "messages": []}
        await manager.record_validated(record)
        # record_validated with interval_records=1 triggers a flush
        assert manager.records_path.exists()
        lines = manager.records_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == "rec-1"
        assert manager.state_path.exists()
        state = json.loads(manager.state_path.read_text())
        assert state["total_validated"] == 1
        assert state["total_attempted"] == 0

    @pytest.mark.asyncio
    async def test_rejected_and_attempted_counts(self, tmp_path):
        manager = CheckpointManager(tmp_path, interval_records=100, interval_seconds=0)
        await manager.record_attempted()
        await manager.record_attempted()
        await manager.record_rejected()
        await manager.flush()
        state = json.loads(manager.state_path.read_text())
        assert state["total_attempted"] == 2
        assert state["total_rejected"] == 1
        assert state["total_validated"] == 0

    @pytest.mark.asyncio
    async def test_load_existing_records_round_trip(self, tmp_path):
        manager = CheckpointManager(tmp_path, interval_records=100, interval_seconds=0)
        await manager.record_validated({"id": "a", "scenario": "s1", "messages": []})
        await manager.record_validated({"id": "b", "scenario": "s2", "messages": []})
        await manager.finalize()

        manager2 = CheckpointManager(tmp_path, interval_records=100, interval_seconds=0)
        records = manager2.load_existing_records()
        assert {r["id"] for r in records} == {"a", "b"}
        assert manager2.existing_record_ids() == {"a", "b"}

    @pytest.mark.asyncio
    async def test_resume_skips_existing_ids(self, tmp_path):
        """generate_cases_async with checkpoint should not return already-checkpointed records."""
        # Seed a checkpoint with one record already present.
        manager = CheckpointManager(tmp_path, interval_records=100, interval_seconds=0)
        await manager.record_validated({"id": "existing", "scenario": "seed", "messages": []})
        await manager.finalize()
        skip_ids = manager.existing_record_ids()

        async def fake_scenario(*_args, **_kwargs):
            return "scenario"

        async def fake_session(_session, scenario):
            return {
                "scenario": scenario,
                "messages": [
                    {"role": "user", "content": "Patient line"},
                    {"role": "assistant", "content": "Therapist line"},
                ],
            }

        with (
            patch.object(nfg, "generate_nightmare_scenario_async", side_effect=fake_scenario),
            patch.object(nfg, "simulate_therapy_session_async", side_effect=fake_session),
            patch.object(nfg.uuid, "uuid4", return_value="existing"),
        ):
            cases = await nfg.generate_cases_async(num_cases=1, concurrency=1, checkpoint=manager, skip_ids=skip_ids)

        # The one generated case collides with the seeded id and must be skipped.
        assert cases == []

    @pytest.mark.asyncio
    async def test_checkpoint_without_existing_file(self, tmp_path):
        """A brand new checkpoint directory should produce an empty existing set."""
        manager = CheckpointManager(tmp_path / "fresh", interval_records=1, interval_seconds=0)
        assert manager.existing_record_ids() == set()
        assert manager.load_existing_records() == []

    @pytest.mark.asyncio
    async def test_should_flush_respects_record_threshold(self, tmp_path):
        manager = CheckpointManager(tmp_path, interval_records=2, interval_seconds=0)
        await manager.record_validated({"id": "1", "scenario": "s", "messages": []})
        assert not manager.records_path.exists(), "should not flush before threshold"
        await manager.record_validated({"id": "2", "scenario": "s", "messages": []})
        assert manager.records_path.exists(), "should flush at threshold"

    @pytest.mark.asyncio
    async def test_should_flush_respects_time_threshold(self, tmp_path):
        manager = CheckpointManager(tmp_path, interval_records=100, interval_seconds=0.01)
        await manager.record_validated({"id": "1", "scenario": "s", "messages": []})
        assert not manager.records_path.exists()
        await asyncio.sleep(0.02)
        await manager.record_validated({"id": "2", "scenario": "s", "messages": []})
        assert manager.records_path.exists()
        lines = manager.records_path.read_text().strip().splitlines()
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_finalize_writes_pending_records(self, tmp_path):
        manager = CheckpointManager(tmp_path, interval_records=1000, interval_seconds=1000)
        pending = [
            {"id": "p1", "scenario": "s", "messages": []},
            {"id": "p2", "scenario": "s", "messages": []},
        ]
        await manager.finalize(extra_records=pending)
        lines = manager.records_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert manager.state_path.exists()

    @pytest.mark.asyncio
    async def test_malformed_lines_are_skipped_on_load(self, tmp_path):
        manager = CheckpointManager(tmp_path, interval_records=100, interval_seconds=0)
        # Write a valid record then a corrupt line.
        with manager.records_path.open("w") as f:
            f.write(json.dumps({"id": "good"}) + "\n")
            f.write("not json\n")
        records = manager.load_existing_records()
        assert records == [{"id": "good"}]

    @pytest.mark.asyncio
    async def test_corrupt_state_file_falls_back_to_fresh(self, tmp_path):
        manager = CheckpointManager(tmp_path, interval_records=100, interval_seconds=0)
        manager.state_path.write_text("not json")
        manager2 = CheckpointManager(tmp_path, interval_records=100, interval_seconds=0)
        assert manager2.state.total_validated == 0

    @pytest.mark.asyncio
    async def test_generate_cases_async_swallows_exception_and_continues(self, tmp_path):
        """A network failure on one case must not crash the loop or skip finalize."""

        call_count = 0

        async def flaky_generate_case(_session, _sem, _idx, _total):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            return {
                "id": "ok",
                "scenario": "s",
                "messages": [
                    {"role": "user", "content": "Patient line"},
                    {"role": "assistant", "content": "Therapist line"},
                ],
            }

        manager = CheckpointManager(tmp_path, interval_records=100, interval_seconds=0)
        with patch.object(nfg, "_generate_case", side_effect=flaky_generate_case):
            cases = await nfg.generate_cases_async(num_cases=2, concurrency=1, checkpoint=manager)

        # The failed case is recorded as rejected; the successful case is returned.
        assert len(cases) == 1
        assert cases[0]["id"] == "ok"
        assert manager.state.total_rejected == 1
        assert manager.state.total_attempted == 1
        assert manager.state.total_validated == 1
        # finalize ran despite the exception — records.jsonl was written.
        assert manager.records_path.exists()

    @pytest.mark.asyncio
    async def test_resume_false_does_not_skip_existing_ids(self, tmp_path):
        """When skip_ids is empty (resume=False path), existing records are regenerated."""

        async def fake_scenario(*_args, **_kwargs):
            return "scenario"

        async def fake_session(_session, scenario):
            return {
                "scenario": scenario,
                "messages": [
                    {"role": "user", "content": "Patient line"},
                    {"role": "assistant", "content": "Therapist line"},
                ],
            }

        with (
            patch.object(nfg, "generate_nightmare_scenario_async", side_effect=fake_scenario),
            patch.object(nfg, "simulate_therapy_session_async", side_effect=fake_session),
            patch.object(nfg.uuid, "uuid4", return_value="existing"),
        ):
            cases = await nfg.generate_cases_async(num_cases=1, concurrency=1, skip_ids=set())

        # No skip_ids passed → case is returned even though id == "existing".
        assert len(cases) == 1
        assert cases[0]["id"] == "existing"
