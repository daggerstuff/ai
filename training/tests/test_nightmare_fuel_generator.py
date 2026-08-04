"""Tests for nightmare_fuel_generator async rewrite."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from training import nightmare_fuel_generator as nfg


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

        async def fake_scenario(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
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


class TestSyncWrappers:
    def test_generate_nightmare_scenario_sync_wrapper(self):
        with patch.object(
            nfg,
            "generate_nightmare_scenario_async",
            new=AsyncMock(return_value="sync scenario"),
        ):
            assert nfg.generate_nightmare_scenario() == "sync scenario"

    def test_main_runs_async_pipeline(self):
        with (
            patch.object(nfg, "generate_cases_async", new=AsyncMock(return_value=[{"id": "1"}])),
            patch.object(nfg, "_run_clinical_gate", return_value=MagicMock(empty=False, iterrows=lambda: [])),
            patch.object(nfg, "_export_survivors") as export_mock,
        ):
            nfg.main()

        export_mock.assert_called_once()
