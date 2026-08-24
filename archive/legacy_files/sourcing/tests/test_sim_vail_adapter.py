"""Tests for the SIM-VAIL dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai.sourcing.dataset_adapters.sim_vail_adapter import SIMVAILAdapter

_NUM_SAMPLE_TRANSCRIPTS = 2


def _make_transcript(
    transcript_id: str = "test-001",
    target_model: str = "claude-3.7-sonnet",
    description: str = "depressive thinking patterns",
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "metadata": {
            "transcript_id": transcript_id,
            "auditor_model": "openrouter/anthropic/claude-sonnet-4.5",
            "target_model": f"openrouter/anthropic/{target_model}",
            "description": description,
            "judge_output": {"response": "highlights here"},
        },
        "transcript": messages
        or [
            {"role": "user", "content": "I feel hopeless"},
            {"role": "assistant", "content": "Have you considered professional help?"},
        ],
    }


def _setup_repo(raw_dir: Path, transcripts: list[dict[str, Any]]) -> Path:
    repo_dir = raw_dir / "sim-vail"
    model_dir = repo_dir / "data" / "v1" / "claude-3.7-sonnet"
    model_dir.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(transcripts):
        (model_dir / f"transcript_test_{i}.json").write_text(json.dumps(t), encoding="utf-8")
    return repo_dir


@pytest.fixture
def sample_transcripts():
    return [
        _make_transcript(
            transcript_id="t1",
            description="depressive thinking patterns who invites validation",
            messages=[
                {"role": "user", "content": "Nothing ever works out"},
                {"role": "assistant", "content": "That sounds really hard."},
            ],
        ),
        _make_transcript(
            transcript_id="t2",
            description="psychosis phenotype with paranoia",
            messages=[
                {"role": "user", "content": "I hear voices"},
                {"role": "assistant", "content": "Can you tell me more?"},
            ],
        ),
    ]


@pytest.fixture
def adapter(tmp_path):
    return SIMVAILAdapter("sim_vail", tmp_path)


class TestSIMVAILAdapter:
    def test_download_skips_when_repo_exists(self, adapter):
        repo_dir = adapter._raw_dir / "sim-vail"
        repo_dir.mkdir(parents=True, exist_ok=True)
        with patch("subprocess.run") as mock_run:
            adapter.download()
            mock_run.assert_not_called()

    def test_download_calls_git_clone(self, adapter):
        with patch("subprocess.run") as mock_run:
            adapter.download()
            assert mock_run.called
            args = mock_run.call_args[0][0]
            assert "git" in args
            assert "clone" in args

    def test_extract_reads_transcripts(self, adapter, sample_transcripts):
        _setup_repo(adapter._raw_dir, sample_transcripts)
        records = adapter.extract()
        assert len(records) == _NUM_SAMPLE_TRANSCRIPTS
        assert records[0]["_target_model"] == "claude-3.7-sonnet"

    def test_convert_basic(self, adapter, sample_transcripts):
        records = adapter.convert_to_chatml(sample_transcripts)
        assert len(records) == _NUM_SAMPLE_TRANSCRIPTS

        rec0 = records[0]
        assert rec0["source"] == "sim_vail"
        assert rec0["task_type"] == "adversarial_safety"
        assert rec0["messages"][0]["role"] == "system"
        assert "depressive" in rec0["messages"][0]["content"]
        assert rec0["messages"][1]["role"] == "user"
        assert rec0["messages"][2]["role"] == "assistant"
        assert rec0["phenotype"] == "depressive"
        assert rec0["target_model"] == "openrouter/anthropic/claude-3.7-sonnet"

    def test_no_transcript_skipped(self, adapter):
        raw = [{"metadata": {"target_model": "x"}, "transcript": []}]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_missing_roles_skipped(self, adapter):
        raw = [
            {
                "metadata": {"target_model": "x"},
                "transcript": [
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "Anyone?"},
                ],
            }
        ]
        records = adapter.convert_to_chatml(raw)
        assert len(records) == 0

    def test_provenance_present(self, adapter, sample_transcripts):
        records = adapter.convert_to_chatml(sample_transcripts)
        assert "provenance" in records[0]
        assert records[0]["provenance"]["source_url"].startswith("https://github.com/veithweilnhammer")

    def test_full_run(self, adapter, sample_transcripts):
        _setup_repo(adapter._raw_dir, sample_transcripts)
        output_path = adapter.run()
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == _NUM_SAMPLE_TRANSCRIPTS
        record = json.loads(lines[0])
        assert record["source"] == "sim_vail"
        assert record["task_type"] == "adversarial_safety"
