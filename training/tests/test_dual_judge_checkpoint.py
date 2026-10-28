"""Tests for dual_judge calibration checkpoint/resume (DUAL_JUDGE_CHECKPOINT).

Covers the live calibration path writing one JSONL row per completed sample
and resuming from a partially populated checkpoint: pre-checkpointed samples
are never re-judged (no HTTP calls) and still appear in the final report with
their checkpointed scores.

All HTTP calls are mocked — no vLLM or GPU required.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from training.dual_judge import (
    ACCEPT_THRESHOLD,
    CALIB_KAPPA_MIN,
    CALIB_PEARSON_MIN,
    DIMENSIONS,
    _load_checkpoint,
    _run_calibration,
)

SAMPLE_COUNT = 4

# ---------------------------------------------------------------------------
# Helpers (mirroring test_dual_judge_report_out.py patterns)
# ---------------------------------------------------------------------------

def make_mock_aiohttp_response(content: str) -> MagicMock:
    """Create a mock aiohttp response object with status 200."""
    resp = MagicMock()
    resp.status = 200
    resp.text = AsyncMock(return_value="")
    resp.json = AsyncMock(return_value={"choices": [{"message": {"content": content}}]})
    return resp

class _AsyncCM:
    """Minimal async context manager wrapping a value."""

    def __init__(self, value: Any) -> None:
        self.value = value

    async def __aenter__(self) -> Any:
        return self.value

    async def __aexit__(self, *args: Any) -> bool:
        return False

def make_judge_json_response(quality: float) -> str:
    """Produce the JSON string a judge model would return."""
    return json.dumps(
        {
            "quality_score": quality,
            "reject_reason": "",
            "dim_scores": dict.fromkeys(DIMENSIONS, quality),
            "reasoning": "test verdict",
        }
    )

def _write_golden(tmp_path: Path, n: int = SAMPLE_COUNT) -> Path:
    """Write a small golden calibration JSONL with distinct human scores."""
    samples: list[dict[str, Any]] = []
    for i in range(n):
        quality = 0.2 + 0.2 * i
        samples.append(
            {
                "id": f"ckpt-test-{i:03d}",
                "conversation": [
                    {"role": "user", "content": f"client concern {i}"},
                    {"role": "assistant", "content": f"therapist response {i}"},
                ],
                "human_scores": dict.fromkeys(DIMENSIONS, quality),
            }
        )
    path = tmp_path / "golden_checkpoint.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
    return path

def _counting_post_factory(counter: list[int]) -> Any:
    """Mock post that increments a shared counter and scores like the human."""

    def _post(*_args: Any, **kwargs: Any) -> _AsyncCM:
        counter.append(1)
        payload = kwargs["json"]
        marker = "therapist response "
        idx = int(payload["messages"][1]["content"].split(marker, 1)[1][0])
        quality = 0.2 + 0.2 * idx
        return _AsyncCM(make_mock_aiohttp_response(make_judge_json_response(quality)))

    return _post

def _mock_session(post_side_effect: Any) -> MagicMock:
    """Create a mock aiohttp.ClientSession with the given post side_effect."""
    session = MagicMock()
    session.post = MagicMock(side_effect=post_side_effect)
    session.close = AsyncMock()
    return session

@contextmanager
def _generic_transport(session: MagicMock) -> Iterator[None]:
    """Force the generic judge transport and inject the mocked session."""

    with (
        patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "", "CLOUDFLARE_API_TOKEN": ""}),
        patch("training.dual_judge.aiohttp.ClientSession", return_value=_AsyncCM(session)),
    ):
        yield

# ---------------------------------------------------------------------------
# Tests: DUAL_JUDGE_CHECKPOINT durability + resume
# ===========================================================================

class TestCheckpoint:
    """Verify checkpoint writes and resume behavior on the live path."""

    def test_checkpoint_writes_one_row_per_sample(self, tmp_path: Path) -> None:
        """Live calibration with DUAL_JUDGE_CHECKPOINT appends one row per sample."""
        golden_path = _write_golden(tmp_path)
        ckpt_path = tmp_path / "calib_checkpoint.jsonl"
        counter: list[int] = []
        env = {"DUAL_JUDGE_CHECKPOINT": str(ckpt_path)}
        with (
            patch.dict(os.environ, env),
            _generic_transport(_mock_session(_counting_post_factory(counter))),
        ):
            rc = _run_calibration(
                str(golden_path),
                live=True,
                url="http://127.0.0.1:20128/v1/chat/completions",
                primary_model="nvidia/z-ai/glm-5.3",
            )

        assert rc == 0
        lines = [line for line in ckpt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == SAMPLE_COUNT
        rows = [json.loads(line) for line in lines]
        assert {row["id"] for row in rows} == {f"ckpt-test-{i:03d}" for i in range(SAMPLE_COUNT)}
        for row in rows:
            assert set(row) == {"id", "judge_score", "human_score", "judge_dim_scores", "human_dim_scores"}

    def test_resume_skips_checkpointed_samples(self, tmp_path: Path) -> None:
        """Pre-checkpointed samples are not re-judged but appear in the report."""
        golden_path = _write_golden(tmp_path)
        ckpt_path = tmp_path / "calib_checkpoint_resume.jsonl"

        # Pre-populate rows for the first two samples with distinctive scores
        # (close enough to the human labels that the offset-calibrated gate on
        # the resumed+fresh mix still passes: offset +0.025 keeps the eval
        # bins aligned, kappa 1.0).
        pre_rows = [
            {
                "id": f"ckpt-test-{i:03d}",
                "judge_score": 0.25 + 0.05 * i,
                "human_score": 0.2 + 0.2 * i,
                "judge_dim_scores": dict.fromkeys(DIMENSIONS, 0.25 + 0.05 * i),
                "human_dim_scores": dict.fromkeys(DIMENSIONS, 0.2 + 0.2 * i),
            }
            for i in range(2)
        ]
        ckpt_path.write_text(
            "\n".join(json.dumps(r) for r in pre_rows) + "\n", encoding="utf-8"
        )

        counter: list[int] = []
        env = {"DUAL_JUDGE_CHECKPOINT": str(ckpt_path)}
        with (
            patch.dict(os.environ, env),
            _generic_transport(_mock_session(_counting_post_factory(counter))),
        ):
            rc = _run_calibration(
                str(golden_path),
                live=True,
                url="http://127.0.0.1:20128/v1/chat/completions",
                primary_model="nvidia/z-ai/glm-5.3",
                report_out=str(tmp_path / "report.json"),
            )

        assert rc == 0
        # Each fresh sample makes exactly 3 part calls; resumed samples make none.
        assert len(counter) == 3 * (SAMPLE_COUNT - 2)

        report: dict[str, Any] = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["sample_count"] == SAMPLE_COUNT
        by_id = {row["id"]: row for row in report["per_sample"]}
        for i in range(2):
            row = by_id[f"ckpt-test-{i:03d}"]
            assert row["judge_score"] == pytest.approx(0.25 + 0.05 * i)
        for i in range(2, SAMPLE_COUNT):
            row = by_id[f"ckpt-test-{i:03d}"]
            assert row["judge_score"] == pytest.approx(0.2 + 0.2 * i)
        # Gate thresholds unchanged in the persisted report.
        assert report["gate"]["pearson_min"] == CALIB_PEARSON_MIN
        assert report["gate"]["kappa_min"] == CALIB_KAPPA_MIN
        assert report["gate"]["accept_threshold"] == ACCEPT_THRESHOLD

    def test_no_checkpoint_env_makes_no_file(self, tmp_path: Path) -> None:
        """Without DUAL_JUDGE_CHECKPOINT no checkpoint file is written."""
        golden_path = _write_golden(tmp_path)
        counter: list[int] = []
        with (
            patch.dict(os.environ, {"DUAL_JUDGE_CHECKPOINT": ""}),
            _generic_transport(_mock_session(_counting_post_factory(counter))),
        ):
            rc = _run_calibration(
                str(golden_path),
                live=True,
                url="http://127.0.0.1:20128/v1/chat/completions",
                primary_model="nvidia/z-ai/glm-5.3",
            )
        assert rc == 0
        assert not (tmp_path / "calib_checkpoint.jsonl").exists()

    def test_load_checkpoint_skips_malformed_lines(self, tmp_path: Path) -> None:
        """_load_checkpoint tolerates a mid-write killed trailing line."""
        ckpt_path = tmp_path / "torn.jsonl"
        good = json.dumps(
            {
                "id": "ckpt-test-000",
                "judge_score": 0.5,
                "human_score": 0.2,
                "judge_dim_scores": dict.fromkeys(DIMENSIONS, 0.5),
                "human_dim_scores": dict.fromkeys(DIMENSIONS, 0.2),
            }
        )
        ckpt_path.write_text(good + "\n" + '{"id": "ckpt-test-001", "judge_sco', encoding="utf-8")
        rows = _load_checkpoint(str(ckpt_path))
        assert set(rows) == {"ckpt-test-000"}

    def test_load_checkpoint_missing_file_is_empty(self, tmp_path: Path) -> None:
        """A nonexistent checkpoint path loads as an empty dict."""
        assert _load_checkpoint(str(tmp_path / "nope.jsonl")) == {}
        assert _load_checkpoint(None) == {}
