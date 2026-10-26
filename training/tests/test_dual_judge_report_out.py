"""Tests for dual_judge calibration report persistence (--report-out). PIX-4343.

Covers the live calibration path writing the full report — per-sample
judge-vs-human scores, Pearson r, Cohen's kappa, per-dimension correlations,
sample_count, gate thresholds, pass/fail, and the endpoint + model ids
actually used — to a parseable JSON file, while stdout keeps printing the
original summary report shape.

All HTTP calls are mocked — no vLLM or GPU required.
"""

from __future__ import annotations

import json
import os
import re
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
    _run_calibration,
)

SAMPLE_COUNT = 4

# ---------------------------------------------------------------------------
# Helpers (mirroring test_dual_judge.py patterns)
# ---------------------------------------------------------------------------

def make_mock_aiohttp_response(content: str) -> MagicMock:
    """Create a mock aiohttp response object with status 200."""
    resp = MagicMock()
    resp.status = 200
    resp.text = AsyncMock(return_value="")
    resp.json = AsyncMock(return_value={"choices": [{"message": {"content": content}}]})
    return resp

def make_mock_session(post_side_effect: Any) -> MagicMock:
    """Create a mock aiohttp.ClientSession with the given post side_effect."""
    session = MagicMock()
    session.post = MagicMock(side_effect=post_side_effect)
    session.close = AsyncMock()
    return session

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
                "id": f"report-test-{i:03d}",
                "conversation": [
                    {"role": "user", "content": f"client concern {i}"},
                    {"role": "assistant", "content": f"therapist response {i}"},
                ],
                "human_scores": dict.fromkeys(DIMENSIONS, quality),
            }
        )
    path = tmp_path / "golden_report_out.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
    return path

def _variable_quality_post(*_args: Any, **kwargs: Any) -> _AsyncCM:
    """Mock post whose judge score matches the golden sample's quality.

    The candidate content embeds the sample index ("therapist response <i>"),
    so the judge verdict reproduces the human score for that sample — giving
    perfect correlation and full agreement on accept/reject.
    """
    payload = kwargs["json"]
    match = re.search(r"therapist response (\d+)", payload["messages"][1]["content"])
    assert match is not None, "mock post called with unexpected candidate content"
    quality = 0.2 + 0.2 * int(match.group(1))
    return _AsyncCM(make_mock_aiohttp_response(make_judge_json_response(quality)))

@contextmanager
def _generic_transport() -> Iterator[None]:
    """Force the generic judge transport (never the Cloudflare REST path).

    Empty env values make ``CLOUDFLARE_ACCOUNT_ID`` / ``CLOUDFLARE_API_TOKEN``
    falsy, so ``_eval_sample_cf`` always returns None.
    """
    with patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": "", "CLOUDFLARE_API_TOKEN": ""}):
        yield

# ---------------------------------------------------------------------------
# Tests: --report-out persistence on the live calibration path
# ===========================================================================

class TestReportOut:
    """Verify --report-out writes the full calibration report to JSON."""

    def test_report_out_writes_persisted_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Live calibration with report_out writes a parseable JSON file with
        the expected keys; stdout still prints the original summary shape."""
        golden_path = _write_golden(tmp_path)
        report_path = tmp_path / "calibration_report.json"
        mock_session = make_mock_session(post_side_effect=_variable_quality_post)
        with (
            _generic_transport(),
            patch(
                "training.dual_judge.aiohttp.ClientSession",
                return_value=_AsyncCM(mock_session),
            ),
        ):
            rc = _run_calibration(
                str(golden_path),
                live=True,
                url="http://127.0.0.1:20128/v1/chat/completions",
                primary_model="nvidia/z-ai/glm-5.3",
                report_out=str(report_path),
            )

        assert rc == 0
        assert report_path.exists()
        detail: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))

        # Sample counts
        assert detail["sample_count"] == SAMPLE_COUNT
        assert detail["golden_samples"] == SAMPLE_COUNT
        assert len(detail["per_sample"]) == SAMPLE_COUNT
        for row in detail["per_sample"]:
            assert set(row) == {"id", "judge_score", "human_score", "judge_dim_scores", "human_dim_scores"}
            assert row["judge_score"] == pytest.approx(row["human_score"], abs=0.001)

        # Overall metrics: judge reproduced the human scores exactly
        metrics = detail["metrics"]
        assert set(metrics) == {"pearson_r", "cohen_kappa", "mae", "rmse"}
        assert metrics["pearson_r"] == pytest.approx(1.0, abs=0.001)
        assert metrics["cohen_kappa"] == pytest.approx(1.0, abs=0.001)

        # Per-dimension correlations for all five dims
        per_dim = detail["per_dim_correlations"]
        assert set(per_dim) == set(DIMENSIONS)
        assert all(v == pytest.approx(1.0, abs=0.001) for v in per_dim.values())

        # Gate thresholds, pass/fail, endpoint + models actually used
        gate = detail["gate"]
        assert gate["pearson_min"] == CALIB_PEARSON_MIN
        assert gate["kappa_min"] == CALIB_KAPPA_MIN
        assert gate["accept_threshold"] == ACCEPT_THRESHOLD
        assert gate["passed"] is True
        assert detail["verdict"] == "PASSED"
        assert detail["endpoint"] == "http://127.0.0.1:20128/v1/chat/completions"
        assert detail["primary_model"] == "nvidia/z-ai/glm-5.3"

        # Stdout behavior intact: the original summary report shape is printed
        # (after the per-sample progress lines, which carry no braces)
        stdout = capsys.readouterr().out
        summary: dict[str, Any] = json.loads(stdout[stdout.index("{"):])
        assert summary["golden_samples"] == SAMPLE_COUNT
        assert summary["metrics"]["pearson_r"] == pytest.approx(1.0, abs=0.001)
        assert summary["gate"]["passed"] is True
        assert "per_sample" not in summary  # summary shape unchanged (no per-sample rows)

    def test_no_report_out_writes_no_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Without report_out the stdout summary prints and no file is written."""
        golden_path = _write_golden(tmp_path)
        report_path = tmp_path / "should_not_exist.json"
        mock_session = make_mock_session(post_side_effect=_variable_quality_post)
        with (
            _generic_transport(),
            patch(
                "training.dual_judge.aiohttp.ClientSession",
                return_value=_AsyncCM(mock_session),
            ),
        ):
            rc = _run_calibration(
                str(golden_path),
                live=True,
                url="http://127.0.0.1:20128/v1/chat/completions",
                primary_model="nvidia/z-ai/glm-5.3",
            )

        assert rc == 0
        assert not report_path.exists()
        stdout = capsys.readouterr().out
        summary: dict[str, Any] = json.loads(stdout[stdout.index("{"):])
        assert summary["verdict"] == "PASSED"

    def test_report_out_records_failed_gate(self, tmp_path: Path) -> None:
        """A judge that disagrees with the golden labels writes a FAILED
        report and returns a nonzero exit code."""
        golden_path = _write_golden(tmp_path)
        report_path = tmp_path / "failed_report.json"
        # Constant judge score → zero-variance judge vector → pearson 0.0
        mock_resp = make_mock_aiohttp_response(make_judge_json_response(0.7))
        mock_session = make_mock_session(post_side_effect=lambda *_a, **_kw: _AsyncCM(mock_resp))
        with (
            _generic_transport(),
            patch(
                "training.dual_judge.aiohttp.ClientSession",
                return_value=_AsyncCM(mock_session),
            ),
        ):
            rc = _run_calibration(
                str(golden_path),
                live=True,
                url="http://127.0.0.1:20128/v1/chat/completions",
                primary_model="nvidia/z-ai/glm-5.3",
                report_out=str(report_path),
            )

        assert rc == 1
        detail: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        assert detail["verdict"] == "FAILED"
        assert detail["gate"]["passed"] is False
        assert detail["metrics"]["pearson_r"] < CALIB_PEARSON_MIN
