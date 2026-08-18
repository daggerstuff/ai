"""Regression tests for PixelatedEmpathyAPI.wait_for_job failure semantics.

Ports intent of:
  docs/api/clients/javascript_client.test.ts
    - "should resolve immediately if job is already failed"

Vitest cases live in docs/ and are not executed by CI (.github/workflows/ci.yml
runs `uv run pytest tests/` only). These tests pin the Python client's
behaviour so the regression remains enforced under the Python CI suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# tests/api/* / tests/api_tests/* are picked up by `uv run pytest tests/`.
# python_client.py lives under docs/api/clients/ - add repo root to import path
# so `from api_clients.python_client import ...` works in both layouts.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "docs" / "api" / "clients"))

python_client = pytest.importorskip("python_client")
PixelatedEmpathyAPI = python_client.PixelatedEmpathyAPI


class TestWaitForJobFailureSemantics:
    """Mirror Vitest `PixelatedEmpathyAPI Method waitForJob` failure cases."""

    def test_resolves_immediately_when_job_already_completed(self) -> None:
        api = PixelatedEmpathyAPI(api_key="test_key")

        with patch.object(
            api,
            "get_job_status",
            return_value={"status": "completed", "progress": 100},
        ):
            result = api.wait_for_job("job-123")

        assert result == {"status": "completed", "progress": 100}

    def test_resolves_immediately_when_job_already_failed(self) -> None:
        api = PixelatedEmpathyAPI(api_key="test_key")

        with patch.object(
            api,
            "get_job_status",
            return_value={"status": "failed", "progress": None},
        ):
            # Short timeout + short poll means a single call must short-circuit.
            result = api.wait_for_job(
                "job-123", poll_interval=1, timeout=5
            )

        assert result == {"status": "failed", "progress": None}
