"""Contract tests for the Adaption RAW upload's column-mapping guarantee.

Raw ingest renames the source columns to ``original_prompt`` /
``original_completion`` but does not reliably record which one is the prompt: the
217 MB v19 upload came back with ``prompt_column=None`` on the same call that gave
the 11k arc-run2 dataset ``prompt_column=original_prompt`` (2026-11-10). A dataset
without a persisted prompt column only trains when every launch repeats the
mapping, so these tests pin three things: the mapping is written with an
estimate-only (free) call, it is read back rather than assumed, and an upload that
cannot persist it fails instead of handing back a half-configured dataset.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from training.adaption_dataset_upload import (
    CANONICAL_COMPLETION,
    CANONICAL_PROMPT,
    _evidence_path,
    configure_column_mapping,
    ensure_column_mapping,
    main,
    read_column_mapping,
)


class _FakeDatasets:
    """Minimal stand-in for ``client.datasets`` that records adapt calls."""

    def __init__(self, prompt: str | None = None, completion: str | None = None) -> None:
        self.prompt = prompt
        self.completion = completion
        self.adapt_calls: list[dict[str, Any]] = []

    def get(self, dataset_id: str) -> Any:
        if self.prompt is None and self.completion is None:
            configured = None
        else:
            configured = SimpleNamespace(prompt=self.prompt, completion=self.completion)
        return SimpleNamespace(dataset_id=dataset_id, configured_column_mapping=configured)

    def adapt(self, dataset_id: str, **kwargs: Any) -> Any:
        self.adapt_calls.append({"dataset_id": dataset_id, **kwargs})
        mapping = kwargs["column_mapping"]
        self.prompt = mapping["prompt"]
        self.completion = mapping["completion"]
        return SimpleNamespace(estimated_credits_consumed=1.0, estimated_minutes=2.0, estimate=True)


class _FakeClient:
    def __init__(self, datasets: _FakeDatasets) -> None:
        self.datasets = datasets


class _DeafDatasets(_FakeDatasets):
    """Accepts the adapt call but never records the mapping."""

    def adapt(self, dataset_id: str, **kwargs: Any) -> Any:
        self.adapt_calls.append({"dataset_id": dataset_id, **kwargs})
        return SimpleNamespace(estimated_credits_consumed=1.0, estimated_minutes=2.0, estimate=True)


def test_read_column_mapping_reports_missing_roles_as_none() -> None:
    assert read_column_mapping(_FakeClient(_FakeDatasets()), "ds") == (None, None)


def test_ensure_column_mapping_does_not_spend_a_call_when_roles_already_known() -> None:
    datasets = _FakeDatasets(CANONICAL_PROMPT, CANONICAL_COMPLETION)

    assert ensure_column_mapping(_FakeClient(datasets), "ds") is True
    assert datasets.adapt_calls == []


def test_ensure_column_mapping_persists_roles_with_a_free_estimate() -> None:
    datasets = _FakeDatasets()

    assert ensure_column_mapping(_FakeClient(datasets), "ds", max_rows=50_000) is True
    assert len(datasets.adapt_calls) == 1
    call = datasets.adapt_calls[0]
    assert call["estimate"] is True
    assert call["column_mapping"] == {"prompt": CANONICAL_PROMPT, "completion": CANONICAL_COMPLETION}
    assert call["job_specification"] == {"max_rows": 50_000}


def test_ensure_column_mapping_fails_loudly_when_the_mapping_does_not_stick() -> None:
    datasets = _DeafDatasets()

    assert ensure_column_mapping(_FakeClient(datasets), "ds") is False


def test_configure_column_mapping_omits_the_row_cap_when_none_is_given() -> None:
    datasets = _FakeDatasets()

    configure_column_mapping(_FakeClient(datasets), "ds")

    assert datasets.adapt_calls[0]["job_specification"] == {}


def test_main_repairs_an_existing_dataset_without_uploading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ADAPTION_API_KEY", "test-key")
    datasets = _FakeDatasets()

    assert main(["--dataset-id", "ds-1"], client=_FakeClient(datasets)) == 0
    assert datasets.adapt_calls[0]["dataset_id"] == "ds-1"
    assert "column mapping persisted" in capsys.readouterr().out


def test_main_returns_failure_when_an_existing_dataset_stays_unmapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADAPTION_API_KEY", "test-key")

    assert main(["--dataset-id", "ds-1"], client=_FakeClient(_DeafDatasets())) == 1


def test_main_requires_a_file_and_name_without_dataset_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADAPTION_API_KEY", "test-key")

    assert main([], client=_FakeClient(_FakeDatasets())) == 1


def test_main_refuses_to_run_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADAPTION_API_KEY", raising=False)

    assert main(["--dataset-id", "ds-1"], client=_FakeClient(_FakeDatasets())) == 1


def test_evidence_files_are_named_after_the_dataset(tmp_path: Path) -> None:
    """A new upload must not overwrite the previous run's record (v19 clobbered run-2)."""
    first = _evidence_path(tmp_path, "adaption_upload", "pixelated-v19-selection-20261110")
    second = _evidence_path(tmp_path, "adaption_upload", "arc-run2-train-v1")

    assert first != second
    assert first.name == "adaption_upload_pixelated-v19-selection-20261110.json"
