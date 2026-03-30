"""
Data access helpers for standard therapeutic source files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import orjson

from ai.pipelines.orchestrator.orchestration.standard_therapeutic_record_format import (
    is_standard_therapeutic_record,
)


class StandardTherapeuticRepository:
    """Load raw standard-therapeutic records from JSON and JSONL files."""

    # JSON fallback is a narrow compatibility path; keep the eager-parse ceiling low
    # so large sources are forced back onto the canonical JSONL ingestion route.
    _MAX_EAGER_JSON_BYTES = 8 * 1024 * 1024

    def __init__(
        self,
        *,
        record_array_keys: tuple[str, ...] = ("conversations", "records", "items", "data"),
        single_record_predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self.record_array_keys = record_array_keys
        self.single_record_predicate = single_record_predicate or is_standard_therapeutic_record

    def load_json_file(self, file_path: Path) -> list[Any]:
        file_size = file_path.stat().st_size
        if file_size > self._MAX_EAGER_JSON_BYTES:
            raise ValueError(
                "Standard therapeutic JSON fallback is too large for eager loading; "
                "use the canonical JSONL source instead"
            )
        payload = orjson.loads(file_path.read_bytes())
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in self.record_array_keys:
                if key not in payload:
                    continue
                records = payload[key]
                if isinstance(records, list):
                    return records
                raise ValueError(
                    f"Expected object with a '{key}' array in {file_path}"
                )
            if self.single_record_predicate(payload):
                return [payload]
            raise ValueError(
                "Expected object with one of "
                f"{self.record_array_keys} arrays or a single therapeutic record in {file_path}"
            )
        raise ValueError(
            f"Expected JSON array or object root in {file_path}, found {type(payload).__name__}"
        )

    def iter_jsonl_records(
        self,
        file_path: Path,
        max_samples: int | None = None,
        on_decode_error: Callable[[int, json.JSONDecodeError], None] | None = None,
    ):
        with file_path.open(encoding="utf-8") as handle:
            yielded = 0
            for line_number, line in enumerate(handle, start=1):
                if max_samples and yielded >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    yield line_number, json.loads(line)
                    yielded += 1
                except json.JSONDecodeError as exc:
                    if on_decode_error is not None:
                        on_decode_error(line_number, exc)
                    continue

__all__ = ["StandardTherapeuticRepository"]
