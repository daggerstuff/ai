"""Abstract base class for dataset adapters.

Each adapter inherits from BaseDatasetAdapter and implements download(),
extract(), and convert_to_chatml(). The run() method orchestrates the full
pipeline: download -> extract -> convert -> validate -> save as JSONL.

Output records conform to the standardized ChatML schema:

    {
        "messages": [{"role": "system|user|assistant", "content": "..."}],
        "source": "dataset_name",
        "task_type": "symptom_classification|...",
        "diagnostic_tag": "string|null",
        "demographic_tags": ["age_18_25", ...],
        "linguistic_style": "formal|informal|mixed",
        "clinical_reviewed": false,
        "provenance": {
            "source_url": "...",
            "access_method": "direct_download|github|huggingface|kaggle|request",
            "original_format": "csv|json|sharegpt|sql|audio_transcripts|hf_dataset",
            "transformations": ["download", "extract", "convert_to_chatml", "validate"],
            "extracted_at": "ISO-8601"
        }
    }
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai.sourcing.utils.converters import save_jsonl
from ai.sourcing.utils.validators import filter_valid


class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    Subclasses must implement:
        - download(): Acquire raw data (skip if already present).
        - extract(): Parse raw files into a list of intermediate dicts.
        - convert_to_chatml(): Transform intermediate dicts to ChatML records.

    The base class provides:
        - validate(): Filter invalid records.
        - run(): Orchestrate full pipeline.
        - _save_as_jsonl(): Write output to <output_dir>/<dataset_name>.jsonl.
        - _build_provenance(): Construct provenance metadata block.
    """

    def __init__(self, dataset_name: str, output_dir: str | Path) -> None:
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir) / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._raw_dir = self.output_dir / "raw"
        self._raw_dir.mkdir(exist_ok=True)

    @abstractmethod
    def download(self) -> None:
        """Download or acquire the raw dataset if not already present."""

    @abstractmethod
    def extract(self) -> list[dict[str, Any]]:
        """Extract raw data from downloaded files into intermediate dicts."""

    @abstractmethod
    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert intermediate dicts to standardized ChatML records."""

    def validate(self, chatml_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out invalid records."""
        return filter_valid(chatml_data)

    def run(self) -> Path:
        """Execute full pipeline: download -> extract -> convert -> validate -> save.

        Returns:
            Path to the output JSONL file.
        """
        self.download()
        raw_data = self.extract()
        chatml_data = self.convert_to_chatml(raw_data)
        validated_data = self.validate(chatml_data)
        output_path = self._save_as_jsonl(validated_data)
        print(f"[{self.dataset_name}] {len(validated_data)} records -> {output_path}")
        return output_path

    def _save_as_jsonl(self, data: list[dict[str, Any]]) -> Path:
        """Save validated records as JSONL in the output directory."""
        output_file = self.output_dir / f"{self.dataset_name}.jsonl"
        save_jsonl(data, output_file)
        return output_file

    def _build_provenance(
        self,
        *,
        source_url: str,
        access_method: str,
        original_format: str,
        transformations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Construct a provenance metadata block for each record."""
        return {
            "source_url": source_url,
            "access_method": access_method,
            "original_format": original_format,
            "transformations": transformations or ["download", "extract", "convert_to_chatml", "validate"],
            "extracted_at": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read a JSONL file into a list of dicts."""
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
