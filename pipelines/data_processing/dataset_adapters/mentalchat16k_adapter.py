"""Adapter for the MentalChat16K dataset.

Source: S3 (whitebat:training/pixelated-empathy/ingestion/hf/mentalchat16k/)
Original: HuggingFace (ShenLab/mentalchat16k) + S3 cache
Paper: https://arxiv.org/abs/2405.06147

Format: CSV with columns:
  - instruction: str (system prompt — counselling assistant role)
  - input: str (patient's description / question)
  - output: str (counsellor's response)

Files:
  - Interview_Data_6K.csv (~6K rows — real interview data)
  - Synthetic_Data_10K.csv (~10K rows — synthetic augmentation)

Size: ~16K total rows

Output task_type: therapy_response_generation
Each record: single-turn system/user/assistant conversation.
"""

from __future__ import annotations

import csv
import subprocess
from typing import Any

from ai.pipelines.data_processing.dataset_adapters.adapter_factory import register_adapter
from ai.pipelines.data_processing.dataset_adapters.base_adapter import BaseDatasetAdapter

_S3_PREFIX = "whitebat:training/pixelated-empathy/ingestion/hf/mentalchat16k/"
_SOURCE_URL = "https://huggingface.co/datasets/ShenLab/mentalchat16k"

_CSV_FILES = ["Interview_Data_6K.csv", "Synthetic_Data_10K.csv"]


@register_adapter("mentalchat16k")
class MentalChat16KAdapter(BaseDatasetAdapter):
    """Adapter for MentalChat16K counseling Q&A dataset from S3."""

    def download(self) -> None:
        """Download CSVs from S3 whitebat remote if not already present."""
        existing = [self._raw_dir / f for f in _CSV_FILES]
        if all(p.exists() and p.stat().st_size > 0 for p in existing):
            return

        try:
            for fname in _CSV_FILES:
                target = self._raw_dir / fname
                if target.exists() and target.stat().st_size > 0:
                    continue
                subprocess.run(
                    ["rclone", "copy", _S3_PREFIX + fname, str(self._raw_dir)],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError) as e:
            readme = self._raw_dir / "README.txt"
            readme.write_text(
                f"Download failed: {e}\nManual: rclone copy {_S3_PREFIX} <target>\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Read CSV files into intermediate dicts."""
        records: list[dict[str, Any]] = []
        for fname in _CSV_FILES:
            path = self._raw_dir / fname
            if not path.exists():
                continue
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["_source_file"] = fname
                    records.append(row)
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert instruction/input/output CSV rows to ChatML."""
        records: list[dict[str, Any]] = []

        for row in raw_data:
            instruction = (row.get("instruction") or "").strip()
            user_input = (row.get("input") or "").strip()
            output = (row.get("output") or "").strip()

            if not user_input or not output:
                continue

            system_content = instruction or (
                "You are a compassionate mental health counsellor. "
                "Respond with empathy, active listening, and practical guidance."
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": output},
            ]

            records.append(
                {
                    "messages": messages,
                    "source": "mentalchat16k",
                    "task_type": "therapy_response_generation",
                    "diagnostic_tag": None,
                    "demographic_tags": [],
                    "linguistic_style": "formal",
                    "clinical_reviewed": False,
                    "provenance": self._build_provenance(
                        source_url=_SOURCE_URL,
                        access_method="s3",
                        original_format="csv",
                    ),
                }
            )

        return records
