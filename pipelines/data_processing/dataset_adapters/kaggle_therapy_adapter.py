"""Adapter for Kaggle therapy/counseling datasets.

Handles three dataset formats downloaded from Kaggle:
1. Synthetic Therapy Conversations (train.csv) — multi-turn dialogues with
   conversations column containing [{'from': 'human'|'gpt', 'value': ...}] lists.
2. Mental Health Counseling Conversations (combined_dataset.json) — Context/Response JSONL.
3. CounselChat archives (counsel_chat2.csv, counselchat-data.csv) — questionText/answerText CSVs.

Output task_type: therapy_response_generation
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from ai.pipelines.data_processing.dataset_adapters.base_adapter import BaseDatasetAdapter

# Increase CSV field size limit for large text fields in therapy datasets
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _parse_python_list(s: str) -> list[dict[str, str]] | None:
    """Parse a Python repr list-of-dicts string that uses newlines instead of commas between items."""
    s = re.sub(r"\}\s*\{", "}, {", s)
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass
    return None

_SYSTEM_PROMPT = (
    "You are a compassionate mental health counselor providing supportive, "
    "evidence-based responses to someone seeking help. Respond with empathy, "
    "active listening, and practical guidance."
)

_KAGGLE_URL = "https://www.kaggle.com/datasets/thedevastator/synthetic-therapy-conversations-dataset"


class KaggleTherapyAdapter(BaseDatasetAdapter):
    """Adapter for Kaggle therapy/counseling datasets."""

    def __init__(self, dataset_name: str, output_dir: str | Path, raw_dir: Path | None = None) -> None:
        super().__init__(dataset_name, output_dir)
        if raw_dir is not None:
            self._raw_dir = raw_dir

    def download(self) -> None:
        """Data is already downloaded by the extraction script — skip."""

    def extract(self) -> list[dict[str, Any]]:
        """Extract records from all Kaggle dataset files."""
        records: list[dict[str, Any]] = []
        records.extend(self._extract_synthetic_train())
        records.extend(self._extract_combined_json())
        records.extend(self._extract_counselchat_csvs())
        return records

    def _extract_synthetic_train(self) -> list[dict[str, Any]]:
        """Extract multi-turn dialogues from train.csv (synthetic therapy conversations)."""
        path = self._raw_dir / "train.csv"
        if not path.exists():
            return []

        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                conv_raw = row.get("conversations", "")
                if not conv_raw.strip():
                    continue
                conversations = _parse_python_list(conv_raw)
                if not isinstance(conversations, list) or len(conversations) < 2:
                    continue

                messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
                for turn in conversations:
                    role_raw = turn.get("from", "")
                    value = (turn.get("value", "") or "").strip()
                    if not value:
                        continue
                    role = "user" if role_raw == "human" else "assistant"
                    messages.append({"role": role, "content": value})

                if len(messages) < 3:
                    continue
                records.append({"messages": messages, "_source_file": "train.csv"})
        return records

    def _extract_combined_json(self) -> list[dict[str, Any]]:
        """Extract Context/Response pairs from combined_dataset.json."""
        path = self._raw_dir / "combined_dataset.json"
        if not path.exists():
            return []

        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                context = (row.get("Context") or "").strip()
                response = (row.get("Response") or "").strip()
                if not context or not response:
                    continue
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": response},
                ]
                records.append({"messages": messages, "_source_file": "combined_dataset.json"})
        return records

    def _extract_counselchat_csvs(self) -> list[dict[str, Any]]:
        """Extract question/answer pairs from CounselChat archive CSVs."""
        archive_dir = self._raw_dir / "archive"
        records: list[dict[str, Any]] = []

        for csv_name in ("counsel_chat2.csv", "counselchat-data.csv"):
            path = archive_dir / csv_name
            if not path.exists():
                continue

            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question = (row.get("questionText") or row.get("questionText") or "").strip()
                    answer = (row.get("answerText") or "").strip()
                    if not question or not answer:
                        continue
                    messages = [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                    records.append({"messages": messages, "_source_file": csv_name})
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert intermediate records to standardized ChatML."""
        records: list[dict[str, Any]] = []
        for raw in raw_data:
            messages = raw["messages"]
            if len(messages) < 3:
                continue
            record: dict[str, Any] = {
                "messages": messages,
                "source": "kaggle_therapy",
                "task_type": "therapy_response_generation",
                "diagnostic_tag": None,
                "demographic_tags": [],
                "linguistic_style": "informal",
                "clinical_reviewed": False,
                "provenance": self._build_provenance(
                    source_url=_KAGGLE_URL,
                    access_method="kaggle",
                    original_format="csv+json",
                ),
            }
            records.append(record)
        return records
