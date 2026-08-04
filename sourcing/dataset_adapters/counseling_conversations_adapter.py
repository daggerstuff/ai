"""Adapter for the Mental Health Counseling Conversations dataset.

Source: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
Size: 3,512 counseling Q&A pairs
Format: JSONL with columns `Context` (user message) and `Response` (therapist response)
Access: HuggingFace (open download)

Output task_type: therapy_response_generation
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_HF_REPO_ID = "Amod/mental_health_counseling_conversations"
_HF_FILENAME = "combined_dataset.json"
_SOURCE_URL = "https://huggingface.co/datasets/Amod/mental_health_counseling_conversations"

_SYSTEM_PROMPT = (
    "You are a compassionate mental health counselor providing supportive, "
    "evidence-based responses to someone seeking help. Respond with empathy, "
    "active listening, and practical guidance."
)


@register_adapter("counseling_conversations")
class CounselingConversationsAdapter(BaseDatasetAdapter):
    """Adapter for mental health counseling conversations from HuggingFace."""

    def download(self) -> None:
        """Download the dataset from HuggingFace Hub."""
        target = self._raw_dir / _HF_FILENAME
        if target.exists() and target.stat().st_size > 0:
            return

        cache_dir = self.output_dir.parent / ".hf_cache"
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                _HF_REPO_ID,
                _HF_FILENAME,
                repo_type="dataset",
                cache_dir=str(cache_dir / "hub"),
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
        except Exception as e:
            readme = self._raw_dir / "README.txt"
            if not readme.exists():
                readme.write_text(
                    f"Download failed: {e}\n"
                    f"Manual: huggingface-cli download {_HF_REPO_ID} {_HF_FILENAME} "
                    f"--repo-type dataset\n",
                    encoding="utf-8",
                )

    def extract(self) -> list[dict[str, Any]]:
        """Read JSONL file into intermediate dicts."""
        target = self._raw_dir / _HF_FILENAME
        if not target.exists():
            return []

        records: list[dict[str, Any]] = []
        with open(target, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert raw Q&A pairs to ChatML records."""
        records: list[dict[str, Any]] = []

        for row in raw_data:
            context = (row.get("Context") or "").strip()
            response = (row.get("Response") or "").strip()
            if not context or not response:
                continue

            messages: list[dict[str, str]] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": context},
                {"role": "assistant", "content": response},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "counseling_conversations",
                "task_type": "therapy_response_generation",
                "diagnostic_tag": None,
                "demographic_tags": [],
                "linguistic_style": "informal",
                "clinical_reviewed": False,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="huggingface",
                    original_format="jsonl",
                ),
            }
            records.append(record)

        return records
