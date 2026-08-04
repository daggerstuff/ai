"""Adapter for the PsyDial privacy-preserving counseling dataset.

Source: HuggingFace (qiuhuachuan/PsyDial-D101, PsyDial-D4, PsyDial-D1)
Original paper: ACL 2025 (aclanthology.org/2025.acl-long.1049).
Format: JSON dialogues with messages[] + golden response.
Size: PsyDial-D101 (1278), PsyDial-D4, PsyDial-D1
Method: RMRR (Retrieve, Mask, Reconstruct, Refine) -- retrieves chief
  complaints from PsyQA, masks all client utterances, reconstructs with
  GPT-4o, refines counselor utterances.
Language: Chinese (zh)

Output task_type: therapy_response_generation
Tagged privacy_preserving=True. RMRR methodology note in system prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://huggingface.co/qiuhuachuan/PsyDial-D101"

_HF_REPOS: list[tuple[str, str]] = [
    ("qiuhuachuan/PsyDial-D101", "PsyDial-D101.json"),
    ("qiuhuachuan/PsyDial-D4", "PsyDial-D4.json"),
    ("qiuhuachuan/PsyDial-D1", "PsyDial-D1.json"),
]

_RMRR_NOTE = (
    "Privacy-preserving. Counseling dialogues reconstructed via RMRR methodology "
    "(Retrieve chief complaints from PsyQA, Mask all client utterances, "
    "Reconstruct client utterances with GPT-4o, Refine counselor utterances). "
    "No real client text is included."
)


@register_adapter("psydial")
class PsyDialAdapter(BaseDatasetAdapter):
    """Adapter for PsyDial privacy-preserving counseling dataset.

    Each entry has a list of user messages and a golden assistant response.
    Combined into a single ChatML record per entry.
    """

    def download(self) -> None:
        """Download PsyDial JSON files from HuggingFace."""
        import os
        import tempfile

        cache_dir = (
            os.environ.get("HF_HOME")
            or os.environ.get("HF_DATASETS_CACHE")
            or str(self.output_dir.parent / ".hf_cache")
        )
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))

        from huggingface_hub import hf_hub_download

        for repo_id, filename in _HF_REPOS:
            target = self._raw_dir / filename
            if target.exists():
                continue
            try:
                path = hf_hub_download(
                    repo_id,
                    filename,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
                import shutil

                shutil.copy(path, target)
            except Exception:
                pass

    def extract(self) -> list[dict[str, Any]]:
        """Extract dialogues from all downloaded JSON files."""
        records: list[dict[str, Any]] = []
        for repo_id, filename in _HF_REPOS:
            path = self._raw_dir / filename
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                entry["_source_file"] = filename
                entry["_source_repo"] = repo_id
                records.append(entry)
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert extracted entries to ChatML records."""
        records: list[dict[str, Any]] = []

        for entry in raw_data:
            messages_raw = entry.get("messages", [])
            golden = entry.get("golden", {})

            if not isinstance(golden, dict) or not golden:
                continue

            golden_role = golden.get("role", "assistant")
            golden_content = (golden.get("content") or "").strip()
            if not golden_content:
                continue

            system_content = f"{_RMRR_NOTE} Language: zh (Chinese). You are a supportive counselor providing guidance."

            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

            for msg in messages_raw:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "user")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                messages.append({"role": role, "content": content})

            messages.append({"role": golden_role, "content": golden_content})

            if len(messages) < 3:
                continue

            record: dict[str, Any] = {
                "messages": messages,
                "source": "psydial",
                "task_type": "therapy_response_generation",
                "diagnostic_tag": None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "privacy_preserving": True,
                "rmrr_methodology": _RMRR_NOTE,
                "language": "zh",
                "source_file": entry.get("_source_file", ""),
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="huggingface",
                    original_format="json",
                ),
            }
            records.append(record)

        return records
