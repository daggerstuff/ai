"""Adapter for the KokoroChat Japanese counseling dialogue dataset.

Source: S3 (whitebat:training/pixelated-empathy/output/kokorochat/raw/kokorochat/kokorochat_dialogues/)
Size: 6,589 JSON dialogue files
Format: Each file contains {dialogue: [{role: "counselor"|"client", time: ISO, utterance: str}]}
Language: Japanese (ja)

Output task_type: therapy_response_generation
Each record: multi-turn Japanese counseling dialogue with system prompt.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from ai.pipelines.data_processing.dataset_adapters.adapter_factory import register_adapter
from ai.pipelines.data_processing.dataset_adapters.base_adapter import BaseDatasetAdapter

_S3_PREFIX = "whitebat:training/pixelated-empathy/output/kokorochat/raw/kokorochat/kokorochat_dialogues/"
_SOURCE_URL = "https://github.com/kokorochat/kokorochat"

_SYSTEM_PROMPT = (
    "あなたは共感的で専門的なカウンセラーです。クライアントの話を注意深く聞き、"
    "理解し、サポートを提供してください。日本語で応答してください。"
)


@register_adapter("kokorochat")
class KokoroChatAdapter(BaseDatasetAdapter):
    """Adapter for KokoroChat Japanese counseling dialogues from S3."""

    def download(self) -> None:
        """Download dialogue JSON files from S3 if not already present."""
        if any(self._raw_dir.glob("*.json")):
            return

        try:
            subprocess.run(
                ["rclone", "copy", _S3_PREFIX, str(self._raw_dir)],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError) as e:
            readme = self._raw_dir / "README.txt"
            readme.write_text(
                f"Download failed: {e}\nManual: rclone copy {_S3_PREFIX} <target>\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Read JSON dialogue files into intermediate dicts."""
        records: list[dict[str, Any]] = []
        for json_path in sorted(self._raw_dir.glob("*.json")):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                dialogue = data.get("dialogue", [])
                if not dialogue:
                    continue
                records.append(
                    {
                        "_source_file": json_path.stem,
                        "dialogue": dialogue,
                    }
                )
            except (json.JSONDecodeError, OSError):
                continue
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert dialogue arrays to ChatML with alternating client/counselor turns."""
        records: list[dict[str, Any]] = []

        for row in raw_data:
            dialogue = row.get("dialogue", [])
            if not dialogue:
                continue

            messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

            for turn in dialogue:
                role_str = (turn.get("role") or "").strip().lower()
                utterance = (turn.get("utterance") or "").strip()
                if not utterance:
                    continue

                if role_str == "client":
                    messages.append({"role": "user", "content": utterance})
                elif role_str == "counselor":
                    messages.append({"role": "assistant", "content": utterance})
                else:
                    continue

            if len(messages) < 3:
                continue

            # Ensure at least one user and one assistant message
            has_user = any(m["role"] == "user" for m in messages)
            has_assistant = any(m["role"] == "assistant" for m in messages)
            if not has_user or not has_assistant:
                continue

            records.append(
                {
                    "messages": messages,
                    "source": "kokorochat",
                    "task_type": "therapy_response_generation",
                    "diagnostic_tag": None,
                    "demographic_tags": [],
                    "linguistic_style": "formal",
                    "clinical_reviewed": False,
                    "provenance": self._build_provenance(
                        source_url=_SOURCE_URL,
                        access_method="s3",
                        original_format="json",
                    ),
                }
            )

        return records
