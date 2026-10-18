"""Adapter for mental_health_multiagent — AI-generated MH assessment chat logs.

Source: S3 (whitebat:training/pixelated-empathy/output/mental_health_multiagent/)
Format: JSON files with {timestamp, questionnaire, conversation: [{role, content}]}
Size: 8,012 chat log files from GPT-4.1 multiagent conversations
Language: English

Output task_type: therapy_response_generation
Each record: multi-turn mental health assessment conversation.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_S3_PREFIX = "whitebat:training/pixelated-empathy/output/mental_health_multiagent/raw/"
_SOURCE_URL = "https://github.com/mental-health-multiagent/mental_health_multiagent"

_SYSTEM_PROMPT = (
    "You are a compassionate mental health professional conducting an assessment. "
    "Ask thoughtful questions, listen actively, and provide supportive guidance."
)


@register_adapter("mental_health_multiagent")
class MentalHealthMultiagentAdapter(BaseDatasetAdapter):
    """Adapter for AI-generated mental health chat logs from S3."""

    def download(self) -> None:
        """Download chat logs from S3 if not already present."""
        if any(self._raw_dir.rglob("*.json")):
            return

        try:
            subprocess.run(
                ["rclone", "copy", _S3_PREFIX, str(self._raw_dir), "--include", "*.json"],
                check=True,
                capture_output=True,
                timeout=600,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError) as e:
            readme = self._raw_dir / "README.txt"
            readme.write_text(f"Download failed: {e}\n", encoding="utf-8")

    def extract(self) -> list[dict[str, Any]]:
        """Read JSON chat log files."""
        records: list[dict[str, Any]] = []
        for json_path in sorted(self._raw_dir.rglob("*.json")):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                if "conversation" in data:
                    records.append(
                        {
                            "_source_file": json_path.name,
                            "conversation": data["conversation"],
                            "timestamp": data.get("timestamp", ""),
                            "questionnaire": data.get("questionnaire", ""),
                        }
                    )
            except (json.JSONDecodeError, OSError):
                continue
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert chat logs to ChatML."""
        records: list[dict[str, Any]] = []

        for entry in raw_data:
            conversation = entry.get("conversation", [])
            if not conversation:
                continue

            messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

            for turn in conversation:
                role = (turn.get("role") or "").strip().lower()
                content = (turn.get("content") or "").strip()
                if not content:
                    continue

                # Map roles to ChatML
                if role in ("assistant", "counselor", "therapist", "supporter"):
                    messages.append({"role": "assistant", "content": content})
                elif role in ("user", "client", "seeker", "patient"):
                    messages.append({"role": "user", "content": content})
                else:
                    # Unknown role — treat as user if system is already present
                    messages.append({"role": "user", "content": content})

            if len(messages) < 3:
                continue

            has_user = any(m["role"] == "user" for m in messages)
            has_assistant = any(m["role"] == "assistant" for m in messages)
            if not has_user or not has_assistant:
                continue

            records.append(
                {
                    "messages": messages,
                    "source": "mental_health_multiagent",
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
