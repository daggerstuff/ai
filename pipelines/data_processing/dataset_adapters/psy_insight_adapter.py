"""Adapter for PsyInsight — psychology therapy dialogue dataset.

Source: S3 (whitebat:training/pixelated-empathy/output/psy_insight/)
Format: JSON with 520 entries, each containing dialog[] with speaker/participant/content/observation
Fields: dialog_id, theme, psychotherapy, topic, stage, guide, background, reasoning, dialog, summary
Speakers: Seeker (Client) ↔ Supporter (Therapist)
Size: 520 dialogues, ~11 turns each
Language: English + Chinese versions available

Output task_type: therapy_response_generation
Each record: multi-turn therapy dialogue with topic + background context.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_S3_PREFIX = "whitebat:training/pixelated-empathy/output/psy_insight/raw/Psy-Insight/data/"
_SOURCE_URL = "https://github.com/Psy-Insight/Psy-Insight"

_SYSTEM_PROMPT = (
    "You are a skilled psychotherapist conducting a therapy session. "
    "Use evidence-based techniques appropriate to the therapeutic approach. "
    "Listen actively, reflect empathically, and guide the client toward insight."
)


@register_adapter("psy_insight")
class PsyInsightAdapter(BaseDatasetAdapter):
    """Adapter for PsyInsight therapy dialogues from S3."""

    def download(self) -> None:
        """Download PsyInsight data from S3 if not already present."""
        en_path = self._raw_dir / "Psy-Insight" / "data" / "en_data_version7.json"
        if en_path.exists() and en_path.stat().st_size > 0:
            return

        try:
            subprocess.run(
                ["rclone", "copy", _S3_PREFIX, str(self._raw_dir / "Psy-Insight" / "data")],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError) as e:
            readme = self._raw_dir / "README.txt"
            readme.write_text(f"Download failed: {e}\n", encoding="utf-8")

    def extract(self) -> list[dict[str, Any]]:
        """Read PsyInsight JSON data."""
        records: list[dict[str, Any]] = []
        for lang in ["en", "cn"]:
            path = self._raw_dir / "Psy-Insight" / "data" / f"{lang}_data_version7.json"
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        entry["_language"] = lang
                        records.append(entry)
            except (json.JSONDecodeError, OSError):
                continue
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert PsyInsight entries to ChatML."""
        records: list[dict[str, Any]] = []

        for entry in raw_data:
            dialog = entry.get("dialog", [])
            if not dialog:
                continue

            psychotherapy = entry.get("psychotherapy", "")
            topic = entry.get("topic", "")
            background = entry.get("background", "")
            theme = entry.get("theme", "")
            lang = entry.get("_language", "en")

            system_parts = [_SYSTEM_PROMPT]
            if psychotherapy:
                system_parts.append(f"Therapeutic approach: {psychotherapy}")
            if topic:
                system_parts.append(f"Topic: {topic}")
            if background:
                system_parts.append(f"Context: {background}")
            system_content = "\n".join(system_parts)

            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

            for turn in dialog:
                participant = (turn.get("participant") or "").strip().lower()
                speaker = (turn.get("speaker") or "").strip().lower()
                content = (turn.get("content") or "").strip()
                if not content:
                    continue

                # Map participant/speaker to ChatML roles
                if "client" in participant or "seeker" in speaker:
                    messages.append({"role": "user", "content": content})
                elif "therapist" in participant or "supporter" in speaker:
                    messages.append({"role": "assistant", "content": content})
                else:
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
                    "source": "psy_insight",
                    "task_type": "therapy_response_generation",
                    "diagnostic_tag": topic or psychotherapy or None,
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
