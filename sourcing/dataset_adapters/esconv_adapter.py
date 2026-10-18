"""Adapter for the ESConv (Emotional Support Conversation) dataset.

Source: https://github.com/thu-coai/Emotional-Support-Conversation
Size: 1,300 conversations, 38,365 utterances
Format: JSON (ESConv.json + FailedESConv.json with 196 negative samples)
Labels: 12 problem categories, 8 support strategies
Paper: ACL 2021

Output task_type: therapy_response_generation
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/thu-coai/Emotional-Support-Conversation"
_RAW_URL = "https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json"

# FailedESConv negative samples
_FAILED_URL = "https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/FailedESConv.json"


@register_adapter("esconv")
class ESConvAdapter(BaseDatasetAdapter):
    """Adapter for ESConv dataset.

    Converts 1,300 emotional support conversations to ChatML with:
    - System prompt containing situation, emotion type, and problem type
    - User/assistant turns from seeker/supporter dialog
    - FailedESConv conversations tagged as adversarial_safety
    """

    def download(self) -> None:
        """Download ESConv.json and FailedESConv.json if not present."""
        esconv_file = self._raw_dir / "ESConv.json"
        failed_file = self._raw_dir / "FailedESConv.json"

        if not esconv_file.exists():
            urllib.request.urlretrieve(_RAW_URL, esconv_file)

        if not failed_file.exists():
            try:
                urllib.request.urlretrieve(_FAILED_URL, failed_file)
            except Exception:
                # FailedESConv may not be available in all repo states
                pass

    def extract(self) -> list[dict[str, Any]]:
        """Extract conversation data from JSON files."""
        records: list[dict[str, Any]] = []

        esconv_file = self._raw_dir / "ESConv.json"
        if esconv_file.exists():
            with open(esconv_file, encoding="utf-8") as f:
                data = json.load(f)
            for conv in data:
                records.append({**conv, "_source_file": "ESConv.json", "_is_negative": False})

        failed_file = self._raw_dir / "FailedESConv.json"
        if failed_file.exists():
            with open(failed_file, encoding="utf-8") as f:
                data = json.load(f)
            for conv in data:
                records.append({**conv, "_source_file": "FailedESConv.json", "_is_negative": True})

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert ESConv conversations to ChatML records."""
        records: list[dict[str, Any]] = []

        for conv in raw_data:
            dialog = conv.get("dialog") or conv.get("conversation") or []
            if not dialog:
                continue

            is_negative = conv.get("_is_negative", False)
            task_type = "adversarial_safety" if is_negative else "therapy_response_generation"

            situation = conv.get("situation", "")
            emotion_type = conv.get("emotion_type", "")
            problem_type = conv.get("problem_type", "")

            system_content = (
                f"Context: {situation}. Emotion: {emotion_type}. Problem: {problem_type}."
                if situation or emotion_type or problem_type
                else None
            )

            messages: list[dict[str, str]] = []
            if system_content:
                messages.append({"role": "system", "content": system_content})

            for turn in dialog:
                speaker = str(turn.get("speaker", "")).lower()
                utterance = (turn.get("utterance") or turn.get("text") or turn.get("content") or "").strip()
                if not utterance:
                    continue
                role = "user" if speaker in ("seeker", "user", "client", "human") else "assistant"
                messages.append({"role": role, "content": utterance})

            if len(messages) < 2:
                continue

            # Ensure at least one user and one assistant
            roles = {m["role"] for m in messages}
            if "user" not in roles or "assistant" not in roles:
                continue

            record: dict[str, Any] = {
                "messages": messages,
                "source": "esconv",
                "task_type": task_type,
                "diagnostic_tag": None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "emotion_type": emotion_type,
                "problem_type": problem_type,
                "is_negative_sample": is_negative,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="json",
                ),
            }
            records.append(record)

        return records
