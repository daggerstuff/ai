"""Adapter for the PsyDial privacy-preserving counseling dataset.

Source: ACL 2025 (aclanthology.org/2025.acl-long.1049).
Format: JSON dialogues (reconstructed via RMRR methodology).
Size: 2,382 long-term counseling dialogues, avg 37.8 turns/dialogue.
Method: RMRR (Retrieve, Mask, Reconstruct, Refine) -- retrieves chief
  complaints from PsyQA, masks all client utterances, reconstructs with
  GPT-4o, refines counselor utterances.
Paper: ACL 2025.

Output task_type: therapy_response_generation
Tagged privacy_preserving=True. RMRR methodology note in system prompt.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://aclanthology.org/2025.acl-long.1049"

# Expected raw files (place manually if no public mirror available):
#   dialogues.json  - 2,382 reconstructed counseling dialogues
#   metadata.json   - chief complaints + RMRR provenance per dialogue
_RAW_URLS: dict[str, str] = {
    "dialogues.json": "https://example.invalid/psydial/dialogues.json",
    "metadata.json": "https://example.invalid/psydial/metadata.json",
}

_RMRR_NOTE = (
    "Privacy-preserving. Counseling dialogues reconstructed via RMRR methodology "
    "(Retrieve chief complaints from PsyQA, Mask all client utterances, "
    "Reconstruct client utterances with GPT-4o, Refine counselor utterances). "
    "No real client text is included."
)


@register_adapter("psydial")
class PsyDialAdapter(BaseDatasetAdapter):
    """Adapter for PsyDial privacy-preserving counseling dataset.

    Each dialogue becomes one ChatML record. User turns are reconstructed
    client utterances; assistant turns are refined counselor utterances.
    System prompt includes RMRR methodology note + chief complaint when
    available. Records tagged privacy_preserving=True.
    """

    def download(self) -> None:
        """Download dialogues/metadata if not already present."""
        for filename, url in _RAW_URLS.items():
            target = self._raw_dir / filename
            if target.exists():
                continue
            try:
                urllib.request.urlretrieve(url, target)
            except Exception:
                # Public mirror may not exist; manual placement supported.
                pass

    def extract(self) -> list[dict[str, Any]]:
        """Extract dialogues joined with metadata (chief complaint, RMRR provenance)."""
        dialogues = self._load_json("dialogues.json", default=[])
        metadata = self._load_json("metadata.json", default=[])

        meta_index: dict[str, dict[str, Any]] = {}
        for entry in metadata:
            did = entry.get("dialogue_id") or entry.get("id")
            if did is not None:
                meta_index[str(did)] = entry

        records: list[dict[str, Any]] = []
        for dialogue in dialogues:
            did = dialogue.get("dialogue_id") or dialogue.get("id")
            meta = meta_index.get(str(did)) if did is not None else None
            records.append({**dialogue, "_metadata": meta})
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert extracted dialogues to ChatML records."""
        records: list[dict[str, Any]] = []

        for dialogue in raw_data:
            meta = dialogue.get("_metadata") or {}
            chief_complaint = (meta.get("chief_complaint") or dialogue.get("chief_complaint") or "").strip()
            language = (dialogue.get("language") or meta.get("language") or "zh").strip()

            system_content = self._build_system(
                chief_complaint=chief_complaint,
                language=language,
            )

            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

            turns = (
                dialogue.get("dialog")
                or dialogue.get("turns")
                or dialogue.get("conversation")
                or dialogue.get("messages")
                or []
            )
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("speaker") or turn.get("role") or turn.get("from") or "").lower()
                utterance = (
                    turn.get("utterance") or turn.get("text") or turn.get("content") or turn.get("value") or ""
                ).strip()
                if not utterance:
                    continue
                role = self._map_role(speaker)
                messages.append({"role": role, "content": utterance})

            if len(messages) < 2:
                continue

            roles = {m["role"] for m in messages}
            if "user" not in roles or "assistant" not in roles:
                continue

            turns_count = len(messages) - 1  # exclude system

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
                "chief_complaint": chief_complaint,
                "language": language,
                "turn_count": turns_count,
                "dialogue_id": dialogue.get("dialogue_id") or dialogue.get("id"),
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="request",
                    original_format="json",
                ),
            }
            records.append(record)

        return records

    def _load_json(self, filename: str, *, default: Any) -> Any:
        path = self._raw_dir / filename
        if not path.exists():
            return default
        if filename.endswith(".csv"):
            rows: list[dict[str, Any]] = []
            with open(path, encoding="utf-8") as csv_file:
                for row in csv.DictReader(csv_file):
                    rows.append(row)
            return rows
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _build_system(*, chief_complaint: str, language: str) -> str:
        parts = [_RMRR_NOTE]
        if chief_complaint:
            parts.append(f"Chief complaint: {chief_complaint}.")
        if language:
            parts.append(f"Language: {language}.")
        return " ".join(parts)

    @staticmethod
    def _map_role(speaker: str) -> str:
        client_tokens = {"client", "user", "patient", "seeker", "human"}
        counselor_tokens = {"counselor", "therapist", "assistant", "supporter", "gpt", "system"}
        if speaker in client_tokens:
            return "user"
        if speaker in counselor_tokens:
            return "assistant"
        # Default: turns alternate; unknown speaker treated as assistant
        # (reconstructed client turns should be explicitly tagged "client").
        return "assistant"
