"""Adapter for the Empathetic Dialogues dataset.

Source: HuggingFace (Adapting/empathetic_dialogues_v2)
Original: Facebook Research (facebook/empathetic_dialogues)
Paper: ACL 2019 (https://aclanthology.org/2020.acl-main.54)

Format: JSONL with columns:
  - id: int
  - chat_history: string repr of list[str] (alternating user/assistant, odd length)
  - sys_response: str (final assistant reply)
  - situation: str (context description)
  - emotion: str (emotion label)
  - question or not: string repr of list (mostly [None])
  - behavior: str (behavioral instruction, or [None])

Size: 40,245 train + 5,734 validation + 5,255 test = 51,234 total

Output task_type: empathy_response_generation
Each record: multi-turn dialogue ending with empathetic assistant response.
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any

from ai.pipelines.data_processing.dataset_adapters.adapter_factory import register_adapter
from ai.pipelines.data_processing.dataset_adapters.base_adapter import BaseDatasetAdapter

_HF_REPO_ID = "Adapting/empathetic_dialogues_v2"
_SOURCE_URL = "https://huggingface.co/datasets/Adapting/empathetic_dialogues_v2"

_SPLITS = ["train", "validation", "test"]

_SYSTEM_PROMPT = (
    "You are an empathetic conversational partner who responds with emotional "
    "understanding and appropriate behavioral framing. Listen actively, "
    "acknowledge feelings, and provide supportive responses."
)


def _parse_list(val: str) -> list[str]:
    """Parse a string representation of a Python list into an actual list."""
    if not val or val == "[None]":
        return []
    try:
        result = ast.literal_eval(val)
        if isinstance(result, list):
            return [str(x) for x in result if x is not None]
    except (ValueError, SyntaxError):
        pass
    return []


@register_adapter("empath")
class EmpathAdapter(BaseDatasetAdapter):
    """Adapter for Empathetic Dialogues from HuggingFace."""

    def download(self) -> None:
        """Download dataset from HuggingFace if not already present."""
        existing = [self._raw_dir / f"{s}.jsonl" for s in _SPLITS]
        if all(p.exists() and p.stat().st_size > 0 for p in existing):
            return

        cache_dir = self.output_dir.parent / ".hf_cache"
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

        try:
            from datasets import load_dataset

            ds = load_dataset(_HF_REPO_ID, cache_dir=str(cache_dir))
            for split in _SPLITS:
                if split not in ds:
                    continue
                out_path = self._raw_dir / f"{split}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for row in ds[split]:
                        f.write(json.dumps(dict(row)) + "\n")
        except Exception as e:
            readme = self._raw_dir / "README.txt"
            readme.write_text(
                f"Download failed: {e}\nManual download: huggingface-cli download {_HF_REPO_ID}\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Read JSONL files into intermediate dicts."""
        records: list[dict[str, Any]] = []
        for split in _SPLITS:
            path = self._raw_dir / f"{split}.jsonl"
            if not path.exists():
                continue
            for line in self._read_jsonl(path):
                records.append({**line, "_split": split})
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to ChatML with alternating user/assistant turns."""
        records: list[dict[str, Any]] = []

        for row in raw_data:
            chat_history = _parse_list(row.get("chat_history", "[]"))
            sys_response = (row.get("sys_response") or "").strip()
            if not sys_response:
                continue

            situation = (row.get("situation") or "").strip()
            emotion = (row.get("emotion") or "").strip()
            behavior = (row.get("behavior") or "").strip()
            behavior = "" if behavior == "[None]" else behavior

            system_content = _SYSTEM_PROMPT
            if situation:
                system_content += f"\n\nSituation: {situation}"
            if emotion:
                system_content += f"\nEmotion: {emotion}"
            if behavior:
                system_content += f"\nBehavioral framing: {behavior}"

            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

            for i, turn in enumerate(chat_history):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": turn.strip()})

            messages.append({"role": "assistant", "content": sys_response})

            if len(messages) < 3:
                continue

            records.append(
                {
                    "messages": messages,
                    "source": "empath",
                    "task_type": "therapy_response_generation",
                    "diagnostic_tag": emotion or None,
                    "demographic_tags": [],
                    "linguistic_style": "informal",
                    "clinical_reviewed": False,
                    "provenance": self._build_provenance(
                        source_url=_SOURCE_URL,
                        access_method="huggingface",
                        original_format="jsonl",
                    ),
                }
            )

        return records
