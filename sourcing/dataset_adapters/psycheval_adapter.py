"""Adapter for the PsychEval dataset — Chinese clinical counseling dialogues.

Source: S3 (whitebat:training/pixelated-empathy/output/psycheval/raw/PsychEval/)
Format: JSON files organized by therapy modality (bt, cbt, het, integrative, pdt, pmt)
Each file: {theoretical, client_id, client_info, global_plan, sessions[]}
Each session: {session_number, session_goals, suggest_skills, session_dialogue[], session_summary}
session_dialogue: [{role: "Counselor"|"Client", text: str}]
Counselor text may contain XML-like meta tags (<assessment>, <client_state>, <skill>, <strategy>)
  before the actual response — these are stripped for ChatML content.

Size: 369 files × ~8 sessions × ~63 turns = ~186K dialogue turns
Language: Chinese (zh)
Therapy modalities: BT, CBT, HET, Integrative, PDT, PMT

Output task_type: therapy_response_generation
Each record: one counseling session as a multi-turn dialogue.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_S3_PREFIX = "whitebat:training/pixelated-empathy/output/psycheval/raw/PsychEval/"
_SOURCE_URL = "https://github.com/PsychEval/PsychEval"

_MODALITIES = ["bt", "cbt", "het", "integrative", "pdt", "pmt"]

_MODALITY_NAMES = {
    "bt": "Behavior Therapy (BT)",
    "cbt": "Cognitive Behavioral Therapy (CBT)",
    "het": "Humanistic/Existential Therapy (HET)",
    "integrative": "Integrative Therapy",
    "pdt": "Psychodynamic Therapy (PDT)",
    "pmt": "Psychodynamic/Motivational Therapy (PMT)",
}

# Pattern to strip XML-like meta tags from counselor text
_META_TAG_PATTERN = re.compile(r"<[^>]+>.*?</[^>]+>\s*")


def _strip_meta_tags(text: str) -> str:
    """Remove XML-like meta tags from counselor dialogue text."""
    # Remove all <tag>...</tag> patterns, keeping text after the last tag
    cleaned = _META_TAG_PATTERN.sub("", text).strip()
    if not cleaned:
        # If all text was in tags, try to extract from the last tag
        match = re.findall(r">([^<]+)<", text)
        if match:
            cleaned = match[-1].strip()
    return cleaned


@register_adapter("psycheval")
class PsychEvalAdapter(BaseDatasetAdapter):
    """Adapter for PsychEval Chinese clinical counseling dialogues from S3."""

    def download(self) -> None:
        """Download JSON data from S3 if not already present."""
        data_dir = self._raw_dir / "PsychEval" / "data"
        if data_dir.exists() and any(data_dir.rglob("*.json")):
            return

        try:
            subprocess.run(
                ["rclone", "copy", _S3_PREFIX + "data/", str(data_dir)],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError) as e:
            readme = self._raw_dir / "README.txt"
            readme.write_text(
                f"Download failed: {e}\nManual: rclone copy {_S3_PREFIX}data/ <target>\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Read JSON files organized by modality into intermediate dicts."""
        data_dir = self._raw_dir / "PsychEval" / "data"
        records: list[dict[str, Any]] = []

        for modality in _MODALITIES:
            mod_dir = data_dir / modality
            if not mod_dir.exists():
                continue
            for json_path in sorted(mod_dir.glob("*.json")):
                try:
                    with open(json_path, encoding="utf-8") as f:
                        data = json.load(f)
                    records.append(
                        {
                            "_source_file": str(json_path.relative_to(self._raw_dir)),
                            "_modality": modality,
                            **data,
                        }
                    )
                except (json.JSONDecodeError, OSError):
                    continue

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert PsychEval JSON files to ChatML sessions."""
        records: list[dict[str, Any]] = []

        for entry in raw_data:
            modality = entry.get("_modality", "")
            client_info = entry.get("client_info", {})
            client_id = entry.get("client_id", 0)
            sessions = entry.get("sessions", [])

            modality_name = _MODALITY_NAMES.get(modality, modality)

            # Build system prompt with client context
            traits = client_info.get("static_traits", {})
            main_problem = client_info.get("main_problem", "")
            topic = client_info.get("topic", "")

            system_parts = [
                f"あなたは{modality_name}の臨床カウンセラーです。",
                f"クライアント: {traits.get('name', '不明')}, {traits.get('age', '不明')}, {traits.get('gender', '不明')}",
            ]
            if main_problem:
                system_parts.append(f"主訴: {main_problem}")
            if topic:
                system_parts.append(f"トピック: {topic}")
            system_content = "\n".join(system_parts)

            for session in sessions:
                dialogue = session.get("session_dialogue", [])
                if not dialogue:
                    continue

                session_num = session.get("session_number", 0)

                messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

                for turn in dialogue:
                    role_str = (turn.get("role") or "").strip()
                    text = turn.get("text") or ""

                    if role_str == "Counselor":
                        text = _strip_meta_tags(text)
                        if text:
                            messages.append({"role": "assistant", "content": text})
                    elif role_str == "Client":
                        text = text.strip()
                        if text:
                            messages.append({"role": "user", "content": text})

                if len(messages) < 3:
                    continue

                has_user = any(m["role"] == "user" for m in messages)
                has_assistant = any(m["role"] == "assistant" for m in messages)
                if not has_user or not has_assistant:
                    continue

                records.append(
                    {
                        "messages": messages,
                        "source": "psycheval",
                        "task_type": "therapy_response_generation",
                        "diagnostic_tag": topic or None,
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
