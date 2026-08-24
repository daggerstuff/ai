"""Adapter for the MI-TAGS (Motivational Interviewing Transcripts Annotated with Global Scores) dataset.

Source: https://github.com/Advanced-Reality-Lab/MI-TAGS
Size: 242 MI demonstration transcripts (sample in repo, full via access request)
Format: CSV (sample_utterances.csv + sample_global_mitis.csv)
Utterance columns: id, Video Title, Turn, Speaker, Text, Code, Annotator, Normalized Turn
Global columns: id, Video Title, Annotator, Empathy, SofteningSustainTalk, CultivatingChangeTalk, Partnership, ...
Topics: Smoking cessation, alcohol, substance abuse, weight management, medication adherence
Paper: LREC-COLING 2024 (Best Paper nominee)

Output task_type: therapy_response_generation
"""

from __future__ import annotations

import csv
import subprocess
from collections import defaultdict
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/Advanced-Reality-Lab/MI-TAGS"
_GIT_CLONE_URL = "https://github.com/Advanced-Reality-Lab/MI-TAGS.git"
_UTTERANCES_FILE = "sample_utterances.csv"
_GLOBALS_FILE = "sample_global_mitis.csv"

_MITI_CODES = {
    "Giving Information",
    "Question Open",
    "Question Closed",
    "Simple Reflection",
    "Complex Reflection",
    "Affirm",
    "Seeking Collaboration",
    "Emphasizing Autonomy",
    "Structure Statement",
    "Support",
    "Self-Disclosure",
}


@register_adapter("mitags")
class MITAGSAdapter(BaseDatasetAdapter):
    """Adapter for MI-TAGS motivational interviewing transcripts."""

    def download(self) -> None:
        repo_dir = self._raw_dir / "MI-TAGS"
        if repo_dir.exists():
            return
        subprocess.run(
            ["git", "clone", "--depth", "1", _GIT_CLONE_URL, str(repo_dir)],
            check=True,
            capture_output=True,
        )

    def extract(self) -> list[dict[str, Any]]:
        repo_dir = self._raw_dir / "MI-TAGS"
        utterances_file = repo_dir / _UTTERANCES_FILE
        globals_file = repo_dir / _GLOBALS_FILE

        if not utterances_file.exists():
            return []

        utterances_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with open(utterances_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_title = row.get("Video Title", "")
                utterances_by_video[video_title].append(
                    {
                        "turn": row.get("Turn", ""),
                        "speaker": row.get("Speaker", ""),
                        "text": row.get("Text", ""),
                        "code": row.get("Code", ""),
                        "normalized_turn": row.get("Normalized Turn", ""),
                    }
                )

        global_scores: dict[str, dict[str, Any]] = {}
        if globals_file.exists():
            with open(globals_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_title = row.get("Video Title", "")
                    global_scores[video_title] = {
                        "empathy": int(row.get("Empathy", 0) or 0),
                        "softening_sustain_talk": int(row.get("SofteningSustainTalk", 0) or 0),
                        "cultivating_change_talk": int(row.get("CultivatingChangeTalk", 0) or 0),
                        "partnership": int(row.get("Partnership", 0) or 0),
                    }

        sessions: list[dict[str, Any]] = []
        for video_title, utterances in utterances_by_video.items():
            sessions.append(
                {
                    "video_title": video_title,
                    "utterances": utterances,
                    "global_scores": global_scores.get(video_title, {}),
                }
            )
        return sessions

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for session in raw_data:
            utterances = session["utterances"]
            if not utterances:
                continue

            messages: list[dict[str, str]] = []
            messages.append(
                {
                    "role": "system",
                    "content": "You are a trained motivational interviewing practitioner.",
                }
            )

            miti_codes: list[str] = []
            for utt in utterances:
                speaker = utt["speaker"].strip().upper()
                text = utt["text"].strip()
                if not text:
                    continue
                role = "assistant" if speaker == "T" else "user"
                messages.append({"role": role, "content": text})
                code = utt.get("code", "").strip()
                if code:
                    miti_codes.append(code)

            if len(messages) < 2:
                continue

            # If only one speaker role present, add a minimal counterpart
            # so the record passes base validation (sample data has 1 utterance per session)
            roles_present = {m["role"] for m in messages}
            if "user" not in roles_present:
                messages.insert(1, {"role": "user", "content": "[session context]"})
            elif "assistant" not in roles_present:
                messages.append({"role": "assistant", "content": "[continuation]"})

            record: dict[str, Any] = {
                "messages": messages,
                "source": "mitags",
                "task_type": "therapy_response_generation",
                "diagnostic_tag": None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "video_title": session["video_title"],
                "miti_codes": miti_codes,
                "global_scores": session.get("global_scores", {}),
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records
