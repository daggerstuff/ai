"""Adapter for the AnnoMI dataset.

Source: https://github.com/uccollab/AnnoMI
HF: https://huggingface.co/datasets/to-be/annomi-motivational-interviewing-therapy-conversations
Format: CSV (AnnoMI-simple.csv + AnnoMI-full.csv)
Size: 133 MI transcripts, 9,699 utterances, 10 topics
Labels: mi_quality, main_therapist_behaviour, client_talk_type
License: Public Domain (GDPR compliant)
Paper: ICASSP 2022

Output task_type: therapy_response_generation
"""

from __future__ import annotations

import csv
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/uccollab/AnnoMI"

_SIMPLE_URL = "https://raw.githubusercontent.com/uccollab/AnnoMI/main/AnnoMI-simple.csv"
_FULL_URL = "https://raw.githubusercontent.com/uccollab/AnnoMI/main/AnnoMI-full.csv"

_TOPIC_DIAG_MAP: dict[str, str] = {
    "smoking": "smoking_cessation",
    "alcohol": "alcohol_use",
    "substance": "substance_abuse",
    "weight": "weight_management",
    "medication": "medication_adherence",
}


@register_adapter("annomi")
class AnnoMIAdapter(BaseDatasetAdapter):
    """Adapter for AnnoMI motivational interviewing transcripts.

    Downloads CSV files from GitHub raw URLs. Converts therapist/client
    turns into ChatML conversations with MI quality and behavioral labels.
    """

    def download(self) -> None:
        """Download AnnoMI CSV files from GitHub."""
        import urllib.request

        for filename, url in [
            ("AnnoMI-simple.csv", _SIMPLE_URL),
            ("AnnoMI-full.csv", _FULL_URL),
        ]:
            target = self._raw_dir / filename
            if target.exists():
                continue
            try:
                urllib.request.urlretrieve(url, target)
            except Exception:
                pass

    def extract(self) -> list[dict[str, Any]]:
        """Extract utterances from CSV files."""
        csv_files = list(self._raw_dir.glob("*.csv"))
        if not csv_files:
            return []

        utterances: list[dict[str, Any]] = []
        for cf in sorted(csv_files):
            is_full = "full" in cf.stem.lower()
            with open(cf, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entry: dict[str, Any] = {
                        "utterance_id": row.get("utterance_id", row.get("ID", "")),
                        "transcript_id": row.get("transcript_id", ""),
                        "interlocutor": row.get("interlocutor", "").strip().lower(),
                        "utterance": row.get("utterance_text", row.get("utterance", row.get("utterance", ""))).strip(),
                        "mi_quality": row.get("mi_quality", "").strip().lower(),
                        "main_therapist_behaviour": row.get("main_therapist_behaviour", "").strip().lower(),
                        "client_talk_type": row.get("client_talk_type", "").strip().lower(),
                        "topic": row.get("topic", "").strip().lower(),
                        "is_full": is_full,
                    }
                    if is_full:
                        entry["therapist_input_exists"] = row.get("therapist_input_exists", "").strip().lower()
                        entry["reflection_subtype"] = row.get("reflection_subtype", "").strip().lower()
                        entry["question_subtype"] = row.get("question_subtype", "").strip().lower()
                    utterances.append(entry)
        return utterances

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        # Group by transcript_id
        transcripts: dict[str, list[dict[str, Any]]] = {}
        for utt in raw_data:
            tid = utt.get("transcript_id") or utt.get("utterance_id", "")
            if not tid:
                continue
            transcripts.setdefault(tid, []).append(utt)

        for tid, utterances in transcripts.items():
            messages: list[dict[str, str]] = [
                {"role": "system", "content": "You are a motivational interviewing therapist conducting a session."}
            ]

            mi_qualities: list[str] = []
            therapist_behaviours: list[str] = []
            client_talk_types: list[str] = []
            topics: list[str] = []

            for utt in utterances:
                content = utt["utterance"]
                if not content:
                    continue
                interlocutor = utt["interlocutor"]
                role = "assistant" if interlocutor in ("therapist", "t") else "user"
                messages.append({"role": role, "content": content})

                if utt["mi_quality"]:
                    mi_qualities.append(utt["mi_quality"])
                if utt["main_therapist_behaviour"]:
                    therapist_behaviours.append(utt["main_therapist_behaviour"])
                if utt["client_talk_type"]:
                    client_talk_types.append(utt["client_talk_type"])
                if utt["topic"]:
                    topics.append(utt["topic"])

            roles = {m["role"] for m in messages}
            if "user" not in roles or "assistant" not in roles:
                continue

            topic = topics[0] if topics else ""
            diagnostic_tag = _TOPIC_DIAG_MAP.get(topic, topic or None)

            record: dict[str, Any] = {
                "messages": messages,
                "source": "annomi",
                "task_type": "therapy_response_generation",
                "diagnostic_tag": diagnostic_tag,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": True,
                "transcript_id": tid,
                "mi_quality": mi_qualities[0] if mi_qualities else None,
                "therapist_behaviours": list(set(therapist_behaviours)),
                "client_talk_types": list(set(client_talk_types)),
                "topic": topic,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records
