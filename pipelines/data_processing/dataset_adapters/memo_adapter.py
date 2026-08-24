"""Adapter for the MEMO (Mental hEalth suMmarizatOn) dataset.

Source: https://github.com/LCS2-IIITD/MEMO
Size: 12.9K utterances, 212 counseling sessions (extends HOPE)
Format: CSV (request-based access via Google Form)
Labels: 4 psychotherapy elements: symptom_and_history, patient_discovery, reflecting, discussion_filler
Paper: KDD 2022

NOTE: MEMO data is request-based. Users must fill out the Google Form at
https://forms.gle/RarCVxAdmGUP3Pmh7 and place the received CSV files in
the raw directory before running this adapter.

Output task_type: therapy_response_generation
"""

from __future__ import annotations

import csv
from typing import Any

from ai.pipelines.data_processing.dataset_adapters.adapter_factory import register_adapter
from ai.pipelines.data_processing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/LCS2-IIITD/MEMO"

_PSYCHOTHERAPY_ELEMENTS = {
    "symptom_and_history",
    "patient_discovery",
    "reflecting",
    "discussion_filler",
}


@register_adapter("memo")
class MEMOAdapter(BaseDatasetAdapter):
    """Adapter for MEMO counseling summarization dataset.

    Expects CSV files placed in the raw directory after access request.
    Supports HOPE-format CSVs (ID, Type, Utterance, Dialog_Act) with an
    optional 'Component' column for psychotherapy element labels.
    """

    def download(self) -> None:
        """No-op: MEMO data is request-based.

        Users must obtain access via Google Form and place CSV files
        in the raw directory manually.
        """
        readme = self._raw_dir / "README.txt"
        if not readme.exists():
            readme.write_text(
                "MEMO dataset requires access request.\n"
                "1. Fill out: https://forms.gle/RarCVxAdmGUP3Pmh7\n"
                "2. Email: aseems@iiitd.ac.in\n"
                "3. Place received CSV files in this directory.\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract sessions from CSV files in the raw directory."""
        csv_files = list(self._raw_dir.glob("*.csv"))
        if not csv_files:
            return []

        sessions: list[dict[str, Any]] = []
        for csv_file in sorted(csv_files):
            session_id = csv_file.stem
            utterances: list[dict[str, Any]] = []
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    utterances.append(
                        {
                            "id": row.get("ID", ""),
                            "type": row.get("Type", ""),
                            "utterance": row.get("Utterance", ""),
                            "dialog_act": row.get("Dialog_Act", ""),
                            "component": row.get("Component", ""),
                        }
                    )
            if utterances:
                sessions.append({"session_id": session_id, "utterances": utterances})
        return sessions

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for session in raw_data:
            utterances = session["utterances"]
            if len(utterances) < 2:
                continue

            messages: list[dict[str, str]] = []
            messages.append(
                {
                    "role": "system",
                    "content": "You are a trained counselor conducting a therapy session.",
                }
            )

            components: list[str] = []
            for utt in utterances:
                speaker = utt["type"].strip().upper()
                content = utt["utterance"].strip()
                if not content:
                    continue
                role = "assistant" if speaker == "T" else "user"
                messages.append({"role": role, "content": content})
                comp = utt.get("component", "").strip().lower()
                if comp and comp in _PSYCHOTHERAPY_ELEMENTS:
                    components.append(comp)

            roles = {m["role"] for m in messages}
            if "user" not in roles or "assistant" not in roles:
                continue

            record: dict[str, Any] = {
                "messages": messages,
                "source": "memo",
                "task_type": "therapy_response_generation",
                "diagnostic_tag": None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "session_id": session["session_id"],
                "psychotherapy_elements": components,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="request",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records
