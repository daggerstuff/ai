"""Adapter for the HOPE (Helping Oriented PErsonalized Therapy) dataset.

Source: https://github.com/LCS2-IIITD/SPARTA_WSDM2022 (HOPE_data/ folder)
Size: 12.9K utterances, 212 counseling sessions
Format: CSV (one file per session in HOPE_therapy_session_transcripts/)
Columns: ID, Type (T=Therapist, P=Patient), Utterance, Dialog_Act
Labels: 12 DAC labels (gt, id, ynq, irq, crq, pa, na, cd, ack, gc, qo, da, ok, yna)
Topics: CBT, child therapy, family therapy
Paper: WSDM 2022

Output task_type: therapy_response_generation
"""

from __future__ import annotations

import csv
import subprocess
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/LCS2-IIITD/SPARTA_WSDM2022"
_GIT_CLONE_URL = "https://github.com/LCS2-IIITD/SPARTA_WSDM2022.git"
_TRANSCRIPTS_SUBDIR = "HOPE_data/HOPE_therapy_session_transcripts"

_DAC_LABELS = {
    "gt",
    "id",
    "ynq",
    "irq",
    "crq",
    "pa",
    "na",
    "cd",
    "ack",
    "gc",
    "qo",
    "da",
    "ok",
    "yna",
}


@register_adapter("hope")
class HOPEAdapter(BaseDatasetAdapter):
    """Adapter for HOPE counseling session transcripts."""

    def download(self) -> None:
        """Clone the SPARTA repo if not already present."""
        repo_dir = self._raw_dir / "SPARTA_WSDM2022"
        if repo_dir.exists():
            return
        subprocess.run(
            ["git", "clone", "--depth", "1", _GIT_CLONE_URL, str(repo_dir)],
            check=True,
            capture_output=True,
        )

    def extract(self) -> list[dict[str, Any]]:
        """Extract session CSVs from the cloned repo."""
        repo_dir = self._raw_dir / "SPARTA_WSDM2022"
        transcripts_dir = repo_dir / _TRANSCRIPTS_SUBDIR

        if not transcripts_dir.exists():
            return []

        sessions: list[dict[str, Any]] = []
        for csv_file in sorted(transcripts_dir.glob("*.csv")):
            session_id = csv_file.stem
            utterances: list[dict[str, str]] = []
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    utterances.append(
                        {
                            "id": row.get("ID", ""),
                            "type": row.get("Type", ""),
                            "utterance": row.get("Utterance", ""),
                            "dialog_act": row.get("Dialog_Act", ""),
                        }
                    )
            if utterances:
                sessions.append(
                    {
                        "session_id": session_id,
                        "utterances": utterances,
                    }
                )
        return sessions

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert HOPE sessions to ChatML records."""
        records: list[dict[str, Any]] = []

        for session in raw_data:
            utterances = session["utterances"]
            if len(utterances) < 2:
                continue

            messages: list[dict[str, str]] = []
            messages.append(
                {
                    "role": "system",
                    "content": "You are a trained therapist conducting a counseling session.",
                }
            )

            dac_labels: list[str] = []
            for utt in utterances:
                speaker = utt["type"].strip().upper()
                content = utt["utterance"].strip()
                if not content:
                    continue
                role = "assistant" if speaker == "T" else "user"
                messages.append({"role": role, "content": content})
                da = utt.get("dialog_act", "").strip().lower()
                if da and da in _DAC_LABELS:
                    dac_labels.append(da)

            roles = {m["role"] for m in messages}
            if "user" not in roles or "assistant" not in roles:
                continue

            record: dict[str, Any] = {
                "messages": messages,
                "source": "hope",
                "task_type": "therapy_response_generation",
                "diagnostic_tag": None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "session_id": session["session_id"],
                "dac_labels": dac_labels,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records
