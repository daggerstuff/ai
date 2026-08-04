"""Adapter for the DAIC-WOZ / E-DAIC clinical interview dataset.

Source: https://dcapswoz.ict.usc.edu/
Format: Transcripts CSV + Audio WAV + Features
Data: 189 sessions, PTSD/depression. Labels: PHQ-8, PCL-C.
License: Academic license (request-based access)

Output task_type: severity_estimation
Uses transcript CSV only (text training). Audio modality ignored for text pipeline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://dcapswoz.ict.usc.edu/"

_README_TEXT = """\
DAIC-WOZ / E-DAIC — Clinical Interview Dataset
===============================================

Source: https://dcapswoz.ict.usc.edu/

Stats:
  - 189 sessions
  - PTSD and depression detection
  - Labels: PHQ-8, PCL-C
  - Format: transcripts CSV + audio WAV + features

Acquisition:
  Request access from USC ICT. After approval, download the transcript
  CSV files and place them in this directory. The adapter reads any *.csv
  files with transcript data.

Expected transcript CSV columns:
  - participant_id  (or speaker)
  - transcript       (or utterance, text)
  - timestamp        (optional)

Labels should be in a separate file (e.g., labels.csv) with:
  - participant_id
  - phq8_score
  - pcl_c_score (optional)
"""


@register_adapter("daic_woz")
class DAICWozAdapter(BaseDatasetAdapter):
    """Adapter for DAIC-WOZ clinical interview transcripts.

    Request-based access. Uses transcript CSV files only (no audio).
    Groups utterances by participant/session. PHQ-8 → severity_estimation.
    """

    def download(self) -> None:
        """Create README with access instructions."""
        readme = self._raw_dir / "README.txt"
        if not readme.exists():
            readme.write_text(_README_TEXT, encoding="utf-8")

    def extract(self) -> list[dict[str, Any]]:
        """Extract transcript sessions from CSV files."""
        records: list[dict[str, Any]] = []

        # Load labels if available
        labels: dict[str, dict[str, str]] = {}
        for cf in sorted(self._raw_dir.glob("*label*")):
            if not cf.name.endswith(".csv"):
                continue
            try:
                with open(cf, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pid = str(row.get("participant_id", row.get("id", ""))).strip()
                        if pid:
                            labels[pid] = {k.lower(): v for k, v in row.items()}
            except Exception:
                pass

        # Load transcripts grouped by session
        sessions: dict[str, list[dict[str, str]]] = {}
        for cf in sorted(self._raw_dir.glob("*.csv")):
            if "label" in cf.name.lower() or cf.name == "README.txt":
                continue
            try:
                with open(cf, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        lower = {k.lower(): v for k, v in row.items()}
                        session_id = str(lower.get("participant_id", lower.get("session_id", lower.get("id", cf.stem))))
                        sessions.setdefault(session_id, []).append({**lower, "_source_file": cf.stem})
            except Exception:
                pass

        for session_id, utterances in sessions.items():
            records.append(
                {
                    "session_id": session_id,
                    "utterances": utterances,
                    "labels": labels.get(session_id, {}),
                    "_source_file": utterances[0].get("_source_file", "") if utterances else "",
                }
            )

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for session in raw_data:
            utterances = session.get("utterances", [])
            if not utterances:
                continue

            labels = session.get("labels", {})
            phq8 = labels.get("phq8_score", "")
            pclc = labels.get("pcl_c_score", "")

            # Build conversation from utterances
            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": self._build_system(phq8=phq8, pclc=pclc),
                }
            ]

            has_user = False
            has_assistant = False
            for utt in utterances:
                speaker = str(utt.get("speaker", utt.get("participant", utt.get("role", "")))).lower()
                text = str(utt.get("transcript", utt.get("utterance", utt.get("text", "")))).strip()
                if not text:
                    continue

                if speaker in ("ellie", "interviewer", "therapist", "counselor", "assistant"):
                    messages.append({"role": "assistant", "content": text})
                    has_assistant = True
                elif speaker in ("participant", "patient", "client", "user"):
                    messages.append({"role": "user", "content": text})
                    has_user = True
                else:
                    # Default to user for unknown speakers
                    messages.append({"role": "user", "content": text})
                    has_user = True

            if not has_user or not has_assistant:
                continue

            # Determine severity from PHQ-8
            severity = self._phq8_to_severity(phq8)

            record: dict[str, Any] = {
                "messages": messages,
                "source": "daic_woz",
                "task_type": "severity_estimation",
                "diagnostic_tag": "depression" if phq8 else None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": True,
                "session_id": session.get("session_id", ""),
                "phq8_score": phq8,
                "pcl_c_score": pclc,
                "severity": severity,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="request",
                    original_format="csv_audio",
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _build_system(phq8: str, pclc: str) -> str:
        parts: list[str] = ["DAIC-WOZ clinical interview transcript (USC ICT)."]
        if phq8:
            parts.append(f"PHQ-8 score: {phq8}.")
        if pclc:
            parts.append(f"PCL-C score: {pclc}.")
        parts.append("Assess depression and PTSD symptom severity from the interview.")
        return " ".join(parts)

    @staticmethod
    def _phq8_to_severity(phq8: str) -> str:
        if not phq8:
            return "unknown"
        try:
            score = int(phq8)
            if score < 5:
                return "minimal"
            elif score < 10:
                return "mild"
            elif score < 15:
                return "moderate"
            elif score < 20:
                return "moderately_severe"
            else:
                return "severe"
        except ValueError:
            return "unknown"
