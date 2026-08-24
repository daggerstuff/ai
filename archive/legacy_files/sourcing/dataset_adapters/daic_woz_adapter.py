"""Adapter for the DAIC-WOZ / E-DAIC clinical interview dataset.

Source: https://dcapswoz.ict.usc.edu/
HuggingFace mirror: saeedzou/DAIC-WOZ (parquet, 46,721 utterance-level rows)
Format: Transcripts + Audio (audio column ignored for text pipeline)
Data: 189 sessions, PTSD/depression. Labels: PHQ-8, PCL-C.
License: Academic license (HF mirror available)

Output task_type: severity_estimation
Uses transcript text only (no audio). Groups utterances by participant.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_HF_REPO_ID = "saeedzou/DAIC-WOZ"
_SOURCE_URL = "https://dcapswoz.ict.usc.edu/"

_README_TEXT = """\
DAIC-WOZ / E-DAIC — Clinical Interview Dataset
==============================================

Source: https://dcapswoz.ict.usc.edu/
HuggingFace mirror: saeedzou/DAIC-WOZ

Stats:
  - 46,721 utterance-level rows (train: 25,209, dev: 8,949, test: 12,563)
  - 189 participants
  - PTSD and depression detection
  - Labels: PHQ-8, PCL-C
  - Format: parquet (text + audio; audio ignored for text pipeline)

HF columns:
  participant_id, speaker, text, start_time, stop_time,
  PHQ8_Binary, PHQ8_Score, PTSD_severity, PTSD_label, Gender, age,
  PHQ8 sub-scores, PCL-C sub-scores

Downloaded via `datasets.load_dataset('saeedzou/DAIC-WOZ')`.
Audio column removed to avoid torchcodec dependency.
"""


@register_adapter("daic_woz")
class DAICWozAdapter(BaseDatasetAdapter):
    """Adapter for DAIC-WOZ clinical interview transcripts.

    Downloads from HuggingFace mirror (saeedzou/DAIC-WOZ).
    Groups utterances by participant_id. PHQ-8 → severity_estimation.
    """

    def download(self) -> None:
        """Download from HuggingFace or create README if HF unavailable."""
        jsonl_files = list(self._raw_dir.glob("*.jsonl"))
        if jsonl_files:
            return  # Already downloaded

        # Try HF download
        readme = self._raw_dir / "README.txt"
        try:
            import datasets

            cache_dir = str(self.output_dir.parent / ".hf_cache")
            os.environ.setdefault("HF_HOME", cache_dir)
            os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
            ds = datasets.load_dataset(_HF_REPO_ID, cache_dir=cache_dir)
            ds = ds.remove_columns("audio")  # Avoid torchcodec dependency
            self._raw_dir.mkdir(parents=True, exist_ok=True)
            for split in ds:
                path = self._raw_dir / f"{split}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for row in ds[split]:
                        f.write(json.dumps(row) + "\n")
        except Exception:
            if not readme.exists():
                readme.write_text(_README_TEXT, encoding="utf-8")

    def extract(self) -> list[dict[str, Any]]:
        """Extract sessions from JSONL files, grouped by participant_id."""
        sessions: dict[str, list[dict[str, Any]]] = {}
        labels: dict[str, dict[str, Any]] = {}

        for jf in sorted(self._raw_dir.glob("*.jsonl")):
            with open(jf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    pid = str(row.get("participant_id", ""))
                    if not pid:
                        continue
                    sessions.setdefault(pid, []).append(row)
                    # Collect labels from first row of each participant
                    if pid not in labels:
                        labels[pid] = {
                            "PHQ8_Binary": row.get("PHQ8_Binary"),
                            "PHQ8_Score": row.get("PHQ8_Score"),
                            "PTSD_severity": row.get("PTSD_severity"),
                            "PTSD_label": row.get("PTSD_label"),
                            "Gender": row.get("Gender"),
                            "age": row.get("age"),
                        }

        records: list[dict[str, Any]] = []
        for pid, utterances in sessions.items():
            records.append(
                {
                    "session_id": pid,
                    "utterances": utterances,
                    "labels": labels.get(pid, {}),
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
            phq8_score = labels.get("PHQ8_Score", "")
            phq8_binary = labels.get("PHQ8_Binary", "")
            ptsd_severity = labels.get("PTSD_severity", "")
            ptsd_label = labels.get("PTSD_label", "")
            gender = labels.get("Gender", "")
            age = labels.get("age", "")

            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": self._build_system(
                        phq8_score=phq8_score,
                        ptsd_severity=ptsd_severity,
                        gender=gender,
                        age=age,
                    ),
                }
            ]

            has_user = False
            has_assistant = False
            for utt in utterances:
                speaker = str(utt.get("speaker", "")).strip()
                text = str(utt.get("text", "")).strip()
                if not text:
                    continue

                # In DAIC-WOZ, "Ellie" is the virtual interviewer, participant is the patient
                if speaker.lower() == "ellie":
                    messages.append({"role": "assistant", "content": text})
                    has_assistant = True
                else:
                    messages.append({"role": "user", "content": text})
                    has_user = True

            if not has_user or not has_assistant:
                # Single-speaker session — add synthetic counterpart
                if has_user and not has_assistant:
                    messages.append({"role": "assistant", "content": "[continuation]"})
                elif has_assistant and not has_user:
                    messages.insert(1, {"role": "user", "content": "[session context]"})
                else:
                    continue

            severity = self._phq8_to_severity(str(phq8_score))

            record: dict[str, Any] = {
                "messages": messages,
                "source": "daic_woz",
                "task_type": "severity_estimation",
                "diagnostic_tag": "depression" if phq8_binary else "ptsd",
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": True,
                "session_id": session.get("session_id", ""),
                "phq8_score": str(phq8_score) if phq8_score != "" else None,
                "phq8_binary": str(phq8_binary) if phq8_binary != "" else None,
                "ptsd_severity": str(ptsd_severity) if ptsd_severity != "" else None,
                "ptsd_label": str(ptsd_label) if ptsd_label != "" else None,
                "gender": str(gender) if gender != "" else None,
                "age": str(age) if age != "" else None,
                "severity": severity,
                "provenance": self._build_provenance(
                    source_url="https://huggingface.co/datasets/" + _HF_REPO_ID,
                    access_method="huggingface",
                    original_format="parquet",
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _build_system(phq8_score: Any, ptsd_severity: Any, gender: Any, age: Any) -> str:
        parts: list[str] = [
            "DAIC-WOZ clinical interview transcript (USC ICT).",
            "Virtual interviewer Ellie conducted automated depression/PTSD screening.",
        ]
        if phq8_score not in (None, "", "None"):
            parts.append(f"PHQ-8 score: {phq8_score}.")
        if ptsd_severity not in (None, "", "None"):
            parts.append(f"PTSD severity: {ptsd_severity}.")
        if age not in (None, "", "None"):
            parts.append(f"Age: {age}.")
        if gender not in (None, "", "None"):
            gender_str = (
                "female" if str(gender) in ("0", "0.0") else "male" if str(gender) in ("1", "1.0") else str(gender)
            )
            parts.append(f"Gender: {gender_str}.")
        parts.append("Assess depression and PTSD symptom severity from the interview.")
        return " ".join(parts)

    @staticmethod
    def _phq8_to_severity(phq8: str) -> str:
        if not phq8:
            return "unknown"
        try:
            score = float(phq8)
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
        except (ValueError, TypeError):
            return "unknown"
