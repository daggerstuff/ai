"""Adapter for the BBRD (BPD and Behaviour Reddit Dataset).

Source: https://research.lancaster-university.uk/en/datasets/bpd-and-behaviour-reddit-dataset-bbrd/
Format: CSV
Data: 992 BPD Reddit users, 68,590 posts (2011-2023), 17K manually annotated.
Labels: suicidality, self-harm, substance use, therapy behaviors.
License: CC BY-NC (non-commercial)

Output task_type: symptom_classification
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from ai.pipelines.data_processing.dataset_adapters.adapter_factory import register_adapter
from ai.pipelines.data_processing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://research.lancaster-university.uk/en/datasets/bpd-and-behaviour-reddit-dataset-bbrd/"
_BEHAVIOR_COLUMNS = [
    "suicide",
    "self-harm",
    "medication",
    "therapy",
    "substances",
    "self-control",
    "psychotic_symptoms",
    "social_interactions",
    "intense_emotions",
]


@register_adapter("bbrd")
class BBRDAdapter(BaseDatasetAdapter):
    """Adapter for BBRD BPD Reddit dataset.

    CC BY-NC license. Downloads CSV from Lancaster University. 992 users,
    68K posts, 17K annotated. Labels: suicidality, self-harm, substance use,
    therapy behaviors. task_type: symptom_classification.
    """

    def download(self) -> None:
        """Create README with access instructions or locate local file."""
        readme = self._raw_dir / "README.txt"
        if not readme.exists():
            readme.write_text(
                "BBRD — BPD and Behaviour Reddit Dataset\n"
                "=========================================\n\n"
                f"Source: {_SOURCE_URL}\n\n"
                "Stats:\n"
                "  - 992 BPD Reddit users, 68,590 posts (2011-2023)\n"
                "  - 17K manually annotated posts\n"
                "  - Labels: suicidality, self-harm, substance use, therapy behaviors\n"
                "  - License: CC BY-NC (non-commercial)\n\n"
                "Acquisition:\n"
                "  Download CSV files from the Lancaster University research dataset page.\n"
                "  Place them in this directory.\n\n"
                "Expected CSV columns:\n"
                "  - user_id (or user, username)\n"
                "  - post_text (or text, content, body)\n"
                "  - label (or behavioral annotation columns: suicide, self-harm, etc.)\n"
                "  - timestamp (or date_time, date)\n",
                encoding="utf-8",
            )

        # Ingest local CSV from user home directory if present and raw dir has no CSVs yet
        if not list(self._raw_dir.glob("*.csv")):
            target_csv = self._raw_dir / "BPD_and_Behaviour_Reddit_Dataset_2011-2023_-_Anonymised.csv"
            home_csv = Path.home() / "BPD_and_Behaviour_Reddit_Dataset_2011-2023_-_Anonymised.csv"
            if home_csv.exists():
                shutil.copy2(home_csv, target_csv)

    def extract(self) -> list[dict[str, Any]]:
        """Extract posts from CSV files."""
        records: list[dict[str, Any]] = []

        for cf in sorted(self._raw_dir.glob("*.csv")):
            if cf.name == "README.txt":
                continue
            try:
                with open(cf, encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_id = str(row.get("id") or "").strip().lower()
                        row_user = str(row.get("user") or "").strip().lower()
                        if row_id.startswith("unique post id") or row_user.startswith("user identifier"):
                            continue
                        records.append({**{k.lower(): v for k, v in row.items()}, "_source_file": cf.stem})
            except Exception:
                pass

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for row in raw_data:
            post_text = (row.get("post_text") or row.get("text") or row.get("content") or row.get("body") or "").strip()
            if not post_text:
                continue

            label = self._extract_label(row)
            diagnostic = self._map_label(label)
            demographic_tags = self._extract_demographics(row)

            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical NLP system classifying Reddit posts from "
                        "BPD users. Identify behavioral indicators: suicidality, self-harm, "
                        "substance use, therapy behaviors."
                    ),
                },
                {"role": "user", "content": post_text},
                {"role": "assistant", "content": label},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "bbrd",
                "task_type": "symptom_classification",
                "diagnostic_tag": diagnostic,
                "demographic_tags": demographic_tags,
                "linguistic_style": "informal",
                "clinical_reviewed": True,
                "user_id": row.get("user_id") or row.get("user") or row.get("username", ""),
                "label": label,
                "timestamp": row.get("timestamp") or row.get("date_time") or row.get("date", ""),
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="request",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _extract_label(row: dict[str, Any]) -> str:
        explicit = (
            row.get("label") or row.get("annotation") or row.get("category") or row.get("behavior_type") or ""
        ).strip()
        if explicit:
            return explicit

        active = [f"{col}: {row[col].strip()}" for col in _BEHAVIOR_COLUMNS if (row.get(col) or "").strip()]
        return "; ".join(active) if active else "unclassified"

    @staticmethod
    def _extract_demographics(row: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        age = (row.get("age") or "").strip()
        if age and age.isdigit():
            tags.append(f"age_{age}")
        gender = (row.get("gender") or "").strip()
        if gender:
            tags.append(f"gender_{gender.lower()}")
        ethnicity = (row.get("ethnicity") or "").strip()
        if ethnicity:
            tags.append(f"ethnicity_{ethnicity.lower().replace(' ', '_')}")
        country = (row.get("country") or "").strip()
        if country:
            tags.append(f"country_{country.lower().replace(' ', '_')}")
        if (row.get("depression") or "").strip() == "1":
            tags.append("comorbid_depression")
        if (row.get("ptsd") or "").strip() == "1":
            tags.append("comorbid_ptsd")
        return tags

    @staticmethod
    def _map_label(label: str) -> str:
        norm = label.lower()
        mapping = (
            (("suicid",), "suicidal_ideation"),
            (("self_harm", "self-harm", "selfharm"), "self_harm"),
            (("substance", "drug", "alcohol"), "substance_use"),
            (("therapy", "treatment"), "therapy_behavior"),
            (("psychotic", "psychosis"), "psychotic_symptoms"),
            (("recovery",), "recovery_trajectory"),
        )
        for needles, tag in mapping:
            if any(needle in norm for needle in needles):
                return tag
        return "borderline_personality_disorder"
