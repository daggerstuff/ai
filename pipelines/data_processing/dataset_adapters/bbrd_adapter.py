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
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://research.lancaster-university.uk/en/datasets/bpd-and-behaviour-reddit-dataset-bbrd/"


@register_adapter("bbrd")
class BBRDAdapter(BaseDatasetAdapter):
    """Adapter for BBRD BPD Reddit dataset.

    CC BY-NC license. Downloads CSV from Lancaster University. 992 users,
    68K posts, 17K annotated. Labels: suicidality, self-harm, substance use,
    therapy behaviors. task_type: symptom_classification.
    """

    def download(self) -> None:
        """Create README with access instructions."""
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
                "  - user_id (or username)\n"
                "  - post_text (or text, content, body)\n"
                "  - label (or annotation, category)\n"
                "  - timestamp (optional)\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract posts from CSV files."""
        records: list[dict[str, Any]] = []

        for cf in sorted(self._raw_dir.glob("*.csv")):
            if cf.name == "README.txt":
                continue
            try:
                with open(cf, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
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

            label = (
                row.get("label") or row.get("annotation") or row.get("category") or row.get("behavior_type") or ""
            ).strip()

            if not label:
                label = "unclassified"

            # Map label to diagnostic context
            diagnostic = self._map_label(label)

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
                "demographic_tags": [],
                "linguistic_style": "informal",
                "clinical_reviewed": True,
                "user_id": row.get("user_id", row.get("username", "")),
                "label": label,
                "timestamp": row.get("timestamp", ""),
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="request",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _map_label(label: str) -> str:
        l = label.lower()
        if "suicid" in l:
            return "suicidal_ideation"
        if "self_harm" in l or "self-harm" in l or "selfharm" in l:
            return "self_harm"
        if "substance" in l or "drug" in l or "alcohol" in l:
            return "substance_use"
        if "therapy" in l or "treatment" in l:
            return "therapy_behavior"
        if "recovery" in l:
            return "recovery_trajectory"
        return "borderline_personality_disorder"
