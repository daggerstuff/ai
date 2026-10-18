"""Adapter for the Mental Disorders Identification Reddit NLP (Kaggle) dataset.

Source: https://www.kaggle.com/datasets/kamaruladha/mental-disorders-identification-reddit-nlp
Format: CSV
Data: Multi-disorder classification from Reddit posts.
Labels: Multiple mental disorder categories.
License: Kaggle (requires API key)

Output task_type: symptom_classification
"""

from __future__ import annotations

import csv
import subprocess
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://www.kaggle.com/datasets/kamaruladha/mental-disorders-identification-reddit-nlp"
_DATASET_SLUG = "kamaruladha/mental-disorders-identification-reddit-nlp"


@register_adapter("reddit_mental_nlp")
class RedditMentalNLPAdapter(BaseDatasetAdapter):
    """Adapter for Mental Disorders Identification Reddit NLP (Kaggle).

    Downloads via Kaggle API. Multi-disorder classification from Reddit posts.
    task_type: symptom_classification. Maps disorder labels to diagnostic tags.
    """

    def download(self) -> None:
        """Download via Kaggle API, or create README with instructions."""
        csv_files = list(self._raw_dir.glob("*.csv"))
        if csv_files:
            return

        # Try Kaggle API
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", _DATASET_SLUG, "-p", str(self._raw_dir), "--unzip"],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except Exception:
            readme = self._raw_dir / "README.txt"
            if not readme.exists():
                readme.write_text(
                    "Mental Disorders Identification Reddit NLP\n"
                    "=============================================\n\n"
                    f"Source: {_SOURCE_URL}\n\n"
                    "Acquisition:\n"
                    "  1. Set up Kaggle API: https://github.com/Kaggle/kaggle-api#api-credentials\n"
                    "  2. Run: kaggle datasets download -d "
                    f"{_DATASET_SLUG} -p <this_dir> --unzip\n"
                    "  3. Or download manually from the Kaggle page.\n\n"
                    "Expected CSV columns:\n"
                    "  - text (or post, content, body)\n"
                    "  - label (or disorder, category, diagnosis)\n",
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
        seen: set[str] = set()

        for row in raw_data:
            text = (
                row.get("text") or row.get("selftext") or row.get("post") or row.get("content") or row.get("body") or ""
            ).strip()
            title = (row.get("title") or "").strip()
            if not text and title:
                text = title
            if not text:
                continue

            label = (
                row.get("label")
                or row.get("disorder")
                or row.get("category")
                or row.get("diagnosis")
                or row.get("subreddit")
                or ""
            ).strip()

            if not label:
                label = "unspecified"

            dedup_key = f"{' '.join(text.lower().split())}|{label.lower()}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            diagnostic = self._map_diagnosis(label)

            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical NLP system classifying Reddit posts for "
                        "mental disorder identification. Determine the most likely "
                        "disorder category from the text."
                    ),
                },
                {"role": "user", "content": text},
                {"role": "assistant", "content": label},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "reddit_mental_nlp",
                "task_type": "symptom_classification",
                "diagnostic_tag": diagnostic,
                "demographic_tags": [],
                "linguistic_style": "informal",
                "clinical_reviewed": False,
                "label": label,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="kaggle",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _map_diagnosis(label: str) -> str:
        l = label.lower().replace(" ", "_").replace("-", "_")
        mapping = {
            "depression": "major_depressive_disorder",
            "adhd": "adhd",
            "anxiety": "anxiety_disorder",
            "bpd": "borderline_personality_disorder",
            "borderline": "borderline_personality_disorder",
            "ptsd": "ptsd",
            "ocd": "obsessive_compulsive_disorder",
            "bipolar": "bipolar_disorder",
            "schizophrenia": "schizophrenia",
            "eating_disorder": "eating_disorder",
            "autism": "autism_spectrum_disorder",
            "asd": "autism_spectrum_disorder",
        }
        for key, val in mapping.items():
            if key in l:
                return val
        return l if l else "unspecified"
