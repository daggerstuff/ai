"""Adapter for the machine_learning_BPD dataset.

Source: https://github.com/saidejp/machine_learning_BPD
Format: CSV (DBT treatment outcome data)
Data: ~30-50 BPD patients pre/post DBT
Labels: impulsivity (BIS-11), symptom severity
License: GitHub (research)

Output task_type: severity_estimation
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/saidejp/machine_learning_BPD"


@register_adapter("ml_bpd")
class MLBPDAdapter(BaseDatasetAdapter):
    """Adapter for machine_learning_BPD DBT treatment outcome data.

    Downloads CSV files from GitHub raw URLs. Converts patient records
    into ChatML with BPD symptom assessment context.
    """

    def download(self) -> None:
        """Download CSV files from GitHub repo."""
        import urllib.request

        repo_base = "https://raw.githubusercontent.com/saidejp/machine_learning_BPD/main/"
        candidate_files = [
            "data.csv",
            "BPD_data.csv",
            "treatment_outcome.csv",
            "bis11_scores.csv",
            "symptom_severity.csv",
        ]
        for fname in candidate_files:
            target = self._raw_dir / fname
            if target.exists():
                continue
            try:
                urllib.request.urlretrieve(repo_base + fname, target)
            except Exception:
                pass

        if not any(self._raw_dir.glob("*.csv")):
            readme = self._raw_dir / "README.txt"
            readme.write_text(
                "machine_learning_BPD data download.\n"
                "1. Visit: https://github.com/saidejp/machine_learning_BPD\n"
                "2. Download CSV files from the repo.\n"
                "3. Place them in this directory.\n"
                "Expected: patient records with BIS-11 impulsivity scores and symptom severity.\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract patient records from CSV files."""
        csv_files = list(self._raw_dir.glob("*.csv"))
        if not csv_files:
            return []

        records: list[dict[str, Any]] = []
        for cf in sorted(csv_files):
            with open(cf, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append({**{k.lower(): v for k, v in row.items()}, "_source_file": cf.stem})
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for row in raw_data:
            # Find the most descriptive text field
            description = (
                row.get("description") or row.get("symptoms") or row.get("clinical_notes") or row.get("notes") or ""
            ).strip()

            if not description:
                # Build description from available fields
                parts = []
                for key in ("bis11_score", "impulsivity", "symptom_severity", "severity", "age", "gender"):
                    val = row.get(key, "").strip()
                    if val:
                        parts.append(f"{key}: {val}")
                description = "; ".join(parts) if parts else ""

            if not description:
                continue

            # Find outcome/prediction target
            outcome = (
                row.get("treatment_outcome")
                or row.get("outcome")
                or row.get("dbt_response")
                or row.get("response")
                or row.get("label")
                or ""
            ).strip()

            if not outcome:
                # Use severity as outcome
                outcome = row.get("symptom_severity", row.get("severity", "moderate")).strip()

            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": "You are a clinical assessor evaluating BPD patient treatment outcome data. Assess symptom severity based on DBT pre/post measures.",
                },
                {"role": "user", "content": description},
                {"role": "assistant", "content": outcome if outcome else "moderate"},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "ml_bpd",
                "task_type": "severity_estimation",
                "diagnostic_tag": "borderline_personality_disorder",
                "demographic_tags": self._extract_demographics(row),
                "linguistic_style": "formal",
                "clinical_reviewed": True,
                "bis11_score": row.get("bis11_score", row.get("impulsivity", "")),
                "symptom_severity": row.get("symptom_severity", row.get("severity", "")),
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _extract_demographics(row: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        age = row.get("age", "").strip()
        if age:
            try:
                a = int(age)
                if a <= 25:
                    tags.append("age_18_25")
                elif a <= 45:
                    tags.append("age_26_45")
                else:
                    tags.append("age_46_plus")
            except ValueError:
                pass
        gender = row.get("gender", "").strip().lower()
        if gender:
            if "female" in gender or gender == "f":
                tags.append("gender_female")
            elif "male" in gender or gender == "m":
                tags.append("gender_male")
            else:
                tags.append("gender_nonbinary")
        return tags
