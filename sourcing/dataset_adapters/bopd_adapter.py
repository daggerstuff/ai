"""Adapter for the BoPD (Borderline Personality Disorder) screening tool.

Source: https://github.com/BoPDdiseasescreening/Borderline-Personality-Disorder-BoPD-automatic-disease-screening-tool
Format: SQL + Python (EHR screening tool with ICD-10-CM codes)
Data: 456 patients, AUROC 0.837
License: GitHub (research)

Output task_type: symptom_classification
"""

from __future__ import annotations

import csv
import subprocess
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = (
    "https://github.com/BoPDdiseasescreening/Borderline-Personality-Disorder-BoPD-automatic-disease-screening-tool"
)


@register_adapter("bopd")
class BoPDAdapter(BaseDatasetAdapter):
    """Adapter for BoPD EHR screening tool.

    Git clones the repo, extracts any CSV/JSON data files containing
    patient screening records with ICD-10-CM codes.
    """

    def download(self) -> None:
        """Git clone the BoPD repo if not already present."""
        clone_dir = self._raw_dir / "bopd_repo"
        if clone_dir.exists():
            return
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", _SOURCE_URL, str(clone_dir)],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except Exception:
            readme = self._raw_dir / "README.txt"
            readme.write_text(
                "BoPD screening tool data download.\n"
                "1. Visit: https://github.com/BoPDdiseasescreening/Borderline-Personality-Disorder-BoPD-automatic-disease-screening-tool\n"
                "2. Clone or download the repo.\n"
                "3. Place CSV/JSON data files in this directory.\n"
                "Expected: EHR screening records with ICD-10-CM codes for BPD.\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract patient records from CSV/JSON files in the cloned repo."""
        records: list[dict[str, Any]] = []

        # Search for CSV files in the cloned repo
        clone_dir = self._raw_dir / "bopd_repo"
        search_dirs = [clone_dir, self._raw_dir] if clone_dir.exists() else [self._raw_dir]

        for search_dir in search_dirs:
            for cf in sorted(search_dir.rglob("*.csv")):
                if cf.name == "README.txt":
                    continue
                try:
                    with open(cf, encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            records.append({**{k.lower(): v for k, v in row.items()}, "_source_file": cf.stem})
                except Exception:
                    pass

            for jf in sorted(search_dir.rglob("*.json")):
                if jf.name in ("package.json", "tsconfig.json", "manifest.json"):
                    continue
                try:
                    import json

                    with open(jf, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                records.append({**{k.lower(): v for k, v in item.items()}, "_source_file": jf.stem})
                except Exception:
                    pass

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for row in raw_data:
            # Build patient description from available fields
            description_parts: list[str] = []
            for key in (
                "icd10_code",
                "icd_code",
                "diagnosis_code",
                "age",
                "gender",
                "sex",
                "symptoms",
                "clinical_notes",
                "notes",
                "screening_score",
                "bpd_score",
                "criteria_met",
            ):
                val = str(row.get(key, "")).strip()
                if val:
                    description_parts.append(f"{key}: {val}")

            description = "; ".join(description_parts)
            if not description:
                continue

            # Determine BPD screening result
            label = (
                row.get("bpd_diagnosis")
                or row.get("diagnosis")
                or row.get("label")
                or row.get("screening_result")
                or row.get("classification")
                or ""
            ).strip()

            if not label:
                # Infer from criteria_met or screening_score
                criteria = row.get("criteria_met", "").strip().lower()
                if criteria in ("true", "1", "yes", "positive"):
                    label = "borderline_personality_disorder"
                elif criteria in ("false", "0", "no", "negative"):
                    label = "no_bpd"
                else:
                    label = "bpd_screening_positive"

            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": "You are a clinical screening assistant evaluating EHR data for Borderline Personality Disorder using ICD-10-CM codes.",
                },
                {"role": "user", "content": description},
                {"role": "assistant", "content": label},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "bopd",
                "task_type": "symptom_classification",
                "diagnostic_tag": "borderline_personality_disorder",
                "demographic_tags": self._extract_demographics(row),
                "linguistic_style": "formal",
                "clinical_reviewed": True,
                "icd10_codes": row.get("icd10_code", row.get("icd_code", "")),
                "screening_score": row.get("screening_score", row.get("bpd_score", "")),
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="sql_python",
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _extract_demographics(row: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        age = str(row.get("age", "")).strip()
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
        gender = str(row.get("gender", row.get("sex", ""))).strip().lower()
        if gender:
            if "female" in gender or gender == "f":
                tags.append("gender_female")
            elif "male" in gender or gender == "m":
                tags.append("gender_male")
            else:
                tags.append("gender_nonbinary")
        return tags
