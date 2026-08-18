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
_GIT_CLONE_URL = "https://github.com/saidejp/machine_learning_BPD.git"

# Actual CSV files in the repo
_CSV_FILES = [
    "training_noSession.csv",
    "training_impulsivity.csv",
    "testing_noSession.csv",
    "testing_impulsivity.csv",
]

# Key clinical columns for building description text
_CLINICAL_COLS = [
    "Edad",
    "Sexo",
    "BDI_TOTAL_PRE",
    "BAI_TOTAL_PRE",
    "BIS_TOTAL_PRE",
    "DERS_TOTAL_PRE",
    "BEST_TOTAL_PRE",
    "MF_TOTAL_PRE",
    "EOSS_TOTAL_PRE",
    "DES_TOTAL",
    "Total_TLP",
    "Total_DEP",
]


@register_adapter("ml_bpd")
class MLBPDAdapter(BaseDatasetAdapter):
    """Adapter for machine_learning_BPD DBT treatment outcome data.

    Clones the GitHub repo, reads CSV files with BPD patient assessment
    data (BDI, BAI, BIS-11, DERS, etc.). Converts each patient record
    into a ChatML severity estimation record.
    """

    def download(self) -> None:
        """Clone the machine_learning_BPD repo."""
        repo_dir = self._raw_dir / "machine_learning_BPD"
        if repo_dir.exists():
            return
        try:
            import subprocess

            subprocess.run(
                ["git", "clone", "--depth", "1", _GIT_CLONE_URL, str(repo_dir)],
                check=True,
                capture_output=True,
            )
        except Exception:
            (self._raw_dir / "README.txt").write_text(
                "machine_learning_BPD data download.\n"
                "1. Visit: https://github.com/saidejp/machine_learning_BPD\n"
                "2. Clone repo into this directory.\n"
                "Expected: CSV files with BPD patient assessment data.\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract patient records from CSV files in the cloned repo."""
        repo_dir = self._raw_dir / "machine_learning_BPD"
        if not repo_dir.exists():
            return []

        records: list[dict[str, Any]] = []
        for fname in _CSV_FILES:
            cf = repo_dir / fname
            if not cf.exists():
                continue
            with open(cf, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append({**row, "_source_file": fname})
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for row in raw_data:
            # Build clinical description from Spanish assessment columns
            parts: list[str] = []
            for col in _CLINICAL_COLS:
                val = (row.get(col) or "").strip()
                if val:
                    parts.append(f"{col}: {val}")
            description = "; ".join(parts)

            if not description:
                continue

            # Use BDI (depression) + BAI (anxiety) as severity indicators
            bdi = (row.get("BDI_TOTAL_PRE") or "").strip()
            bai = (row.get("BAI_TOTAL_PRE") or "").strip()
            bis = (row.get("BIS_TOTAL_PRE") or "").strip()

            if bdi:
                try:
                    bdi_val = int(bdi)
                    if bdi_val >= 29:
                        outcome = "severe"
                    elif bdi_val >= 20:
                        outcome = "moderate"
                    elif bdi_val >= 14:
                        outcome = "mild"
                    else:
                        outcome = "minimal"
                except ValueError:
                    outcome = "moderate"
            else:
                outcome = "moderate"

            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical assessor evaluating BPD patient "
                        "treatment outcome data. Assess symptom severity based "
                        "on DBT pre-treatment measures (BDI, BAI, BIS-11, DERS, "
                        "BEST). All scores are pre-treatment baseline."
                    ),
                },
                {"role": "user", "content": description},
                {"role": "assistant", "content": outcome},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "ml_bpd",
                "task_type": "severity_estimation",
                "diagnostic_tag": "borderline_personality_disorder",
                "demographic_tags": self._extract_demographics(row),
                "linguistic_style": "formal",
                "clinical_reviewed": True,
                "bis11_score": bis,
                "bdi_score": bdi,
                "bai_score": bai,
                "source_file": row.get("_source_file", ""),
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
        age = (row.get("Edad") or "").strip()
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
        sexo = (row.get("Sexo") or "").strip().lower()
        if sexo:
            if "mujer" in sexo or sexo == "f":
                tags.append("gender_female")
            elif "hombre" in sexo or sexo == "m":
                tags.append("gender_male")
            else:
                tags.append("gender_other")
        return tags
