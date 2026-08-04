"""Adapter for DSM-in-Vector-Space — DSM-5 diagnostic taxonomy data.

Source: S3 (whitebat:training/pixelated-empathy/output/dsm_vector_space/)
Format: JSON files per DSM-5 diagnostic category. Each file is a list of diagnoses.
Each diagnosis: {diagnosis_id, diagnosis_name, diagnostic_code, chapter_category,
                 threshold_count, duration_rule, symptoms}
Size: 21+ files, 167 total diagnoses in diagnoses.json alone
Language: English

Output task_type: risk_assessment
Each record: DSM-5 diagnosis as system knowledge for clinical reasoning.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_S3_PREFIX = "whitebat:training/pixelated-empathy/output/dsm_vector_space/raw/dsm-in-vector-space/data/"
_SOURCE_URL = "https://github.com/savelee/dsm-in-vector-space"

_SYSTEM_PROMPT = (
    "You are a clinical diagnostic assistant with expert knowledge of DSM-5 criteria. "
    "Use the provided diagnostic category information to assess symptoms, "
    "determine diagnoses, and explain clinical reasoning."
)


@register_adapter("dsm_vector_space")
class DsmVectorSpaceAdapter(BaseDatasetAdapter):
    """Adapter for DSM-5 diagnostic taxonomy from S3."""

    def download(self) -> None:
        """Download DSM taxonomy JSONs from S3 if not already present."""
        data_dir = self._raw_dir / "dsm-in-vector-space" / "data"
        if data_dir.exists() and any(data_dir.glob("*.json")):
            return

        try:
            subprocess.run(
                ["rclone", "copy", _S3_PREFIX, str(data_dir)],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError) as e:
            readme = self._raw_dir / "README.txt"
            readme.write_text(f"Download failed: {e}\n", encoding="utf-8")

    def extract(self) -> list[dict[str, Any]]:
        """Read DSM taxonomy JSON files. Each file may be a list of diagnoses."""
        records: list[dict[str, Any]] = []
        data_dir = self._raw_dir / "dsm-in-vector-space" / "data"
        if not data_dir.exists():
            return records

        for json_path in sorted(data_dir.glob("*.json")):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            records.append({"_category_file": json_path.stem, **item})
                elif isinstance(data, dict):
                    records.append({"_category_file": json_path.stem, **data})
            except (json.JSONDecodeError, OSError):
                continue
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert DSM diagnosis entries to ChatML as diagnostic knowledge Q&A."""
        records: list[dict[str, Any]] = []

        for entry in raw_data:
            dx_name = entry.get("diagnosis_name") or entry.get("name") or ""
            dx_code = entry.get("diagnostic_code") or ""
            chapter = entry.get("chapter_category") or entry.get("_category_file", "").replace("_", " ").title()
            symptoms = entry.get("symptoms") or ""
            threshold = entry.get("threshold_count") or ""
            duration = entry.get("duration_rule") or ""

            if not dx_name and not chapter:
                continue

            criteria_parts: list[str] = []
            if dx_name:
                criteria_parts.append(f"Diagnosis: {dx_name}")
            if dx_code:
                criteria_parts.append(f"ICD-10 Code: {dx_code}")
            if chapter:
                criteria_parts.append(f"Category: {chapter}")
            if threshold:
                criteria_parts.append(f"Threshold: {threshold}")
            if duration:
                criteria_parts.append(f"Duration: {duration}")
            if symptoms:
                if isinstance(symptoms, list):
                    criteria_parts.append(f"Symptoms: {'; '.join(str(s) for s in symptoms[:10])}")
                else:
                    criteria_parts.append(f"Symptoms: {symptoms}")

            if len(criteria_parts) < 2:
                continue

            criteria_text = "\n".join(criteria_parts)
            system_content = f"{_SYSTEM_PROMPT}\n\n{criteria_text}"

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"What are the diagnostic criteria for {dx_name or chapter}?"},
                {"role": "assistant", "content": criteria_text},
            ]

            records.append(
                {
                    "messages": messages,
                    "source": "dsm_vector_space",
                    "task_type": "risk_assessment",
                    "diagnostic_tag": dx_name or chapter,
                    "demographic_tags": [],
                    "linguistic_style": "formal",
                    "clinical_reviewed": False,
                    "provenance": self._build_provenance(
                        source_url=_SOURCE_URL,
                        access_method="s3",
                        original_format="json",
                    ),
                }
            )

        return records
