"""Adapter for the MHSafeEval mental-health safety evaluation dataset.

Source: https://huggingface.co/datasets/Suhyunlee/MHSafeEval
Format: Hugging Face dataset (JSON/CSV export).
Data: 7 harm categories x 4 counselor roles
       (Perpetrator, Instigator, Facilitator, Enabler)
       across depression / delusion / psychosis.
       QD grid, kappa = 0.327-0.387, recall 91.5-95.2%.

Output task_type: adversarial_safety
Each harm scenario becomes one ChatML record. System describes the
counselor role and harm category. User = patient scenario. Assistant =
harmful response (for adversarial training). diagnostic_tag = harm
category (depression, delusion, psychosis). Metadata: counselor_role,
harm_category.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://huggingface.co/datasets/Suhyunlee/MHSafeEval"
_ACCESS_METHOD = "huggingface"
_ORIGINAL_FORMAT = "hf_dataset"

_VALID_COUNSELOR_ROLES = {"perpetrator", "instigator", "facilitator", "enabler"}
_VALID_HARM_CATEGORIES = {"depression", "delusion", "psychosis"}

_README_TEXT = """\
MHSafeEval — Mental-Health Safety Evaluation Dataset
====================================================

Source: https://huggingface.co/datasets/Suhyunlee/MHSafeEval

Stats:
  - 7 harm categories x 4 counselor roles
    (Perpetrator, Instigator, Facilitator, Enabler)
  - Across depression / delusion / psychosis
  - QD grid, kappa = 0.327-0.387, recall 91.5-95.2%

Acquisition:
  Download via the HuggingFace `datasets` library:

      from datasets import load_dataset
      ds = load_dataset("Suhyunlee/MHSafeEval")
      # Export splits to JSON or CSV and place them in this directory.

  The adapter reads any *.json or *.csv files placed here.

Expected fields (per row/record):
  - counselor_role   : one of Perpetrator, Instigator, Facilitator, Enabler
  - harm_category   : one of depression, delusion, psychosis
  - patient_input   : the patient scenario / input text
  - model_response  : the harmful response produced by the counselor role
  - harm_category_detail (optional)
"""


@register_adapter("mhsafeeval")
class MHSafeEvalAdapter(BaseDatasetAdapter):
    """Adapter for MHSafeEval.

    Harm scenarios -> ChatML adversarial_safety records. System prompt
    notes the counselor role + harm category. User = patient scenario,
    Assistant = harmful response. diagnostic_tag = harm category.
    Metadata: counselor_role, harm_category.
    """

    def download(self) -> None:
        """Write a README with the HF dataset link + download instructions."""
        readme = self._raw_dir / "README.txt"
        if not readme.exists():
            readme.write_text(_README_TEXT, encoding="utf-8")

    def extract(self) -> list[dict[str, Any]]:
        """Read harm-scenario rows from *.json and *.csv files in the raw dir."""
        records: list[dict[str, Any]] = []

        # JSON files
        for path in sorted(self._raw_dir.glob("*.json")):
            if path.name == "README.txt":
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    records.append({**item, "_source_file": path.name})

        # CSV files
        for path in sorted(self._raw_dir.glob("*.csv")):
            try:
                with open(path, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        records.append({**row, "_source_file": path.name})
            except OSError:
                continue

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert harm-scenario rows to ChatML adversarial_safety records."""
        records: list[dict[str, Any]] = []

        for row in raw_data:
            patient_input = (
                row.get("patient_input") or row.get("patient_scenario") or row.get("scenario") or ""
            ).strip()
            model_response = (
                row.get("model_response") or row.get("counselor_response") or row.get("response") or ""
            ).strip()
            if not patient_input or not model_response:
                continue

            counselor_role = self._normalize_counselor_role(row.get("counselor_role") or row.get("role") or "")
            harm_category = self._normalize_harm_category(
                row.get("harm_category") or row.get("harm_type") or row.get("category") or ""
            )

            system_content = self._build_system(
                counselor_role=counselor_role,
                harm_category=harm_category,
            )

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": patient_input},
                {"role": "assistant", "content": model_response},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "mhsafeeval",
                "task_type": "adversarial_safety",
                "diagnostic_tag": harm_category,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "counselor_role": counselor_role,
                "harm_category": harm_category,
                "is_harmful_sample": True,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method=_ACCESS_METHOD,
                    original_format=_ORIGINAL_FORMAT,
                ),
            }
            # Preserve optional detail if present.
            detail = row.get("harm_category_detail")
            if isinstance(detail, str) and detail.strip():
                record["harm_category_detail"] = detail.strip()
            records.append(record)

        return records

    @staticmethod
    def _normalize_counselor_role(value: str) -> str:
        v = value.strip().lower()
        if v in _VALID_COUNSELOR_ROLES:
            return v
        # fall back to raw lowercased string (still useful for audit)
        return v or "unknown"

    @staticmethod
    def _normalize_harm_category(value: str) -> str:
        v = value.strip().lower()
        if v in _VALID_HARM_CATEGORIES:
            return v
        return v or None  # type: ignore[return-value]

    @staticmethod
    def _build_system(*, counselor_role: str, harm_category: str | None) -> str:
        parts: list[str] = ["MHSafeEval adversarial safety sample."]
        if counselor_role and counselor_role != "unknown":
            label = counselor_role.capitalize()
            parts.append(f"Counselor role: {label} (harm-enabling stance).")
        else:
            parts.append("Counselor role: unknown.")
        if harm_category:
            parts.append(f"Harm category: {harm_category}.")
        parts.append("The assistant turn below is a known harmful response; it must NOT be reproduced as safe advice.")
        return " ".join(parts)
