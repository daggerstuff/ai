"""Adapter for the MIT ai-psychosis adversarial safety dataset.

Source: https://github.com/mitmedialab/ai-psychosis
Format: CSV (harmful-responses.csv + taxonomy.csv)
Size: 6 conditions, 18 real harm cases -> 2,160 scenarios, 157K turns
       51,693 harmful responses, 15 failure patterns in 4 categories
Clinical staging: Stage 0-N models

Output task_type: adversarial_safety
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/mitmedialab/ai-psychosis"
_HARMFUL_URL = "https://raw.githubusercontent.com/mitmedialab/ai-psychosis/main/data/harmful-responses.csv"
_TAXONOMY_URL = "https://raw.githubusercontent.com/mitmedialab/ai-psychosis/main/data/taxonomy.csv"

# Expected CSV column names in harmful-responses.csv
_HARMFUL_COLS = (
    "condition",
    "scenario",
    "stage",
    "patient_input",
    "llm_response",
    "failure_pattern",
    "failure_category",
    "model",
)


@register_adapter("mit_psychosis")
class MITPsychosisAdapter(BaseDatasetAdapter):
    """Adapter for MIT ai-psychosis dataset.

    Converts harmful LLM responses to ChatML with:
    - System prompt noting the clinical condition and scenario
    - User turn = patient input
    - Assistant turn = harmful LLM response
    - Metadata: failure_pattern, failure_category, stage, condition, model
    - task_type = adversarial_safety
    """

    def download(self) -> None:
        """Download harmful-responses.csv and taxonomy.csv if not present."""
        harmful_file = self._raw_dir / "harmful-responses.csv"
        taxonomy_file = self._raw_dir / "taxonomy.csv"

        if not harmful_file.exists():
            try:
                urllib.request.urlretrieve(_HARMFUL_URL, harmful_file)
            except Exception:
                (self._raw_dir / "README.txt").write_text(
                    "MIT ai-psychosis data download.\n"
                    "1. Visit: https://github.com/mitmedialab/ai-psychosis\n"
                    "2. Download data/harmful-responses.csv and data/taxonomy.csv\n"
                    "3. Place them in this directory.\n",
                    encoding="utf-8",
                )

        if not taxonomy_file.exists():
            try:
                urllib.request.urlretrieve(_TAXONOMY_URL, taxonomy_file)
            except Exception:
                pass

    def extract(self) -> list[dict[str, Any]]:
        """Extract harmful response rows from CSV into intermediate dicts."""
        records: list[dict[str, Any]] = []

        harmful_file = self._raw_dir / "harmful-responses.csv"
        if not harmful_file.exists():
            return records

        with open(harmful_file, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({**row, "_source_file": "harmful-responses.csv"})

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert harmful response rows to ChatML records."""
        records: list[dict[str, Any]] = []

        for row in raw_data:
            patient_input = (row.get("patient_input") or row.get("patient_turn") or "").strip()
            llm_response = (row.get("llm_response") or row.get("response") or "").strip()
            if not patient_input or not llm_response:
                continue

            condition = (row.get("condition") or "").strip()
            scenario = (row.get("scenario") or "").strip()
            stage = (row.get("stage") or "").strip()
            failure_pattern = (row.get("failure_pattern") or "").strip()
            failure_category = (row.get("failure_category") or "").strip()
            model = (row.get("model") or "").strip()

            system_parts: list[str] = []
            if condition:
                system_parts.append(f"Condition: {condition}")
            if scenario:
                system_parts.append(f"Scenario: {scenario}")
            if stage:
                system_parts.append(f"Stage: {stage}")
            system_parts.append(
                "Adversarial safety sample: the assistant response below is a known harmful LLM output."
            )
            system_content = ". ".join(system_parts)

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": patient_input},
                {"role": "assistant", "content": llm_response},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "mit_psychosis",
                "task_type": "adversarial_safety",
                "diagnostic_tag": condition or None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "condition": condition,
                "scenario": scenario,
                "stage": stage,
                "failure_pattern": failure_pattern,
                "failure_category": failure_category,
                "model": model,
                "is_harmful_sample": True,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records
