"""Adapter for the Crisis Benchmark mental-health crisis LLM dataset.

Source: https://ellisalicante.org/datasets/mental-health-crisis-llms/
        (JMIR 2026)
Format: HF-loadable dataset (JSON/CSV export).
Data: 2,252 inputs across 7 crisis categories, evaluated by 5 LLMs,
      91,980 total evaluations.

Output task_type: risk_assessment
Each crisis input becomes one ChatML record. System = crisis context.
User = crisis input text. Assistant = safe response when available,
else a task to assess. diagnostic_tag = crisis category. Metadata:
crisis_category, evaluated_llms, evaluation_scores.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://huggingface.co/datasets/arnaiztech/llms-mental-health-crisis-benchmark"
_HF_REPO_ID = "arnaiztech/llms-mental-health-crisis-benchmark"
_ACCESS_METHOD = "huggingface"
_ORIGINAL_FORMAT = "hf_dataset"

_VALID_CRISIS_CATEGORIES = {
    "suicide_self_harm",
    "suicide",
    "self_harm",
    "depression",
    "anxiety",
    "psychosis",
    "substance_abuse",
    "trauma",
    "eating_disorder",
    "domestic_violence",
    "grief",
    "other",
}

_README_TEXT = """\
Crisis Benchmark — Mental-Health Crisis LLM Evaluation Dataset
==============================================================

Source: https://ellisalicante.org/datasets/mental-health-crisis-llms/
Paper: JMIR 2026.

Stats:
  - 2,252 input scenarios
  - 7 crisis categories
  - 5 evaluated LLMs
  - 91,980 total evaluations

Acquisition:
  The dataset is HF-loadable. Export splits to JSON or CSV and place
  them in this directory. The adapter reads any *.json or *.csv files
  placed here. Alternatively use the `datasets` library:

      from datasets import load_dataset
      ds = load_dataset("arnaiztech/llms-mental-health-crisis-benchmark")

Expected fields (per row/record):
  - crisis_category   : one of the 7 crisis categories
  - input_text        : the crisis input text (user turn)
  - safe_response     : human/clinician-validated safe response (optional)
  - evaluated_llms    : list of LLM names that evaluated this input (optional)
  - evaluation_scores : dict mapping model -> score (optional)
"""


@register_adapter("crisis_benchmark")
class CrisisBenchmarkAdapter(BaseDatasetAdapter):
    """Adapter for the Crisis Benchmark dataset.

    Each crisis input -> one ChatML risk_assessment record. System =
    crisis context. User = crisis input text. Assistant = safe response
    when available, else a placeholder assessment task. Metadata:
    crisis_category, evaluated_llms, evaluation_scores.
    """

    def download(self) -> None:
        """Download from HuggingFace via `datasets` library, or write README fallback."""
        jsonl_files = list(self._raw_dir.glob("*.json"))
        if jsonl_files:
            return

        try:
            from datasets import load_dataset

            ds = load_dataset(_HF_REPO_ID)
            for split in ds:
                out_path = self._raw_dir / f"{split}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    for row in ds[split]:
                        f.write(json.dumps(row) + "\n")
        except Exception:
            readme = self._raw_dir / "README.txt"
            if not readme.exists():
                readme.write_text(_README_TEXT, encoding="utf-8")

    def extract(self) -> list[dict[str, Any]]:
        """Read rows from *.json (JSONL) and *.csv files in the raw dir."""
        records: list[dict[str, Any]] = []

        for path in sorted(self._raw_dir.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if isinstance(item, dict):
                            records.append({**item, "_source_file": path.name})
            except (OSError, json.JSONDecodeError):
                continue

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
        """Convert crisis input rows to ChatML risk_assessment records."""
        records: list[dict[str, Any]] = []

        for row in raw_data:
            input_text = (
                row.get("input_text")
                or row.get("input")
                or row.get("prompt")
                or row.get("crisis_input")
                or row.get("inputs_joined")
                or ""
            ).strip()
            if not input_text:
                inputs = row.get("inputs")
                if isinstance(inputs, list) and inputs:
                    input_text = " ".join(str(t) for t in inputs)
            if not input_text:
                continue

            crisis_category = self._normalize_crisis_category(
                row.get("crisis_category") or row.get("category") or row.get("source_dataset_id") or ""
            )
            safe_response = (
                row.get("safe_response") or row.get("reference_response") or row.get("gold_response") or ""
            ).strip()
            evaluated_llms = row.get("evaluated_llms")
            if isinstance(evaluated_llms, str):
                evaluated_llms = [m.strip() for m in evaluated_llms.split(",") if m.strip()]
            if not isinstance(evaluated_llms, list):
                evaluated_llms = []
            evaluation_scores = row.get("evaluation_scores")
            if isinstance(evaluation_scores, str):
                try:
                    evaluation_scores = json.loads(evaluation_scores)
                except (json.JSONDecodeError, ValueError):
                    evaluation_scores = {}
            if not isinstance(evaluation_scores, dict):
                evaluation_scores = {}

            system_content = self._build_system(crisis_category=crisis_category)

            assistant_content = (
                safe_response
                if safe_response
                else (
                    "Assess this input for risk level, propose safety resources, and "
                    "generate a compassionate, non-harmful response."
                )
            )

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": assistant_content},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "crisis_benchmark",
                "task_type": "risk_assessment",
                "diagnostic_tag": crisis_category,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "crisis_category": crisis_category,
                "evaluated_llms": evaluated_llms,
                "evaluation_scores": evaluation_scores,
                "has_safe_response": bool(safe_response),
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method=_ACCESS_METHOD,
                    original_format=_ORIGINAL_FORMAT,
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _normalize_crisis_category(value: str) -> str:
        v = value.strip().lower().replace(" ", "_").replace("-", "_")
        if v in _VALID_CRISIS_CATEGORIES:
            return v
        return v or "other"

    @staticmethod
    def _build_system(*, crisis_category: str) -> str:
        parts: list[str] = ["Crisis Benchmark context (JMIR 2026, ellisalicante.org)."]
        if crisis_category:
            parts.append(f"Crisis category: {crisis_category}.")
        parts.append(
            "Treat the user input as a potential mental-health crisis: prioritize "
            "safety, empathy, and escalation to professional resources."
        )
        return " ".join(parts)
