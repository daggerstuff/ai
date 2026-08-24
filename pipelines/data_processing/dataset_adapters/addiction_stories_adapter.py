"""Adapter for the Addiction Stories dataset.

Source: https://huggingface.co/datasets/KerenHaruvi/Addiction_Stories
Format: HuggingFace dataset (291 train + 200 test = 491 records)
Columns: example_id, text (personal addiction stories), label (trauma category),
         annotator labels (3 annotators, 2 rounds)
License: CC BY

Output task_type: therapy_response_generation
"""

from __future__ import annotations

import json
from typing import Any

from ai.pipelines.data_processing.dataset_adapters.adapter_factory import register_adapter
from ai.pipelines.data_processing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://huggingface.co/datasets/KerenHaruvi/Addiction_Stories"
_HF_DATASET_NAME = "KerenHaruvi/Addiction_Stories"


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(x) for x in value).strip()
    return str(value).strip()


@register_adapter("addiction_stories")
class AddictionStoriesAdapter(BaseDatasetAdapter):
    """Adapter for Addiction Stories dataset.

    Downloads via HuggingFace datasets API. Each personal addiction story
    is converted to a ChatML therapy_response_generation record with the
    story as user input and the annotated label as the assistant response.
    """

    def download(self) -> None:
        """Download the Addiction Stories dataset via HuggingFace API."""
        cache_file = self._raw_dir / "addiction_stories.json"
        if cache_file.exists():
            return

        try:
            from datasets import load_dataset

            ds = load_dataset(_HF_DATASET_NAME)
            all_records: list[dict[str, Any]] = []
            for split_name in ds:
                for row in ds[split_name]:
                    row_dict = dict(row)
                    row_dict["_split"] = split_name
                    all_records.append(row_dict)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False)
        except Exception as e:
            (self._raw_dir / "README.txt").write_text(
                f"Addiction Stories download failed: {e}\n"
                "1. Visit: https://huggingface.co/datasets/KerenHaruvi/Addiction_Stories\n"
                "2. Ensure `datasets` library is installed: pip install datasets\n"
                "3. Run: python -c \"from datasets import load_dataset; ds = load_dataset('KerenHaruvi/Addiction_Stories')\"\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract addiction stories from cached JSON."""
        cache_file = self._raw_dir / "addiction_stories.json"
        if not cache_file.exists():
            return []

        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert addiction stories to ChatML therapy_response_generation records."""
        records: list[dict[str, Any]] = []

        for row in raw_data:
            text = _to_str(row.get("text"))
            label = _to_str(row.get("label"))
            split = _to_str(row.get("_split"))

            if not text:
                continue

            # Collect annotator labels for context
            annotator_labels: list[str] = []
            for i in range(1, 4):
                for t in (1, 2):
                    key = f"annotator{i}_t{t}_label"
                    val = row.get(key)
                    if val is not None:
                        annotator_labels.append(f"Annotator{i}_T{t}: {_to_str(val)}")

            system_content = (
                "Addiction Stories: Personal addiction narratives with trauma classification. "
                "Respond with the categorized trauma type and therapeutic framing."
            )

            assistant_parts: list[str] = []
            if label:
                assistant_parts.append(f"Trauma category: {label}")
            if annotator_labels:
                assistant_parts.append("Annotations:\n" + "\n".join(f"  - {a}" for a in annotator_labels))

            assistant_content = "\n\n".join(assistant_parts) if assistant_parts else "Unable to classify."

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": text},
                {"role": "assistant", "content": assistant_content},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "addiction_stories",
                "task_type": "therapy_response_generation",
                "diagnostic_tag": "addiction",
                "demographic_tags": [],
                "linguistic_style": "informal",
                "clinical_reviewed": True,
                "trauma_label": label,
                "split": split,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="huggingface",
                    original_format="hf_dataset",
                ),
            }
            records.append(record)

        return records
