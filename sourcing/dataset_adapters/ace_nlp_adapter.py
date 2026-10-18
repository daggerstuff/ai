"""Adapter for the ACE-NLP (Adverse Childhood Experiences NLP) dataset.

Source: https://github.com/knowlab/ace-nlp
Format: JSON (ACE concept annotations from Reddit Mental Health corpus)
Size: 780+ documents, 322 ACE concepts
License: MIT
Paper: ACL 2024

The dataset contains mention-level annotations of ACE concepts in Reddit
mental health posts, with concept CUIs, start/end offsets, and covered text.

Output task_type: symptom_classification
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/knowlab/ace-nlp"
_GIT_CLONE_URL = "https://github.com/knowlab/ace-nlp.git"
_ZIP_PATH = "Reddit-MH/aces_reddits_mental_health_322_ACEs.zip"


@register_adapter("ace_nlp")
class ACE_NLPAdapter(BaseDatasetAdapter):
    """Adapter for ACE-NLP dataset.

    Clones the GitHub repo, extracts the Reddit-MH ACE annotations zip,
    and converts ACE concept mentions into ChatML symptom classification records.
    Each record presents a Reddit post with identified ACE concepts.
    """

    def download(self) -> None:
        """Clone the ACE-NLP repo and extract the annotations zip."""
        repo_dir = self._raw_dir / "ace-nlp"
        if not repo_dir.exists():
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", _GIT_CLONE_URL, str(repo_dir)],
                    check=True,
                    capture_output=True,
                )
            except Exception:
                (self._raw_dir / "README.txt").write_text(
                    "ACE-NLP data download.\n"
                    "1. Visit: https://github.com/knowlab/ace-nlp\n"
                    "2. Clone repo into this directory.\n"
                    "Expected: Reddit-MH/aces_reddits_mental_health_322_ACEs.zip\n",
                    encoding="utf-8",
                )
                return

        # Extract zip if not already extracted
        zip_file = repo_dir / _ZIP_PATH
        extract_dir = repo_dir / "Reddit-MH" / "extracted"
        if zip_file.exists() and not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_file) as zf:
                zf.extractall(extract_dir)

    def extract(self) -> list[dict[str, Any]]:
        """Extract ACE annotation data from JSON files."""
        repo_dir = self._raw_dir / "ace-nlp"
        extract_dir = repo_dir / "Reddit-MH" / "extracted"
        if not extract_dir.exists():
            return []

        records: list[dict[str, Any]] = []
        for json_file in sorted(extract_dir.rglob("*.json")):
            # Skip macOS metadata files
            if "__MACOSX" in str(json_file):
                continue
            try:
                with open(json_file, encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Handle both single-doc and multi-doc formats
            if isinstance(data, list):
                for doc in data:
                    if isinstance(doc, dict):
                        doc["_source_file"] = json_file.name
                        records.append(doc)
            elif isinstance(data, dict):
                data["_source_file"] = json_file.name
                records.append(data)

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert ACE annotations to ChatML symptom classification records."""
        records: list[dict[str, Any]] = []

        for doc in raw_data:
            # Extract text and annotations
            text = doc.get("text") or doc.get("content") or doc.get("post") or ""
            mentions = doc.get("mentions") or doc.get("annotations") or doc.get("aces") or []

            if not text and not mentions:
                continue

            # Build ACE concept list
            ace_concepts: list[str] = []
            for m in mentions:
                if isinstance(m, dict):
                    concept = m.get("concept") or m.get("cui") or m.get("concept_id") or ""
                    covered_text = m.get("covered_text") or m.get("text") or m.get("value") or ""
                    if concept and covered_text:
                        ace_concepts.append(f'{concept}: "{covered_text}"')
                    elif concept:
                        ace_concepts.append(concept)
                    elif covered_text:
                        ace_concepts.append(covered_text)

            if not text and not ace_concepts:
                continue

            # Build ChatML
            user_content = text if text else "Analyze the following Reddit post for ACE concepts."
            assistant_content = (
                "Identified ACE concepts:\n" + "\n".join(f"- {c}" for c in ace_concepts)
                if ace_concepts
                else "No ACE concepts identified."
            )

            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": "ACE-NLP: Adverse Childhood Experiences concept detection from Reddit mental health posts.",
                },
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "ace_nlp",
                "task_type": "symptom_classification",
                "diagnostic_tag": "ace_detection",
                "demographic_tags": [],
                "linguistic_style": "informal",
                "clinical_reviewed": False,
                "ace_concept_count": len(ace_concepts),
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="json",
                ),
            }
            records.append(record)

        return records
