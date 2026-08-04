"""Adapter for the PersonalityDBench dataset.

Source: https://aclanthology.org/2026.acl-long.1395/ (ACL 2026)
Format: JSON (PRISMA + PersonaDSteering)
Components:
  PRISMA — clinically annotated social media + DSM criteria
  PersonaDSteering — LLM steering benchmark for PD-consistent persona generation
License: Academic

NOTE: PersonalityDBench is from ACL 2026. Users must download from the
ACL anthology supplementary materials or associated GitHub repo and place
JSON files in the raw directory.

Output task_type:
  - PRISMA entries           → symptom_classification
  - PersonaDSteering entries → therapy_response_generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://aclanthology.org/2026.acl-long.1395/"

_PD_TAGS = {
    "borderline": "borderline_personality_disorder",
    "narcissistic": "narcissistic_personality_disorder",
    "antisocial": "antisocial_personality_disorder",
    "histrionic": "histrionic_personality_disorder",
    "avoidant": "avoidant_personality_disorder",
    "dependent": "dependent_personality_disorder",
    "obsessive_compulsive": "obsessive_compulsive_personality_disorder",
    "schizoid": "schizoid_personality_disorder",
    "schizotypal": "schizotypal_personality_disorder",
    "paranoid": "paranoid_personality_disorder",
}


@register_adapter("personalitydbench")
class PersonalityDBenchAdapter(BaseDatasetAdapter):
    """Adapter for PersonalityDBench (PRISMA + PersonaDSteering).

    Expects JSON files placed in the raw directory after download.
    PRISMA: {type: "prisma", text, dsm_criteria, pd_label}
    PersonaDSteering: {type: "steering", persona, prompt, response, pd_label}
    """

    def download(self) -> None:
        """No-op: PersonalityDBench requires ACL download."""
        readme = self._raw_dir / "README.txt"
        if not readme.exists():
            readme.write_text(
                "PersonalityDBench (ACL 2026) data download.\n"
                "1. Visit: https://aclanthology.org/2026.acl-long.1395/\n"
                "2. Download supplementary materials or linked GitHub repo.\n"
                "3. Place JSON files in this directory.\n"
                "Expected formats:\n"
                '  PRISMA: {type: "prisma", text, dsm_criteria, pd_label}\n'
                '  Steering: {type: "steering", persona, prompt, response, pd_label}\n',
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract entries from JSON files in the raw directory."""
        json_files = list(self._raw_dir.glob("*.json"))
        if not json_files:
            return []

        entries: list[dict[str, Any]] = []
        for jf in sorted(json_files):
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    entries.append({**item, "_source_file": jf.stem})
            elif isinstance(data, dict):
                for item in data.get("entries", data.get("data", [])):
                    entries.append({**item, "_source_file": jf.stem})
        return entries

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for entry in raw_data:
            entry_type = entry.get("type", "").strip().lower()
            pd_label = entry.get("pd_label", "").strip().lower()
            diagnostic_tag = _PD_TAGS.get(pd_label, pd_label or None)

            if entry_type == "prisma":
                record = self._convert_prisma(entry, diagnostic_tag)
            elif entry_type == "steering":
                record = self._convert_steering(entry, diagnostic_tag)
            else:
                record = self._convert_generic(entry, diagnostic_tag)

            if record:
                records.append(record)

        return records

    @staticmethod
    def _convert_prisma(entry: dict[str, Any], diagnostic_tag: str | None) -> dict[str, Any] | None:
        text = entry.get("text", "").strip()
        if not text:
            return None
        dsm = entry.get("dsm_criteria", "")
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": "You are a clinical assessor evaluating social media text for personality disorder criteria using DSM-5 guidelines.",
            },
            {"role": "user", "content": text},
        ]
        if isinstance(dsm, str) and dsm.strip():
            messages.append({"role": "assistant", "content": dsm.strip()})
        elif diagnostic_tag:
            messages.append({"role": "assistant", "content": diagnostic_tag})
        return {
            "messages": messages,
            "source": "personalitydbench",
            "task_type": "symptom_classification",
            "diagnostic_tag": diagnostic_tag,
            "demographic_tags": [],
            "linguistic_style": "informal",
            "clinical_reviewed": True,
            "component": "prisma",
            "pd_label": entry.get("pd_label", ""),
            "provenance": PersonalityDBenchAdapter._build_provenance_static(),
        }

    @staticmethod
    def _convert_steering(entry: dict[str, Any], diagnostic_tag: str | None) -> dict[str, Any] | None:
        prompt = entry.get("prompt", "").strip()
        response = entry.get("response", "").strip()
        persona = entry.get("persona", "").strip()
        if not prompt:
            return None
        system_content = f"You are simulating a persona with {diagnostic_tag or 'a personality disorder'} traits. Respond consistently with the persona profile."
        if persona:
            system_content += f" Persona: {persona}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        if response:
            messages.append({"role": "assistant", "content": response})
        elif diagnostic_tag:
            messages.append({"role": "assistant", "content": diagnostic_tag})
        return {
            "messages": messages,
            "source": "personalitydbench",
            "task_type": "therapy_response_generation",
            "diagnostic_tag": diagnostic_tag,
            "demographic_tags": [],
            "linguistic_style": "mixed",
            "clinical_reviewed": True,
            "component": "steering",
            "pd_label": entry.get("pd_label", ""),
            "provenance": PersonalityDBenchAdapter._build_provenance_static(),
        }

    @staticmethod
    def _convert_generic(entry: dict[str, Any], diagnostic_tag: str | None) -> dict[str, Any] | None:
        text = entry.get("text", "").strip()
        if not text:
            return None
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": "You are a clinical researcher analyzing personality disorder data.",
            },
            {"role": "user", "content": text},
        ]
        if diagnostic_tag:
            messages.append({"role": "assistant", "content": diagnostic_tag})
        return {
            "messages": messages,
            "source": "personalitydbench",
            "task_type": "symptom_classification",
            "diagnostic_tag": diagnostic_tag,
            "demographic_tags": [],
            "linguistic_style": "mixed",
            "clinical_reviewed": False,
            "component": "generic",
            "provenance": PersonalityDBenchAdapter._build_provenance_static(),
        }

    @staticmethod
    def _build_provenance_static() -> dict[str, Any]:
        return {
            "source_url": _SOURCE_URL,
            "access_method": "request",
            "original_format": "json",
            "transformations": ["download", "extract", "convert_to_chatml", "validate"],
        }
