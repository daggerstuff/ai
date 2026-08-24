"""Adapter for the VERA-MH dataset.

Source: https://github.com/SpringCare/VERA-MH
Format: TSV (clinical personas + evaluation rubric)
Size: ~30 clinical personas with suicide risk profiles
Key concept: Adversarial safety benchmark for mental health chatbot
             responses to suicidal ideation.

Data files in repo:
  data/personas.tsv  — clinical persona definitions
  data/rubric.tsv    — evaluation rubric for chatbot response quality

Output task_type: adversarial_safety
"""

from __future__ import annotations

import csv
import subprocess
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/SpringCare/VERA-MH"
_GIT_CLONE_URL = "https://github.com/SpringCare/VERA-MH.git"
_SHORT_TEXT_THRESHOLD = 280


@register_adapter("vera_mh")
class VERAMHAdapter(BaseDatasetAdapter):
    """Adapter for VERA-MH adversarial safety benchmark.

    Clones the GitHub repo, reads clinical personas from data/personas.tsv.
    Each persona defines a patient profile with suicide risk level, diagnoses,
    communication style, and a seed phrase for initiating conversation.
    The rubric in data/rubric.tsv provides evaluation criteria for assessing
    chatbot responses to these personas.
    """

    def download(self) -> None:
        """Clone the VERA-MH repo if not already present."""
        repo_dir = self._raw_dir / "VERA-MH"
        if repo_dir.exists():
            return
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", _GIT_CLONE_URL, str(repo_dir)],
                check=True,
                capture_output=True,
            )
        except Exception:
            (self._raw_dir / "README.txt").write_text(
                "VERA-MH data download.\n"
                "1. Visit: https://github.com/SpringCare/VERA-MH\n"
                "2. Clone repo into this directory.\n"
                "Expected: data/personas.tsv, data/rubric.tsv\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract persona records from personas.tsv in cloned repo."""
        repo_dir = self._raw_dir / "VERA-MH"
        personas_file = repo_dir / "data" / "personas.tsv"
        if not personas_file.exists():
            return []

        records: list[dict[str, Any]] = []
        with open(personas_file, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row["_source_file"] = "data/personas.tsv"
                records.append(row)
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert persona records to ChatML adversarial safety records."""
        records: list[dict[str, Any]] = []

        for persona in raw_data:
            name = (persona.get("Name") or "").strip()
            age = (persona.get("Age") or "").strip()
            gender = (persona.get("Gender") or "").strip()
            short_risk = (persona.get("Short Current Suicide Risk Level") or "").strip()
            diagnoses = (persona.get("Diagnoses and Symptoms") or "").strip()
            comm_style = (persona.get("Primary Communication Style") or "").strip()
            reaction = (persona.get("Reaction to Chatbot Responses") or "").strip()
            treatment = (persona.get("Treatment Engagement / Access") or "").strip()
            stigma = (persona.get("Mental Health Stigma") or "").strip()
            background = (persona.get("Background") or "").strip()
            triggers = (persona.get("Recent Triggers and Stressors") or "").strip()
            seed_phrase = (persona.get("Final Seed Phrase") or "").strip()

            if not seed_phrase:
                continue

            first_diagnosis = diagnoses.split("|")[0].strip() if diagnoses else ""

            system_parts: list[str] = [
                "VERA-MH adversarial safety evaluation.",
                f"Persona: {name}." if name else "",
                f"Suicide risk: {short_risk}." if short_risk else "",
                f"Diagnoses: {diagnoses}." if diagnoses else "",
                f"Communication style: {comm_style}." if comm_style else "",
                "Assistant responses may be unsafe; do not emulate clinically.",
            ]
            system_content = " ".join(p for p in system_parts if p)

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": seed_phrase},
                {
                    "role": "assistant",
                    "content": (
                        "[EVALUATION RUBRIC APPLIES] Response quality assessed "
                        "against VERA-MH rubric for suicide risk detection, "
                        "confirmation, and care guidance."
                    ),
                },
            ]

            demographic_tags: list[str] = []
            if "18-29" in age:
                demographic_tags.append("age_18_29")
            elif "30-44" in age:
                demographic_tags.append("age_30_44")
            elif "45-59" in age:
                demographic_tags.append("age_45_59")
            elif "60+" in age:
                demographic_tags.append("age_60_plus")

            gender_lower = gender.lower()
            if "woman" in gender_lower:
                demographic_tags.append("gender_female")
            elif "man" in gender_lower:
                demographic_tags.append("gender_male")
            elif "transgender" in gender_lower:
                demographic_tags.append("gender_transgender")

            record: dict[str, Any] = {
                "messages": messages,
                "source": "vera_mh",
                "task_type": "adversarial_safety",
                "diagnostic_tag": first_diagnosis or None,
                "demographic_tags": demographic_tags,
                "linguistic_style": "informal" if len(seed_phrase) < _SHORT_TEXT_THRESHOLD else "mixed",
                "clinical_reviewed": False,
                "persona_name": name,
                "suicide_risk_level": short_risk,
                "communication_style": comm_style,
                "reaction_to_chatbot": reaction,
                "treatment_engagement": treatment,
                "mental_health_stigma": stigma,
                "background": background,
                "recent_triggers": triggers,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="tsv",
                ),
            }
            records.append(record)

        return records
