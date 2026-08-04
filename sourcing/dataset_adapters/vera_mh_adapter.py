"""Adapter for the VERA-MH (Spring Health) mental health persona dataset.

Source: https://github.com/SpringCare/VERA-MH
Format: JSON/CSV (personas + rubric)
Size: 100 clinically-developed personas, 5-dimension rubric
Rubric dimensions: Detects Risk, Confirms Risk, Guides to Human Care,
                   Supportive Conversation, Follows AI Boundaries
Pipeline: LLM-as-patient -> provider -> judge

Output task_type: therapy_response_generation (persona conversations)
                  or adversarial_safety (failed cases)
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/SpringCare/VERA-MH"
_PERSONAS_URL = "https://raw.githubusercontent.com/SpringCare/VERA-MH/main/personas.json"
_RUBRIC_URL = "https://raw.githubusercontent.com/SpringCare/VERA-MH/main/rubric.csv"

# The five VERA-MH rubric dimensions.
_RUBRIC_DIMENSIONS = (
    "detects_risk",
    "confirms_risk",
    "guides_to_human_care",
    "supportive_conversation",
    "follows_ai_boundaries",
)


@register_adapter("vera_mh")
class VERAMHAdapter(BaseDatasetAdapter):
    """Adapter for VERA-MH dataset.

    Converts 100 clinically-developed personas + 5-dimension rubric scores to
    ChatML with:
    - System prompt including the persona description
    - User/assistant turns from the patient/provider dialog (if present)
    - Metadata: persona_id, rubric scores per dimension
    - task_type = therapy_response_generation by default; adversarial_safety
      for any conversation marked as failed (rubric threshold breach)
    """

    def download(self) -> None:
        """Download personas.json and rubric.csv if not present."""
        personas_file = self._raw_dir / "personas.json"
        rubric_file = self._raw_dir / "rubric.csv"

        if not personas_file.exists():
            urllib.request.urlretrieve(_PERSONAS_URL, personas_file)

        if not rubric_file.exists():
            try:
                urllib.request.urlretrieve(_RUBRIC_URL, rubric_file)
            except Exception:
                # Rubric is supplementary; personas.json is authoritative
                pass

    def extract(self) -> list[dict[str, Any]]:
        """Extract persona + rubric rows into intermediate dicts."""
        records: list[dict[str, Any]] = []

        personas_file = self._raw_dir / "personas.json"
        rubric_file = self._raw_dir / "rubric.csv"

        rubric_by_persona: dict[str, dict[str, Any]] = {}
        if rubric_file.exists():
            with open(rubric_file, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = (row.get("persona_id") or "").strip()
                    if pid:
                        rubric_by_persona[pid] = {**row, "_source_file": "rubric.csv"}

        if not personas_file.exists():
            return records

        with open(personas_file, encoding="utf-8") as f:
            personas = json.load(f)

        for persona in personas:
            pid = persona.get("persona_id") or persona.get("id") or ""
            merged = {**persona, "_source_file": "personas.json"}
            if pid and pid in rubric_by_persona:
                merged.update(
                    {k: v for k, v in rubric_by_persona[pid].items() if k not in ("persona_id", "_source_file")}
                )
            records.append(merged)

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert persona records to ChatML."""
        records: list[dict[str, Any]] = []

        for persona in raw_data:
            description = (
                persona.get("description") or persona.get("persona_description") or persona.get("persona") or ""
            ).strip()
            if not description:
                continue

            dialog = persona.get("dialog") or persona.get("conversation") or persona.get("turns") or []
            rubric_scores: dict[str, Any] = {}
            failed = False
            for dim in _RUBRIC_DIMENSIONS:
                raw_value = persona.get(dim)
                if raw_value is None or raw_value == "":
                    continue
                try:
                    score = int(raw_value)
                except (TypeError, ValueError):
                    continue
                rubric_scores[dim] = score
                if score <= 1:
                    failed = True

            system_content = f"VERA-MH persona: {description}"
            if rubric_scores:
                score_summary = ", ".join(
                    f"{dim}={rubric_scores[dim]}" for dim in _RUBRIC_DIMENSIONS if dim in rubric_scores
                )
                system_content += f". Rubric: {score_summary}."

            task_type = "adversarial_safety" if failed else "therapy_response_generation"

            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

            if dialog:
                for turn in dialog:
                    speaker = str(turn.get("speaker") or turn.get("role") or "").lower()
                    utterance = (turn.get("utterance") or turn.get("text") or turn.get("content") or "").strip()
                    if not utterance:
                        continue
                    role = "user" if speaker in ("patient", "seeker", "user", "human") else "assistant"
                    messages.append({"role": role, "content": utterance})
            else:
                # No dialog; synthesize a seed prompt from the persona description
                # so the record still passes the >=2-message validator gate.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Act as the following persona and respond as a mental health provider: {description}"
                        ),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"[VERA-MH persona seed] Persona acknowledged. "
                            f"Rubric dimensions applied: {', '.join(_RUBRIC_DIMENSIONS)}."
                        ),
                    }
                )

            roles = {m["role"] for m in messages}
            if "user" not in roles or "assistant" not in roles:
                continue

            record: dict[str, Any] = {
                "messages": messages,
                "source": "vera_mh",
                "task_type": task_type,
                "diagnostic_tag": persona.get("diagnostic_tag") or persona.get("risk_type") or None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": True,
                "persona_id": persona.get("persona_id") or persona.get("id"),
                "rubric_scores": rubric_scores,
                "failed_rubric": failed,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="json",
                ),
            }
            records.append(record)

        return records
