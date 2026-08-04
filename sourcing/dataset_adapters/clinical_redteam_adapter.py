"""Adapter for the Clinical Red Teaming (Steenstra) dataset.

Source: arxiv 2602.19948 (clinical red-teaming of AI psychotherapists).
Format: JSON/CSV sessions with failure annotations.
Size: 15 DSM-5 personas, 6 AI psychotherapists, 369 sessions.
Key finding: "AI Psychosis" failure mode (validates delusions),
  cumulative iatrogenic harm.
Paper: 2025.

Output task_type: adversarial_safety
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://arxiv.org/abs/2602.19948"

# Expected raw files (placed manually if no public repo available):
#   sessions.json   - session objects with failure annotations
#   personas.json   - 15 DSM-5 persona definitions
#   therapists.json - 6 AI psychotherapist system prompts/configs
_RAW_URLS: dict[str, str] = {
    "sessions.json": "https://example.invalid/clinical_redteam/sessions.json",
    "personas.json": "https://example.invalid/clinical_redteam/personas.json",
    "therapists.json": "https://example.invalid/clinical_redteam/therapists.json",
}


@register_adapter("clinical_redteam")
class ClinicalRedTeamAdapter(BaseDatasetAdapter):
    """Adapter for Clinical Red Teaming (Steenstra) dataset.

    Each session is one ChatML conversation. System prompt includes the
    DSM-5 persona. User turns are patient utterances (with persona);
    assistant turns are AI psychotherapist responses. Failures tagged
    adversarial_safety. Includes failure mode + iatrogenic harm flag as
    metadata.
    """

    def download(self) -> None:
        """Download sessions/personas/therapists if not already present."""
        for filename, url in _RAW_URLS.items():
            target = self._raw_dir / filename
            if target.exists():
                continue
            try:
                urllib.request.urlretrieve(url, target)
            except Exception:
                # Public artifact may not be hosted; manual placement supported.
                pass

    def extract(self) -> list[dict[str, Any]]:
        """Join sessions with their persona + therapist metadata."""
        sessions = self._load_json("sessions.json", default=[])
        personas = self._load_json("personas.json", default=[])
        therapists = self._load_json("therapists.json", default=[])

        persona_index: dict[str, dict[str, Any]] = {}
        for persona in personas:
            pid = persona.get("persona_id") or persona.get("id")
            if pid is not None:
                persona_index[str(pid)] = persona

        therapist_index: dict[str, dict[str, Any]] = {}
        for therapist in therapists:
            tid = therapist.get("therapist_id") or therapist.get("id")
            if tid is not None:
                therapist_index[str(tid)] = therapist

        records: list[dict[str, Any]] = []
        for session in sessions:
            pid = session.get("persona_id")
            tid = session.get("therapist_id")
            records.append(
                {
                    **session,
                    "_persona": persona_index.get(str(pid)) if pid is not None else None,
                    "_therapist": therapist_index.get(str(tid)) if tid is not None else None,
                }
            )
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert extracted sessions to ChatML adversarial_safety records."""
        records: list[dict[str, Any]] = []

        for session in raw_data:
            persona = session.get("_persona") or {}
            therapist = session.get("_therapist") or {}

            system_content = self._build_system(persona=persona, therapist=therapist)
            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

            turns = (
                session.get("dialog")
                or session.get("turns")
                or session.get("conversation")
                or session.get("messages")
                or []
            )
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("speaker") or turn.get("role") or turn.get("from") or "").lower()
                utterance = (
                    turn.get("utterance") or turn.get("text") or turn.get("content") or turn.get("value") or ""
                ).strip()
                if not utterance:
                    continue
                role = self._map_role(speaker)
                messages.append({"role": role, "content": utterance})

            if len(messages) < 2:
                continue

            roles = {m["role"] for m in messages}
            if "user" not in roles or "assistant" not in roles:
                continue

            failure_mode = session.get("failure_mode") or session.get("failure_type")
            iatrogenic_harm = bool(
                session.get("iatrogenic_harm")
                or session.get("cumulative_iatrogenic_harm")
                or session.get("has_iatrogenic_harm")
                or False
            )

            demographic_tags: list[str] = []
            dsm_category = persona.get("dsm_category") or persona.get("dsm5_category")
            if dsm_category:
                demographic_tags.append(f"dsm5_{str(dsm_category).strip().lower().replace(' ', '_')}")

            record: dict[str, Any] = {
                "messages": messages,
                "source": "clinical_redteam",
                "task_type": "adversarial_safety",
                "diagnostic_tag": dsm_category,
                "demographic_tags": demographic_tags,
                "linguistic_style": "mixed",
                "clinical_reviewed": True,
                "persona_id": session.get("persona_id"),
                "therapist_id": session.get("therapist_id"),
                "session_id": session.get("session_id") or session.get("id"),
                "failure_mode": failure_mode,
                "iatrogenic_harm": iatrogenic_harm,
                "is_ai_psychosis": str(failure_mode).lower().startswith("ai_psychosis") if failure_mode else False,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="request",
                    original_format="json",
                ),
            }
            records.append(record)

        return records

    def _load_json(self, filename: str, *, default: Any) -> Any:
        path = self._raw_dir / filename
        if not path.exists():
            return default
        if filename.endswith(".csv"):
            rows: list[dict[str, Any]] = []
            with open(path, encoding="utf-8") as csv_file:
                for row in csv.DictReader(csv_file):
                    rows.append(row)
            return rows
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _build_system(*, persona: dict[str, Any], therapist: dict[str, Any]) -> str:
        parts: list[str] = []
        name = persona.get("name") or persona.get("persona_id") or "patient"
        dsm = persona.get("dsm_category") or persona.get("dsm5_category") or "unspecified"
        parts.append(f"DSM-5 persona: {name} (category: {dsm}).")

        symptoms = persona.get("symptoms") or persona.get("presentation") or ""
        if symptoms:
            parts.append(f"Presentation: {symptoms}")

        delusions = persona.get("delusions") or persona.get("delusional_content") or ""
        if delusions:
            parts.append(f"Delusional content: {delusions}")

        therapist_name = therapist.get("name") or therapist.get("therapist_id") or "AI psychotherapist"
        parts.append(f"AI psychotherapist: {therapist_name}.")

        t_system = therapist.get("system_prompt") or therapist.get("configuration") or ""
        if t_system:
            parts.append(f"Therapist config: {t_system}")

        return " ".join(parts)

    @staticmethod
    def _map_role(speaker: str) -> str:
        patient_tokens = {"patient", "user", "client", "seeker", "human", "persona"}
        therapist_tokens = {"therapist", "ai", "assistant", "supporter", "counselor", "gpt", "system"}
        if speaker in patient_tokens:
            return "user"
        if speaker in therapist_tokens:
            return "assistant"
        # Default: alternate speaker treated as assistant if turns are paired.
        return "assistant"
