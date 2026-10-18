"""Adapter for the Clinical Red Teaming (Steenstra) dataset.

Source: arxiv 2602.19948 (clinical red-teaming of AI psychotherapists).
Kaggle: steeni/ai-psychotherapy-eval
Format: CSV sessions with failure annotations.
Size: 15 DSM-5 personas, 6 AI psychotherapists, 369 sessions, 27K turns.
Key finding: "AI Psychosis" failure mode (validates delusions),
  cumulative iatrogenic harm.
Paper: 2025.

Output task_type: adversarial_safety
"""

from __future__ import annotations

import csv
import io
import os
import urllib.request
import zipfile
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://arxiv.org/abs/2602.19948"
_KAGGLE_DATASET = "steeni/ai-psychotherapy-eval"

# CSV files in the Kaggle dataset
_CSV_FILES = [
    "conversations.csv",
    "patient_personas.csv",
    "pairings.csv",
    "eval_crisis_detection.csv",
    "eval_crisis_protocol_adherence.csv",
    "eval_mi_behavior_counts.csv",
    "eval_mi_global_ratings.csv",
    "adverse_outcomes.csv",
]

# Only these files are required for the adapter to work
_REQUIRED_FILES = [
    "conversations.csv",
    "patient_personas.csv",
    "pairings.csv",
]


@register_adapter("clinical_redteam")
class ClinicalRedTeamAdapter(BaseDatasetAdapter):
    """Adapter for Clinical Red Teaming (Steenstra) dataset.

    Each session is one ChatML conversation. System prompt includes the
    DSM-5 persona. Patient turns are user messages; therapist turns are
    assistant responses. Includes crisis detection labels and MI behavior
    counts as metadata.
    """

    def download(self) -> None:
        """Download CSV files from Kaggle API if not already present."""
        if all((self._raw_dir / f).exists() for f in _REQUIRED_FILES):
            return

        token = os.environ.get("KAGGLE_API_TOKEN", "")
        url = f"https://www.kaggle.com/api/v1/datasets/download/{_KAGGLE_DATASET}"
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(self._raw_dir)
        except Exception:
            # Manual placement supported — create README if download fails
            readme = self._raw_dir / "README.txt"
            if not readme.exists():
                readme.write_text(
                    f"Download from Kaggle: {_KAGGLE_DATASET}\n"
                    f"URL: https://www.kaggle.com/datasets/{_KAGGLE_DATASET}\n"
                    "Place CSV files in this directory.\n",
                    encoding="utf-8",
                )

    def extract(self) -> list[dict[str, Any]]:
        """Read conversations.csv and join with persona + pairing metadata."""
        personas = self._read_csv("patient_personas.csv")
        pairings = self._read_csv("pairings.csv")
        conversations = self._read_csv("conversations.csv")
        crisis = self._read_csv("eval_crisis_detection.csv")

        persona_index: dict[str, dict[str, Any]] = {str(p.get("patient_id", "")): p for p in personas}
        pairing_index: dict[str, dict[str, Any]] = {str(p.get("pairing_id", "")): p for p in pairings}
        # Crisis labels keyed by (pairing_id, session_id, turn)
        crisis_index: dict[tuple[str, str, str], str] = {}
        for c in crisis:
            key = (
                str(c.get("pairing_id", "")),
                str(c.get("session_id", "")),
                str(c.get("turn", "")),
            )
            crisis_index[key] = c.get("classification", "")

        # Group turns by (pairing_id, session_id)
        sessions: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for turn in conversations:
            key = (
                str(turn.get("pairing_id", "")),
                str(turn.get("session_id", "")),
            )
            sessions.setdefault(key, []).append(turn)

        # Sort turns by turn number within each session
        for turns in sessions.values():
            turns.sort(key=lambda t: int(t.get("turn", 0) or 0))

        records: list[dict[str, Any]] = []
        for (pairing_id, session_id), turns in sessions.items():
            pairing = pairing_index.get(pairing_id, {})
            patient_id = str(pairing.get("patient_id", ""))
            persona = persona_index.get(patient_id, {})
            crisis_labels = [crisis_index.get((pairing_id, session_id, str(t.get("turn", ""))), "") for t in turns]
            records.append(
                {
                    "pairing_id": pairing_id,
                    "session_id": session_id,
                    "therapist_id": pairing.get("therapist_id", ""),
                    "patient_id": patient_id,
                    "turns": turns,
                    "persona": persona,
                    "crisis_labels": crisis_labels,
                }
            )
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert extracted sessions to ChatML adversarial_safety records."""
        records: list[dict[str, Any]] = []

        for session in raw_data:
            persona = session.get("persona") or {}
            turns = session.get("turns") or []
            crisis_labels = session.get("crisis_labels") or []

            system_content = self._build_system(persona=persona)
            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

            for i, turn in enumerate(turns):
                speaker = str(turn.get("speaker", "")).lower()
                text = (turn.get("message") or "").strip()
                if not text:
                    continue
                role = self._map_role(speaker)
                messages.append({"role": role, "content": text})

            if len(messages) < 2:
                continue

            # Ensure both user + assistant present (synthetic counterpart if needed)
            roles = {m["role"] for m in messages}
            if "user" not in roles and "assistant" in roles:
                messages.insert(1, {"role": "user", "content": "[session context]"})
            elif "assistant" not in roles and "user" in roles:
                messages.append({"role": "assistant", "content": "[continuation]"})

            # Crisis flag: any turn labeled as crisis
            has_crisis = any("crisis" in c.lower() and "no" not in c.lower() for c in crisis_labels)

            # Intensity scores from persona
            intensity_scores = {}
            for key in [
                "hopelessness_intensity",
                "negative_core_belief_intensity",
                "cognitive_preoccupation_with_use_intensity",
                "self_efficacy_intensity",
                "distress_tolerance_intensity",
                "substance_craving_intensity",
                "motivational_intensity",
                "ambivalence_about_change_intensity",
                "perceived_burdensomeness_intensity",
                "thwarted_belongingness_intensity",
            ]:
                val = persona.get(key)
                if val:
                    try:
                        intensity_scores[key] = float(val)
                    except (ValueError, TypeError):
                        pass

            demographic_tags: list[str] = []
            subtype = persona.get("subtype_name")
            if subtype:
                demographic_tags.append(f"aud_subtype_{str(subtype).strip().lower().replace(' ', '_')}")
            stage = persona.get("stage_of_change")
            if stage:
                demographic_tags.append(f"stage_{str(stage).strip().lower().replace(' ', '_')}")

            record: dict[str, Any] = {
                "messages": messages,
                "source": "clinical_redteam",
                "task_type": "adversarial_safety",
                "diagnostic_tag": "aud",
                "demographic_tags": demographic_tags,
                "linguistic_style": "formal",
                "clinical_reviewed": True,
                "pairing_id": session.get("pairing_id"),
                "session_id": session.get("session_id"),
                "therapist_id": session.get("therapist_id"),
                "patient_id": session.get("patient_id"),
                "persona_name": persona.get("name"),
                "persona_subtype": persona.get("subtype_name"),
                "has_crisis": has_crisis,
                "intensity_scores": intensity_scores if intensity_scores else None,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="kaggle",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records

    def _read_csv(self, filename: str) -> list[dict[str, Any]]:
        path = self._raw_dir / filename
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _build_system(*, persona: dict[str, Any]) -> str:
        parts: list[str] = []
        name = persona.get("name") or "patient"
        subtype = persona.get("subtype_name") or "unspecified"
        parts.append(f"DSM-5 AUD persona: {name} (subtype: {subtype}).")

        desc = persona.get("persona_description") or persona.get("ad_subtype_description") or ""
        if desc:
            parts.append(f"Description: {desc}")

        onset = persona.get("age_onset") or ""
        if onset:
            parts.append(f"Age/onset: {onset}")

        severity = persona.get("aud_severity_symptoms") or ""
        if severity:
            parts.append(f"AUD severity: {severity}")

        comorbid = persona.get("comorbid_psychiatric_disorders") or ""
        if comorbid:
            parts.append(f"Comorbid: {comorbid}")

        stage = persona.get("stage_of_change") or ""
        if stage:
            parts.append(f"Stage of change: {stage}")

        return " ".join(parts)

    @staticmethod
    def _map_role(speaker: str) -> str:
        if speaker in {"patient", "user", "client", "seeker", "human", "persona"}:
            return "user"
        if speaker in {"therapist", "ai", "assistant", "supporter", "counselor", "gpt", "system"}:
            return "assistant"
        return "assistant"
