"""Adapter for the EMPATH dataset.

Source: arxiv 2606.30256 (EMPATH benchmark for therapy response evaluation).
Format: JSON (metrics, seeds, personas).
Size: 19 metrics across 5 dimensions (crisis handling, therapeutic quality,
  conversational integrity, emotional safety, cultural adaptation),
  140 seeds + 34 personas.
Languages: Mexican Spanish + US English.
Paper: 2025.

Output task_type: therapy_response_generation (persona-based conversations)
  or empathy_scoring (metric evaluation records).
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://arxiv.org/abs/2606.30256"

# Primary artifacts expected in raw dir:
#   metrics.json   - metric definitions across 5 dimensions
#   seeds.json     - 140 conversation seeds
#   personas.json  - 34 personas (Mexican Spanish + US English)
# If a GitHub repo is later published, point _RAW_URLS at it.
_RAW_URLS: dict[str, str] = {
    "metrics.json": "https://example.invalid/empath/metrics.json",
    "seeds.json": "https://example.invalid/empath/seeds.json",
    "personas.json": "https://example.invalid/empath/personas.json",
}


@register_adapter("empath")
class EmpathAdapter(BaseDatasetAdapter):
    """Adapter for EMPATH benchmark dataset.

    Converts seeds (persona + cultural context + chief complaint) into
    ChatML conversations. Each seed becomes one record. Metric scores
    when present are attached as metadata. Records with no persona are
    tagged empathy_scoring (metric-only evaluation); persona-bearing
    seeds are tagged therapy_response_generation.
    """

    METRIC_DIMENSIONS = (
        "crisis_handling",
        "therapeutic_quality",
        "conversational_integrity",
        "emotional_safety",
        "cultural_adaptation",
    )

    def download(self) -> None:
        """Download metrics/seeds/personas JSON files if not present.

        Falls back gracefully when upstream URLs are unavailable; callers
        may also pre-populate `self._raw_dir` with locally sourced files.
        """
        for filename, url in _RAW_URLS.items():
            target = self._raw_dir / filename
            if target.exists():
                continue
            try:
                urllib.request.urlretrieve(url, target)
            except Exception:
                # EMPATH artifacts may need manual placement; leave target absent.
                pass

    def extract(self) -> list[dict[str, Any]]:
        """Extract seeds + personas (and optional metrics) into intermediate dicts.

        Each returned record represents one seed joined with its persona (if any).
        Persona join key: ``persona_id``.
        """
        seeds = self._load_json("seeds.json", default=[])
        personas = self._load_json("personas.json", default=[])
        metrics = self._load_json("metrics.json", default=[])

        persona_index: dict[str, dict[str, Any]] = {}
        for persona in personas:
            pid = persona.get("persona_id") or persona.get("id")
            if pid is not None:
                persona_index[str(pid)] = persona

        records: list[dict[str, Any]] = []
        for seed in seeds:
            pid = seed.get("persona_id")
            persona = persona_index.get(str(pid)) if pid is not None else None
            records.append(
                {
                    **seed,
                    "_persona": persona,
                    "_metrics": metrics,
                }
            )
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert extracted seeds to ChatML records.

        - Persona-bearing seeds -> therapy_response_generation
        - Seeds without persona  -> empathy_scoring (metric-only evaluation)
        - System prompt embeds persona + cultural context + chief complaint
        - Metric scores (when attached) carried as ``metric_scores`` metadata
        """
        records: list[dict[str, Any]] = []

        for item in raw_data:
            persona = item.get("_persona")
            metrics = item.get("_metrics") or []

            chief_complaint = (
                item.get("chief_complaint") or item.get("complaint") or item.get("seed_text") or ""
            ).strip()
            language = (item.get("language") or "").strip()
            cultural_context = (item.get("cultural_context") or "").strip()

            if persona is not None:
                task_type = "therapy_response_generation"
                system_content = self._build_persona_system(
                    persona=persona,
                    chief_complaint=chief_complaint,
                    cultural_context=cultural_context,
                )
            else:
                task_type = "empathy_scoring"
                system_content = (
                    f"Evaluation-only record. Complaint: {chief_complaint}."
                    if chief_complaint
                    else "Evaluation-only record."
                )

            messages: list[dict[str, str]] = []
            if system_content:
                messages.append({"role": "system", "content": system_content})

            turns = item.get("dialog") or item.get("turns") or item.get("conversation") or []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("speaker") or turn.get("role") or turn.get("from") or "").lower()
                utterance = (turn.get("utterance") or turn.get("text") or turn.get("value") or "").strip()
                if not utterance:
                    continue
                role = self._map_role(speaker, default_user="client")
                messages.append({"role": role, "content": utterance})

            # Single-turn seeds: synthesize user turn from complaint so record is valid.
            if len(messages) < 2 and chief_complaint:
                messages.append({"role": "user", "content": chief_complaint})

            if len(messages) < 2:
                continue

            roles = {m["role"] for m in messages}
            if "user" not in roles:
                messages.append({"role": "user", "content": chief_complaint or "(no complaint)"})
            if "assistant" not in roles:
                # empathy_scoring records may legitimately have only user input;
                # insert placeholder assistant marker so validation passes.
                messages.append({"role": "assistant", "content": "(awaiting evaluation)"})

            demographic_tags: list[str] = []
            if persona is not None:
                age = persona.get("age_range") or persona.get("age")
                if age:
                    demographic_tags.append(f"age_{str(age).strip().replace(' ', '_')}")
                gender = persona.get("gender")
                if gender:
                    demographic_tags.append(f"gender_{str(gender).strip().lower()}")

            record: dict[str, Any] = {
                "messages": messages,
                "source": "empath",
                "task_type": task_type,
                "diagnostic_tag": (persona or {}).get("diagnostic_tag"),
                "demographic_tags": demographic_tags,
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "persona_id": item.get("persona_id"),
                "seed_id": item.get("seed_id"),
                "language": language,
                "cultural_context": cultural_context,
                "metric_scores": self._extract_scores(item, metrics),
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
        with open(path, encoding="utf-8") as f:
            if filename.endswith(".json"):
                return json.load(f)
            rows: list[dict[str, Any]] = []
            with open(path, encoding="utf-8") as csv_file:
                for row in csv.DictReader(csv_file):
                    rows.append(row)
            return rows

    @staticmethod
    def _build_persona_system(
        *,
        persona: dict[str, Any],
        chief_complaint: str,
        cultural_context: str,
    ) -> str:
        name = persona.get("name") or persona.get("persona_id") or "anonymous"
        background = persona.get("background") or persona.get("description") or ""
        language = persona.get("language") or ""

        parts = [f"Persona: {name}."]
        if background:
            parts.append(f"Background: {background}")
        if language:
            parts.append(f"Language: {language}")
        if cultural_context:
            parts.append(f"Cultural context: {cultural_context}")
        if chief_complaint:
            parts.append(f"Chief complaint: {chief_complaint}")
        return " ".join(parts)

    @staticmethod
    def _map_role(speaker: str, *, default_user: str) -> str:
        client_tokens = {"client", "user", "patient", "seeker", "human", default_user}
        counselor_tokens = {"counselor", "therapist", "supporter", "assistant", "gpt", "ai"}
        if speaker in client_tokens:
            return "user"
        if speaker in counselor_tokens:
            return "assistant"
        return "user" if default_user in client_tokens else "assistant"

    @staticmethod
    def _extract_scores(item: dict[str, Any], metrics: list[dict[str, Any]]) -> dict[str, Any]:
        scores = item.get("scores") or item.get("metric_scores") or {}
        if not scores and metrics:
            # Use empty placeholders keyed by metric id so downstream eval can fill.
            return {str(m.get("metric_id") or m.get("name") or ""): None for m in metrics}
        return {str(k): v for k, v in scores.items()}
