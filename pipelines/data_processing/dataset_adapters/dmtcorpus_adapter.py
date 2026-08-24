"""Adapter for the DMTCorpus synthetic multi-session CBT dataset.

Source: arxiv 2606.03132 (synthetic dataset; not yet publicly released).
Format: JSON (synthetic multi-session CBT dialogues).
Size: 4,317 sessions, 768 conditions, 6 sessions/condition.
      148 CBT cases from PsychEval, 383 homework items.
Method: GPT-4.1-mini generation; cross-session homework continuity.

Output task_type: therapy_response_generation
Each session becomes one ChatML record. System prompt embeds session
number, condition, and cross-session homework context. Therapist -> assistant,
Patient -> user. Metadata preserves multi-session ordering info to support
longitudinal / Foresight Continuity work.
"""

from __future__ import annotations

import json
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://arxiv.org/abs/2606.03132"
_ACCESS_METHOD = "request"
_ORIGINAL_FORMAT = "json"

_README_TEXT = """\
DMTCorpus — Synthetic Multi-Session CBT Dataset
================================================

Source: arXiv 2606.03132 (synthetic dataset, not yet publicly released).

Stats:
  - 4,317 sessions across 768 conditions
  - 6 sessions per condition (multi-session CBT)
  - 148 CBT cases from PsychEval
  - 383 homework items with cross-session continuity
  - Generated with GPT-4.1-mini

Acquisition:
  The dataset is synthetic and not yet publicly distributed.
  To obtain the raw JSON files, reproduce the generation pipeline described
  in the arXiv paper (https://arxiv.org/abs/2606.03132) and place the
  resulting JSON files into this directory. The adapter reads any *.json
  files placed here.

Expected raw file layout (place any of these in this folder):
  sessions.json   - list of session objects (see schema below)
  conditions.json - optional condition metadata
  homework.json   - optional homework metadata

Session JSON schema (per session object):
  {
    "condition_id": "c-001",
    "condition": "major depressive disorder",
    "session_number": 1,
    "total_sessions": 6,
    "homework_items": [...],          # optional, assigned this session
    "cross_session_state": {...},    # optional, carried-over homework
    "dialog": [
      {"speaker": "Therapist", "utterance": "..."},
      {"speaker": "Patient",  "utterance": "..."},
      ...
    ]
  }
"""


@register_adapter("dmtcorpus")
class DMTCorpusAdapter(BaseDatasetAdapter):
    """Adapter for the DMTCorpus synthetic multi-session CBT dataset.

    Each session -> one ChatML record. System prompt carries session
    number, condition, and cross-session homework context (BEST MATCH for
    longitudinal / Foresight Continuity — session ordering preserved).
    Therapist -> assistant, Patient -> user. task_type:
    therapy_response_generation.
    """

    def download(self) -> None:
        """Write a README with arXiv link + generation instructions."""
        readme = self._raw_dir / "README.txt"
        if not readme.exists():
            readme.write_text(_README_TEXT, encoding="utf-8")

    def extract(self) -> list[dict[str, Any]]:
        """Read session objects from any *.json file in the raw dir."""
        records: list[dict[str, Any]] = []
        for path in sorted(self._raw_dir.glob("*.json")):
            if path.name == "README.txt":
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        records.append({**item, "_source_file": path.name})
            elif isinstance(data, dict):
                # Dict-of-sessions or single session object.
                sessions_val = data.get("sessions")
                if isinstance(sessions_val, list):
                    for item in sessions_val:
                        if isinstance(item, dict):
                            records.append({**item, "_source_file": path.name})
                else:
                    records.append({**data, "_source_file": path.name})
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert session dicts to ChatML — one record per session."""
        records: list[dict[str, Any]] = []

        for session in raw_data:
            dialog = (
                session.get("dialog")
                or session.get("turns")
                or session.get("conversation")
                or session.get("messages")
                or []
            )
            if not dialog:
                continue

            condition = (session.get("condition") or session.get("condition_name") or "").strip()
            condition_id = (session.get("condition_id") or session.get("case_id") or "").strip()
            session_number = session.get("session_number") or session.get("session")
            total_sessions = session.get("total_sessions")
            homework_items = session.get("homework_items") or []
            cross_session_state = session.get("cross_session_state") or {}

            system_content = self._build_system(
                condition=condition,
                session_number=session_number,
                total_sessions=total_sessions,
                homework_items=homework_items,
                cross_session_state=cross_session_state,
            )

            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
            for turn in dialog:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("speaker") or turn.get("role") or "").lower()
                utterance = (turn.get("utterance") or turn.get("text") or turn.get("content") or "").strip()
                if not utterance:
                    continue
                role = self._map_role(speaker)
                messages.append({"role": role, "content": utterance})

            if len(messages) < 2:
                continue
            roles = {m["role"] for m in messages}
            if "user" not in roles or "assistant" not in roles:
                continue

            sn_val = session_number if isinstance(session_number, int) else None
            ts_val = total_sessions if isinstance(total_sessions, int) else None

            record: dict[str, Any] = {
                "messages": messages,
                "source": "dmtcorpus",
                "task_type": "therapy_response_generation",
                "diagnostic_tag": condition or None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "session_number": sn_val,
                "total_sessions": ts_val,
                "condition": condition,
                "condition_id": condition_id,
                "homework_items": homework_items,
                "cross_session_state": cross_session_state,
                "is_synthetic": True,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method=_ACCESS_METHOD,
                    original_format=_ORIGINAL_FORMAT,
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _map_role(speaker: str) -> str:
        therapist_tokens = {"therapist", "counselor", "assistant", "supporter", "gpt"}
        patient_tokens = {"patient", "client", "user", "seeker", "human"}
        if speaker in therapist_tokens:
            return "assistant"
        if speaker in patient_tokens:
            return "user"
        return "assistant"

    @staticmethod
    def _build_system(
        *,
        condition: str,
        session_number: Any,
        total_sessions: Any,
        homework_items: list[Any],
        cross_session_state: dict[str, Any],
    ) -> str:
        parts: list[str] = ["Synthetic multi-session CBT (DMTCorpus, arXiv 2606.03132, GPT-4.1-mini)."]
        if condition:
            parts.append(f"Condition: {condition}.")
        if isinstance(session_number, int) and isinstance(total_sessions, int):
            parts.append(f"Session {session_number} of {total_sessions}.")
        elif isinstance(session_number, int):
            parts.append(f"Session {session_number}.")
        if homework_items:
            parts.append(f"Homework assigned this session: {len(homework_items)} item(s).")
        if cross_session_state:
            keys = list(cross_session_state.keys())
            parts.append(f"Cross-session state keys: {', '.join(keys)}.")
        return " ".join(parts)
