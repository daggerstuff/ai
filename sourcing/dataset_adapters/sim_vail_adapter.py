"""Adapter for the SIM-VAIL dataset.

Source: arxiv 2602.01347 (VAILs = vulnerability-amplifying interaction loops)
Format: JSON/CSV (conversations + turn-level ratings)
Size: 30 psychiatric phenotypes x 9 chatbots x 13 risk dimensions
       = 810 conversations, 90K+ turn-level ratings
Key concept: VAILs = vulnerability-amplifying interaction loops.
             Risk accumulates over turns.

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

_SOURCE_URL = "https://arxiv.org/abs/2602.01347"
_REPO_URL = "https://github.com/sim-vail/dataset"
_CONVERSATIONS_URL = "https://raw.githubusercontent.com/sim-vail/dataset/main/conversations.json"
_RATINGS_URL = "https://raw.githubusercontent.com/sim-vail/dataset/main/turn_ratings.csv"

# SIM-VAIL risk dimensions.
_RISK_DIMENSIONS = (
    "self_harm",
    "suicide_risk",
    "psychosis",
    "depression",
    "anxiety",
    "mania",
    "substance_abuse",
    "eating_disorder",
    "trauma",
    "personality_disorder",
    "dissociation",
    "paranoia",
    "social_isolation",
)


@register_adapter("sim_vail")
class SIMVAILAdapter(BaseDatasetAdapter):
    """Adapter for SIM-VAIL dataset.

    Converts conversations + turn-level risk ratings to ChatML with:
    - System prompt including phenotype and chatbot identity
    - User turn = patient with psychiatric phenotype
    - Assistant turn = chatbot response
    - Metadata: phenotype, chatbot, risk_dimensions, vail_detected flag,
      turn-level ratings
    - task_type = adversarial_safety (risk accumulates over turns)
    """

    def download(self) -> None:
        """Download conversations.json and turn_ratings.csv if present.

        Falls back gracefully to local files placed in the raw dir, since the
        public SIM-VAIL release may not yet expose canonical raw URLs.
        """
        conv_file = self._raw_dir / "conversations.json"
        ratings_file = self._raw_dir / "turn_ratings.csv"

        if not conv_file.exists():
            try:
                urllib.request.urlretrieve(_CONVERSATIONS_URL, conv_file)
            except Exception:
                # No public raw URL yet; adapter can work with locally
                # placed conversations.json in the raw dir.
                pass

        if not ratings_file.exists():
            try:
                urllib.request.urlretrieve(_RATINGS_URL, ratings_file)
            except Exception:
                # Ratings file is supplementary; conversations.json is authoritative
                pass

    def extract(self) -> list[dict[str, Any]]:
        """Extract conversations and merge turn-level ratings."""
        records: list[dict[str, Any]] = []

        conv_file = self._raw_dir / "conversations.json"
        ratings_file = self._raw_dir / "turn_ratings.csv"

        ratings_by_turn: dict[str, dict[str, Any]] = {}
        if ratings_file.exists():
            with open(ratings_file, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    turn_id = (row.get("turn_id") or "").strip()
                    if turn_id:
                        ratings_by_turn[turn_id] = {**row, "_source_file": "turn_ratings.csv"}

        if not conv_file.exists():
            return records

        with open(conv_file, encoding="utf-8") as f:
            conversations = json.load(f)

        for conv in conversations:
            merged = {**conv, "_source_file": "conversations.json"}
            # Attach turn-level ratings to each turn when available
            turns = conv.get("turns") or conv.get("dialog") or conv.get("conversation") or []
            enriched_turns: list[dict[str, Any]] = []
            for turn in turns:
                turn_id = str(turn.get("turn_id") or turn.get("id") or "")
                if turn_id and turn_id in ratings_by_turn:
                    enriched_turns.append({**turn, **ratings_by_turn[turn_id]})
                else:
                    enriched_turns.append(turn)
            if enriched_turns:
                merged["turns"] = enriched_turns
            records.append(merged)

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert conversations to ChatML with VAIL metadata."""
        records: list[dict[str, Any]] = []

        for conv in raw_data:
            turns = conv.get("turns") or conv.get("dialog") or conv.get("conversation") or []
            if not turns:
                continue

            phenotype = (conv.get("phenotype") or conv.get("psychiatric_phenotype") or "").strip()
            chatbot = (conv.get("chatbot") or conv.get("model") or "").strip()
            conversation_id = conv.get("conversation_id") or conv.get("id") or ""
            vail_detected = bool(conv.get("vail_detected") or conv.get("vail_flag") or False)

            risk_scores: dict[str, Any] = {}
            for dim in _RISK_DIMENSIONS:
                raw_value = conv.get(dim)
                if raw_value is None or raw_value == "":
                    continue
                try:
                    risk_scores[dim] = float(raw_value)
                except (TypeError, ValueError):
                    continue

            system_parts: list[str] = ["SIM-VAIL adversarial simulation."]
            if phenotype:
                system_parts.append(f"Phenotype: {phenotype}")
            if chatbot:
                system_parts.append(f"Chatbot: {chatbot}")
            if vail_detected:
                system_parts.append("VAIL detected: vulnerability-amplifying interaction loop.")
            system_parts.append("Assistant responses may be unsafe; do not emulate clinically.")
            system_content = " ".join(system_parts)

            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

            turn_ratings: list[dict[str, Any]] = []
            for turn in turns:
                speaker = str(turn.get("speaker") or turn.get("role") or "").lower()
                utterance = (turn.get("utterance") or turn.get("text") or turn.get("content") or "").strip()
                if not utterance:
                    continue
                role = "user" if speaker in ("patient", "user", "seeker", "human") else "assistant"
                messages.append({"role": role, "content": utterance})

                turn_id = str(turn.get("turn_id") or turn.get("id") or "")
                if turn_id:
                    rating_entry: dict[str, Any] = {"turn_id": turn_id, "role": role}
                    for dim in _RISK_DIMENSIONS:
                        raw_value = turn.get(dim)
                        if raw_value is None or raw_value == "":
                            continue
                        try:
                            rating_entry[dim] = float(raw_value)
                        except (TypeError, ValueError):
                            continue
                    turn_ratings.append(rating_entry)

            roles = {m["role"] for m in messages}
            if "user" not in roles or "assistant" not in roles:
                continue

            record: dict[str, Any] = {
                "messages": messages,
                "source": "sim_vail",
                "task_type": "adversarial_safety",
                "diagnostic_tag": phenotype or None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "conversation_id": conversation_id,
                "phenotype": phenotype,
                "chatbot": chatbot,
                "vail_detected": vail_detected,
                "risk_dimensions": risk_scores,
                "turn_ratings": turn_ratings,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="json",
                ),
            }
            records.append(record)

        return records
