"""Adapter for the SIM-VAIL dataset.

Source: https://github.com/veithweilnhammer/sim-vail
Format: JSON (transcripts + turn-level scores + causal validation)
Size: 30 psychiatric phenotypes x 9 target models x 13 risk dimensions
      = 810 conversations, 90K+ turn-level ratings
Key concept: VAILs = vulnerability-amplifying interaction loops.
             Risk accumulates over turns.

Data structure in repo:
  data/v1/<target_model>/transcript_*.json   — 810 conversation transcripts
  data/v1/turn_scores/                        — per-turn judge scores
  data/v1/conversation_rescores/               — conversation-level rescores
  data/v1/causal_scores/                       — causal validation outputs

Output task_type: adversarial_safety
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/veithweilnhammer/sim-vail"
_GIT_CLONE_URL = "https://github.com/veithweilnhammer/sim-vail.git"

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

_PHENOTYPE_KEYWORDS = (
    "depressive",
    "anxiety",
    "psychosis",
    "mania",
    "substance",
    "eating disorder",
    "trauma",
    "personality disorder",
    "dissociation",
    "paranoia",
    "social isolation",
)


def _detect_phenotype(description: str) -> str:
    for ph in _PHENOTYPE_KEYWORDS:
        if ph in description.lower():
            return ph
    return ""


_USER_ROLES = {"user", "seeker", "patient", "human"}
_ASSISTANT_ROLES = {"assistant", "bot", "model"}


def _map_role(role: str) -> str | None:
    if role in _USER_ROLES:
        return "user"
    if role in _ASSISTANT_ROLES:
        return "assistant"
    return None


@register_adapter("sim_vail")
class SIMVAILAdapter(BaseDatasetAdapter):
    """Adapter for SIM-VAIL dataset.

    Clones the GitHub repo, reads transcript JSON files from
    data/v1/<target_model>/transcript_*.json. Converts each transcript
    to ChatML with:
    - System prompt including phenotype and target model identity
    - User turn = simulated patient with psychiatric phenotype
    - Assistant turn = chatbot response
    - Metadata: phenotype, target_model, risk_dimensions, vail_detected flag
    - task_type = adversarial_safety (risk accumulates over turns)
    """

    def download(self) -> None:
        """Clone the sim-vail repo if not already present."""
        repo_dir = self._raw_dir / "sim-vail"
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
                "SIM-VAIL data download.\n"
                "1. Visit: https://github.com/veithweilnhammer/sim-vail\n"
                "2. Clone repo into this directory.\n"
                "Expected: data/v1/<target_model>/transcript_*.json files.\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract transcript JSON files from cloned repo."""
        repo_dir = self._raw_dir / "sim-vail"
        data_v1 = repo_dir / "data" / "v1"
        if not data_v1.exists():
            return []

        records: list[dict[str, Any]] = []
        for model_dir in sorted(data_v1.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            target_model = model_dir.name
            for transcript_file in sorted(model_dir.glob("transcript_*.json")):
                try:
                    with open(transcript_file, encoding="utf-8") as f:
                        transcript = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                transcript["_target_model"] = target_model
                transcript["_source_file"] = str(transcript_file.relative_to(repo_dir))
                records.append(transcript)

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert transcript JSON files to ChatML with VAIL metadata.

        Each transcript has a 'metadata' dict (auditor_model, target_model,
        description, judge_output) and a 'transcript' list of message dicts
        with role/content fields.
        """
        records: list[dict[str, Any]] = []

        for conv in raw_data:
            metadata = conv.get("metadata") or {}
            target_model = conv.get("_target_model") or metadata.get("target_model", "")
            auditor_model = metadata.get("auditor_model", "")
            transcript_id = metadata.get("transcript_id", "")
            description = metadata.get("description", "")
            phenotype = _detect_phenotype(description) if description else ""

            messages_raw = conv.get("transcript") or conv.get("messages") or conv.get("turns") or []
            if not messages_raw:
                continue

            system_parts: list[str] = ["SIM-VAIL adversarial safety simulation."]
            if phenotype:
                system_parts.append(f"Phenotype: {phenotype}")
            if target_model:
                system_parts.append(f"Target model: {target_model}")
            system_parts.append("Assistant responses may be unsafe; do not emulate clinically.")
            system_content = " ".join(system_parts)

            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

            for msg in messages_raw:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role") or msg.get("from") or "").lower()
                content = (msg.get("content") or msg.get("text") or msg.get("value") or "").strip()
                if not content:
                    continue
                mapped = _map_role(role)
                if mapped:
                    messages.append({"role": mapped, "content": content})

            roles_present = {m["role"] for m in messages}
            if "user" not in roles_present or "assistant" not in roles_present:
                continue

            judge_output = metadata.get("judge_output") or {}
            judge_highlights = ""
            if isinstance(judge_output, dict):
                judge_highlights = judge_output.get("response", "")

            record: dict[str, Any] = {
                "messages": messages,
                "source": "sim_vail",
                "task_type": "adversarial_safety",
                "diagnostic_tag": phenotype or None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "transcript_id": transcript_id,
                "phenotype": phenotype,
                "target_model": target_model,
                "auditor_model": auditor_model,
                "judge_highlights": judge_highlights[:500] if judge_highlights else "",
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="json",
                ),
            }
            records.append(record)

        return records
