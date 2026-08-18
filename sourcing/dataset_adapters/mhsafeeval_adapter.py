"""Adapter for the MHSafeEval dataset.

Source: https://github.com/suhyun565/MHSafeEval
Format: JSON (patient cognitive profiles for CBT evaluation)
Size: 58 patient profiles across 3 disorders (delusion, depression, psychosis)
Key concept: Structured clinical profiles with core beliefs, intermediate
    beliefs, coping strategies, and cognitive models (situation → thought
    → emotion → behavior).

Data files in repo:
  config/CCD/{disorder}/patient{N}.json

Output task_type: symptom_classification
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/suhyun565/MHSafeEval"
_GIT_CLONE_URL = "https://github.com/suhyun565/MHSafeEval.git"


def _to_str(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(x) for x in value).strip()
    return str(value or "").strip()


@register_adapter("mhsafeeval")
class MHSafeEvalAdapter(BaseDatasetAdapter):
    """Adapter for MHSafeEval clinical patient profiles.

    Clones the GitHub repo, reads patient JSON profiles from
    config/CCD/{disorder}/patient{N}.json. Each profile contains
    life history, core beliefs, intermediate beliefs, coping strategies,
    and cognitive models (situation/automatic_thoughts/emotion/behavior).
    """

    def download(self) -> None:
        """Clone the MHSafeEval repo if not already present."""
        repo_dir = self._raw_dir / "MHSafeEval"
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
                "MHSafeEval data download.\n"
                "1. Visit: https://github.com/suhyun565/MHSafeEval\n"
                "2. Clone repo into this directory.\n"
                "Expected: config/CCD/{disorder}/patient{N}.json files.\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract patient JSON profiles from cloned repo."""
        repo_dir = self._raw_dir / "MHSafeEval"
        ccd_dir = repo_dir / "config" / "CCD"
        if not ccd_dir.exists():
            return []

        records: list[dict[str, Any]] = []
        for disorder_dir in sorted(ccd_dir.iterdir()):
            if not disorder_dir.is_dir():
                continue
            disorder = disorder_dir.name
            for patient_file in sorted(disorder_dir.glob("patient*.json")):
                try:
                    with open(patient_file, encoding="utf-8") as f:
                        profile = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                profile["_disorder"] = disorder
                profile["_source_file"] = str(patient_file.relative_to(repo_dir))
                records.append(profile)
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert patient profiles to ChatML symptom classification records."""
        records: list[dict[str, Any]] = []

        for profile in raw_data:
            disorder = profile.get("_disorder", "")
            life_history = _to_str(profile.get("life_history"))
            core_beliefs = _to_str(profile.get("core_beliefs"))
            core_belief_desc = _to_str(profile.get("core_belief_description"))
            intermediate_beliefs = _to_str(profile.get("intermediate_beliefs"))
            coping = _to_str(profile.get("coping_strategies"))
            cognitive_models = profile.get("cognitive_models") or []

            if not life_history:
                continue

            user_content_parts: list[str] = []
            if life_history:
                user_content_parts.append(f"Life History: {life_history}")
            if coping:
                user_content_parts.append(f"Coping Strategies: {coping}")
            user_content = "\n\n".join(user_content_parts)

            assistant_parts: list[str] = []
            if core_belief_desc:
                assistant_parts.append(f"Core Belief: {core_belief_desc}")
            if intermediate_beliefs:
                assistant_parts.append(f"Intermediate Belief: {intermediate_beliefs}")
            for i, model in enumerate(cognitive_models, 1):
                if isinstance(model, dict):
                    situation = model.get("situation", "")
                    thought = model.get("automatic_thoughts", "")
                    emotion = model.get("emotion", "")
                    behavior = model.get("behavior", "")
                    assistant_parts.append(
                        f"Cognitive Model {i}:\n"
                        f"  Situation: {situation}\n"
                        f"  Automatic Thought: {thought}\n"
                        f"  Emotion: {emotion}\n"
                        f"  Behavior: {behavior}"
                    )
            assistant_content = (
                "\n\n".join(assistant_parts) if assistant_parts else "No cognitive model data available."
            )

            system_content = (
                f"MHSafeEval clinical profile evaluation. Disorder: {disorder}. Core belief: {core_belief_desc}."
                if core_belief_desc
                else f"MHSafeEval clinical profile evaluation. Disorder: {disorder}."
            )

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "mhsafeeval",
                "task_type": "symptom_classification",
                "diagnostic_tag": disorder,
                "demographic_tags": [],
                "linguistic_style": "formal",
                "clinical_reviewed": True,
                "disorder": disorder,
                "core_beliefs": core_beliefs,
                "core_belief_description": core_belief_desc,
                "coping_strategies": coping,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="json",
                ),
            }
            records.append(record)

        return records
