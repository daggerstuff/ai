"""Adapter for the MIT ai-psychosis adversarial safety dataset.

Source: https://github.com/mitmedialab/ai-psychosis
Format: CSV (harmful-responses.csv) + JSON scenario files
Size: 2,160 scenarios, 587K harmful responses across multiple LLMs
       15 failure patterns in 4 categories
Clinical staging: Stage 0-N models

Output task_type: adversarial_safety
"""

from __future__ import annotations

import csv
import json
import subprocess
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://github.com/mitmedialab/ai-psychosis"
_GIT_CLONE_URL = "https://github.com/mitmedialab/ai-psychosis.git"


@register_adapter("mit_psychosis")
class MITPsychosisAdapter(BaseDatasetAdapter):
    """Adapter for MIT ai-psychosis dataset.

    Clones the repo, reads scenario JSON files and harmful-responses.csv.
    Each scenario has pairs of (patient message, AI response). The
    harmful-responses.csv contains model-specific responses keyed by
    scenario_idx + message_idx. Joins them to produce ChatML records
    with adversarial safety metadata.
    """

    def download(self) -> None:
        """Clone the ai-psychosis repo if not already present."""
        repo_dir = self._raw_dir / "ai-psychosis"
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
                "MIT ai-psychosis data download.\n"
                "1. Visit: https://github.com/mitmedialab/ai-psychosis\n"
                "2. Clone repo into this directory.\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract scenario + harmful response data, joined on idx+msg_idx."""
        repo_dir = self._raw_dir / "ai-psychosis"
        scenarios_dir = repo_dir / "scenarios"
        harmful_csv = repo_dir / "harmful-responses.csv"

        if not scenarios_dir.exists() or not harmful_csv.exists():
            return []

        # Load scenario JSON files — build lookup: (scenario_idx, message_idx) -> patient_message
        scenario_lookup: dict[tuple[int, int], str] = {}
        scenario_meta: dict[int, dict[str, Any]] = {}

        for json_path in sorted(
            scenarios_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 999999
        ):
            if not json_path.stem.isdigit():
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            scen = data.get("scenario", {})
            file_idx = int(json_path.stem)

            inner = scen.get("scenario", {})
            pairs = scen.get("pairs", [])
            for msg_idx, pair in enumerate(pairs):
                msg = pair.get("message", "")
                if msg:
                    scenario_lookup[(file_idx, msg_idx)] = msg

            scenario_meta[file_idx] = {
                "description": inner.get("description", ""),
                "harm_type": inner.get("harm_type", ""),
                "age": inner.get("age", ""),
                "gender": inner.get("gender", ""),
                "socioeconomic": inner.get("socioeconomic", ""),
                "action": inner.get("action", ""),
                "outcome": inner.get("outcome", ""),
                "n_pairs": len(pairs),
            }

        # Read harmful-responses.csv and join with scenario lookup
        records: list[dict[str, Any]] = []
        with open(harmful_csv, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    scen_idx = int(row.get("scenario_idx", 0))
                    msg_idx = int(row.get("message_idx", 0))
                except (ValueError, TypeError):
                    continue

                patient_input = scenario_lookup.get((scen_idx, msg_idx), "")
                if not patient_input:
                    continue

                meta = scenario_meta.get(scen_idx, {})
                records.append(
                    {
                        "scenario_idx": scen_idx,
                        "message_idx": msg_idx,
                        "patient_input": patient_input,
                        "response": row.get("response", ""),
                        "stage": row.get("stage", ""),
                        "harm_type": row.get("harm_type", "") or meta.get("harm_type", ""),
                        "model": row.get("model", ""),
                        "cluster_l1": row.get("cluster_l1", ""),
                        "cluster_l2": row.get("cluster_l2", ""),
                        "cluster_id": row.get("cluster_id", ""),
                        "scenario_description": meta.get("description", ""),
                        "scenario_action": meta.get("action", ""),
                        "scenario_outcome": meta.get("outcome", ""),
                        "age": meta.get("age", ""),
                        "gender": meta.get("gender", ""),
                        "socioeconomic": meta.get("socioeconomic", ""),
                    }
                )

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert joined harmful response rows to ChatML records."""
        records: list[dict[str, Any]] = []

        for row in raw_data:
            patient_input = row.get("patient_input", "").strip()
            llm_response = row.get("response", "").strip()
            if not patient_input or not llm_response:
                continue

            harm_type = row.get("harm_type", "").strip()
            stage = row.get("stage", "").strip()
            model = row.get("model", "").strip()

            system_parts: list[str] = []
            if harm_type:
                system_parts.append(f"Harm type: {harm_type}")
            if stage:
                system_parts.append(f"Stage: {stage}")
            desc = row.get("scenario_description", "").strip()
            if desc:
                system_parts.append(f"Scenario: {desc[:200]}")
            system_parts.append(
                "Adversarial safety sample: the assistant response below is a known harmful LLM output."
            )
            system_content = ". ".join(system_parts)

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": patient_input},
                {"role": "assistant", "content": llm_response},
            ]

            demographic_tags: list[str] = []
            age = row.get("age", "").strip().lower()
            if age:
                demographic_tags.append(f"age_{age}")
            gender = row.get("gender", "").strip().lower()
            if gender:
                demographic_tags.append(f"gender_{gender}")

            record: dict[str, Any] = {
                "messages": messages,
                "source": "mit_psychosis",
                "task_type": "adversarial_safety",
                "diagnostic_tag": harm_type or None,
                "demographic_tags": demographic_tags,
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "harm_type": harm_type,
                "stage": stage,
                "model": model,
                "cluster_l1": row.get("cluster_l1", ""),
                "cluster_l2": row.get("cluster_l2", ""),
                "scenario_idx": row.get("scenario_idx"),
                "message_idx": row.get("message_idx"),
                "is_harmful_sample": True,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="github",
                    original_format="csv+json",
                ),
            }
            records.append(record)

        return records
