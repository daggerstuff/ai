"""Adapter for the CLPsych Shared Tasks dataset.

Source: https://clpsych.org/shared-task/
Format: JSON (Reddit mental health corpora)
Tasks:
  2024 — suicide risk evidence (125 users, r/SuicideWatch)
  2025 — ABCD self-state labels (Affect / Behavior / Cognition / Desire)
  2026 — ADHD symptom ranking
License: Research

NOTE: CLPsych data requires registration at https://clpsych.org/shared-task/.
Users must register, download the JSON files, and place them in the raw
directory before running this adapter.

Output task_type varies by task:
  - suicide risk   → risk_assessment
  - ABCD labels    → symptom_classification
  - ADHD ranking   → symptom_classification
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://clpsych.org/shared-task/"

_TASK_MAP: dict[str, str] = {
    "suicide_risk": "risk_assessment",
    "abcd_self_state": "symptom_classification",
    "adhd_symptoms": "symptom_classification",
}

_DIAGNOSTIC_MAP: dict[str, str] = {
    "suicide_risk": "suicide_risk",
    "abcd_self_state": "abcd_self_state",
    "adhd_symptoms": "adhd_symptoms",
}


@register_adapter("clpsych")
class CLPsychAdapter(BaseDatasetAdapter):
    """Adapter for CLPsych Shared Tasks Reddit mental health corpora.

    Expects JSON files placed in the raw directory after registration.
    Each file should contain an array of post objects with at minimum
    ``user_id``, ``post_id``, ``text``, and ``label`` fields.
    """

    def download(self) -> None:
        """No-op: CLPsych data requires registration download."""
        readme = self._raw_dir / "README.txt"
        if not readme.exists():
            readme.write_text(
                "CLPsych Shared Tasks data requires registration.\n"
                "1. Visit: https://clpsych.org/shared-task/\n"
                "2. Register and download the JSON data files.\n"
                "3. Place them in this directory.\n"
                "Expected fields per post: user_id, post_id, text, label.\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract posts from JSON files in the raw directory."""
        json_files = list(self._raw_dir.glob("*.json"))
        if not json_files:
            return []

        posts: list[dict[str, Any]] = []
        for jf in sorted(json_files):
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    posts.append({**item, "_source_file": jf.stem})
            elif isinstance(data, dict):
                for item in data.get("posts", data.get("data", [])):
                    posts.append({**item, "_source_file": jf.stem})
        return posts

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for post in raw_data:
            text = post.get("text", "").strip()
            if not text:
                continue

            label = post.get("label", "").strip().lower()
            task_key = self._match_task(label)
            task_type = _TASK_MAP.get(task_key, "symptom_classification")
            diagnostic_tag = _DIAGNOSTIC_MAP.get(task_key, label or None)

            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": self._system_prompt(task_key, label),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ]

            # If a response/annotation is present, add as assistant
            response = post.get("response") or post.get("annotation") or post.get("reasoning")
            if isinstance(response, str) and response.strip():
                messages.append({"role": "assistant", "content": response.strip()})
            elif label:
                # Use label as the classification answer
                messages.append({"role": "assistant", "content": label})

            record: dict[str, Any] = {
                "messages": messages,
                "source": "clpsych",
                "task_type": task_type,
                "diagnostic_tag": diagnostic_tag,
                "demographic_tags": [],
                "linguistic_style": "informal",
                "clinical_reviewed": False,
                "user_id": post.get("user_id"),
                "post_id": post.get("post_id"),
                "clpsych_label": label,
                "task_year": post.get("task_year"),
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="request",
                    original_format="json",
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _match_task(label: str) -> str:
        label_lower = label.lower()
        if "suicide" in label_lower or "self_harm" in label_lower:
            return "suicide_risk"
        if (
            "abcd" in label_lower
            or "affect" in label_lower
            or "behavior" in label_lower
            or "cognition" in label_lower
            or "desire" in label_lower
        ):
            return "abcd_self_state"
        if "adhd" in label_lower:
            return "adhd_symptoms"
        return ""

    @staticmethod
    def _system_prompt(task_key: str, label: str) -> str:
        if task_key == "suicide_risk":
            return (
                "You are a clinical mental health screener analyzing a social media post "
                "for suicide risk evidence. Assess the level of risk and provide reasoning."
            )
        if task_key == "abcd_self_state":
            return (
                "You are a clinical annotator classifying a social media post into "
                "ABCD self-state labels: Affect, Behavior, Cognition, or Desire."
            )
        if task_key == "adhd_symptoms":
            return "You are a clinical assessor ranking ADHD symptom evidence in a social media post."
        return "You are a clinical mental health researcher analyzing social media text."
