"""Adapter for the eRisk (CLEF) dataset.

Source: https://erisk.irlab.org/
Format: JSON sequential social media posts
Tasks: depression, anorexia, self-harm, pathological gambling,
       BDI-II ranking, ADHD symptom ranking (2026), conversational depression
License: Research (CLEF registration required)

NOTE: eRisk data requires CLEF registration. Users must register at
https://erisk.irlab.org/ and place JSON files in the raw directory.

Output task_type varies by task:
  - depression          → severity_estimation
  - anorexia            → symptom_classification
  - self-harm           → risk_assessment
  - pathological_gambling → symptom_classification
  - adhd                → symptom_classification
"""

from __future__ import annotations

import json
from typing import Any

from ai.pipelines.data_processing.dataset_adapters.adapter_factory import register_adapter
from ai.pipelines.data_processing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://erisk.irlab.org/"

_TASK_TYPE_MAP: dict[str, str] = {
    "depression": "severity_estimation",
    "anorexia": "symptom_classification",
    "self_harm": "risk_assessment",
    "selfharm": "risk_assessment",
    "pathological_gambling": "symptom_classification",
    "gambling": "symptom_classification",
    "adhd": "symptom_classification",
    "conversational_depression": "severity_estimation",
}

_DIAGNOSTIC_MAP: dict[str, str] = {
    "depression": "depression",
    "anorexia": "anorexia",
    "self_harm": "self_harm",
    "selfharm": "self_harm",
    "pathological_gambling": "pathological_gambling",
    "gambling": "pathological_gambling",
    "adhd": "adhd",
    "conversational_depression": "depression",
}


@register_adapter("erisk")
class ERISKAdapter(BaseDatasetAdapter):
    """Adapter for eRisk early risk detection from social media.

    Expects JSON files placed in the raw directory after CLEF registration.
    Each file should contain an array of user objects with sequential posts.
    """

    def download(self) -> None:
        """No-op: eRisk data requires CLEF registration."""
        readme = self._raw_dir / "README.txt"
        if not readme.exists():
            readme.write_text(
                "eRisk data requires CLEF registration.\n"
                "1. Visit: https://erisk.irlab.org/\n"
                "2. Register for the challenge.\n"
                "3. Download JSON data files.\n"
                "4. Place them in this directory.\n"
                "Expected: array of {user_id, task, posts: [{date, text, label}]}\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Extract user post sequences from JSON files."""
        json_files = list(self._raw_dir.glob("*.json"))
        if not json_files:
            return []

        users: list[dict[str, Any]] = []
        for jf in sorted(json_files):
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    users.append({**item, "_source_file": jf.stem})
            elif isinstance(data, dict):
                for item in data.get("users", data.get("data", [])):
                    users.append({**item, "_source_file": jf.stem})
        return users

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for user in raw_data:
            posts = user.get("posts", [])
            if not posts or not isinstance(posts, list):
                continue

            task = user.get("task", "").strip().lower()
            task_type = _TASK_TYPE_MAP.get(task, "symptom_classification")
            diagnostic_tag = _DIAGNOSTIC_MAP.get(task, task or None)

            messages: list[dict[str, str]] = [
                {"role": "system", "content": self._system_prompt(task)},
            ]

            valid_posts = 0
            for post in posts:
                text = ""
                if isinstance(post, dict):
                    text = post.get("text", "").strip()
                elif isinstance(post, str):
                    text = post.strip()
                if not text:
                    continue
                messages.append({"role": "user", "content": text})
                valid_posts += 1

            if valid_posts < 1:
                continue

            label = user.get("label", "")
            if isinstance(label, str) and label.strip():
                messages.append({"role": "assistant", "content": label.strip()})
            elif task:
                messages.append({"role": "assistant", "content": task})

            record: dict[str, Any] = {
                "messages": messages,
                "source": "erisk",
                "task_type": task_type,
                "diagnostic_tag": diagnostic_tag,
                "demographic_tags": [],
                "linguistic_style": "informal",
                "clinical_reviewed": False,
                "user_id": user.get("user_id"),
                "erisk_task": task,
                "num_posts": valid_posts,
                "erde_metric": "latency-aware",
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="request",
                    original_format="json",
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _system_prompt(task: str) -> str:
        if "depression" in task:
            return (
                "You are a clinical assessor evaluating a sequence of social media posts "
                "for early signs of depression. Assess severity based on temporal patterns."
            )
        if "anorexia" in task:
            return "You are a clinical assessor evaluating social media posts for early signs of anorexia nervosa."
        if "self_harm" in task or "selfharm" in task:
            return "You are a clinical risk assessor evaluating social media posts for self-harm risk indicators."
        if "gambling" in task:
            return "You are a clinical assessor evaluating social media posts for pathological gambling behavior."
        if "adhd" in task:
            return "You are a clinical assessor ranking ADHD symptom evidence in social media posts."
        return "You are a clinical assessor evaluating social media posts for mental health risk."
