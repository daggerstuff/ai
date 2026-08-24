"""Adapter for the Reddit Mental Health Posts dataset.

Source: https://huggingface.co/datasets/solomonk/reddit_mental_health_posts
Size: 151,288 posts across 5 subreddits (adhd, aspergers, depression, ocd, ptsd)
Format: CSV (one per subreddit) with columns: author, body, created_utc, id,
        num_comments, score, subreddit, title, upvote_ratio, url
Access: HuggingFace (open download)

Output task_type: symptom_classification
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_HF_REPO_ID = "solomonk/reddit_mental_health_posts"
_SOURCE_URL = "https://huggingface.co/datasets/solomonk/reddit_mental_health_posts"
_CSV_FILES = ["adhd.csv", "aspergers.csv", "depression.csv", "ocd.csv", "ptsd.csv"]

_SYSTEM_PROMPT = (
    "You are a mental health professional analyzing a Reddit post from a "
    "mental health support subreddit. Classify the primary concern and "
    "provide an empathetic, supportive response."
)

_SUBREDDIT_TO_DIAGNOSIS: dict[str, str] = {
    "adhd": "adhd",
    "aspergers": "autism_spectrum_disorder",
    "depression": "major_depressive_disorder",
    "ocd": "obsessive_compulsive_disorder",
    "ptsd": "ptsd",
}


@register_adapter("reddit_mental_health_posts")
class RedditMentalHealthPostsAdapter(BaseDatasetAdapter):
    """Adapter for Reddit mental health posts from HuggingFace."""

    def download(self) -> None:
        """Download CSV files from HuggingFace Hub."""
        # Skip if any CSV already exists with content
        existing = [f for f in _CSV_FILES if (self._raw_dir / f).exists() and (self._raw_dir / f).stat().st_size > 0]
        if len(existing) >= len(_CSV_FILES):
            return

        cache_dir = self.output_dir.parent / ".hf_cache"
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

        try:
            from huggingface_hub import hf_hub_download

            for fname in _CSV_FILES:
                target = self._raw_dir / fname
                if target.exists() and target.stat().st_size > 0:
                    continue
                path = hf_hub_download(
                    _HF_REPO_ID,
                    fname,
                    repo_type="dataset",
                    cache_dir=str(cache_dir / "hub"),
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(path).read_bytes())
        except Exception as e:
            readme = self._raw_dir / "README.txt"
            if not readme.exists():
                readme.write_text(
                    f"Download failed: {e}\n"
                    f"Manual: huggingface-cli download {_HF_REPO_ID} "
                    f"--repo-type dataset --local-dir {self._raw_dir}\n",
                    encoding="utf-8",
                )

    def extract(self) -> list[dict[str, Any]]:
        """Read all CSV files into intermediate dicts."""
        records: list[dict[str, Any]] = []

        for fname in _CSV_FILES:
            path = self._raw_dir / fname
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))
        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Reddit posts to ChatML records."""
        records: list[dict[str, Any]] = []
        seen: set[str] = set()

        for row in raw_data:
            body = (row.get("body") or "").strip()
            title = (row.get("title") or "").strip()
            text = body if body else title
            if not text:
                continue

            subreddit = (row.get("subreddit") or "").strip().lower()

            dedup_key = f"{' '.join(text.lower().split())}|{subreddit}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            diagnosis = _SUBREDDIT_TO_DIAGNOSIS.get(subreddit, subreddit or "unspecified")

            messages: list[dict[str, str]] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
                {"role": "assistant", "content": f"[{diagnosis}] Post from r/{subreddit}. See full text for details."},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "reddit_mental_health_posts",
                "task_type": "symptom_classification",
                "diagnostic_tag": diagnosis,
                "demographic_tags": [],
                "linguistic_style": "informal",
                "clinical_reviewed": False,
                "label": subreddit,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="huggingface",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records
