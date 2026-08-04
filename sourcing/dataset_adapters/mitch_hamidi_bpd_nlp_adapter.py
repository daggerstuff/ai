"""Adapter for the mitch_hamidi_bpd_nlp dataset — Reddit mental health discussions.

Source: S3 (whitebat:training/pixelated-empathy/output/mitch_hamidi_bpd_nlp/)
Original: github.com/mitchellhamidi/may-2024-mental-health-nlp
Format: CSV files in Scraping_Reddit_for_Data/ directory
  - *_comments.csv: columns Comment, Author, Post (parent post text)
  - *_search.csv: columns Title, Post Text, Post Creation Time, Content Type, etc.
Categories: diagnosed, drug, medicine, prescribed, recovery, therapy, treatment

Size: ~40K comments + ~9K search posts = ~49K total rows
Language: English

Output task_type: therapy_response_generation
Comments paired with parent posts create 2-turn dialogues.
Search posts create single-turn entries with title + body as user message.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_S3_PREFIX = "whitebat:training/pixelated-empathy/output/mitch_hamidi_bpd_nlp/raw/may-2024-mental-health-nlp/Scraping_Reddit_for_Data/"
_SOURCE_URL = "https://github.com/mitchellhamidi/may-2024-mental-health-nlp"

_CATEGORIES = ["diagnosed", "drug", "medicine", "prescribed", "recovery", "therapy", "treatment"]

_SYSTEM_PROMPT = (
    "You are a supportive community member responding to mental health discussions. "
    "Respond with empathy, understanding, and practical advice based on lived experience."
)


@register_adapter("mitch_hamidi_bpd_nlp")
class MitchHamidiBpdNlpAdapter(BaseDatasetAdapter):
    """Adapter for Reddit mental health discussions from S3."""

    def download(self) -> None:
        """Download CSVs from S3 if not already present."""
        target_dir = self._raw_dir / "Scraping_Reddit_for_Data"
        if target_dir.exists() and any(target_dir.glob("*.csv")):
            return

        try:
            subprocess.run(
                ["rclone", "copy", _S3_PREFIX, str(target_dir)],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError) as e:
            readme = self._raw_dir / "README.txt"
            readme.write_text(
                f"Download failed: {e}\nManual: rclone copy {_S3_PREFIX} <target>\n",
                encoding="utf-8",
            )

    def extract(self) -> list[dict[str, Any]]:
        """Read CSV files into intermediate dicts."""
        records: list[dict[str, Any]] = []
        target_dir = self._raw_dir / "Scraping_Reddit_for_Data"
        if not target_dir.exists():
            return records

        for csv_path in sorted(target_dir.glob("*.csv")):
            category = csv_path.stem.replace("_comments", "").replace("_search", "")
            is_comments = "_comments" in csv_path.stem

            try:
                with open(csv_path, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["_category"] = category
                        row["_is_comment"] = is_comments
                        records.append(row)
            except OSError:
                continue

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Reddit comments + posts to ChatML."""
        records: list[dict[str, Any]] = []

        for row in raw_data:
            category = row.get("_category", "")
            is_comment = row.get("_is_comment", False)

            if is_comment:
                # Comment paired with parent post
                comment = (row.get("Comment") or "").strip()
                parent_post = (row.get("Post") or "").strip()
                if not comment:
                    continue

                user_content = parent_post if parent_post else f"[{category} discussion]"
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": comment},
                ]
            else:
                # Search post: title + body
                title = (row.get("Title") or "").strip()
                body = (row.get("Post Text") or "").strip()
                if not title and not body:
                    continue

                user_content = f"{title}\n\n{body}" if title and body else (title or body)

                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": f"[Community discussion about {category}]"},
                ]

            records.append(
                {
                    "messages": messages,
                    "source": "mitch_hamidi_bpd_nlp",
                    "task_type": "therapy_response_generation",
                    "diagnostic_tag": category,
                    "demographic_tags": [],
                    "linguistic_style": "informal",
                    "clinical_reviewed": False,
                    "provenance": self._build_provenance(
                        source_url=_SOURCE_URL,
                        access_method="s3",
                        original_format="csv",
                    ),
                }
            )

        return records
