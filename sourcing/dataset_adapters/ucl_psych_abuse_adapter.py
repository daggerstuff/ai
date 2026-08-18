"""Adapter for the UCL Psychological Abuse dataset.

Source: https://api.figshare.com/v2/articles/31587925 (Figshare, open access)
Format: CSV (4 annotator files + 1 aggregate XLSX)
Size: 1,500 Reddit posts from r/abusiverelationships, r/domesticviolence, r/emotionalabuse
Categories: 6 non-mutually exclusive psychological abuse categories
License: Open access (CC BY 4.0)

Output task_type: symptom_classification
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

_SOURCE_URL = "https://doi.org/10.5522/04/31587925"
_FIGSHARE_API = "https://api.figshare.com/v2/articles/31587925"

# Direct download URLs for each annotator CSV
_FILES = {
    "a1_full.csv": "https://ndownloader.figshare.com/files/62552977",
    "a2_full.csv": "https://ndownloader.figshare.com/files/62552980",
    "a3_full.csv": "https://ndownloader.figshare.com/files/62552983",
    "a4_full.csv": "https://ndownloader.figshare.com/files/62552989",
}

# The 6 psychological abuse categories (column name → display name)
_ABUSE_CATEGORIES = {
    "1_Rules": "Coercive Control",
    "2_Justify": "Minimising Abuse",
    "3_Threat": "Threats",
    "4_Shame": "Emotional Abuse",
    "5_Isolate": "Isolation",
    "6_Surveil": "Surveillance",
}

_TEXT_COLUMN = "sampled_text"


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


@register_adapter("ucl_psych_abuse")
class UCLPsychAbuseAdapter(BaseDatasetAdapter):
    """Adapter for UCL Psychological Abuse dataset.

    Downloads 4 annotator CSV files from Figshare (open access). Each file
    contains Reddit posts from abuse subreddits with 6 binary abuse category
    labels. Converts to ChatML symptom_classification records.
    """

    def download(self) -> None:
        """Download annotator CSV files from Figshare."""
        for filename, url in _FILES.items():
            filepath = self._raw_dir / filename
            if filepath.exists():
                continue
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception:
                (self._raw_dir / "README.txt").write_text(
                    "UCL Psychological Abuse download.\n"
                    f"Failed to download {filename} from {url}\n"
                    "1. Visit: https://doi.org/10.5522/04/31587925\n"
                    "2. Download all a*_full.csv files into this directory.\n",
                    encoding="utf-8",
                )
                return

    def extract(self) -> list[dict[str, Any]]:
        """Extract Reddit posts and abuse annotations from CSV files."""
        records: list[dict[str, Any]] = []

        for filename in _FILES:
            filepath = self._raw_dir / filename
            if not filepath.exists():
                continue

            annotator_id = filename[1]  # a1 -> "1"
            with open(filepath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    post_id = _to_str(row.get("id"))
                    text = _to_str(row.get(_TEXT_COLUMN))

                    labels: dict[str, int] = {}
                    for col, display_name in _ABUSE_CATEGORIES.items():
                        if col in row:
                            val = row[col]
                            if isinstance(val, str):
                                labels[display_name] = 1 if val.strip().lower() in ("1", "true", "yes") else 0
                            elif isinstance(val, (int, float)):
                                labels[display_name] = int(val)

                    records.append(
                        {
                            "post_id": post_id,
                            "text": text,
                            "subreddit": "",
                            "annotator_id": annotator_id,
                            "labels": labels,
                            "source_file": filename,
                        }
                    )

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert annotated Reddit posts to ChatML symptom classification records."""
        # Group by post_id to aggregate annotator labels
        posts: dict[str, dict[str, Any]] = {}
        for rec in raw_data:
            post_id = rec["post_id"]
            if not post_id or not rec["text"]:
                continue
            if post_id not in posts:
                posts[post_id] = {
                    "post_id": post_id,
                    "text": rec["text"],
                    "subreddit": rec["subreddit"],
                    "annotator_labels": {},
                }
            # Merge: majority vote across annotators
            for cat, val in rec["labels"].items():
                if cat not in posts[post_id]["annotator_labels"]:
                    posts[post_id]["annotator_labels"][cat] = []
                posts[post_id]["annotator_labels"][cat].append(val)

        records: list[dict[str, Any]] = []
        for post_id, post in posts.items():
            text = post["text"]
            subreddit = post["subreddit"]

            # Majority vote per category
            confirmed_abuse: list[str] = []
            for cat, votes in post["annotator_labels"].items():
                if votes and sum(votes) > len(votes) / 2:
                    confirmed_abuse.append(cat)

            if not text:
                continue

            system_content = (
                "UCL Psychological Abuse: Classify psychological abuse types in Reddit posts "
                "from abuse-support subreddits."
            )

            assistant_content = (
                "Identified abuse categories:\n" + "\n".join(f"- {c}" for c in confirmed_abuse)
                if confirmed_abuse
                else "No psychological abuse categories identified."
            )
            if subreddit:
                assistant_content = f"Subreddit: r/{subreddit}\n\n" + assistant_content

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": text},
                {"role": "assistant", "content": assistant_content},
            ]

            record: dict[str, Any] = {
                "messages": messages,
                "source": "ucl_psych_abuse",
                "task_type": "symptom_classification",
                "diagnostic_tag": "psychological_abuse",
                "demographic_tags": [],
                "linguistic_style": "informal",
                "clinical_reviewed": True,
                "subreddit": subreddit,
                "confirmed_abuse_categories": confirmed_abuse,
                "provenance": self._build_provenance(
                    source_url=_SOURCE_URL,
                    access_method="direct_download",
                    original_format="csv",
                ),
            }
            records.append(record)

        return records
