"""Flexible adapter for extracting data from arbitrary GitHub repositories.

Designed for PIX-4239: extracts raw CSV/JSON/JSONL data from GitHub repos
identified in the hackathon research map (PIX-4238) and converts them to
standardized ChatML format.

Supports common mental health hackathon data patterns:
  - CSV with text/label columns (classification datasets)
  - JSON conversation arrays with speaker/utterance turns
  - ShareGPT-format conversations
  - JSONL with pre-formatted message arrays
  - Generic text/label pair CSVs

The adapter uses the GitHub API (`gh api`) to browse repo trees and
`raw.githubusercontent.com` for file downloads, avoiding full clones for
repos under ~50 MB.

Output records conform to the standardized ChatML schema defined in
base_adapter.py.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ai.sourcing.dataset_adapters.adapter_factory import register_adapter
from ai.sourcing.dataset_adapters.base_adapter import BaseDatasetAdapter

# File extensions we consider as potential data files
_DATA_EXTENSIONS = {".csv", ".json", ".jsonl", ".tsv"}

# Directories to skip when scanning repo trees
_SKIP_DIRS = {
    ".git",
    ".github",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
    "dist",
    "build",
}

# Max file size to download (50 MB) — larger files should use git clone
_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Common column name patterns for mental health datasets
_TEXT_COLUMN_CANDIDATES = [
    "text",
    "content",
    "body",
    "post",
    "message",
    "utterance",
    "statement",
    "tweet",
    "comment",
    "response",
    "answer",
    "input",
    "prompt",
    "seeker",
    "user",
    "patient",
    "client",
]

_LABEL_COLUMN_CANDIDATES = [
    "label",
    "category",
    "class",
    "target",
    "diagnosis",
    "disorder",
    "condition",
    "sentiment",
    "emotion",
    "mental_state",
    "mental_health",
    "status",
    "flag",
]

_ASSISTANT_COLUMN_CANDIDATES = [
    "response",
    "answer",
    "reply",
    "supporter",
    "therapist",
    "assistant",
    "counselor",
    "output",
    "target",
    "completion",
]

# Speaker role mappings for conversation data
_SPEAKER_USER_VALUES = {
    "seeker",
    "user",
    "client",
    "patient",
    "human",
    "speaker_a",
    "speaker1",
    "person",
    "speaker",
}

_SPEAKER_ASSISTANT_VALUES = {
    "supporter",
    "therapist",
    "assistant",
    "counselor",
    "gpt",
    "system",
    "agent",
    "speaker_b",
    "speaker2",
    "bot",
}


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _gh_api(args: list[str]) -> str:
    """Run a gh api command and return stdout. Raises on failure."""
    env = {**os.environ, "NO_COLOR": "1"}
    result = subprocess.run(
        ["gh", "api", *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return _ANSI_RE.sub("", result.stdout)


def _gh_api_json(args: list[str]) -> Any:
    """Run a gh api command and parse JSON output."""
    return json.loads(_gh_api(args))


def _raw_download_url(owner: str, repo: str, branch: str, path: str) -> str:
    """Construct a raw.githubusercontent.com download URL."""
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def _should_skip_path(path: str) -> bool:
    """Check if a file path should be skipped (in a skip directory)."""
    parts = Path(path).parts
    return any(part in _SKIP_DIRS for part in parts)


def _is_data_file(path: str) -> bool:
    """Check if a file path has a data file extension."""
    return Path(path).suffix.lower() in _DATA_EXTENSIONS


def _find_column(headers: list[str], candidates: list[str]) -> str | None:
    """Find a column name from candidates, case-insensitive."""
    lower_headers = {h.lower(): h for h in headers}
    for candidate in candidates:
        if candidate in lower_headers:
            return lower_headers[candidate]
    # Partial match
    for candidate in candidates:
        for h in headers:
            if candidate in h.lower():
                return h
    return None


def _classify_csv(
    csv_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Parse a CSV and classify its structure.

    Returns (rows, classification) where classification is one of:
      - "text_label"  : single text column + label column
      - "conversation" : user_column + assistant_column
      - "unknown"      : no recognized columns
    """
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], "unknown"
        headers = list(reader.fieldnames)
        rows = list(reader)

    # Check for conversation pattern: user + assistant columns
    user_col = _find_column(headers, _TEXT_COLUMN_CANDIDATES)
    assistant_col = _find_column(headers, _ASSISTANT_COLUMN_CANDIDATES)

    if user_col and assistant_col and user_col != assistant_col:
        return rows, "conversation"

    # Check for text + label pattern
    text_col = _find_column(headers, _TEXT_COLUMN_CANDIDATES)
    label_col = _find_column(headers, _LABEL_COLUMN_CANDIDATES)

    if text_col and label_col and text_col != label_col:
        return rows, "text_label"

    if text_col:
        # At least we have a text column — treat label as None
        return rows, "text_label"

    return rows, "unknown"


def _csv_to_chatml_records(
    rows: list[dict[str, Any]],
    source_name: str,
    provenance_fn: Any,
) -> list[dict[str, Any]]:
    """Convert CSV rows to ChatML records, handling both classification and conversation patterns."""
    records: list[dict[str, Any]] = []

    if not rows:
        return records

    headers = list(rows[0].keys())
    user_col = _find_column(headers, _TEXT_COLUMN_CANDIDATES)
    assistant_col = _find_column(headers, _ASSISTANT_COLUMN_CANDIDATES)
    label_col = _find_column(headers, _LABEL_COLUMN_CANDIDATES)

    # Conversation pattern
    if user_col and assistant_col and user_col != assistant_col:
        for row in rows:
            user_text = (row.get(user_col) or "").strip()
            assistant_text = (row.get(assistant_col) or "").strip()
            if not user_text or not assistant_text:
                continue
            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
            record: dict[str, Any] = {
                "messages": messages,
                "source": source_name,
                "task_type": "therapy_response_generation",
                "diagnostic_tag": (row.get(label_col) or None) if label_col else None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "provenance": provenance_fn(),
            }
            records.append(record)
        return records

    # Classification pattern (text + label -> symptom_classification)
    if user_col:
        for row in rows:
            text = (row.get(user_col) or "").strip()
            if not text:
                continue
            label = (row.get(label_col) or "").strip() if label_col else ""

            messages = [
                {
                    "role": "user",
                    "content": text,
                },
                {
                    "role": "assistant",
                    "content": f"[{label}] Classification: {text[:200]}..." if label else text[:200],
                },
            ]
            record = {
                "messages": messages,
                "source": source_name,
                "task_type": "symptom_classification" if label else "therapy_response_generation",
                "diagnostic_tag": label if label else None,
                "demographic_tags": [],
                "linguistic_style": "informal" if len(text) < 280 else "mixed",
                "clinical_reviewed": False,
                "provenance": provenance_fn(),
            }
            records.append(record)
        return records

    return records


def _json_to_chatml_records(
    data: Any,
    source_name: str,
    provenance_fn: Any,
) -> list[dict[str, Any]]:
    """Convert JSON data to ChatML records, auto-detecting structure."""
    records: list[dict[str, Any]] = []

    # Normalize to a list of items
    if isinstance(data, dict):
        # Could be a single conversation or a wrapper
        if any(k in data for k in ("dialog", "turns", "conversation", "messages")):
            items = [data]
        elif "data" in data and isinstance(data["data"], list):
            items = data["data"]
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        return records

    for item in items:
        if not isinstance(item, dict):
            continue

        # Pattern 1: Pre-formatted messages array (already ChatML-like)
        messages = item.get("messages")
        if isinstance(messages, list) and len(messages) >= 2:
            valid = True
            for msg in messages:
                if not isinstance(msg, dict):
                    valid = False
                    break
                role = msg.get("role")
                content = msg.get("content")
                if role not in ("system", "user", "assistant"):
                    valid = False
                    break
                if not isinstance(content, str) or not content.strip():
                    valid = False
                    break
            if valid:
                roles = {m["role"] for m in messages}
                if "user" in roles and "assistant" in roles:
                    record: dict[str, Any] = {
                        "messages": messages,
                        "source": source_name,
                        "task_type": item.get("task_type", "therapy_response_generation"),
                        "diagnostic_tag": item.get("diagnostic_tag"),
                        "demographic_tags": item.get("demographic_tags", []),
                        "linguistic_style": item.get("linguistic_style", "mixed"),
                        "clinical_reviewed": False,
                        "provenance": provenance_fn(),
                    }
                    records.append(record)
                    continue

        # Pattern 2: Conversation with dialog/turns/conversation key
        turns = item.get("dialog") or item.get("turns") or item.get("conversation") or []
        if isinstance(turns, list) and len(turns) >= 2:
            chatml_messages: list[dict[str, str]] = []

            # Add system message if situation/context exists
            situation = item.get("situation") or item.get("context") or ""
            if situation:
                chatml_messages.append({"role": "system", "content": str(situation)})

            has_user = False
            has_assistant = False
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                speaker = str(
                    turn.get("speaker")
                    or turn.get("from")
                    or turn.get("role")
                    or ""
                ).lower()
                utterance = (
                    turn.get("utterance")
                    or turn.get("value")
                    or turn.get("text")
                    or turn.get("content")
                    or ""
                )
                utterance = str(utterance).strip()
                if not utterance:
                    continue

                if speaker in _SPEAKER_USER_VALUES:
                    chatml_messages.append({"role": "user", "content": utterance})
                    has_user = True
                elif speaker in _SPEAKER_ASSISTANT_VALUES:
                    chatml_messages.append({"role": "assistant", "content": utterance})
                    has_assistant = True
                # Unknown speaker — alternate based on position
                elif not has_user:
                    chatml_messages.append({"role": "user", "content": utterance})
                    has_user = True
                elif not has_assistant:
                    chatml_messages.append({"role": "assistant", "content": utterance})
                    has_assistant = True
                else:
                    # Continue alternating
                    last_role = chatml_messages[-1]["role"]
                    next_role = "assistant" if last_role == "user" else "user"
                    chatml_messages.append({"role": next_role, "content": utterance})
                    if next_role == "user":
                        has_user = True
                    else:
                        has_assistant = True

            if has_user and has_assistant and len(chatml_messages) >= 2:
                record = {
                    "messages": chatml_messages,
                    "source": source_name,
                    "task_type": "therapy_response_generation",
                    "diagnostic_tag": item.get("emotion_type") or item.get("problem_type"),
                    "demographic_tags": [],
                    "linguistic_style": "mixed",
                    "clinical_reviewed": False,
                    "provenance": provenance_fn(),
                }
                records.append(record)
                continue

        # Pattern 3: ShareGPT format (conversations key)
        conversations = item.get("conversations") or []
        if isinstance(conversations, list) and len(conversations) >= 2:
            chatml_messages = []
            has_user = False
            has_assistant = False
            for turn in conversations:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("from", "")).lower()
                value = (turn.get("value") or "").strip()
                if not value:
                    continue
                if speaker in ("human", "user"):
                    chatml_messages.append({"role": "user", "content": value})
                    has_user = True
                else:
                    chatml_messages.append({"role": "assistant", "content": value})
                    has_assistant = True

            if has_user and has_assistant and len(chatml_messages) >= 2:
                record = {
                    "messages": chatml_messages,
                    "source": source_name,
                    "task_type": "therapy_response_generation",
                    "diagnostic_tag": None,
                    "demographic_tags": [],
                    "linguistic_style": "mixed",
                    "clinical_reviewed": False,
                    "provenance": provenance_fn(),
                }
                records.append(record)
                continue

        # Pattern 4: Simple text + label JSON (classification)
        text = item.get("text") or item.get("content") or item.get("body") or ""
        text = str(text).strip()
        label = item.get("label") or item.get("category") or item.get("class") or ""
        label = str(label).strip()
        if text and label:
            record = {
                "messages": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": f"[{label}] Classification."},
                ],
                "source": source_name,
                "task_type": "symptom_classification",
                "diagnostic_tag": label,
                "demographic_tags": [],
                "linguistic_style": "informal" if len(text) < 280 else "mixed",
                "clinical_reviewed": False,
                "provenance": provenance_fn(),
            }
            records.append(record)
            continue

    return records


@register_adapter("github_repo")
class GitHubRepoAdapter(BaseDatasetAdapter):
    """Flexible adapter for extracting data from arbitrary GitHub repos.

    Unlike dataset-specific adapters, this one discovers data files at
    runtime by browsing the repo tree via the GitHub API. It auto-detects
    CSV and JSON formats and converts them to ChatML.

    Usage:
        adapter = GitHubRepoAdapter(
            "github_repo",
            output_dir,
            repo_full_name="Karan-g-2003/deep-mental-health-voice",
        )
        adapter.run()
    """

    def __init__(
        self,
        dataset_name: str,
        output_dir: str | Path,
        *,
        repo_full_name: str = "",
        branch: str = "",
        license_name: str = "",
        matched_categories: list[str] | None = None,
    ) -> None:
        super().__init__(dataset_name, output_dir)
        self.repo_full_name = repo_full_name
        self.branch = branch or "main"
        self.license = license_name
        self.matched_categories = matched_categories or []
        self._repo_url = f"https://github.com/{repo_full_name}" if repo_full_name else ""
        self._file_list: list[dict[str, Any]] = []
        self._manifest: list[dict[str, Any]] = []

    def _list_repo_tree(self) -> list[dict[str, Any]]:
        """List all files in the repo tree via gh api."""
        try:
            data = _gh_api_json(
                [
                    f"repos/{self.repo_full_name}/git/trees/{self.branch}?recursive=1",
                ]
            )
            return data.get("tree", [])
        except Exception:
            return []

    def download(self) -> None:
        """Download data files from the repo to the raw directory."""
        if not self.repo_full_name:
            return

        # Check if files already downloaded
        existing = list(self._raw_dir.glob("*"))
        if existing:
            return

        tree = self._list_repo_tree()
        if not tree:
            return

        # Also fetch repo metadata for provenance
        try:
            repo_info = _gh_api_json([f"repos/{self.repo_full_name}"])
            self._repo_sha = repo_info.get("pushed_at", "")
        except Exception:
            self._repo_sha = ""

        owner, _, repo = self.repo_full_name.partition("/")

        for entry in tree:
            path = entry.get("path", "")
            if not path or entry.get("type") != "blob":
                continue
            if _should_skip_path(path) or not _is_data_file(path):
                continue

            size = entry.get("size", 0)
            if size > _MAX_FILE_SIZE_BYTES:
                continue

            # Download via raw URL
            url = _raw_download_url(owner, repo, self.branch, path)
            safe_name = path.replace("/", "_")
            target = self._raw_dir / safe_name

            try:
                import urllib.request

                urllib.request.urlretrieve(url, target)
                self._file_list.append(
                    {
                        "path": path,
                        "size": size,
                        "sha": entry.get("sha", ""),
                        "local_file": safe_name,
                    }
                )
            except Exception:
                pass

    def extract(self) -> list[dict[str, Any]]:
        """Extract data from downloaded files into intermediate dicts."""
        if not self._file_list:
            # Scan raw dir for downloaded files
            for f in self._raw_dir.iterdir():
                if f.is_file() and _is_data_file(f.name):
                    self._file_list.append(
                        {"path": f.name, "local_file": f.name, "size": f.stat().st_size, "sha": ""}
                    )

        records: list[dict[str, Any]] = []
        for file_info in self._file_list:
            local_path = self._raw_dir / file_info["local_file"]
            if not local_path.exists():
                continue
            ext = local_path.suffix.lower()

            if ext in {".csv", ".tsv"}:
                rows, classification = _classify_csv(local_path)
                records.append(
                    {
                        "type": "csv",
                        "classification": classification,
                        "rows": rows,
                        "file_info": file_info,
                    }
                )
            elif ext in (".json", ".jsonl"):
                try:
                    if ext == ".jsonl":
                        data = []
                        with open(local_path, encoding="utf-8", errors="replace") as f:
                            for line in f:
                                if line.strip():
                                    data.append(json.loads(line))
                    else:
                        with open(local_path, encoding="utf-8", errors="replace") as f:
                            data = json.load(f)
                    records.append(
                        {
                            "type": "json",
                            "data": data,
                            "file_info": file_info,
                        }
                    )
                except (json.JSONDecodeError, Exception):
                    continue

        return records

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert extracted intermediate dicts to ChatML records."""
        records: list[dict[str, Any]] = []

        for item in raw_data:
            file_info = item.get("file_info", {})
            file_path = file_info.get("path", "")
            item_type = item.get("type", "unknown")

            def provenance_fn(file_path: str = file_path, item_type: str = item_type) -> dict[str, Any]:
                return self._build_provenance(
                    source_url=f"{self._repo_url}/blob/{self.branch}/{file_path}",
                    access_method="github",
                    original_format=item_type,
                    transformations=["download", "extract", "convert_to_chatml", "validate"],
                )

            source_name = self.repo_full_name or self.dataset_name

            if item["type"] == "csv":
                csv_records = _csv_to_chatml_records(
                    item["rows"],
                    source_name,
                    provenance_fn,
                )
                records.extend(csv_records)
            elif item["type"] == "json":
                json_records = _json_to_chatml_records(
                    item["data"],
                    source_name,
                    provenance_fn,
                )
                records.extend(json_records)

        return records
