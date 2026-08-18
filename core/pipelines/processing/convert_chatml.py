"""Convert common message schemas to ChatML-style sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def convert_to_chatml(
    data: list[dict[str, Any]] | str | Path,
    *,
    system_prompt: str = "You are a therapeutic conversation assistant.",
) -> list[dict[str, Any]]:
    """Convert raw message batches into a ChatML-like format.

    Input can be:
    - a list of dict records with `messages`
    - a filesystem path to a JSON or JSONL file
    """

    if isinstance(data, (str, Path)):
        path = Path(data)
        records: list[dict[str, Any]] = []
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        content = path.read_text(encoding="utf-8")
        text = content.strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = json.loads(text)
            if isinstance(parsed, list):
                records = [item for item in parsed if isinstance(item, dict)]
        else:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        payload = json.loads(line)
                        if isinstance(payload, dict):
                            records.append(payload)
                    except json.JSONDecodeError:
                        continue
    else:
        records = [entry for entry in data if isinstance(entry, dict)]

    output: list[dict[str, Any]] = []
    for record in records:
        messages = record.get("messages", [])
        chatml_messages = []
        if system_prompt:
            chatml_messages.append({"role": "system", "content": system_prompt})

        if isinstance(messages, list):
            for item in messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "assistant")).lower()
                if role not in {"system", "user", "assistant", "therapist", "client"}:
                    role = "user"
                chatml_messages.append(
                    {
                        "role": "user" if role in {"client", "user"} else ("system" if role == "system" else "assistant"),
                        "content": str(item.get("content", "")),
                    }
                )

        output.append(
            {
                **{k: v for k, v in record.items() if k != "messages"},
                "chatml_messages": chatml_messages,
                "chatml_length": len(chatml_messages),
            }
        )

    return output


__all__ = ["convert_to_chatml"]
