"""Shared scaffolding for the first-party synthetic scenario generators.

The suites differ in their cases and in their manifest prose, but every case is
the same envelope: system prompt, one user turn, the tools the case is allowed
to call, the fixture state those tools read, and the deterministic checks. That
envelope lives here so the case files stay readable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are a helpful assistant for a fictional software workspace. Follow the user request and available tools."
)


def function_tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def make_case(suite: str, split: str, spec: dict[str, Any]) -> dict[str, Any]:
    family = spec["family"]
    variant = spec["variant"]
    case_id = f"{split}-{family}-{variant}"
    system = spec.get("system", SYSTEM_PROMPT)
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": spec["user"]})
    return {
        "id": case_id,
        "scenario": family,
        "suite": suite,
        "group_id": f"{case_id}-group",
        "split": split,
        "messages": messages,
        "tools": spec.get("tools", []),
        "tool_fixtures": spec.get("fixtures", {}),
        "state_checks": spec.get("checks", []),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    encoded = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()
