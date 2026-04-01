from __future__ import annotations

import json
from typing import Optional


def parse_json_object(context_json: Optional[str]) -> dict:
    if not context_json:
        return {}
    parsed = json.loads(context_json)
    if not isinstance(parsed, dict):
        raise ValueError("Context payload must be a JSON object.")
    return parsed


def parse_auth_context(auth_context: str) -> dict:
    parsed = parse_json_object(auth_context)
    required_keys = ("actor_id", "timestamp", "nonce", "signature")
    missing = [key for key in required_keys if not parsed.get(key)]
    if missing:
        raise ValueError(f"Missing auth context fields: {', '.join(missing)}")
    return parsed


def parse_scope_context(scope_context: Optional[str]) -> dict:
    parsed = parse_json_object(scope_context)
    return {
        "org_id": parsed.get("org_id"),
        "project_id": parsed.get("project_id"),
        "agent_id": parsed.get("agent_id"),
        "run_id": parsed.get("run_id"),
        "session_id": parsed.get("session_id"),
        "include_shared": parsed.get("include_shared", True),
        "visibility": parsed.get("visibility", "private"),
    }


def parse_metadata(metadata: Optional[str]) -> dict:
    import contextlib

    parsed: dict = {}
    if metadata:
        with contextlib.suppress(Exception):
            parsed = json.loads(metadata) or {}
    return parsed
