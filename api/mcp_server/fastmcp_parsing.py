from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedAuthContext:
    actor_id: str
    timestamp: str
    nonce: str
    signature: str


@dataclass(frozen=True)
class ParsedScopeContext:
    org_id: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    include_shared: bool = True
    visibility: str = "private"


def parse_json_object(context_json: str | None) -> dict:
    if not context_json:
        return {}
    parsed = json.loads(context_json)
    if not isinstance(parsed, dict):
        raise ValueError("Context payload must be a JSON object.")
    return parsed


def parse_auth_context(auth_context: str) -> ParsedAuthContext:
    parsed = parse_json_object(auth_context)
    required_keys = ("actor_id", "timestamp", "nonce", "signature")
    missing = [key for key in required_keys if not parsed.get(key)]
    if missing:
        raise ValueError(f"Missing auth context fields: {', '.join(missing)}")
    return ParsedAuthContext(
        actor_id=str(parsed["actor_id"]),
        timestamp=str(parsed["timestamp"]),
        nonce=str(parsed["nonce"]),
        signature=str(parsed["signature"]),
    )


def parse_scope_context(scope_context: str | None) -> ParsedScopeContext:
    parsed = parse_json_object(scope_context)
    return ParsedScopeContext(
        org_id=parsed.get("org_id"),
        project_id=parsed.get("project_id"),
        agent_id=parsed.get("agent_id"),
        run_id=parsed.get("run_id"),
        session_id=parsed.get("session_id"),
        include_shared=parsed.get("include_shared", True),
        visibility=parsed.get("visibility", "private"),
    )


def parse_metadata(metadata: str | None) -> dict:

    parsed: dict = {}
    if metadata:
        with contextlib.suppress(Exception):
            parsed = json.loads(metadata) or {}
    return parsed
