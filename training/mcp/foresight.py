"""Foresight memory writes for terminal job states (PIX-5000 provenance).

The epic requires that ``searchMemories(training_job_id:...)`` returns real data
for every job. Relying on an agent to poll a job means jobs nobody polls leave
no trace, so the control plane writes the terminal fact itself.

This module is deliberately self-contained and failure-tolerant: Foresight is a
*provenance* side channel, never a dependency of job execution. Every call is
best-effort and returns a small dict instead of raising.

It speaks the Foresight MCP Streamable HTTP transport directly over ``httpx``
(already a transitive dependency of fastmcp) rather than adding ``fastmcp``
client machinery to the write path.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

FORESIGHT_DEFAULT_URL = "http://127.0.0.1:8764/mcp"
_TIMEOUT_S = 10.0


def _endpoint(settings: Any) -> str:
    return str(settings.env.get("FORESIGHT_URL") or FORESIGHT_DEFAULT_URL)


def _initialize(client: httpx.Client, url: str) -> str | None:
    """Perform the MCP initialize handshake, returning the session id."""
    resp = client.post(
        url,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "training-infra-mcp", "version": "1.0.0"},
            },
        },
        headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    session_id = resp.headers.get("mcp-session-id")
    return session_id if isinstance(session_id, str) else None


def _parse_sse(body: str) -> dict[str, Any] | None:
    """Extract the JSON-RPC payload from either SSE or plain JSON."""
    if not body.strip():
        return None
    for line in body.splitlines():
        if line.startswith("data:"):
            with_suppress = line[5:].strip()
            if with_suppress:
                try:
                    payload = json.loads(with_suppress)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return payload
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def store_terminal_memory(settings: Any, job: dict[str, Any]) -> dict[str, Any]:
    """Write a terminal job outcome to Foresight. Never raises.

    Returns ``{"stored": bool, ...}`` describing what happened so callers can
    audit it, but a Foresight outage must not affect the job result.
    """
    url = _endpoint(settings)
    status = str(job.get("status") or "unknown")
    content = json.dumps(
        {
            "type": "training_job_outcome",
            "training_job_id": job.get("job_id"),
            "job_id": job.get("job_id"),
            "recipe": job.get("name"),
            "kind": job.get("kind"),
            "status": status,
            "stage": job.get("stage"),
            "error": job.get("error"),
            "finished_at": job.get("finished_at"),
            "requested_by": job.get("requested_by"),
            "backend_id": job.get("backend_id"),
            "cost_estimate": job.get("cost_estimate"),
            "artifact_refs": job.get("artifact_refs"),
        },
        default=str,
    )
    arguments = {
        "params": {
            "options": {
                "action": "store",
                "content": content,
                "options": {
                    "category": "fact",
                    "scope": "project",
                    "retention": "long_term",
                    "importance": 0.8,
                    "tags": [
                        "training_job_outcome",
                        f"job:{job.get('job_id')}",
                        f"status:{status}",
                    ],
                },
            }
        }
    }
    try:
        with httpx.Client(timeout=_TIMEOUT_S) as client:
            session_id = _initialize(client, url)
            headers = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
            if session_id:
                headers["mcp-session-id"] = session_id
            resp = client.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "manage_memories", "arguments": arguments},
                },
                headers=headers,
            )
            resp.raise_for_status()
            payload = _parse_sse(resp.text) or {}
    except Exception as exc:  # provenance must never break the job result
        return {"stored": False, "error": f"{type(exc).__name__}: {exc}", "url": url}
    result = payload.get("result") if isinstance(payload, dict) else None
    if isinstance(payload, dict) and payload.get("error"):
        return {"stored": False, "error": str(payload["error"]), "url": url}
    if isinstance(result, dict) and result.get("isError"):
        text = " ".join(str(c.get("text", "")) for c in (result.get("content") or []) if isinstance(c, dict))
        return {"stored": False, "error": text.strip() or "foresight returned isError", "url": url}
    return {"stored": True, "url": url, "result": result}
