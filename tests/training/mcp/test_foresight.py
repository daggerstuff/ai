"""Tests for the Foresight memory write path.

The module is a best-effort provenance side channel: it must never raise
into the job result, it must forward the session id from the initialize
handshake, and it must fail closed when Foresight answers with an error.
These tests pin that contract against a fake httpx client.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from training.mcp import foresight


class _Response:
    def __init__(self, text: str = "", session_id: str | None = None) -> None:
        self.text = text
        self.headers: dict[str, str] = {} if session_id is None else {"mcp-session-id": session_id}

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self, responses: list[_Response], *, raises: Exception | None = None) -> None:
        self._responses = list(responses)
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def post(self, url: str, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> _Response:
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._raises is not None:
            raise self._raises
        return self._responses.pop(0)


class _Settings:
    def __init__(self, env: dict[str, str] | None = None) -> None:
        self.env = env or {}


def _install(monkeypatch: pytest.MonkeyPatch, client: _Client) -> _Client:
    monkeypatch.setattr(foresight.httpx, "Client", lambda **_kwargs: client)
    return client


def test_endpoint_prefers_the_configured_url() -> None:
    assert foresight._endpoint(_Settings({"FORESIGHT_URL": "http://memory:9000/mcp"})) == "http://memory:9000/mcp"
    assert foresight._endpoint(_Settings()) == foresight.FORESIGHT_DEFAULT_URL


def test_parse_sse_accepts_plain_json() -> None:
    assert foresight._parse_sse('{"result": {"ok": true}}') == {"result": {"ok": True}}


def test_parse_sse_reads_the_first_data_frame() -> None:
    body = 'event: message\ndata: {"result": {"ok": true}}\n\n'

    assert foresight._parse_sse(body) == {"result": {"ok": True}}


def test_parse_sse_skips_a_malformed_frame_and_keeps_looking() -> None:
    body = 'data: {not json}\ndata: {"result": {"ok": true}}\n'

    assert foresight._parse_sse(body) == {"result": {"ok": True}}


def test_parse_sse_returns_none_for_empty_garbage_or_non_mapping() -> None:
    assert foresight._parse_sse("") is None
    assert foresight._parse_sse("   \n") is None
    assert foresight._parse_sse("not json at all") is None
    assert foresight._parse_sse("[1, 2, 3]") is None
    assert foresight._parse_sse('data: "a bare string"') is None


def test_store_terminal_memory_forwards_the_session_and_the_job_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install(
        monkeypatch,
        _Client(
            [
                _Response(session_id="session-7"),
                _Response(),
                _Response(text='{"result": {"ok": true}}'),
            ]
        ),
    )
    job = {"job_id": "job-42", "name": "sft-nightly", "kind": "train", "status": "succeeded"}

    report = foresight.store_terminal_memory(_Settings(), job)

    assert report["stored"] is True
    assert [call["json"]["method"] for call in client.calls] == [
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    assert "id" not in client.calls[1]["json"]
    assert client.calls[1]["headers"]["mcp-session-id"] == "session-7"
    assert client.calls[2]["headers"]["mcp-session-id"] == "session-7"
    params = client.calls[2]["json"]["params"]
    assert params["name"] == "manage_memories"
    options = params["arguments"]["params"]["options"]
    assert options["action"] == "store"
    content = json.loads(options["content"])
    assert content["training_job_id"] == "job-42"
    assert content["status"] == "succeeded"
    assert options["options"]["tags"] == [
        "training_job_outcome",
        "job:job-42",
        "status:succeeded",
    ]


def test_store_terminal_memory_fails_closed_on_a_jsonrpc_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _Client([_Response(), _Response(), _Response(text='{"error": {"message": "boom"}}')]))

    report = foresight.store_terminal_memory(_Settings(), {"job_id": "job-1", "status": "failed"})

    assert report["stored"] is False
    assert "boom" in report["error"]


def test_store_terminal_memory_fails_closed_on_a_tool_level_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        _Client(
            [
                _Response(),
                _Response(),
                _Response(text='{"result": {"isError": true, "content": [{"text": "rejected"}]}}'),
            ]
        ),
    )

    report = foresight.store_terminal_memory(_Settings(), {"job_id": "job-2", "status": "failed"})

    assert report["stored"] is False
    assert report["error"] == "rejected"


def test_store_terminal_memory_never_raises_when_the_transport_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _Client([], raises=RuntimeError("connection refused")))

    report = foresight.store_terminal_memory(_Settings(), {"job_id": "job-3", "status": "failed"})

    assert report["stored"] is False
    assert report["error"] == "RuntimeError: connection refused"
    assert report["url"] == foresight.FORESIGHT_DEFAULT_URL
