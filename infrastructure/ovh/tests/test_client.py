"""Tests for the OVHcloud API client."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ai.infrastructure.ovh.client import (
    MAX_ATTEMPTS,
    MissingTokenError,
    OvhApiError,
    OvhClient,
    find_dotenv,
    read_env_file,
    resolve_token,
    resolve_token_with_source,
)

TOKEN = "tok-abc123"
ME_PAYLOAD = {"nichandle": "ab12345-ovh", "email": "user@example.com"}
BACKOFF_CALLS = 2
UNAUTHORIZED = 401
BAD_GATEWAY = 502
NOT_FOUND = 404


class _Recorder:
    """httpx.MockTransport handler that records requests and replays responses."""

    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def calls(self) -> int:
        return len(self.requests)


def _client(recorder: _Recorder) -> OvhClient:
    client = OvhClient(TOKEN, transport=httpx.MockTransport(recorder))
    client.sleep = lambda _seconds: None
    return client


# -- token resolution ------------------------------------------------------
def test_env_token_wins(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("OVH_TOKEN=from-file\n", encoding="utf-8")
    token, source = resolve_token_with_source(env={"OVH_TOKEN": "from-env"}, dotenv_path=dotenv)
    assert token == "from-env"
    assert source == "environment OVH_TOKEN"


def test_dotenv_token_is_used_when_env_is_empty(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("# comment\nOVH_TOKEN='from-file'\n", encoding="utf-8")
    token, source = resolve_token_with_source(env={}, dotenv_path=dotenv)
    assert token == "from-file"
    assert source.endswith(".env:OVH_TOKEN")


def test_ovhai_context_is_the_last_resort(tmp_path: Path) -> None:
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"access_tokens": {"US-EAST-VA": "from-ovhai"}}), encoding="utf-8")
    missing_env = tmp_path / "missing.env"
    token, source = resolve_token_with_source(env={}, dotenv_path=missing_env, ovhai_context=context)
    assert token == "from-ovhai"
    assert "access_tokens[US-EAST-VA]" in source


def test_missing_token_raises_with_guidance(tmp_path: Path) -> None:
    with pytest.raises(MissingTokenError, match="OVH_TOKEN"):
        resolve_token(
            env={},
            dotenv_path=tmp_path / "nope.env",
            ovhai_context=tmp_path / "nope.json",
        )


def test_read_env_file_handles_export_quotes_and_comments(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        '# leading comment\nexport OVH_TOKEN="quoted"\nOTHER=plain\nbroken line\n',
        encoding="utf-8",
    )
    values = read_env_file(dotenv)
    assert values["OVH_TOKEN"] == "quoted"
    assert values["OTHER"] == "plain"
    assert "broken line" not in values


def test_find_dotenv_walks_up(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OVH_TOKEN=x\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_dotenv(nested) == tmp_path / ".env"


# -- request plumbing ------------------------------------------------------
def test_get_sends_bearer_token_and_version_prefix() -> None:
    recorder = _Recorder([httpx.Response(200, json=ME_PAYLOAD)])
    with _client(recorder) as client:
        assert client.get("/me") == ME_PAYLOAD
    request = recorder.requests[0]
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert str(request.url) == "https://api.us.ovhcloud.com/1.0/me"


def test_api_version_can_be_overridden_per_call() -> None:
    recorder = _Recorder([httpx.Response(200, json=[])])
    with _client(recorder) as client:
        client.get("/me/identity", api_version="v2")
    assert str(recorder.requests[0].url) == "https://api.us.ovhcloud.com/v2/me/identity"


def test_none_valued_params_are_dropped() -> None:
    recorder = _Recorder([httpx.Response(200, json=[])])
    with _client(recorder) as client:
        client.get("/me/api/application", params={"limit": 10, "offset": None})
    assert str(recorder.requests[0].url).endswith("?limit=10")


def test_post_sends_a_json_body() -> None:
    recorder = _Recorder([httpx.Response(200, json={"id": 7})])
    with _client(recorder) as client:
        assert client.post("/me/api/application", {"name": "cli"}) == {"id": 7}
    assert json.loads(recorder.requests[0].content) == {"name": "cli"}


def test_empty_response_body_decodes_to_none() -> None:
    recorder = _Recorder([httpx.Response(204)])
    with _client(recorder) as client:
        assert client.delete("/me/api/application/1") is None


# -- error handling --------------------------------------------------------
def test_unauthorized_is_not_retried_and_keeps_the_api_class() -> None:
    recorder = _Recorder(
        [httpx.Response(401, json={"class": "Client::Unauthorized", "message": "You must login first"})]
    )
    with _client(recorder) as client, pytest.raises(OvhApiError) as caught:
        client.get("/me")
    error = caught.value
    assert error.status_code == UNAUTHORIZED
    assert error.klass == "Client::Unauthorized"
    assert error.message == "You must login first"
    assert error.is_auth_error is True
    assert recorder.calls == 1


def test_server_error_is_retried_then_succeeds() -> None:
    recorder = _Recorder([httpx.Response(503), httpx.Response(200, json=ME_PAYLOAD)])
    with _client(recorder) as client:
        assert client.get("/me") == ME_PAYLOAD
    assert recorder.calls == BACKOFF_CALLS


def test_a_post_is_not_retried_after_a_5xx() -> None:
    # Re-sending a POST can create the resource twice, so a 5xx surfaces at once.
    recorder = _Recorder([httpx.Response(503)])
    with _client(recorder) as client, pytest.raises(OvhApiError):
        client.post("/cloud/project/x/instance", {"flavor": "b3-8"})
    assert recorder.calls == 1


def test_retries_are_bounded() -> None:
    recorder = _Recorder([httpx.Response(502) for _ in range(MAX_ATTEMPTS)])
    with _client(recorder) as client, pytest.raises(OvhApiError) as caught:
        client.get("/me")
    assert caught.value.status_code == BAD_GATEWAY
    assert recorder.calls == MAX_ATTEMPTS


def test_transport_error_is_retried_then_surfaces() -> None:
    recorder = _Recorder([httpx.ConnectError("boom") for _ in range(MAX_ATTEMPTS)])
    with _client(recorder) as client, pytest.raises(RuntimeError, match="failed after 3 attempts"):
        client.get("/me")
    assert recorder.calls == MAX_ATTEMPTS


def test_non_json_error_body_still_produces_a_useful_message() -> None:
    recorder = _Recorder([httpx.Response(404, text="Not Found")])
    with _client(recorder) as client, pytest.raises(OvhApiError) as caught:
        client.get("/nope")
    assert caught.value.status_code == NOT_FOUND
    assert "Not Found" in caught.value.message


# -- pagination ------------------------------------------------------------
def test_paginate_walks_pages_until_a_short_page() -> None:
    page_one = [{"id": index} for index in range(2)]
    page_two = [{"id": 2}]
    recorder = _Recorder([httpx.Response(200, json=page_one), httpx.Response(200, json=page_two)])
    with _client(recorder) as client:
        rows = list(client.paginate("/me/api/application", limit=2))
    assert rows == [*page_one, *page_two]
    assert [str(r.url) for r in recorder.requests] == [
        "https://api.us.ovhcloud.com/1.0/me/api/application?limit=2&offset=0",
        "https://api.us.ovhcloud.com/1.0/me/api/application?limit=2&offset=2",
    ]
