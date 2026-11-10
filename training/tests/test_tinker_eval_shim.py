"""Tests for the OpenAI-compatible Tinker sampling shim."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from training import tinker_eval_shim as shim


class FakeSampler:
    """Return a fixed reply and record what the handler asked for."""

    def __init__(self, text: str = "Hold the line.") -> None:
        self.text = text
        self.requests: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> shim.Generation:
        self.requests.append(kwargs)
        return shim.Generation(
            text=self.text,
            prompt_tokens=11,
            completion_tokens=4,
            finish_reason="stop",
        )


class ExplodingSampler:
    """Raise on every call so the handler has to answer 502."""

    def generate(self, **kwargs: Any) -> shim.Generation:
        raise RuntimeError("sampler is down")


def config(token: str | None = None) -> shim.ShimConfig:
    return shim.ShimConfig(weights="weights://fake", model_name="tinker-stub", token=token)


def start(sampler: Any, cfg: shim.ShimConfig) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), shim.make_handler(sampler, cfg))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def post(url: str, body: dict[str, Any], *, token: str | None = None) -> tuple[int, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_render_prompt_rejects_an_empty_conversation() -> None:
    with pytest.raises(shim.ShimError, match="messages"):
        shim.render_prompt(object(), [])


def test_render_prompt_unwraps_a_hf_batch_encoding() -> None:
    class EncodingTokenizer:
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> dict[str, list[int]]:
            assert kwargs["add_generation_prompt"] is True
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    assert shim.render_prompt(EncodingTokenizer(), [{"role": "user", "content": "hi"}]) == [1, 2, 3]


def test_parse_chat_request_normalizes_stop_and_defaults() -> None:
    request = shim.parse_chat_request(
        {"messages": [{"role": "user", "content": "hi"}], "stop": "END"},
        default_max_tokens=64,
    )

    assert request["stop"] == ["END"]
    assert request["max_tokens"] == 64
    assert request["temperature"] == 1.0


def test_parse_chat_request_rejects_tool_requests() -> None:
    with pytest.raises(shim.ShimError, match="tool calls"):
        shim.parse_chat_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function"}],
            },
            default_max_tokens=64,
        )


def test_parse_chat_request_rejects_multiple_choices() -> None:
    with pytest.raises(shim.ShimError, match="n=1"):
        shim.parse_chat_request(
            {"messages": [{"role": "user", "content": "hi"}], "n": 2},
            default_max_tokens=64,
        )


def test_chat_completion_round_trip_uses_the_openai_shape() -> None:
    sampler = FakeSampler()
    server, base = start(sampler, config())
    try:
        status, payload = post(
            f"{base}/v1/chat/completions",
            {
                "model": "tinker-stub",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0,
                "max_tokens": 32,
                "stop": ["END"],
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "tinker-stub"
    choice = payload["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hold the line."}
    assert choice["finish_reason"] == "stop"
    assert payload["usage"] == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
    assert sampler.requests == [
        {
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.0,
            "max_tokens": 32,
            "stop": ["END"],
            "chat_template_kwargs": None,
        }
    ]


def test_chat_completion_requires_the_bearer_token_when_configured() -> None:
    server, base = start(FakeSampler(), config(token="s3cret"))
    body = {"messages": [{"role": "user", "content": "hello"}]}
    try:
        unauthorized, error = post(f"{base}/v1/chat/completions", body)
        authorized, payload = post(f"{base}/v1/chat/completions", body, token="s3cret")
    finally:
        server.shutdown()
        server.server_close()

    assert unauthorized == 401
    assert "bearer token" in error["error"]["message"]
    assert authorized == 200
    assert payload["choices"][0]["message"]["content"] == "Hold the line."


def test_sampler_failures_surface_as_bad_gateway() -> None:
    server, base = start(ExplodingSampler(), config())
    try:
        status, payload = post(f"{base}/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    finally:
        server.shutdown()
        server.server_close()

    assert status == 502
    assert "RuntimeError" in payload["error"]["message"]


def test_health_and_model_routes_answer() -> None:
    server, base = start(FakeSampler(), config())
    try:
        with urllib.request.urlopen(f"{base}/healthz", timeout=10) as response:
            health = json.loads(response.read())
        with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as response:
            models = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()

    assert health == {"status": "ok", "model": "tinker-stub"}
    assert models["data"][0]["id"] == "tinker-stub"


def test_unknown_routes_are_rejected() -> None:
    server, base = start(FakeSampler(), config())
    try:
        status, payload = post(f"{base}/v1/completions", {"messages": []})
    finally:
        server.shutdown()
        server.server_close()

    assert status == 404
    assert "unknown route" in payload["error"]["message"]


def test_config_from_run_record_reads_the_sampler_weights(tmp_path: Path) -> None:
    record = tmp_path / shim.RUN_NAME
    record.write_text(
        json.dumps(
            {
                "run_name": "tinker-v19",
                "base_model": "Qwen/Qwen3-8B",
                "sampling_path": "sampler://tinker-v19",
            }
        ),
        encoding="utf-8",
    )

    cfg = shim.config_from_run_record(record, max_tokens=128)

    assert cfg.weights == "sampler://tinker-v19"
    assert cfg.model_name == "tinker-v19"
    assert cfg.base_model == "Qwen/Qwen3-8B"
    assert cfg.max_tokens == 128


def test_config_from_run_record_rejects_a_run_without_weights(tmp_path: Path) -> None:
    record = tmp_path / shim.RUN_NAME
    record.write_text(json.dumps({"run_name": "tinker-v19"}), encoding="utf-8")

    with pytest.raises(shim.ShimError, match="sampling_path"):
        shim.config_from_run_record(record)


def test_config_from_run_record_rejects_a_missing_record(tmp_path: Path) -> None:
    with pytest.raises(shim.ShimError, match="missing"):
        shim.config_from_run_record(tmp_path / shim.RUN_NAME)
