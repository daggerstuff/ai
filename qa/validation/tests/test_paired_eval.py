"""Offline contract tests for paired inference; no network or model weights."""

from __future__ import annotations

import hashlib
import json
from email.message import Message
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from ai.qa.validation.therapy_bench.paired_eval import (
    ConservativeAsciiByteCounter,
    EvaluationInputError,
    OpenAIChatCompletions,
    RunnerOptions,
    RunRequest,
    _run_mock_tool,
    build_dry_run_report,
    load_cases,
    run_paired,
    score_state_checks,
)

EXPECTED_CALLS = 4
CASES_PER_ARM = 2
TOTAL_USAGE_TOKENS = 20
RESUME_SUCCESS_COUNT = 3
OPENAI_RESPONSE_TOKENS = 5
HTTP_BAD_REQUEST = 400
MAX_SAFE_EXCERPT_CHARS = 1024
TWO_COMPLETION_CALLS = 2
AGGREGATED_TOOL_TOKENS = 34
TWO_CASE_TOOL_ROUND_COST = 0.009216
FAILED_TOOL_LOOP_COST = 0.000032
MAX_ALLOWED_TOOL_ROUNDS = 8


def write_inputs(tmp_path: Path, cases: list[dict[str, Any]] | None = None) -> tuple[Path, Path]:
    cases_path = tmp_path / "cases.jsonl"
    rows = cases or [
        {
            "id": "format-1",
            "scenario": "synthetic_format",
            "messages": [{"role": "user", "content": "Return JSON with keys answer and source."}],
            "diagnostic_checks": [{"type": "json_required_keys", "required_keys": ["answer", "source"]}],
        },
        {
            "id": "tool-1",
            "scenario": "synthetic_tool",
            "messages": [{"role": "user", "content": "Look up item A-1."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_mock_catalog",
                        "description": "Read-only synthetic catalog lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "diagnostic_checks": [{"type": "tool_called", "tool_name": "lookup_mock_catalog"}],
        },
    ]
    cases_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "offline-test",
                "decoding": {"temperature": 0, "max_tokens": 128, "stop": []},
                "pricing": {
                    "input_usd_per_million_tokens": None,
                    "output_usd_per_million_tokens": None,
                    "hard_cap_usd": None,
                    "require_usage": False,
                    "require_finish_reason": False,
                },
                "arms": [
                    {
                        "id": "base",
                        "provider": "fake",
                        "base_url": "https://base.example/v1",
                        "model": "base-model",
                        "api_key_env": "BASE_TEST_KEY",
                    },
                    {
                        "id": "sft",
                        "provider": "fake",
                        "base_url": "https://sft.example/v1",
                        "model": "sft-model",
                        "api_key_env": "SFT_TEST_KEY",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return cases_path, config_path


class FakeTransport:
    def __init__(self, failures: set[tuple[str, str]] | None = None) -> None:
        self.calls: list[tuple[str, str, list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]] = []
        self.failures = failures or set()

    def complete(
        self,
        arm: dict[str, str],
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        decoding: dict[str, Any],
    ) -> dict[str, Any]:
        case_id = "format-1" if "JSON" in messages[-1]["content"] else "tool-1"
        self.calls.append((arm["id"], case_id, messages, tools, decoding))
        if (arm["id"], case_id) in self.failures:
            raise RuntimeError("transient transport failure")
        if case_id == "format-1":
            content = '{"answer":"ready","source":"local note"}'
            tool_calls = []
        else:
            content = ""
            tool_calls = [{"id": "call-1", "name": "lookup_mock_catalog", "arguments": '{"query":"A-1"}'}]
        return {
            "response": content,
            "tool_calls": tool_calls,
            "finish_reason": "stop" if content else "tool_calls",
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }


def run(
    paths: tuple[Path, Path, Path],
    transport: FakeTransport,
    options: RunnerOptions | None = None,
    token_counter: Any = None,
):
    request = RunRequest(*paths, options or RunnerOptions())
    return run_paired(request, transport=transport, token_counter=token_counter)


def test_pairs_identical_cases_and_decoding_and_keeps_checks_private(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    transport = FakeTransport()
    results = tmp_path / "results"

    result = run((cases_path, config_path, results), transport)

    assert result["status"] == "complete"
    assert result["comparison_mode"] == "paired"
    assert result["estimated_cost_usd"] is None
    assert result["latency_ms_by_arm"]["base"]["count"] == CASES_PER_ARM
    assert result["latency_ms_by_arm"]["base"]["p95_ms"] >= result["latency_ms_by_arm"]["base"]["p50_ms"]
    assert result["usage_totals_by_arm"]["sft"]["total_tokens"] == TOTAL_USAGE_TOKENS * CASES_PER_ARM
    assert result["finish_reason_length_count_by_arm"]["base"] == 0
    assert len(transport.calls) == EXPECTED_CALLS
    by_case = {}
    for arm_id, case_id, messages, tools, decoding in transport.calls:
        by_case.setdefault(case_id, []).append((arm_id, messages, tools, decoding))
    assert set(by_case) == {"format-1", "tool-1"}
    for pair in by_case.values():
        assert [item[0] for item in pair] == ["base", "sft"]
        assert pair[0][1:] == pair[1][1:]
    output = (results / "results.jsonl").read_text(encoding="utf-8")
    assert '"required_keys":[' not in output
    assert "diagnostic_unvalidated" in output
    assert '"actual_billed_usd":null' in output
    rows = [json.loads(line) for line in output.splitlines()]
    assert all(row["prompt_sha256"] and row["model_sha256"] for row in rows)
    assert all(row["latency_ms"] >= 0 and row["retry_count"] == 0 for row in rows)
    assert all(row["usage"]["total_tokens"] == TOTAL_USAGE_TOKENS for row in rows)
    assert (results / "manifest.json").stat().st_mode & 0o077 == 0


def test_chat_template_thinking_option_is_preserved_in_config_and_calls(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["decoding"]["chat_template_kwargs"] = {"enable_thinking": False}
    config["decoding"]["response_format"] = {"type": "json_object"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    transport = FakeTransport()

    result = run((cases_path, config_path, tmp_path / "results"), transport)

    assert result["status"] == "complete"
    assert all(call[4]["chat_template_kwargs"] == {"enable_thinking": False} for call in transport.calls)


def test_resume_retries_only_failed_case_and_keeps_partial_success(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    results = tmp_path / "results"
    first = FakeTransport(failures={("sft", "tool-1")})
    outcome = run((cases_path, config_path, results), first)
    assert outcome["status"] == "incomplete"
    assert outcome["success_count"] == RESUME_SUCCESS_COUNT
    assert outcome["error_count"] == 1

    retry = FakeTransport()
    outcome = run((cases_path, config_path, results), retry)
    assert outcome["status"] == "complete"
    assert [(call[0], call[1]) for call in retry.calls] == [("sft", "tool-1")]
    rows = [json.loads(line) for line in (results / "results.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows if row["arm_id"] == "sft" and row["case_id"] == "tool-1"] == [
        "error",
        "success",
    ]


def test_http_client_error_stops_before_next_case(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)

    class Http400Transport(FakeTransport):
        def complete(
            self,
            arm: dict[str, str],
            messages: list[dict[str, str]],
            tools: list[dict[str, Any]],
            decoding: dict[str, Any],
        ) -> dict[str, Any]:
            self.calls.append((arm["id"], "format-1", messages, tools, decoding))
            raise RuntimeError("http_400")

    transport = Http400Transport()
    outcome = run(
        (cases_path, config_path, tmp_path / "results"),
        transport,
        RunnerOptions("base"),
    )
    rows = [json.loads(line) for line in (tmp_path / "results" / "results.jsonl").read_text().splitlines()]
    assert outcome["status"] == "incomplete"
    assert len(transport.calls) == 1
    assert rows[0]["error_code"] == "http_400"
    assert rows[0]["fatal"] is True


def test_resume_rejects_changed_inputs_or_arm_selection(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    results = tmp_path / "results"
    run((cases_path, config_path, results), FakeTransport(), RunnerOptions("base"))
    cases_path.write_text(cases_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="identity changed"):
        run((cases_path, config_path, results), FakeTransport(), RunnerOptions("base"))


def test_tool_round_bound_allows_eight_and_rejects_nine(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    accepted = RunRequest(
        cases_path,
        config_path,
        tmp_path / "accepted",
        RunnerOptions("base", max_tool_rounds=MAX_ALLOWED_TOOL_ROUNDS),
    )
    report = build_dry_run_report(accepted, token_counter=StaticTokenCounter(1))
    assert report["max_tool_rounds"] == MAX_ALLOWED_TOOL_ROUNDS

    rejected = RunRequest(cases_path, config_path, tmp_path / "rejected", RunnerOptions("base", max_tool_rounds=9))
    with pytest.raises(EvaluationInputError, match=r"max_tool_rounds must be an integer in \[0, 8\]"):
        build_dry_run_report(rejected, token_counter=StaticTokenCounter(1))


def test_rendered_glimmer_prompt_is_rejected_for_chat_transport(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(
        tmp_path,
        [
            {
                "id": "rendered",
                "scenario": "legacy_bundle",
                "prompt": "<|begin_of_text|><|start|>user<|message|>hello<|eot|>",
            }
        ],
    )
    with pytest.raises(EvaluationInputError, match="provider-rendered prompt"):
        run((cases_path, config_path, tmp_path / "results"), FakeTransport())


class StaticTokenCounter:
    def __init__(self, count: int) -> None:
        self.value = count

    def count(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> int:
        _ = messages, tools
        return self.value


class ScriptedTransport:
    def __init__(self, completions: list[dict[str, Any]]) -> None:
        self.completions = list(completions)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(
        self,
        arm: dict[str, str],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        decoding: dict[str, Any],
    ) -> dict[str, Any]:
        _ = arm, tools, decoding
        self.calls.append(messages)
        completion = self.completions.pop(0)
        if isinstance(completion, Exception):
            raise completion
        return completion


def test_input_token_cap_rejects_before_any_transport_call(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    transport = FakeTransport()
    with pytest.raises(EvaluationInputError, match="token cap exceeded"):
        run(
            (cases_path, config_path, tmp_path / "results"),
            transport,
            RunnerOptions(max_input_tokens=512),
            StaticTokenCounter(513),
        )
    assert not transport.calls


def test_preflight_counts_null_content_assistant_tool_call_messages(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    cases_path.write_text(json.dumps(cases[1]) + "\n", encoding="utf-8")
    tool_completion = {
        "response": "",
        "tool_calls": [{"id": "call-1", "name": "lookup_mock_catalog", "arguments": '{"query":"A-1"}'}],
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }
    final_completion = {
        "response": "A-1 is available.",
        "tool_calls": [],
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }
    transport = ScriptedTransport([tool_completion, final_completion])

    result = run(
        (cases_path, config_path, tmp_path / "null-content-tool-results"),
        transport,
        RunnerOptions("base", max_cases=1, max_input_tokens=2048, max_tool_rounds=1),
        ConservativeAsciiByteCounter(),
    )

    assert result["status"] == "complete"
    assert len(transport.calls) == TWO_COMPLETION_CALLS
    assert transport.calls[1][-2]["content"] is None


def test_conservative_preflight_includes_chat_and_tool_framing(tmp_path: Path) -> None:
    cases_path, _ = write_inputs(tmp_path)
    counts = {
        case["id"]: ConservativeAsciiByteCounter().count(case["messages"], case.get("tools", []))
        for case in load_cases(cases_path)
    }
    assert all(count > 0 for count in counts.values())
    assert counts["tool-1"] > counts["format-1"]


def test_provider_usage_above_preflight_stops_after_one_call(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    config = json.loads(config_path.read_text())
    config["pricing"] = {
        "input_usd_per_million_tokens": 0.25,
        "output_usd_per_million_tokens": 1.25,
        "hard_cap_usd": 0.01,
        "require_usage": True,
        "require_finish_reason": True,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class OversizedUsageTransport(FakeTransport):
        def complete(
            self,
            arm: dict[str, str],
            messages: list[dict[str, str]],
            tools: list[dict[str, Any]],
            decoding: dict[str, Any],
        ) -> dict[str, Any]:
            self.calls.append((arm["id"], "format-1", messages, tools, decoding))
            return {
                "response": "ok",
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 513, "completion_tokens": 1, "total_tokens": 514},
            }

    transport = OversizedUsageTransport()
    outcome = run(
        (cases_path, config_path, tmp_path / "results"),
        transport,
        RunnerOptions("base", max_input_tokens=512),
        StaticTokenCounter(512),
    )
    assert outcome["status"] == "incomplete"
    assert len(transport.calls) == 1
    record = json.loads((tmp_path / "results" / "results.jsonl").read_text().splitlines()[0])
    assert record["error_type"] == "ProviderUsageExceededPreflight"
    assert record["fatal"] is True


def test_hard_cost_cap_uses_input_preflight_and_output_limit(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    config = json.loads(config_path.read_text())
    config["pricing"] = {
        "input_usd_per_million_tokens": 0.25,
        "output_usd_per_million_tokens": 1.25,
        "hard_cap_usd": 0.0005,
        "require_usage": True,
        "require_finish_reason": True,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="exceeds hard cap"):
        run(
            (cases_path, config_path, tmp_path / "results"),
            FakeTransport(),
            RunnerOptions(max_input_tokens=512),
            StaticTokenCounter(512),
        )


def test_missing_required_metadata_stops_after_one_billed_response(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    config = json.loads(config_path.read_text())
    config["pricing"] = {
        "input_usd_per_million_tokens": 0.25,
        "output_usd_per_million_tokens": 1.25,
        "hard_cap_usd": 0.01,
        "require_usage": True,
        "require_finish_reason": True,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class MissingUsageTransport(FakeTransport):
        def complete(
            self,
            arm: dict[str, str],
            messages: list[dict[str, str]],
            tools: list[dict[str, Any]],
            decoding: dict[str, Any],
        ) -> dict[str, Any]:
            self.calls.append((arm["id"], "format-1", messages, tools, decoding))
            return {"response": "ok", "tool_calls": [], "finish_reason": "stop", "usage": None}

    transport = MissingUsageTransport()
    result = run(
        (cases_path, config_path, tmp_path / "results"),
        transport,
        RunnerOptions("base", max_input_tokens=512),
        StaticTokenCounter(20),
    )
    assert result["status"] == "incomplete"
    assert len(transport.calls) == 1
    record = json.loads((tmp_path / "results" / "results.jsonl").read_text().splitlines()[0])
    assert record["error_type"] == "RequiredProviderMetadataMissing"
    assert record["fatal"] is True
    assert record["retry_count"] == 0


def test_openai_transport_keeps_bearer_secret_out_of_run_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "never-persist-this-token"
    monkeypatch.setenv("BASE_TEST_KEY", secret)
    cases_path, config_path = write_inputs(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["arms"][0]["project"] = "team/project"
    config["decoding"]["chat_template_kwargs"] = {"enable_thinking": False}
    config["decoding"]["response_format"] = {"type": "json_object"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    observed: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                }
            ).encode()

    def fake_urlopen(request: Any, **_kwargs: Any) -> FakeResponse:
        observed["authorization"] = request.get_header("Authorization")
        observed["url"] = request.full_url
        observed["project"] = request.get_header("Openai-project")
        observed["body"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    transport = OpenAIChatCompletions()
    arm = config["arms"][0]
    output = transport.complete(arm, [{"role": "user", "content": "hello"}], [], config["decoding"])
    results = tmp_path / "results"
    run((cases_path, config_path, results), FakeTransport(), RunnerOptions("base"))
    persisted = (results / "manifest.json").read_text() + (results / "results.jsonl").read_text()
    assert observed["authorization"] == f"Bearer {secret}"
    assert observed["url"] == "https://base.example/v1/chat/completions"
    assert observed["project"] == "team/project"
    assert observed["body"]["model"] == "base-model"
    assert observed["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert observed["body"]["response_format"] == {"type": "json_object"}
    assert "stop" not in observed["body"]
    assert output["usage"]["total_tokens"] == OPENAI_RESPONSE_TOKENS
    transport.complete(
        arm,
        [{"role": "user", "content": "hello"}],
        [],
        {**config["decoding"], "stop": ["END"]},
    )
    assert observed["body"]["stop"] == ["END"]
    assert secret not in persisted
    assert secret not in config_path.read_text()


def test_provider_http_error_persists_only_allowlisted_redacted_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "never-persist-this-token"
    monkeypatch.setenv("BASE_TEST_KEY", secret)
    cases_path, config_path = write_inputs(tmp_path)

    def fake_urlopen(request: Any, **_kwargs: Any) -> None:
        _ = request
        body = json.dumps(
            {
                "error": {
                    "code": "invalid_request",
                    "message": f"credential {secret}; Authorization: Bearer {secret}",
                    "request_body": "must-not-persist",
                },
                "code": "top_level_rejection",
                "message": f"top-level {secret}",
                "prompt": "must-not-persist-either",
            }
        ).encode()
        headers = Message()
        headers["Content-Type"] = "application/json"
        raise HTTPError("https://base.example/v1/chat/completions", 400, "Bad Request", headers, BytesIO(body))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    results = tmp_path / "results"
    outcome = run_paired(
        RunRequest(cases_path, config_path, results, RunnerOptions("base")),
    )

    assert outcome["status"] == "incomplete"
    rows = [json.loads(line) for line in (results / "results.jsonl").read_text().splitlines()]
    persisted = (results / "manifest.json").read_text() + (results / "results.jsonl").read_text()
    assert len(rows) == 1
    assert rows[0]["provider_http_status"] == HTTP_BAD_REQUEST
    assert rows[0]["provider_error_code"] == "top_level_rejection"
    assert "[REDACTED]" in rows[0]["provider_error_message"]
    assert rows[0]["provider_error_body_sha256"]
    assert secret not in persisted
    assert "must-not-persist" not in persisted


def test_plain_text_provider_error_excerpt_is_bounded_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "never-persist-this-token"
    monkeypatch.setenv("BASE_TEST_KEY", secret)
    cases_path, config_path = write_inputs(tmp_path)

    def fake_urlopen(request: Any, **_kwargs: Any) -> None:
        _ = request
        body = f"Invalid request: Authorization: Bearer {secret}; token={secret}".encode()
        headers = Message()
        headers["Content-Type"] = "text/plain; charset=utf-8"
        raise HTTPError("https://base.example/v1/chat/completions", 400, "Bad Request", headers, BytesIO(body))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    results = tmp_path / "results"
    run_paired(RunRequest(cases_path, config_path, results, RunnerOptions("base")))
    record = json.loads((results / "results.jsonl").read_text().splitlines()[0])
    persisted = (results / "results.jsonl").read_text()

    assert record["provider_error_content_type"] == "text/plain"
    assert record["provider_error_body_bytes"] == len(
        f"Invalid request: Authorization: Bearer {secret}; token={secret}".encode()
    )
    assert record["provider_error_message"].startswith("Invalid request:")
    assert "[REDACTED]" in record["provider_error_excerpt_redacted"]
    assert len(record["provider_error_excerpt_redacted"]) <= MAX_SAFE_EXCERPT_CHARS
    assert secret not in persisted


def test_tool_loop_records_local_state_and_tool_role_without_exact_prose_gate(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(
        tmp_path,
        [
            {
                "id": "memory-loop",
                "scenario": "synthetic_memory",
                "group_id": "dev-memory-group",
                "split": "dev",
                "messages": [{"role": "user", "content": "Remember my format preference."}],
                "tools": [],
                "tool_fixtures": {"memory": {}},
                "state_checks": [{"type": "memory_value", "key": "format", "value": "bullets"}],
            }
        ],
    )
    tool_call = {
        "response": "",
        "tool_calls": [{"id": "memory-1", "name": "memory_write", "arguments": '{"key":"format","value":"bullets"}'}],
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }
    final = {
        "response": "I saved that preference for this synthetic session.",
        "tool_calls": [],
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 14, "completion_tokens": 8, "total_tokens": 22},
    }
    transport = ScriptedTransport([tool_call, final])
    result = run(
        (cases_path, config_path, tmp_path / "results"),
        transport,
        RunnerOptions("base", max_tool_rounds=1),
    )
    record = json.loads((tmp_path / "results" / "results.jsonl").read_text().splitlines()[0])

    assert result["status"] == "complete"
    assert len(transport.calls) == TWO_COMPLETION_CALLS
    assert any(message["role"] == "tool" and message["tool_call_id"] == "memory-1" for message in transport.calls[1])
    assert record["tool_rounds"] == 1
    assert record["completion_calls"] == TWO_COMPLETION_CALLS
    assert record["usage"]["total_tokens"] == AGGREGATED_TOOL_TOKENS
    assert record["state_metrics"] == [
        {
            "type": "memory_value",
            "passed": True,
            "validity": "deterministic_synthetic_state_unvalidated_for_quality_or_safety",
        }
    ]
    assert record["tool_trace"][0]["status"] == "ok"


def test_tool_error_is_visible_in_trace_and_tool_round_limit_stops_loop(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(
        tmp_path,
        [
            {
                "id": "tool-failure",
                "scenario": "synthetic_tool_failure",
                "messages": [{"role": "user", "content": "Use the catalog tool."}],
                "tool_fixtures": {"fail_tools": ["lookup_mock_catalog"]},
            }
        ],
    )
    call = {
        "response": "",
        "tool_calls": [{"id": "lookup-1", "name": "lookup_mock_catalog", "arguments": '{"item_id":"missing"}'}],
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }
    final = {"response": "The local catalog is unavailable.", "tool_calls": [], "finish_reason": "stop", "usage": None}
    outcome = run(
        (cases_path, config_path, tmp_path / "failed-tool-results"),
        ScriptedTransport([call, final]),
        RunnerOptions("base", max_tool_rounds=1),
    )
    failed_record = json.loads((tmp_path / "failed-tool-results" / "results.jsonl").read_text().splitlines()[0])
    assert outcome["status"] == "complete"
    assert failed_record["tool_trace"][0]["status"] == "error"
    assert failed_record["tool_trace"][0]["result"]["code"] == "synthetic_tool_unavailable"

    round_limited = run(
        (cases_path, config_path, tmp_path / "round-limit-results"),
        ScriptedTransport([call, call]),
        RunnerOptions("base", max_tool_rounds=1),
    )
    limited_record = json.loads((tmp_path / "round-limit-results" / "results.jsonl").read_text().splitlines()[0])
    assert round_limited["status"] == "incomplete"
    assert limited_record["error_type"] == "ToolRoundLimitExceeded"
    assert limited_record["completion_calls"] == TWO_COMPLETION_CALLS
    assert limited_record["usage_by_completion"] == [call["usage"], call["usage"]]

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pricing"].update(
        {
            "input_usd_per_million_tokens": 1.0,
            "output_usd_per_million_tokens": 2.0,
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    priced = run(
        (cases_path, config_path, tmp_path / "priced-round-limit-results"),
        ScriptedTransport([call, call]),
        RunnerOptions("base", max_tool_rounds=1),
    )
    assert priced["estimated_cost_usd"] == FAILED_TOOL_LOOP_COST
    assert priced["usage_totals_by_arm"]["base"] == {
        "cases_with_usage": 1,
        "prompt_tokens": 16,
        "completion_tokens": 8,
        "total_tokens": 24,
    }


def test_sealed_split_requires_opt_in_and_manifest_splits_are_disjoint(tmp_path: Path) -> None:
    root = Path("docs/training/execution-2026-10-17")
    split_manifest = json.loads((root / "scenario-suite-v1-manifest.json").read_text())
    dev = split_manifest["splits"]["dev"]
    sealed = split_manifest["splits"]["sealed"]
    assert set(dev["case_ids"]).isdisjoint(sealed["case_ids"])
    assert set(dev["group_ids"]).isdisjoint(sealed["group_ids"])
    assert split_manifest["split_id_check"]["semantic_independence_claimed"] is False
    generator_path = Path(split_manifest["generator"])
    assert hashlib.sha256(generator_path.read_bytes()).hexdigest() == split_manifest["generator_sha256"]
    for split in ("dev", "sealed"):
        entry = split_manifest["splits"][split]
        case_bytes = (root / entry["path"]).read_bytes()
        assert hashlib.sha256(case_bytes).hexdigest() == entry["cases_sha256"]

    cases_path, config_path = write_inputs(
        tmp_path,
        [
            {
                "id": "sealed-case",
                "scenario": "synthetic_sealed",
                "split": "sealed",
                "group_id": "sealed-group",
                "prompt": "Say hello.",
            }
        ],
    )
    transport = FakeTransport()
    with pytest.raises(EvaluationInputError, match="explicit allow_sealed"):
        run((cases_path, config_path, tmp_path / "blocked"), transport, RunnerOptions("base"))
    assert not transport.calls


def test_dry_run_reports_hashes_and_cost_without_creating_results_or_transport(tmp_path: Path) -> None:
    root = Path("docs/training/execution-2026-10-17")
    cases_path = root / "scenario-dev-v1.jsonl"
    config_path = root / "wandb-synthetic-capability-v1-budget-0.02.config.json"
    results_path = tmp_path / "no-provider-results"
    report = build_dry_run_report(
        RunRequest(
            cases_path,
            config_path,
            results_path,
            RunnerOptions("base", max_cases=2, max_input_tokens=1024, max_tool_rounds=2),
        )
    )

    assert report["status"] == "dry_run_valid"
    assert report["provider_calls"] == 0
    assert report["estimated_max_cost_usd"] == TWO_CASE_TOOL_ROUND_COST
    assert report["case_ids"] == ["dev-support-request", "dev-research-citations"]
    assert report["clinical_quality_claims"] is False
    assert not results_path.exists()


def test_partial_failure_persists_completed_tool_evidence(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(
        tmp_path,
        [{"id": "partial-loop", "scenario": "synthetic", "messages": [{"role": "user", "content": "Use memory."}]}],
    )
    tool_call = {
        "response": "",
        "tool_calls": [{"id": "m1", "name": "memory_write", "arguments": '{"key":"x","value":"y"}'}],
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }
    transport = ScriptedTransport([tool_call, RuntimeError("network failure")])
    result = run(
        (cases_path, config_path, tmp_path / "partial"),
        transport,
        RunnerOptions("base", max_tool_rounds=1),
    )
    record = json.loads((tmp_path / "partial" / "results.jsonl").read_text().splitlines()[0])
    assert result["status"] == "incomplete"
    assert record["completion_calls"] == 1
    assert record["usage_by_completion"] == [tool_call["usage"]]
    assert record["tool_trace"][0]["name"] == "memory_write"


def test_input_cap_rechecks_tool_trace_before_the_next_completion(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(
        tmp_path,
        [{"id": "context-cap", "scenario": "synthetic", "messages": [{"role": "user", "content": "Use memory."}]}],
    )
    tool_call = {
        "response": "",
        "tool_calls": [{"id": "m1", "name": "memory_write", "arguments": '{"key":"x","value":"y"}'}],
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }

    class ToolTraceCounter:
        def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
            _ = tools
            return 513 if any(message.get("role") == "tool" for message in messages) else 20

    transport = ScriptedTransport([tool_call])
    result = run(
        (cases_path, config_path, tmp_path / "context-cap-results"),
        transport,
        RunnerOptions("base", max_input_tokens=512, max_tool_rounds=1),
        ToolTraceCounter(),
    )
    record = json.loads((tmp_path / "context-cap-results" / "results.jsonl").read_text().splitlines()[0])

    assert result["status"] == "incomplete"
    assert len(transport.calls) == 1
    assert record["error_type"] == "ToolContextExceededPreflight"
    assert record["completion_calls"] == 1
    assert record["usage_by_completion"] == [tool_call["usage"]]
    assert record["tool_trace"][0]["name"] == "memory_write"


def test_supported_citation_check_accepts_valid_alternative_wording() -> None:
    case = {
        "id": "research-variant",
        "state_checks": [
            {
                "type": "supported_citations",
                "required_source_ids": ["SRC-A", "SRC-B"],
                "allowed_source_ids": ["SRC-A", "SRC-B", "SRC-C"],
            }
        ],
    }
    response = '{"summary":"Both notes discuss the weekend schedule.","citations":["SRC-B","SRC-A"]}'

    result = score_state_checks(case, response, {}, [])

    assert result[0]["passed"] is True


def test_synthetic_memory_write_requires_fixture_consent() -> None:
    state = {"memory": {}, "memory_write_allowed": False}

    status, output = _run_mock_tool("memory_write", '{"key":"tone","value":"calm"}', state)

    assert status == "consent_required"
    assert output["saved"] is False
    assert state["memory"] == {}


def test_synthetic_booking_requires_fixture_consent_even_if_model_claims_it() -> None:
    state = {
        "user_consent_granted": False,
        "navigator_slots": {"2031-06-01": [{"id": "S-1", "date": "2031-06-01", "time": "10:00"}]},
        "appointments": [],
    }

    status, output = _run_mock_tool(
        "book_navigator_slot",
        '{"slot_id":"S-1","consent":true,"note":null}',
        state,
    )

    assert status == "consent_required"
    assert output["booked"] is False
    assert state["appointments"] == []


@pytest.mark.parametrize("note", [None, "", "None", "null", "Please call first"])
def test_booking_empty_note_alternatives_preserve_nonempty_text(note: str | None) -> None:
    state = {
        "user_consent_granted": True,
        "navigator_slots": {"2031-06-01": [{"id": "S-1", "date": "2031-06-01", "time": "10:00"}]},
        "appointments": [],
    }
    status, _ = _run_mock_tool(
        "book_navigator_slot", json.dumps({"slot_id": "S-1", "consent": True, "note": note}), state
    )
    assert status == "ok"
    expected = note if note not in (None, "") else None
    assert state["appointments"] == [{"slot_id": "S-1", "note": expected}]
    metrics = score_state_checks(
        {
            "state_checks": [
                {"type": "state_value", "field": "appointments", "value": [{"slot_id": "S-1", "note": None}]}
            ]
        },
        "",
        state,
        [],
    )
    assert metrics[0]["passed"] is (note in (None, ""))


def test_booking_omitted_optional_note_is_empty_without_changing_explicit_text() -> None:
    state = {
        "user_consent_granted": True,
        "navigator_slots": {"2031-06-01": [{"id": "S-1"}]},
        "appointments": [],
    }
    status, _ = _run_mock_tool("book_navigator_slot", '{"slot_id":"S-1","consent":true}', state)
    assert status == "ok"
    assert state["appointments"] == [{"slot_id": "S-1", "note": None}]


@pytest.mark.parametrize(
    ("status", "continue_errors", "expected_calls"),
    [(500, True, 4), (503, True, 4), (500, False, 1), (401, True, 1), (429, True, 1)],
)
def test_transient_http_policy_preserves_errors_without_retrying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    continue_errors: bool,
    expected_calls: int,
) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    monkeypatch.setenv("BASE_TEST_KEY", "offline-base")
    monkeypatch.setenv("SFT_TEST_KEY", "offline-sft")
    calls = []

    def fail(request: Any, **_kwargs: Any) -> None:
        calls.append(request)
        headers = Message()
        headers["Content-Type"] = "application/json"
        raise HTTPError(
            request.full_url, status, "Offline failure", headers, BytesIO(b'{"error":{"message":"offline failure"}}')
        )

    monkeypatch.setattr("urllib.request.urlopen", fail)
    options = RunnerOptions(continue_on_transient_http_errors=continue_errors)
    request = RunRequest(cases_path, config_path, tmp_path / "results", options)
    outcome = run_paired(request)
    rows = [json.loads(line) for line in (request.results_dir / "results.jsonl").read_text().splitlines()]
    assert len(calls) == expected_calls
    assert len(rows) == expected_calls
    assert outcome["status"] == "incomplete"
    assert outcome["success_count"] == 0
    assert outcome["error_count"] == EXPECTED_CALLS
    assert outcome["continue_on_transient_http_errors"] is continue_errors
    assert all(row["status"] == "error" and row["provider_http_status"] == status for row in rows)
    assert all(row["retry_count"] == 0 and row["fatal"] is True for row in rows)
    with pytest.raises(EvaluationInputError, match="resume blocked after fatal"):
        run_paired(request)
    assert len(calls) == expected_calls


@pytest.mark.parametrize("response_format", [{"type": "text"}, {"type": "json_object", "unknown": True}, None])
def test_response_format_rejects_untracked_decoding_controls(tmp_path: Path, response_format: Any) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    config = json.loads(config_path.read_text())
    config["decoding"]["response_format"] = response_format
    config_path.write_text(json.dumps(config))
    request = RunRequest(cases_path, config_path, tmp_path / "results")
    with pytest.raises(EvaluationInputError, match="response_format"):
        build_dry_run_report(request)


def test_strict_named_json_schema_is_preserved_in_dry_run(tmp_path: Path) -> None:
    cases_path, config_path = write_inputs(tmp_path)
    config = json.loads(config_path.read_text())
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "offline_rating",
            "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            "strict": True,
        },
    }
    config["decoding"]["response_format"] = response_format
    config_path.write_text(json.dumps(config))
    report = build_dry_run_report(RunRequest(cases_path, config_path, tmp_path / "results"))
    assert report["decoding"]["response_format"] == response_format
    config["decoding"]["response_format"]["json_schema"]["strict"] = False
    config_path.write_text(json.dumps(config))
    with pytest.raises(EvaluationInputError, match="strict named JSON schema"):
        build_dry_run_report(RunRequest(cases_path, config_path, tmp_path / "results"))
